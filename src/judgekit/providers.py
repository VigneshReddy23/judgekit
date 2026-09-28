"""LLM providers behind one tiny interface, plus a config model to pick one.

Everything that talks to a model depends on the `Provider` Protocol, so any
backend works: AWS Bedrock, Anthropic's API, or any server that speaks the
OpenAI-compatible /chat/completions format (OpenAI, Ollama, vLLM, Groq,
Together, OpenRouter, LM Studio, ...). Tests use `FakeProvider`: no network,
no keys.

API keys never appear in config files: a suite names the *environment
variable* that holds the key (e.g. `api_key_env: OPENAI_API_KEY`).
"""

import json
import os
import threading
import time
import urllib.error
import urllib.request
from collections.abc import Callable
from typing import Any, Literal, Protocol

import anthropic
import boto3
from botocore.config import Config
from pydantic import BaseModel, ConfigDict, model_validator

# Claude Haiku 4.5 via a US cross-region inference profile on Bedrock.
# VERIFY in the AWS console: Bedrock -> Cross-region inference.
DEFAULT_JUDGE_MODEL_ID = "us.anthropic.claude-haiku-4-5-20251001-v1:0"

ProviderName = Literal["bedrock", "anthropic", "openai_compatible"]


class Provider(Protocol):
    """Anything with `complete(prompt, system) -> str` is a provider (structural typing)."""

    def complete(self, prompt: str, system: str = "") -> str: ...


class MeteredProvider(Provider, Protocol):
    """A real provider that also reports which model it calls and tokens used."""

    model_id: str
    input_tokens: int
    output_tokens: int


class FakeProvider:
    """Returns canned responses in order, cycling when it runs out.

    Records every call so tests can assert on the exact prompt a judge sent.
    A lock keeps it safe under the runner's ThreadPoolExecutor.
    """

    def __init__(self, responses: list[str]) -> None:
        if not responses:
            raise ValueError("FakeProvider needs at least one response")
        self.responses = responses
        self.calls: list[tuple[str, str]] = []
        self._lock = threading.Lock()

    def complete(self, prompt: str, system: str = "") -> str:
        with self._lock:
            response = self.responses[len(self.calls) % len(self.responses)]
            self.calls.append((prompt, system))
        return response


class _TokenMeter:
    """Shared by real providers: thread-safe running token totals for cost estimates."""

    def __init__(self, model_id: str) -> None:
        self.model_id = model_id
        self.input_tokens = 0
        self.output_tokens = 0
        self._lock = threading.Lock()

    def _record(self, input_tokens: int, output_tokens: int) -> None:
        with self._lock:
            self.input_tokens += input_tokens
            self.output_tokens += output_tokens


class BedrockProvider(_TokenMeter):
    """Calls a model on Amazon Bedrock through the model-agnostic `converse` API."""

    def __init__(
        self,
        model_id: str,
        region: str | None = None,
        temperature: float | None = 0.0,
        max_tokens: int = 1024,
        client: Any = None,
    ) -> None:
        super().__init__(model_id)
        self.temperature = temperature
        self.max_tokens = max_tokens
        # `client` is injectable so tests can pass a fake and never touch AWS.
        # Adaptive retries back off automatically when Bedrock throttles us.
        self.client = client or boto3.client(
            "bedrock-runtime",
            region_name=region,
            config=Config(retries={"max_attempts": 5, "mode": "adaptive"}, read_timeout=60),
        )

    def complete(self, prompt: str, system: str = "") -> str:
        inference: dict[str, Any] = {"maxTokens": self.max_tokens}
        if self.temperature is not None:
            inference["temperature"] = self.temperature
        request: dict[str, Any] = {
            "modelId": self.model_id,
            "messages": [{"role": "user", "content": [{"text": prompt}]}],
            "inferenceConfig": inference,
        }
        if system:
            request["system"] = [{"text": system}]

        response = self.client.converse(**request)
        usage = response.get("usage", {})
        self._record(int(usage.get("inputTokens", 0)), int(usage.get("outputTokens", 0)))
        blocks = response["output"]["message"]["content"]
        return "".join(str(block.get("text", "")) for block in blocks)


class AnthropicProvider(_TokenMeter):
    """Calls Claude through Anthropic's own API with the official SDK."""

    def __init__(
        self,
        model_id: str,
        api_key: str | None = None,
        temperature: float | None = 0.0,
        max_tokens: int = 1024,
        client: Any = None,
    ) -> None:
        super().__init__(model_id)
        self.temperature = temperature
        self.max_tokens = max_tokens
        # The SDK retries 429/5xx/connection errors with backoff by itself.
        self.client = client or anthropic.Anthropic(api_key=api_key, max_retries=4)

    def complete(self, prompt: str, system: str = "") -> str:
        request: dict[str, Any] = {
            "model": self.model_id,
            "max_tokens": self.max_tokens,
            "messages": [{"role": "user", "content": prompt}],
        }
        if system:
            request["system"] = system
        if self.temperature is not None:
            # SDK 1.x has no temperature argument. Older models (e.g. Haiku 4.5)
            # still honour it via extra_body; newer ones reject it, so set
            # `temperature: null` in the suite for those.
            request["extra_body"] = {"temperature": self.temperature}

        response = self.client.messages.create(**request)
        self._record(response.usage.input_tokens, response.usage.output_tokens)
        if response.stop_reason == "refusal":
            raise RuntimeError("model declined the request (stop_reason=refusal)")
        return "".join(block.text for block in response.content if block.type == "text")


PostFn = Callable[[str, dict[str, str], dict[str, Any]], dict[str, Any]]
RETRYABLE_STATUS = {408, 409, 429, 500, 502, 503, 504}


class OpenAICompatibleProvider(_TokenMeter):
    """Calls any server implementing the OpenAI-style POST {base_url}/chat/completions.

    Plain HTTP with the standard library: this one "template" covers OpenAI,
    Ollama, vLLM, Groq, Together, OpenRouter, LM Studio and more.
    """

    def __init__(
        self,
        model_id: str,
        base_url: str,
        api_key: str | None = None,
        temperature: float | None = 0.0,
        max_tokens: int = 1024,
        max_retries: int = 4,
        backoff_seconds: float = 1.0,
        timeout_seconds: float = 60.0,
        post: PostFn | None = None,
    ) -> None:
        super().__init__(model_id)
        self.url = base_url.rstrip("/") + "/chat/completions"
        self.api_key = api_key
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.max_retries = max_retries
        self.backoff_seconds = backoff_seconds
        self.timeout_seconds = timeout_seconds
        self.post = post or self._http_post  # injectable for tests

    def complete(self, prompt: str, system: str = "") -> str:
        messages = [{"role": "system", "content": system}] if system else []
        messages.append({"role": "user", "content": prompt})
        body: dict[str, Any] = {
            "model": self.model_id,
            "messages": messages,
            "max_tokens": self.max_tokens,
        }
        if self.temperature is not None:
            body["temperature"] = self.temperature
        headers = {"Content-Type": "application/json"}
        if self.api_key:  # local servers (Ollama, vLLM) usually need no key
            headers["Authorization"] = f"Bearer {self.api_key}"

        data = self.post(self.url, headers, body)
        usage = data.get("usage") or {}
        self._record(int(usage.get("prompt_tokens", 0)), int(usage.get("completion_tokens", 0)))
        return str(data["choices"][0]["message"].get("content") or "")

    def _http_post(self, url: str, headers: dict[str, str], body: dict[str, Any]) -> dict[str, Any]:
        """POST JSON, retrying rate limits, server errors and network errors with backoff."""
        payload = json.dumps(body).encode("utf-8")
        for attempt in range(self.max_retries + 1):
            request = urllib.request.Request(url, data=payload, headers=headers, method="POST")
            try:
                with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
                    result: dict[str, Any] = json.loads(response.read())
                    return result
            except urllib.error.HTTPError as exc:
                if exc.code not in RETRYABLE_STATUS or attempt == self.max_retries:
                    # Body only, never the headers: they contain the API key.
                    detail = exc.read()[:300].decode("utf-8", "replace")
                    raise RuntimeError(f"HTTP {exc.code} from {url}: {detail}") from exc
            except urllib.error.URLError as exc:
                if attempt == self.max_retries:
                    raise RuntimeError(f"cannot reach {url}: {exc.reason}") from exc
            time.sleep(self.backoff_seconds * 2**attempt)  # 1s, 2s, 4s, 8s
        raise AssertionError("unreachable")  # pragma: no cover


class ModelConfig(BaseModel):
    """Which model to call and how. Used for both `judge:` and `target:` in a suite."""

    model_config = ConfigDict(extra="forbid")

    provider: ProviderName = "bedrock"
    model: str
    base_url: str | None = None  # openai_compatible only
    api_key_env: str | None = None  # NAME of the env var holding the key, never the key
    region: str | None = None  # bedrock only; None -> AWS_REGION / your AWS config
    temperature: float | None = 0.0  # null for models that reject sampling params
    max_tokens: int = 1024

    @model_validator(mode="after")
    def _check_provider_fields(self) -> "ModelConfig":
        if self.provider == "openai_compatible" and not self.base_url:
            raise ValueError("openai_compatible needs base_url, e.g. https://api.openai.com/v1")
        if self.provider != "openai_compatible" and self.base_url:
            raise ValueError("base_url is only used with provider: openai_compatible")
        if self.provider != "bedrock" and self.region:
            raise ValueError("region is only used with provider: bedrock")
        if self.provider == "bedrock" and self.api_key_env:
            raise ValueError("bedrock uses AWS credentials, not api_key_env")
        if self.provider == "anthropic" and self.api_key_env is None:
            self.api_key_env = "ANTHROPIC_API_KEY"
        return self


def make_provider(config: ModelConfig) -> MeteredProvider:
    """Build the provider a ModelConfig describes. Raises ValueError if its key is missing."""
    api_key = None
    if config.api_key_env:
        api_key = os.environ.get(config.api_key_env)
        if not api_key:
            raise ValueError(f"environment variable {config.api_key_env} is not set")
    common: dict[str, Any] = {"temperature": config.temperature, "max_tokens": config.max_tokens}
    if config.provider == "bedrock":
        return BedrockProvider(config.model, region=config.region, **common)
    if config.provider == "anthropic":
        return AnthropicProvider(config.model, api_key=api_key, **common)
    assert config.base_url is not None  # guaranteed by the validator
    return OpenAICompatibleProvider(config.model, config.base_url, api_key=api_key, **common)
