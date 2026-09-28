"""LLM providers behind one tiny interface.

Everything that talks to a model depends on the `Provider` Protocol, not a
concrete class, so tests swap in `FakeProvider` with no network or API keys.
"""

import threading
from typing import Any, Protocol

import boto3
from botocore.config import Config

# Claude Haiku 4.5 via a US cross-region inference profile.
# VERIFY this exact ID in the AWS console: Bedrock -> Model catalog / Inference profiles.
# It varies by region prefix (us./eu./apac.) and must be enabled for your account.
DEFAULT_JUDGE_MODEL_ID = "us.anthropic.claude-haiku-4-5-20251001-v1:0"


class Provider(Protocol):
    """Anything with `complete(prompt, system) -> str` is a provider (structural typing)."""

    def complete(self, prompt: str, system: str = "") -> str: ...


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


class BedrockProvider:
    """Calls a model on Amazon Bedrock through the `converse` API.

    `converse` is Bedrock's model-agnostic chat API, so swapping Claude for
    another Bedrock model is a config change, not a code change.
    """

    def __init__(
        self,
        model_id: str,
        region: str | None = None,
        temperature: float = 0.0,
        max_tokens: int = 1024,
        client: Any = None,
    ) -> None:
        self.model_id = model_id
        self.temperature = temperature
        self.max_tokens = max_tokens
        # `client` is injectable so tests can pass a fake and never touch AWS.
        # Adaptive retries back off automatically when Bedrock throttles us.
        self.client = client or boto3.client(
            "bedrock-runtime",
            region_name=region,
            config=Config(retries={"max_attempts": 5, "mode": "adaptive"}, read_timeout=60),
        )
        # Running token totals, used later to estimate cost.
        self.input_tokens = 0
        self.output_tokens = 0
        self._lock = threading.Lock()

    def complete(self, prompt: str, system: str = "") -> str:
        request: dict[str, Any] = {
            "modelId": self.model_id,
            "messages": [{"role": "user", "content": [{"text": prompt}]}],
            "inferenceConfig": {"temperature": self.temperature, "maxTokens": self.max_tokens},
        }
        if system:
            request["system"] = [{"text": system}]

        response = self.client.converse(**request)

        usage = response.get("usage", {})
        with self._lock:
            self.input_tokens += int(usage.get("inputTokens", 0))
            self.output_tokens += int(usage.get("outputTokens", 0))

        blocks = response["output"]["message"]["content"]
        return "".join(str(block.get("text", "")) for block in blocks)
