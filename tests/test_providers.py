import json
import threading
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from types import SimpleNamespace
from typing import Any

import pytest
from pydantic import ValidationError
from typer.testing import CliRunner

from judgekit import cli
from judgekit.providers import (
    AnthropicProvider,
    BedrockProvider,
    ModelConfig,
    OpenAICompatibleProvider,
    make_provider,
)

# --- Bedrock -----------------------------------------------------------------


class FakeBedrockClient:
    """Stands in for boto3's bedrock-runtime client; records each request."""

    def __init__(self) -> None:
        self.requests: list[dict[str, Any]] = []

    def converse(self, **kwargs: Any) -> dict[str, Any]:
        self.requests.append(kwargs)
        return {
            "output": {"message": {"role": "assistant", "content": [{"text": "hello"}]}},
            "usage": {"inputTokens": 12, "outputTokens": 3},
            "stopReason": "end_turn",
        }


def test_bedrock_request_shape() -> None:
    client = FakeBedrockClient()
    provider = BedrockProvider("test-model", client=client)
    assert provider.complete("hi", system="be strict") == "hello"
    assert client.requests[0] == {
        "modelId": "test-model",
        "messages": [{"role": "user", "content": [{"text": "hi"}]}],
        "inferenceConfig": {"temperature": 0.0, "maxTokens": 1024},
        "system": [{"text": "be strict"}],
    }


def test_bedrock_omits_empty_system_and_null_temperature() -> None:
    client = FakeBedrockClient()
    BedrockProvider("test-model", temperature=None, client=client).complete("hi")
    assert "system" not in client.requests[0]
    assert client.requests[0]["inferenceConfig"] == {"maxTokens": 1024}


def test_bedrock_accumulates_token_usage() -> None:
    provider = BedrockProvider("test-model", client=FakeBedrockClient())
    provider.complete("a")
    provider.complete("b")
    assert (provider.input_tokens, provider.output_tokens) == (24, 6)


# --- Anthropic ---------------------------------------------------------------


class FakeAnthropicClient:
    """Mimics `anthropic.Anthropic().messages.create` closely enough for our use."""

    def __init__(self, stop_reason: str = "end_turn") -> None:
        self.requests: list[dict[str, Any]] = []
        self.stop_reason = stop_reason
        self.messages = SimpleNamespace(create=self._create)

    def _create(self, **kwargs: Any) -> SimpleNamespace:
        self.requests.append(kwargs)
        return SimpleNamespace(
            stop_reason=self.stop_reason,
            usage=SimpleNamespace(input_tokens=20, output_tokens=4),
            content=[
                SimpleNamespace(type="thinking", thinking=""),  # non-text blocks are skipped
                SimpleNamespace(type="text", text="he"),
                SimpleNamespace(type="text", text="llo"),
            ],
        )


def test_anthropic_request_shape_and_text() -> None:
    client = FakeAnthropicClient()
    provider = AnthropicProvider("claude-haiku-4-5", client=client)
    assert provider.complete("hi", system="be strict") == "hello"
    assert client.requests[0] == {
        "model": "claude-haiku-4-5",
        "max_tokens": 1024,
        "messages": [{"role": "user", "content": "hi"}],
        "system": "be strict",
        "extra_body": {"temperature": 0.0},  # SDK 1.x has no temperature kwarg
    }
    assert (provider.input_tokens, provider.output_tokens) == (20, 4)


def test_anthropic_null_temperature_and_no_system() -> None:
    client = FakeAnthropicClient()
    AnthropicProvider("claude-opus-5-5", temperature=None, client=client).complete("hi")
    assert "extra_body" not in client.requests[0]  # newer models reject sampling params
    assert "system" not in client.requests[0]


def test_anthropic_refusal_raises() -> None:
    provider = AnthropicProvider("m", client=FakeAnthropicClient(stop_reason="refusal"))
    with pytest.raises(RuntimeError, match="refusal"):
        provider.complete("hi")


# --- OpenAI-compatible: fake transport -----------------------------------------


def chat_response(text: str | None = "hello") -> dict[str, Any]:
    return {
        "choices": [{"message": {"role": "assistant", "content": text}}],
        "usage": {"prompt_tokens": 30, "completion_tokens": 5},
    }


def test_openai_compatible_request_shape() -> None:
    sent: list[tuple[str, dict[str, str], dict[str, Any]]] = []

    def post(url: str, headers: dict[str, str], body: dict[str, Any]) -> dict[str, Any]:
        sent.append((url, headers, body))
        return chat_response()

    provider = OpenAICompatibleProvider(
        "gpt-x", "https://api.example.com/v1/", api_key="sk-test", post=post
    )
    assert provider.complete("hi", system="be strict") == "hello"
    url, headers, body = sent[0]
    assert url == "https://api.example.com/v1/chat/completions"  # trailing slash handled
    assert headers["Authorization"] == "Bearer sk-test"
    assert body == {
        "model": "gpt-x",
        "messages": [
            {"role": "system", "content": "be strict"},
            {"role": "user", "content": "hi"},
        ],
        "max_tokens": 1024,
        "temperature": 0.0,
    }
    assert (provider.input_tokens, provider.output_tokens) == (30, 5)


def test_openai_compatible_no_key_no_system_null_temperature() -> None:
    sent: list[tuple[dict[str, str], dict[str, Any]]] = []

    def post(url: str, headers: dict[str, str], body: dict[str, Any]) -> dict[str, Any]:
        sent.append((headers, body))
        return {"choices": [{"message": {"content": None}}]}  # null content, no usage

    provider = OpenAICompatibleProvider(
        "llama", "http://localhost:11434/v1", temperature=None, post=post
    )
    assert provider.complete("hi") == ""
    headers, body = sent[0]
    assert "Authorization" not in headers  # local servers need no key
    assert body["messages"] == [{"role": "user", "content": "hi"}]
    assert "temperature" not in body
    assert (provider.input_tokens, provider.output_tokens) == (0, 0)


# --- OpenAI-compatible: real HTTP against a local server ------------------------


class ScriptedHandler(BaseHTTPRequestHandler):
    """Replies with the next (status, body) from `script`; records every request."""

    script: list[tuple[int, dict[str, Any]]] = []
    seen: list[dict[str, Any]] = []

    def do_POST(self) -> None:
        length = int(self.headers["Content-Length"])
        ScriptedHandler.seen.append(
            {
                "path": self.path,
                "auth": self.headers.get("Authorization"),
                "body": json.loads(self.rfile.read(length)),
            }
        )
        status, body = ScriptedHandler.script.pop(0)
        payload = json.dumps(body).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, format: str, *args: Any) -> None:  # keep test output quiet
        pass


@pytest.fixture
def server() -> Iterator[str]:
    ScriptedHandler.script, ScriptedHandler.seen = [], []
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), ScriptedHandler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{httpd.server_address[1]}/v1"
    httpd.shutdown()


def test_http_retries_server_errors_then_succeeds(server: str) -> None:
    ScriptedHandler.script = [
        (503, {"error": "busy"}),
        (429, {"error": "slow down"}),
        (200, chat_response("ok")),
    ]
    provider = OpenAICompatibleProvider("m", server, api_key="sk-secret", backoff_seconds=0)
    assert provider.complete("hi") == "ok"
    assert len(ScriptedHandler.seen) == 3
    assert ScriptedHandler.seen[0]["path"] == "/v1/chat/completions"
    assert ScriptedHandler.seen[0]["auth"] == "Bearer sk-secret"
    assert ScriptedHandler.seen[0]["body"]["model"] == "m"


def test_http_client_error_is_not_retried_and_hides_key(server: str) -> None:
    ScriptedHandler.script = [(401, {"error": "invalid api key"})]
    provider = OpenAICompatibleProvider("m", server, api_key="sk-secret", backoff_seconds=0)
    with pytest.raises(RuntimeError) as exc_info:
        provider.complete("hi")
    assert "HTTP 401" in str(exc_info.value) and "invalid api key" in str(exc_info.value)
    assert "sk-secret" not in str(exc_info.value)
    assert len(ScriptedHandler.seen) == 1


def test_http_gives_up_after_max_retries(server: str) -> None:
    ScriptedHandler.script = [(500, {"error": "boom"})] * 3
    provider = OpenAICompatibleProvider("m", server, max_retries=2, backoff_seconds=0)
    with pytest.raises(RuntimeError, match="HTTP 500"):
        provider.complete("hi")
    assert len(ScriptedHandler.seen) == 3  # first try + 2 retries


def test_http_unreachable_server() -> None:
    provider = OpenAICompatibleProvider(
        "m", "http://127.0.0.1:9/v1", max_retries=1, backoff_seconds=0
    )
    with pytest.raises(RuntimeError, match="cannot reach"):
        provider.complete("hi")


# --- ModelConfig and make_provider ---------------------------------------------


@pytest.mark.parametrize(
    ("fields", "message"),
    [
        ({"provider": "openai_compatible"}, "needs base_url"),
        (
            {"provider": "bedrock", "base_url": "http://x"},
            "only used with provider: openai_compatible",
        ),
        ({"provider": "anthropic", "region": "us-east-1"}, "region is only used"),
        ({"provider": "bedrock", "api_key_env": "KEY"}, "AWS credentials"),
        ({"provider": "gemini"}, "Input should be"),
        ({"api_key": "sk-oops"}, "Extra inputs"),  # raw keys are rejected outright
    ],
)
def test_invalid_model_configs(fields: dict[str, Any], message: str) -> None:
    with pytest.raises(ValidationError, match=message):
        ModelConfig.model_validate({"model": "m", **fields})


def test_anthropic_defaults_to_standard_key_env() -> None:
    assert ModelConfig(provider="anthropic", model="m").api_key_env == "ANTHROPIC_API_KEY"


def test_make_provider_reads_key_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MY_KEY", "sk-from-env")
    config = ModelConfig(
        provider="openai_compatible", model="m", base_url="http://x/v1", api_key_env="MY_KEY"
    )
    provider = make_provider(config)
    assert isinstance(provider, OpenAICompatibleProvider)
    assert provider.api_key == "sk-from-env"


def test_make_provider_missing_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    with pytest.raises(ValueError, match="ANTHROPIC_API_KEY is not set"):
        make_provider(ModelConfig(provider="anthropic", model="m"))


def test_make_provider_builds_each_kind(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
    anthropic_provider = make_provider(
        ModelConfig(provider="anthropic", model="a", temperature=None)
    )
    assert isinstance(anthropic_provider, AnthropicProvider)
    assert anthropic_provider.temperature is None
    bedrock_provider = make_provider(ModelConfig(model="b", region="us-east-1", max_tokens=99))
    assert isinstance(bedrock_provider, BedrockProvider)
    assert (bedrock_provider.model_id, bedrock_provider.max_tokens) == ("b", 99)
    local = make_provider(ModelConfig(provider="openai_compatible", model="c", base_url="http://x"))
    assert isinstance(local, OpenAICompatibleProvider) and local.api_key is None


# --- CLI: provider problems are config errors (exit 2) --------------------------


def test_cli_run_missing_api_key(tmp_path: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    cases = tmp_path / "c.jsonl"
    cases.write_text('{"id": "a", "input": "q", "output": "hi"}\n')
    suite = tmp_path / "s.yaml"
    suite.write_text(
        f"name: t\ncases_file: {cases}\njudges: [toxicity]\n"
        "judge: {provider: anthropic, model: claude-haiku-4-5}\n"
    )
    result = CliRunner().invoke(cli.app, ["run", str(suite)])
    assert result.exit_code == 2
    assert "ANTHROPIC_API_KEY is not set" in result.output


def test_cli_calibrate_bad_provider(tmp_path: Any) -> None:
    data = tmp_path / "l.jsonl"
    data.write_text('{"id": "a", "input": "q", "output": "x", "human_label": "pass"}\n')
    result = CliRunner().invoke(
        cli.app, ["calibrate", "--judge", "toxicity", "--data", str(data), "--provider", "gemini"]
    )
    assert result.exit_code == 2
    assert "config error" in result.output
