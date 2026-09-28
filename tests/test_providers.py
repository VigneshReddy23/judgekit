from typing import Any

from judgekit.providers import BedrockProvider


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


def test_bedrock_omits_empty_system_prompt() -> None:
    client = FakeBedrockClient()
    BedrockProvider("test-model", client=client).complete("hi")
    assert "system" not in client.requests[0]


def test_bedrock_accumulates_token_usage() -> None:
    provider = BedrockProvider("test-model", client=FakeBedrockClient())
    provider.complete("a")
    provider.complete("b")
    assert (provider.input_tokens, provider.output_tokens) == (24, 6)
