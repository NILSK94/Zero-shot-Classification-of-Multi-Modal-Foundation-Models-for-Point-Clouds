from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from PIL import Image

from mmfm_tool.providers import ProviderError, build_prompt, create_provider, validate_label


def test_label_protocol_accepts_only_one_bare_closed_set_label() -> None:
    labels = ("chair", "flower_pot", "table")

    assert validate_label("chair\n", labels) == "chair"
    assert validate_label("FLOWER_POT", labels) == "flower_pot"

    for invalid in ('"chair"', '{"label":"chair"}', "chair because it has legs", "flower pot"):
        with pytest.raises(ProviderError):
            validate_label(invalid, labels)


def test_prompt_requires_label_only_and_never_requests_ground_truth() -> None:
    prompt = build_prompt("modelnet", ("chair", "table"), view_count=4)

    assert "exactly the selected label" in prompt
    assert "no JSON" in prompt
    assert "confidence" in prompt  # Explicitly prohibited.
    assert "true_label" not in prompt
    assert "Ground Truth" not in prompt


def test_litellm_uses_responses_api_with_images_and_label_only(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    captured: dict[str, object] = {}

    class FakeResponses:
        def create(self, **request: object) -> SimpleNamespace:
            captured["request"] = request
            return SimpleNamespace(
                output_text="chair",
                model="qwen3.6-35b-a3b-awq",
                usage={"input_tokens": 10, "output_tokens": 1},
                status="completed",
            )

    class FakeOpenAI:
        def __init__(self, **config: object) -> None:
            captured["config"] = config
            self.responses = FakeResponses()

    monkeypatch.setitem(sys.modules, "openai", SimpleNamespace(OpenAI=FakeOpenAI))
    image_path = tmp_path / "view.png"
    Image.new("RGB", (16, 16), (10, 20, 30)).save(image_path)

    provider = create_provider(
        "litellm",
        "qwen3.6-35b-a3b-awq",
        api_key="secret-for-test",
        base_url="https://proxy.example.test/v1/",
    )
    result = provider.classify([image_path], "Choose one label.", ("chair", "table"))

    assert result.label == "chair"
    assert captured["config"] == {
        "api_key": "secret-for-test",
        "base_url": "https://proxy.example.test/v1",
    }
    request = captured["request"]
    assert isinstance(request, dict)
    assert request["model"] == "qwen3.6-35b-a3b-awq"
    assert request["store"] is False
    assert request["input"][1]["content"][1]["type"] == "input_image"
