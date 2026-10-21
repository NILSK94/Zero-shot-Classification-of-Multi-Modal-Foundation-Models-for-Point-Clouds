from __future__ import annotations

import base64
import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from .constants import PROMPT_VERSION


@dataclass(frozen=True)
class ProviderResult:
    label: str
    raw_response: str
    model_version: str
    usage: dict[str, Any]


class ProviderError(RuntimeError):
    pass


def validate_label(value: str, labels: Sequence[str]) -> str:
    """Accept only one bare label from the closed class list.

    Leading and trailing whitespace from the transport is ignored. Quotes,
    Markdown, JSON, explanations and alternative spellings are deliberately
    rejected so every successful API response has the exact paper protocol.
    """

    candidate = value.strip()
    allowed = {label.casefold(): label for label in labels}
    if candidate.casefold() not in allowed or any(character.isspace() for character in candidate):
        raise ProviderError(
            "The model response must contain exactly one allowed class name: "
            f"{value!r}"
        )
    return allowed[candidate.casefold()]


def build_prompt(
    dataset: str,
    labels: Sequence[str],
    view_count: int,
    geometry: dict[str, float] | None = None,
    *,
    decision: bool = False,
    photo_check: bool = False,
    criteria_reference: bool = False,
) -> str:
    label_text = ", ".join(sorted(labels))
    if dataset == "modelnet":
        context = (
            f"You are shown {view_count} isometric image view(s) of the same isolated 3D point-cloud object. "
            "Classify the object from its shape. The background is not part of the object."
        )
    elif dataset == "s3dis":
        context = (
            f"You are shown {view_count} isometric image view(s) of the same segmented indoor 3D point-cloud instance. "
            "The point colors are the original RGB measurements. Classify the instance using visual appearance and shape."
        )
    else:
        raise ValueError(f"Unknown dataset: {dataset}")

    geometry_text = ""
    if geometry is not None:
        geometry_text = (
            "\nThe aligned axis-aligned bounding box has these dimensions in meters: "
            f"width (X) = {geometry['width']:.4f}, depth (Y) = {geometry['depth']:.4f}, "
            f"height (Z) = {geometry['height']:.4f}. "
            "Treat these dimensions only as scale and proportion evidence."
        )

    response_instruction = (
        "Select exactly one class using the typed Choice answer. "
        if decision else
        "Your entire response must be exactly the selected label as written in the list. "
        "Return no JSON, quotes, punctuation, Markdown, confidence, explanation, or additional text. "
    )
    selection_instruction = "Choose exactly one label from this closed list:"
    if criteria_reference:
        selection_instruction = "Choose exactly one label from the closed class list in criteria."
        response_instruction = response_instruction.replace("as written in the list", "as written in criteria")
    if photo_check:
        selection_instruction = "Evaluate the candidate classes against the same point-cloud object:"
        response_instruction = (
            "For each candidate class, answer the supplied yes/no visual check using a Noul probability. "
            "Evaluate all supplied views together as evidence about the same object. "
            "A candidate class is a hypothesis to check, not a known description of the object. "
        )
    class_list = "" if criteria_reference else f"{label_text}\n"
    return (
        f"Protocol: {PROMPT_VERSION}\n"
        f"{context}{geometry_text}\n"
        f"{selection_instruction}\n"
        f"{class_list}"
        f"{response_instruction}"
        "Do not invent another class and do not infer the answer from filenames or metadata."
    )


def _encode_image(path: Path) -> str:
    with path.open("rb") as stream:
        return base64.b64encode(stream.read()).decode("ascii")


def _usage_dict(value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")
    if isinstance(value, dict):
        return value
    return {"value": str(value)}


class BaseProvider:
    provider_name = "base"

    def __init__(self, model: str, api_key: str = "", base_url: str = "") -> None:
        self.model = model.strip()
        self.api_key = api_key.strip()
        self.base_url = base_url.strip().rstrip("/")

    def build_prompt(
        self,
        dataset: str,
        labels: Sequence[str],
        view_count: int,
        geometry: dict[str, float] | None = None,
    ) -> str:
        return build_prompt(dataset, labels, view_count, geometry=geometry)

    def classify(
        self,
        image_paths: Sequence[Path],
        prompt: str,
        labels: Sequence[str],
    ) -> ProviderResult:
        raise NotImplementedError


class LiteLLMProvider(BaseProvider):
    """Responses API client for a LiteLLM proxy."""

    provider_name = "litellm"

    def __init__(self, model: str, api_key: str = "", base_url: str = "") -> None:
        super().__init__(model, api_key, base_url)
        if not self.api_key:
            raise ProviderError("LITELLM_API_KEY is missing.")
        if not self.base_url:
            raise ProviderError('LiteLLM base URL is missing.')
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise ProviderError('The openai package is not installed.') from exc
        self.client = OpenAI(api_key=self.api_key, base_url=self.base_url)

    def classify(
        self,
        image_paths: Sequence[Path],
        prompt: str,
        labels: Sequence[str],
    ) -> ProviderResult:
        content: list[dict[str, Any]] = [{"type": "input_text", "text": prompt}]
        for path in image_paths:
            content.append(
                {
                    "type": "input_image",
                    "image_url": f"data:image/png;base64,{_encode_image(path)}",
                    "detail": "high",
                }
            )
        response = self.client.responses.create(
            model=self.model,
            input=[
                {
                    "role": "system",
                    "content": (
                        "You are a precise closed-set 3D object classifier. "
                        "Output exactly one bare label from the supplied list and nothing else."
                    ),
                },
                {"role": "user", "content": content},
            ],
            max_output_tokens=2048,
            store=False,
        )
        raw = str(getattr(response, "output_text", "") or "")
        if not raw.strip():
            status = getattr(response, "status", "unknown")
            incomplete = getattr(response, "incomplete_details", None)
            raise ProviderError(
                "LiteLLM/Qwen did not return a class name "
                f"(Status: {status}, incomplete_details: {_usage_dict(incomplete)})."
            )
        label = validate_label(raw, labels)
        return ProviderResult(
            label=label,
            raw_response=raw,
            model_version=str(getattr(response, "model", self.model)),
            usage=_usage_dict(getattr(response, "usage", None)),
        )


class GeminiProvider(BaseProvider):
    provider_name = "gemini"

    def __init__(self, model: str, api_key: str = "") -> None:
        super().__init__(model, api_key)
        if not self.api_key:
            raise ProviderError("GEMINI_API_KEY is missing.")
        try:
            from google import genai
        except ImportError as exc:
            raise ProviderError('The google-genai package is not installed.') from exc
        self.client = genai.Client(api_key=self.api_key)

    def classify(
        self,
        image_paths: Sequence[Path],
        prompt: str,
        labels: Sequence[str],
    ) -> ProviderResult:
        inputs: list[dict[str, Any]] = [{"type": "text", "text": prompt}]
        for path in image_paths:
            inputs.append(
                {
                    "type": "image",
                    "data": _encode_image(path),
                    "mime_type": "image/png",
                }
            )
        interaction = self.client.interactions.create(
            model=self.model,
            input=inputs,
            store=False,
        )
        raw = str(getattr(interaction, "output_text", "") or "")
        if not raw.strip():
            raise ProviderError("Gemini did not return a class name.")
        label = validate_label(raw, labels)
        return ProviderResult(
            label=label,
            raw_response=raw,
            model_version=str(getattr(interaction, "model", self.model)),
            usage=_usage_dict(getattr(interaction, "usage", None)),
        )


class JevProvider(BaseProvider):
    """OpenRouter Decisions API; the original images are sent without resizing."""

    provider_name = "jev"
    endpoint = "https://openrouter.ai/api/alpha/decisions"

    def __init__(self, model: str, api_key: str = "") -> None:
        super().__init__(model, api_key)
        if not self.api_key:
            raise ProviderError("OPENROUTER_API_KEY is missing.")
        if not self.model:
            raise ProviderError('JEV model ID is missing.')

    def build_prompt(
        self,
        dataset: str,
        labels: Sequence[str],
        view_count: int,
        geometry: dict[str, float] | None = None,
    ) -> str:
        return build_prompt(dataset, labels, view_count, geometry=geometry, criteria_reference=True)

    @staticmethod
    def build_questions(prompt: str, labels: Sequence[str]) -> dict[str, Any]:
        return {
            "classification": {
                "type": "choice",
                "instructions": prompt,
                "criteria": {label: label for label in sorted(labels)},
            },
        }

    def parse_answer(self, data: dict[str, Any], labels: Sequence[str]) -> str:
        answer = data["answers"]["classification"]
        if answer["type"] != "choice" or not isinstance(answer["choice"], str):
            raise ValueError('Choice response is missing')
        return validate_label(answer["choice"], labels)

    def classify(
        self,
        image_paths: Sequence[Path],
        prompt: str,
        labels: Sequence[str],
    ) -> ProviderResult:
        from PIL import Image

        if not image_paths or not labels:
            raise ProviderError('JEV requires images and a closed class list.')
        state: list[Any] = []
        for index, path in enumerate(image_paths, start=1):
            with Image.open(path) as image:
                width, height = image.size
                if image.format not in {"PNG", "JPEG"} or getattr(image, "is_animated", False):
                    raise ProviderError('JEV supports static PNG and JPEG images only.')
                mime = "image/png" if image.format == "PNG" else "image/jpeg"
            if not (280 <= width <= 1568 and 280 <= height <= 1568):
                raise ProviderError(
                    f"Invalid JEV image dimensions: {width} × {height}. Required: 280–1568 pixels "
                    "per side and at least 78,400 pixels in total. Prepare views at a supported size."
                )
            if path.stat().st_size > 16 * 1024 * 1024:
                raise ProviderError('A JEV image must not exceed 16 MiB.')
            state.extend([
                f"View {index} of the same point-cloud object:",
                {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{_encode_image(path)}"}},
            ])
        payload = {
            "model": self.model,
            "state": state,
            "questions": self.build_questions(prompt, labels),
        }
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        if len(body) > 32 * 1024 * 1024:
            raise ProviderError('The JEV request exceeds the 32 MiB limit.')
        request = Request(
            self.endpoint,
            data=body,
            headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urlopen(request, timeout=120) as response:
                raw = response.read().decode("utf-8")
        except HTTPError as exc:
            hints = {401: 'Check the API key', 402: 'Check the OpenRouter balance', 429: 'Rate limit reached'}
            raise ProviderError(
                f"JEV/OpenRouter HTTP {exc.code}: {hints.get(exc.code, 'Request failed')}."
            ) from exc
        except (URLError, TimeoutError) as exc:
            raise ProviderError('JEV/OpenRouter is unavailable or the request timed out.') from exc
        try:
            data = json.loads(raw)
            label = self.parse_answer(data, labels)
            usage = data["usage"]
            model_version = data["model"]
            if not isinstance(usage, dict) or not isinstance(model_version, str) or not model_version:
                raise ValueError("Model version or usage information is missing")
        except (ValueError, KeyError, TypeError) as exc:
            raise ProviderError('JEV/OpenRouter returned an invalid Decisions response.') from exc
        return ProviderResult(label=label, raw_response=raw, model_version=model_version, usage=usage)


def photo_check_scores(raw_response: str) -> dict[str, float]:
    """Read independent P(yes) values; never normalize them to sum to one."""
    data = json.loads(raw_response)
    answers = data["answers"]
    if not isinstance(answers, dict) or not answers:
        raise ValueError("Photo Check responses are missing")
    scores: dict[str, float] = {}
    for key, answer in answers.items():
        if not key.startswith("class_") or not key[6:] or not isinstance(answer, dict):
            raise ValueError('Invalid Photo Check class')
        value = answer.get("noul")
        if (answer.get("type") != "noul" or isinstance(value, bool)
                or not isinstance(value, (float, int)) or not math.isfinite(value) or not 0 <= value <= 1):
            raise ValueError('Invalid yes probability')
        scores[key[6:]] = float(value)
    return scores


class JevPhotoCheckProvider(JevProvider):
    """One Noul question per candidate class, sharing one set of image views."""

    provider_name = "jev_photo_check"

    def build_prompt(
        self,
        dataset: str,
        labels: Sequence[str],
        view_count: int,
        geometry: dict[str, float] | None = None,
    ) -> str:
        return build_prompt(dataset, labels, view_count, geometry=geometry, photo_check=True)

    @staticmethod
    def build_questions(prompt: str, labels: Sequence[str]) -> dict[str, Any]:
        return {
            f"class_{label}": {
                "type": "noul",
                "instructions": (
                    f"{prompt}\nCandidate class: {label}. "
                    f"Does the object in the supplied point-cloud views belong to the class '{label}'?"
                ),
                "criteria": {
                    "true": f"The depicted object belongs to the candidate class '{label}'.",
                    "false": f"The depicted object belongs to a different class, or the views do not support '{label}'.",
                },
            }
            for label in sorted(labels)
        }

    def parse_answer(self, data: dict[str, Any], labels: Sequence[str]) -> str:
        scores = photo_check_scores(json.dumps(data))
        if set(scores) != set(labels):
            raise ValueError('Photo Check responses do not match the class list')
        # Ties use alphabetical order, independent of ground truth and filenames.
        return max(sorted(scores), key=scores.__getitem__)


class MockProvider(BaseProvider):
    """Cost-free deterministic provider for end-to-end pipeline validation."""

    provider_name = "mock"

    def classify(
        self,
        image_paths: Sequence[Path],
        prompt: str,
        labels: Sequence[str],
    ) -> ProviderResult:
        digest = hashlib.sha256(self.model.encode("utf-8"))
        digest.update(prompt.encode("utf-8"))
        for path in image_paths:
            with path.open("rb") as stream:
                digest.update(stream.read())
        value = int.from_bytes(digest.digest()[:8], "big")
        label = sorted(labels)[value % len(labels)]
        return ProviderResult(
            label=label,
            raw_response=label,
            model_version=self.model or "mock-v1",
            usage={"mock": True},
        )


def create_provider(provider: str, model: str, api_key: str = "", base_url: str = "") -> BaseProvider:
    normalized = provider.strip().lower()
    if normalized == "litellm":
        return LiteLLMProvider(model=model, api_key=api_key, base_url=base_url)
    if normalized == "gemini":
        return GeminiProvider(model=model, api_key=api_key)
    if normalized == "jev":
        return JevProvider(model=model, api_key=api_key)
    if normalized == "jev_photo_check":
        return JevPhotoCheckProvider(model=model, api_key=api_key)
    if normalized == "mock":
        return MockProvider(model=model or "mock-v1")
    raise ProviderError(f"Unknown provider: {provider}")


def prompt_fingerprint(prompt: str) -> str:
    return hashlib.sha256(prompt.encode("utf-8")).hexdigest()
