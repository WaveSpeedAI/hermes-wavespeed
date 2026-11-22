"""WaveSpeedAI image and video generation backends for Hermes Agent."""
from __future__ import annotations

from typing import Any

from agent import image_gen_provider as image_api
from agent import video_gen_provider as video_api
from agent.secret_scope import get_secret_str
from hermes_cli.config import load_config_readonly

from .client import (
    WaveSpeedClient,
    WaveSpeedError,
    default_duration,
    image_payload,
    video_payload,
)

PLUGIN_ID = "hermes-wavespeed"
PROVIDER_ID = "wavespeed"
API_KEY_ENV = "WAVESPEED_API_KEY"

DEFAULT_IMAGE_MODEL = "bytedance/seedream-v5.0-pro"
DEFAULT_VIDEO_MODEL = "google/veo3.1-fast/text-to-video"
IMAGE_MODELS = (DEFAULT_IMAGE_MODEL, "google/nano-banana-pro/text-to-image", "wavespeed-ai/z-image/turbo")
VIDEO_MODELS = (
    DEFAULT_VIDEO_MODEL, "kwaivgi/kling-v4.0/text-to-video", "bytedance/seedance-2.5/text-to-video",
    "alibaba/wan-3.0/text-to-video",
)

IMAGE_TIMEOUT_SECONDS = 300.0
VIDEO_TIMEOUT_SECONDS = 900.0

_SETUP = {
    "name": "WaveSpeedAI",
    "badge": "paid",
    "env_vars": [{"key": API_KEY_ENV, "prompt": "WaveSpeedAI API key", "url": "https://wavespeed.ai/accesskey"}],
}


def _api_key() -> str:
    return get_secret_str(API_KEY_ENV)


def _setting(key: str) -> str:
    """``plugins.entries.hermes-wavespeed.settings.<key>`` from config.yaml, or ``""``."""
    try:
        node: Any = load_config_readonly() or {}
        for part in ("plugins", "entries", PLUGIN_ID, "settings", key):
            node = node.get(part) if isinstance(node, dict) else None
    except Exception:
        return ""
    return node.strip() if isinstance(node, str) else ""


class _Backend:
    """What the two providers share: the key check, the client and the picker rows."""

    name = PROVIDER_ID
    display_name = "WaveSpeedAI"
    _models: tuple[str, ...] = ()
    _default_model = ""
    _setting_key = ""

    def __init__(self, client: WaveSpeedClient | None = None):
        self.client = client or WaveSpeedClient(_api_key)

    def is_available(self) -> bool:
        try:
            return bool(_api_key().strip())
        except Exception:
            return False

    def list_models(self) -> list[dict[str, Any]]:
        return [{"id": model, "display": model} for model in self._models]

    def default_model(self) -> str | None:
        return self._default_model

    def _model(self, requested: str | None) -> str:
        """The model Hermes passes (``image_gen.model`` / ``video_gen.model``), then the
        plugin setting, then the built-in default. Any WaveSpeedAI model id is accepted."""
        requested = requested.strip() if isinstance(requested, str) else ""
        return requested or _setting(self._setting_key) or self._default_model


class WaveSpeedImageProvider(_Backend, image_api.ImageGenProvider):
    _models = IMAGE_MODELS
    _default_model = DEFAULT_IMAGE_MODEL
    _setting_key = "image_model"

    def get_setup_schema(self) -> dict[str, Any]:
        return {**_SETUP, "tag": "Text-to-image with a model of your choice"}

    def capabilities(self) -> dict[str, Any]:
        return {"modalities": ["text"], "max_reference_images": 0}

    def generate(self, prompt: str, aspect_ratio: str = image_api.DEFAULT_ASPECT_RATIO, *,
                 image_url: str | None = None, reference_image_urls: list[str] | None = None,
                 **kwargs: Any) -> dict[str, Any]:
        prompt = (prompt or "").strip()
        aspect = image_api.resolve_aspect_ratio(aspect_ratio)
        model = self._model(kwargs.get("model"))

        def fail(error: str, error_type: str) -> dict[str, Any]:
            return image_api.error_response(error=error, error_type=error_type, provider=self.name,
                                            model=model, prompt=prompt, aspect_ratio=aspect)

        if not prompt:
            return fail("A prompt is required", "invalid_input")
        if image_url or reference_image_urls:
            return fail("This backend generates images from text only; image editing is not supported",
                        "modality_unsupported")
        try:
            payload, dropped = image_payload(self.client.model_properties(model), prompt, aspect)
            prediction_id = self.client.submit(model, payload)
            url = self.client.wait(prediction_id, IMAGE_TIMEOUT_SECONDS)[0]
        except WaveSpeedError as exc:
            return fail(str(exc), exc.kind)
        try:
            image = str(image_api.save_url_image(url, prefix=PROVIDER_ID))
        except Exception:
            image = url  # the link still works when the local copy could not be saved
        return image_api.success_response(
            image=image, model=model, prompt=prompt, aspect_ratio=aspect, provider=self.name,
            extra={"prediction_id": prediction_id, "url": url, "dropped": dropped})


class WaveSpeedVideoProvider(_Backend, video_api.VideoGenProvider):
    _models = VIDEO_MODELS
    _default_model = DEFAULT_VIDEO_MODEL
    _setting_key = "video_model"

    def get_setup_schema(self) -> dict[str, Any]:
        return {**_SETUP, "tag": "Text-to-video and image-to-video with a model of your choice"}

    def capabilities(self) -> dict[str, Any]:
        # One answer for every model, without a network call. What a given model does not
        # accept is left out of the request and listed in the result's ``dropped``.
        return {
            "modalities": ["text", "image"], "aspect_ratios": list(video_api.COMMON_ASPECT_RATIOS),
            "resolutions": list(video_api.COMMON_RESOLUTIONS), "min_duration": 2, "max_duration": 30,
            "supports_audio": True, "supports_negative_prompt": True, "supports_seed": True,
            "supports_upscale": False, "max_reference_images": 0,
        }

    def _image_model(self, model: str) -> tuple[str, dict[str, Any] | None]:
        """The model that takes a start image: the selected one when its schema has ``image``,
        else its ``image-to-video`` counterpart when the model list has one."""
        properties = self.client.model_properties(model)
        if properties is None:
            raise WaveSpeedError("The WaveSpeedAI model list could not be read, so the image input "
                                 "cannot be checked. Try again later.", "schema_unavailable")
        if "image" in properties:
            return model, properties
        suffix = "/text-to-video"
        if model.endswith(suffix):
            sibling = model[:-len(suffix)] + "/image-to-video"
            if self.client.has_model(sibling):
                sibling_properties = self.client.model_properties(sibling) or {}
                if "image" in sibling_properties:
                    return sibling, sibling_properties
        raise WaveSpeedError(f"The model '{model}' does not take a start image. Set video_gen.model to an "
                             "image-to-video model id.", "modality_unsupported")

    def generate(self, prompt: str, *, model: str | None = None, image_url: str | None = None,
                 reference_image_urls: list[str] | None = None, duration: int | None = None,
                 aspect_ratio: str = video_api.DEFAULT_ASPECT_RATIO,
                 resolution: str = video_api.DEFAULT_RESOLUTION, negative_prompt: str | None = None,
                 audio: bool | None = None, seed: int | None = None, **kwargs: Any) -> dict[str, Any]:
        prompt = (prompt or "").strip()
        model_id = self._model(model)
        image = (image_url or "").strip() or None

        def fail(error: str, error_type: str) -> dict[str, Any]:
            return video_api.error_response(error=error, error_type=error_type, provider=self.name,
                                            model=model_id, prompt=prompt, aspect_ratio=aspect_ratio)

        if not prompt:
            return fail("A prompt is required", "invalid_input")
        if image and not image.lower().startswith(("http://", "https://")):
            return fail("The start image must be a public http(s) URL; local files are not uploaded",
                        "invalid_input")
        try:
            if image:
                model_id, properties = self._image_model(model_id)
            else:
                properties = self.client.model_properties(model_id)
            payload, dropped = video_payload(
                properties, prompt=prompt, image_url=image, duration=duration, aspect_ratio=aspect_ratio,
                resolution=resolution, negative_prompt=negative_prompt, audio=audio, seed=seed)
            if reference_image_urls:
                dropped.append("reference_image_urls")
            prediction_id = self.client.submit(model_id, payload)
            url = self.client.wait(prediction_id, VIDEO_TIMEOUT_SECONDS)[0]
        except WaveSpeedError as exc:
            return fail(str(exc), exc.kind)
        try:
            video = str(video_api.save_url_video(url, prefix=PROVIDER_ID))
        except Exception:
            video = url  # the link still works when the local copy could not be saved
        return video_api.success_response(
            video=video, model=model_id, prompt=prompt, modality="image" if image else "text",
            aspect_ratio=payload.get("aspect_ratio", ""),
            duration=payload.get("duration") or default_duration(properties), provider=self.name,
            extra={"prediction_id": prediction_id, "url": url, "dropped": dropped})


def register(ctx) -> None:
    ctx.register_image_gen_provider(WaveSpeedImageProvider())
    ctx.register_video_gen_provider(WaveSpeedVideoProvider())
