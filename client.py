"""WaveSpeedAI HTTP client and parameter mapping. Standard library only."""
from __future__ import annotations

import json
import threading
import time
import urllib.error
import urllib.request
from collections.abc import Callable
from typing import Any
from urllib.parse import quote

from hermes_cli.urllib_security import open_credentialed_url

BASE_URL = "https://api.wavespeed.ai"
USER_AGENT = "hermes-wavespeed/0.1.0"
POLL_INTERVAL_SECONDS = 1.0
REQUEST_TIMEOUT_SECONDS = 30.0
CATALOG_TTL_SECONDS = 3600.0

_ASPECT_CANDIDATES = {
    "landscape": ("16:9", "3:2", "4:3"),
    "square": ("1:1",),
    "portrait": ("9:16", "2:3", "3:4"),
}
_ASPECT_VALUE = {"landscape": 16 / 9, "square": 1.0, "portrait": 9 / 16}


class WaveSpeedError(Exception):
    """A failure with a short ``kind`` used as the Hermes ``error_type``."""

    def __init__(self, message: str, kind: str = "api_error"):
        super().__init__(message)
        self.kind = kind


def _error_message(raw: bytes) -> str:
    try:
        body = json.loads(raw.decode("utf-8", "replace"))
    except ValueError:
        return ""
    if isinstance(body, dict):
        return str(body.get("message") or body.get("error") or "")
    return ""


class WaveSpeedClient:
    """Submit a prediction, poll its result, read the model list.

    ``api_key_fn`` is called for every request, so the key is never stored. ``opener`` has the
    signature of ``open_credentialed_url`` and exists so tests can replace the network.
    """

    def __init__(
        self,
        api_key_fn: Callable[[], str],
        *,
        base_url: str = BASE_URL,
        opener: Callable[..., Any] = open_credentialed_url,
        sleep: Callable[[float], None] = time.sleep,
        monotonic: Callable[[], float] = time.monotonic,
    ):
        self._api_key_fn = api_key_fn
        self._base_url = base_url.rstrip("/")
        self._opener = opener
        self._sleep = sleep
        self._monotonic = monotonic
        self._catalog: dict[str, dict[str, Any]] | None = None
        self._catalog_at = 0.0
        self._catalog_lock = threading.Lock()

    def _request(self, method: str, path: str, body: dict[str, Any] | None = None) -> Any:
        """One HTTP request. Returns the envelope's ``data``; raises ``WaveSpeedError`` for an
        API-level failure and lets network errors (``OSError``) through to the caller."""
        key = (self._api_key_fn() or "").strip()
        if not key:
            raise WaveSpeedError("WAVESPEED_API_KEY is not set", "missing_credentials")
        headers = {"Authorization": f"Bearer {key}", "Accept": "application/json", "User-Agent": USER_AGENT}
        data = None
        if body is not None:
            data = json.dumps(body).encode("utf-8")
            headers["Content-Type"] = "application/json"
        request = urllib.request.Request(self._base_url + path, data=data, headers=headers, method=method)
        try:
            with self._opener(request, timeout=REQUEST_TIMEOUT_SECONDS) as response:
                raw = response.read()
        except urllib.error.HTTPError as exc:
            detail = _error_message(exc.read()) or str(exc.reason or "")
            kind = "auth_error" if exc.code in (401, 403) else f"http_{exc.code}"
            raise WaveSpeedError(f"WaveSpeedAI returned HTTP {exc.code}: {detail}".rstrip(": "), kind) from None
        try:
            envelope = json.loads(raw.decode("utf-8"))
        except ValueError:
            raise WaveSpeedError("WaveSpeedAI returned a response that is not JSON", "bad_response") from None
        if not isinstance(envelope, dict):
            raise WaveSpeedError("WaveSpeedAI returned an unexpected response", "bad_response")
        # Some failures arrive as HTTP 200 with a non-200 code in the envelope.
        if envelope.get("code") != 200:
            raise WaveSpeedError(
                f"WaveSpeedAI error {envelope.get('code')}: {envelope.get('message') or 'unknown error'}", "api_error")
        return envelope.get("data")

    def submit(self, model: str, payload: dict[str, Any]) -> str:
        """Start one prediction and return its id. Sent once and never retried: a second POST
        would start a second billed run."""
        try:
            data = self._request("POST", "/api/v3/" + quote(model.strip("/"), safe="/"), payload)
        except OSError as exc:
            raise WaveSpeedError(
                f"WaveSpeedAI could not be reached while submitting ({exc}). The request was not retried, "
                "because the run may have started; check the account before trying again.",
                "network_error") from None
        prediction_id = data.get("id") if isinstance(data, dict) else None
        if not prediction_id:
            raise WaveSpeedError("WaveSpeedAI accepted the request but returned no prediction id", "bad_response")
        return str(prediction_id)

    def wait(self, prediction_id: str, timeout_seconds: float) -> list[str]:
        """Poll the result about once a second until it completes, fails, or the time runs out."""
        path = f"/api/v3/predictions/{quote(prediction_id, safe='')}/result"
        deadline = self._monotonic() + timeout_seconds
        status = "created"
        while True:
            try:
                data = self._request("GET", path)
            except OSError:
                data = None  # reading the result is safe to repeat
            except WaveSpeedError as exc:
                if exc.kind not in ("http_429", "http_500", "http_502", "http_503", "http_504"):
                    raise
                data = None
            if isinstance(data, dict):
                status = str(data.get("status") or status)
                if status == "completed":
                    outputs = [item for item in data.get("outputs") or [] if isinstance(item, str) and item]
                    if not outputs:
                        raise WaveSpeedError(f"Prediction {prediction_id} completed without any output", "empty_response")
                    return outputs
                if status == "failed":
                    raise WaveSpeedError(
                        f"Prediction {prediction_id} failed: {data.get('error') or 'no reason given'}", "job_failed")
            if self._monotonic() >= deadline:
                raise WaveSpeedError(
                    f"Prediction {prediction_id} did not finish within {int(timeout_seconds)} seconds "
                    f"(last status: {status}). It may still complete and be billed; do not submit it again "
                    "without checking.", "timeout")
            self._sleep(POLL_INTERVAL_SECONDS)

    def model_properties(self, model: str) -> dict[str, Any] | None:
        """The model's request properties from the live model list. ``None`` when the list
        could not be read; raises ``model_not_found`` when the list loads without this id."""
        with self._catalog_lock:
            fresh = self._catalog is not None and self._monotonic() - self._catalog_at < CATALOG_TTL_SECONDS
            if not fresh:
                try:
                    self._catalog = _index_models(self._request("GET", "/api/v3/models"))
                    self._catalog_at = self._monotonic()
                except WaveSpeedError as exc:
                    if exc.kind in ("missing_credentials", "auth_error"):
                        raise
                    self._catalog = None
                except OSError:
                    self._catalog = None
            catalog = self._catalog
        if catalog is None:
            return None
        if model not in catalog:
            raise WaveSpeedError(
                f"WaveSpeedAI has no model with the id '{model}'. Model ids are listed at https://wavespeed.ai/models.",
                "model_not_found")
        return catalog[model]

    def has_model(self, model: str) -> bool:
        try:
            return self.model_properties(model) is not None
        except WaveSpeedError as exc:
            if exc.kind == "model_not_found":
                return False
            raise


def _index_models(data: Any) -> dict[str, dict[str, Any]]:
    index: dict[str, dict[str, Any]] = {}
    for entry in data if isinstance(data, list) else []:
        if not isinstance(entry, dict) or not isinstance(entry.get("model_id"), str):
            continue
        schemas = (entry.get("api_schema") or {}).get("api_schemas") or []
        for schema in schemas:
            if isinstance(schema, dict) and schema.get("type") == "model_run":
                properties = (schema.get("request_schema") or {}).get("properties")
                index[entry["model_id"]] = properties if isinstance(properties, dict) else {}
                break
    if not index:
        raise WaveSpeedError("WaveSpeedAI returned an empty model list", "bad_response")
    return index


def _enum(spec: Any) -> list[Any]:
    values = spec.get("enum") if isinstance(spec, dict) else None
    return list(values) if isinstance(values, list) else []


def _pair(value: Any) -> tuple[float, float] | None:
    """``"1280*720"`` or ``"16:9"`` as two positive numbers."""
    if not isinstance(value, str):
        return None
    for separator in ("*", "x", ":"):
        left, found, right = value.partition(separator)
        if found:
            try:
                width, height = float(left), float(right)
            except ValueError:
                return None
            return (width, height) if width > 0 and height > 0 else None
    return None


def _closest_size(choices: list[Any], ratio: float, short_side: float | None = None) -> str | None:
    """The enum entry whose shape is nearest ``ratio``; ``short_side`` breaks ties."""
    best, best_score = None, None
    for choice in choices:
        pair = _pair(choice)
        if pair is None:
            continue
        score = (round(abs(pair[0] / pair[1] - ratio), 3), abs(min(pair) - short_side) if short_side else 0.0)
        if best_score is None or score < best_score:
            best, best_score = choice, score
    return best


def _sized_like_default(spec: dict[str, Any], aspect: str) -> str | None:
    """A ``width*height`` value for a free-form ``size``: the long side of the model's own
    default, the short side from the aspect, both kept inside the declared bounds."""
    default = _pair(spec.get("default"))
    if default is None or "*" not in str(spec.get("default")):
        return None
    long_side = max(default)
    short_side = long_side if aspect == "square" else round(long_side * 9 / 16 / 16) * 16
    low, high = spec.get("minimum"), spec.get("maximum")

    def bounded(value: float) -> int:
        if isinstance(low, (int, float)):
            value = max(value, low)
        if isinstance(high, (int, float)):
            value = min(value, high)
        return int(value)

    width, height = (short_side, long_side) if aspect == "portrait" else (long_side, short_side)
    return f"{bounded(width)}*{bounded(height)}"


def image_payload(properties: dict[str, Any] | None, prompt: str, aspect: str) -> tuple[dict[str, Any], list[str]]:
    """Request body for text-to-image plus the names of inputs the model has no parameter for.
    Only parameters present in the model's live schema are sent."""
    payload: dict[str, Any] = {"prompt": prompt}
    if properties is None:
        return payload, ["aspect_ratio"]
    ratio_choices = _enum(properties.get("aspect_ratio"))
    size_spec = properties.get("size")
    picked = next((c for c in _ASPECT_CANDIDATES[aspect] if c in ratio_choices), None)
    if picked:
        payload["aspect_ratio"] = picked
        return payload, []
    if isinstance(size_spec, dict):
        size = _closest_size(_enum(size_spec), _ASPECT_VALUE[aspect]) if _enum(size_spec) else _sized_like_default(size_spec, aspect)
        if size:
            payload["size"] = size
            return payload, []
    return payload, ["aspect_ratio"]


def _height(resolution: str | None) -> float | None:
    text = (resolution or "").strip().lower()
    if text.endswith("p") and text[:-1].isdigit():
        return float(text[:-1])
    return {"2k": 1440.0, "4k": 2160.0}.get(text)


def video_payload(
    properties: dict[str, Any] | None, *, prompt: str, image_url: str | None = None,
    duration: int | None = None, aspect_ratio: str | None = None, resolution: str | None = None,
    negative_prompt: str | None = None, audio: bool | None = None, seed: int | None = None,
) -> tuple[dict[str, Any], list[str]]:
    """Request body for text-to-video or image-to-video plus the names of inputs that were not
    sent. Only parameters present in the model's live schema are sent; values outside what the
    schema allows are moved to the nearest allowed value or left to the model's default."""
    payload: dict[str, Any] = {"prompt": prompt}
    dropped: list[str] = []
    if properties is None:
        for name, value in (("duration", duration), ("aspect_ratio", aspect_ratio), ("resolution", resolution),
                            ("negative_prompt", negative_prompt), ("audio", audio), ("seed", seed)):
            if value is not None:
                dropped.append(name)
        return payload, dropped
    if image_url:
        payload["image"] = image_url

    ratio_choices = _enum(properties.get("aspect_ratio"))
    size_choices = _enum(properties.get("size"))
    if aspect_ratio:
        if aspect_ratio in ratio_choices:
            payload["aspect_ratio"] = aspect_ratio
        elif not ratio_choices and size_choices and _pair(aspect_ratio):
            width, height = _pair(aspect_ratio)  # type: ignore[misc]
            size = _closest_size(size_choices, width / height, _height(resolution))
            if size:
                payload["size"] = size
            else:
                dropped.append("aspect_ratio")
        else:
            dropped.append("aspect_ratio")

    if resolution:
        match = next((c for c in _enum(properties.get("resolution"))
                      if isinstance(c, str) and c.lower() == resolution.lower()), None)
        if match:
            payload["resolution"] = match
        elif "size" not in payload:
            dropped.append("resolution")

    if duration is not None:
        spec = properties.get("duration")
        if isinstance(spec, dict) and spec.get("type") == "integer":
            choices = [c for c in _enum(spec) if isinstance(c, int)]
            if choices:
                payload["duration"] = min(choices, key=lambda c: (abs(c - duration), c))
            else:
                low, high = spec.get("minimum"), spec.get("maximum")
                value = int(duration)
                if isinstance(low, (int, float)):
                    value = max(value, int(low))
                if isinstance(high, (int, float)):
                    value = min(value, int(high))
                payload["duration"] = value
        else:
            dropped.append("duration")

    optional = (
        ("negative_prompt", "negative_prompt", "string", negative_prompt or None),
        ("audio", "generate_audio", "boolean", audio),
        ("seed", "seed", "integer", seed),
    )
    for name, parameter, kind, value in optional:
        if value is None:
            continue
        spec = properties.get(parameter)
        if isinstance(spec, dict) and spec.get("type") == kind:
            payload[parameter] = value
        else:
            dropped.append(name)
    return payload, dropped


def default_duration(properties: dict[str, Any] | None) -> int:
    spec = (properties or {}).get("duration")
    value = spec.get("default") if isinstance(spec, dict) else None
    return int(value) if isinstance(value, (int, float)) else 0
