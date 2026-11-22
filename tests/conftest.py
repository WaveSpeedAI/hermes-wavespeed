"""Offline test setup: no test reaches the network and no real key is read."""
import importlib.util
import io
import json
import socket
import sys
import urllib.error
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
TEST_KEY = "test-key-not-real"


def _load_plugin():
    name = "hermes_wavespeed_under_test"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(
        name, ROOT / "__init__.py", submodule_search_locations=[str(ROOT)])
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def plugin(monkeypatch, tmp_path):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hermes-home"))
    module = _load_plugin()
    monkeypatch.setattr(module, "get_secret_str", lambda name, default="": TEST_KEY)
    monkeypatch.setattr(module, "load_config_readonly", dict)
    return module


@pytest.fixture
def api(plugin):
    return sys.modules[plugin.__name__ + ".client"]


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    def blocked(*args, **kwargs):
        raise AssertionError("a test tried to open a network connection")

    monkeypatch.setattr(socket.socket, "connect", blocked)
    monkeypatch.setattr(socket, "getaddrinfo", blocked)


class Reply:
    def __init__(self, body):
        self._raw = body if isinstance(body, bytes) else json.dumps(body).encode()

    def read(self):
        return self._raw

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def ok(data):
    return Reply({"code": 200, "message": "success", "data": data})


def http_error(status, body):
    return urllib.error.HTTPError("https://api.wavespeed.ai/x", status, "error", {}, io.BytesIO(json.dumps(body).encode()))


class FakeApi:
    """Stands in for ``open_credentialed_url``: records each request and plays back replies."""

    def __init__(self, models=None):
        self.requests = []
        self.replies = []
        self.models = models

    def queue(self, *replies):
        self.replies.extend(replies)
        return self

    def __call__(self, request, timeout=None):
        body = json.loads(request.data) if request.data else None
        self.requests.append({"method": request.get_method(), "url": request.full_url,
                              "headers": {k.lower(): v for k, v in request.header_items()},
                              "body": body, "timeout": timeout})
        if request.full_url.endswith("/api/v3/models") and self.models is not None:
            return ok(self.models)
        reply = self.replies.pop(0)
        if isinstance(reply, BaseException):
            raise reply
        return reply

    def calls(self, method):
        return [r for r in self.requests if r["method"] == method and not r["url"].endswith("/api/v3/models")]


def model(model_id, **properties):
    return {"model_id": model_id, "api_schema": {"api_schemas": [
        {"type": "model_run", "request_schema": {"properties": {"prompt": {"type": "string"}, **properties}}}]}}


# Shapes copied from the live model list (property names, types, enums and bounds only).
MODELS = [
    model("wavespeed-ai/z-image/turbo",
          size={"type": "string", "default": "1024*1024", "minimum": 256, "maximum": 1536},
          image={"type": "string"}, seed={"type": "integer"}),
    model("bytedance/seedream-v5.0-pro",
          aspect_ratio={"type": "string", "enum": ["1:1", "2:3", "3:2", "3:4", "4:3", "9:16", "16:9"], "default": "1:1"},
          resolution={"type": "string", "enum": ["1k", "1.5k", "2k"]}),
    model("google/veo3.1-fast/text-to-video",
          aspect_ratio={"type": "string", "enum": ["16:9", "9:16"]},
          duration={"type": "integer", "enum": [8, 4, 6], "default": 8},
          generate_audio={"type": "boolean"}, negative_prompt={"type": "string"},
          resolution={"type": "string", "enum": ["720p", "1080p", "4k"]}, seed={"type": "integer"}),
    model("google/veo3.1-fast/image-to-video",
          image={"type": "string"}, duration={"type": "integer", "enum": [8, 4, 6], "default": 8},
          resolution={"type": "string", "enum": ["720p", "1080p", "4k"]}),
    model("kwaivgi/kling-v4.0/text-to-video",
          aspect_ratio={"type": "string", "enum": ["16:9", "9:16", "1:1", "21:9"]},
          duration={"type": "integer", "enum": list(range(3, 31)), "default": 5},
          resolution={"type": "string", "enum": ["720p", "1080p", "4k"]}),
    model("alibaba/wan-3.0/text-to-video",
          aspect_ratio={"type": "string", "enum": ["16:9", "9:16", "1:1", "4:3", "3:4"]},
          duration={"type": "integer", "default": 5, "minimum": 2, "maximum": 30},
          generate_audio={"type": "boolean"}, resolution={"type": "string", "enum": ["480p", "720p", "1080p"]},
          seed={"type": "integer"}),
    model("example/sized/text-to-video",
          size={"type": "string", "enum": ["1280*720", "720*1280", "1920*1080", "1080*1920"]},
          duration={"type": "integer", "enum": [5, 10]}),
]


@pytest.fixture
def fake():
    return FakeApi(MODELS)


@pytest.fixture
def client(api, fake):
    clock = {"now": 0.0}

    def sleep(seconds):
        clock["now"] += seconds

    value = api.WaveSpeedClient(lambda: TEST_KEY, opener=fake, sleep=sleep, monotonic=lambda: clock["now"])
    value.clock = clock
    return value
