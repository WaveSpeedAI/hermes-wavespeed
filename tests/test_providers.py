from pathlib import Path

import pytest
from conftest import ok


@pytest.fixture
def image(plugin, client, monkeypatch):
    monkeypatch.setattr(plugin.image_api, "save_url_image", lambda url, prefix: Path("/cache") / (prefix + ".png"))
    return plugin.WaveSpeedImageProvider(client)


@pytest.fixture
def video(plugin, client, monkeypatch):
    monkeypatch.setattr(plugin.video_api, "save_url_video", lambda url, prefix: Path("/cache") / (prefix + ".mp4"))
    return plugin.WaveSpeedVideoProvider(client)


def done(url):
    return [ok({"id": "p1", "status": "created", "outputs": []}), ok({"status": "completed", "outputs": [url]})]


def test_register_adds_one_image_and_one_video_provider(plugin):
    class Context:
        def __init__(self):
            self.image, self.video = [], []

        def register_image_gen_provider(self, provider):
            self.image.append(provider)

        def register_video_gen_provider(self, provider):
            self.video.append(provider)

    ctx = Context()
    plugin.register(ctx)
    assert [p.name for p in ctx.image] == ["wavespeed"] and [p.name for p in ctx.video] == ["wavespeed"]
    assert isinstance(ctx.image[0], plugin.image_api.ImageGenProvider)
    assert isinstance(ctx.video[0], plugin.video_api.VideoGenProvider)
    assert ctx.image[0].default_model() == "bytedance/seedream-v5.0-pro"
    assert ctx.video[0].default_model() == "google/veo3.1-fast/text-to-video"
    assert ctx.image[0].get_setup_schema()["env_vars"][0]["key"] == "WAVESPEED_API_KEY"


def test_availability_follows_the_key(plugin, monkeypatch):
    provider = plugin.WaveSpeedImageProvider()
    assert provider.is_available() is True
    monkeypatch.setattr(plugin, "get_secret_str", lambda name, default="": "")
    assert provider.is_available() is False
    result = provider.generate("a kite")
    assert result["success"] is False and result["error_type"] == "missing_credentials"


def test_image_generation_with_the_default_model(image, fake):
    fake.queue(*done("https://cdn.example/out.png"))
    result = image.generate("a kite", "portrait")
    (post,) = fake.calls("POST")
    assert post["url"].endswith("/api/v3/bytedance/seedream-v5.0-pro")
    assert post["body"] == {"prompt": "a kite", "aspect_ratio": "9:16"}
    assert result["success"] is True
    assert result["image"] == "/cache/wavespeed.png"
    assert result["url"] == "https://cdn.example/out.png"
    assert result["model"] == "bytedance/seedream-v5.0-pro" and result["provider"] == "wavespeed"


def test_image_model_comes_from_hermes_then_the_plugin_setting(plugin, image, fake, monkeypatch):
    fake.queue(*done("https://cdn.example/a.png"), *done("https://cdn.example/b.png"))
    monkeypatch.setattr(plugin, "load_config_readonly", lambda: {
        "plugins": {"entries": {"hermes-wavespeed": {"settings": {"image_model": "wavespeed-ai/z-image/turbo"}}}}})
    assert image.generate("a kite", "square")["model"] == "wavespeed-ai/z-image/turbo"
    assert image.generate("a kite", model="bytedance/seedream-v5.0-pro")["model"] == "bytedance/seedream-v5.0-pro"
    first, second = fake.calls("POST")
    assert first["body"] == {"prompt": "a kite", "size": "1024*1024"}
    assert second["url"].endswith("/api/v3/bytedance/seedream-v5.0-pro")


def test_image_falls_back_to_the_link_when_the_copy_cannot_be_saved(plugin, image, fake, monkeypatch):
    def broken(url, prefix):
        raise OSError("disk full")

    monkeypatch.setattr(plugin.image_api, "save_url_image", broken)
    fake.queue(*done("https://cdn.example/out.png"))
    assert image.generate("a kite")["image"] == "https://cdn.example/out.png"


def test_unknown_model_is_refused_without_starting_a_run(image, fake):
    result = image.generate("a kite", model="flux-2-klein")
    assert result["success"] is False and result["error_type"] == "model_not_found"
    assert fake.calls("POST") == []


def test_image_inputs_are_refused_without_starting_a_run(image, fake):
    result = image.generate("make it blue", image_url="https://example.com/a.png")
    assert result["success"] is False and result["error_type"] == "modality_unsupported"
    assert fake.requests == []


def test_failed_image_prediction_is_an_error_result(image, fake):
    fake.queue(ok({"id": "p1", "status": "created"}), ok({"status": "failed", "error": "content rejected"}))
    result = image.generate("a kite")
    assert result["success"] is False and result["error_type"] == "job_failed"
    assert "content rejected" in result["error"] and result["image"] is None


def test_video_generation_with_the_default_model(video, fake):
    fake.queue(*done("https://cdn.example/out.mp4"))
    result = video.generate("waves", duration=5, aspect_ratio="16:9", resolution="720p", audio=True)
    (post,) = fake.calls("POST")
    assert post["url"].endswith("/api/v3/google/veo3.1-fast/text-to-video")
    assert post["body"] == {"prompt": "waves", "aspect_ratio": "16:9", "resolution": "720p", "duration": 4,
                            "generate_audio": True}
    assert result["success"] is True and result["video"] == "/cache/wavespeed.mp4"
    assert result["duration"] == 4 and result["modality"] == "text" and result["aspect_ratio"] == "16:9"


def test_video_with_a_start_image_uses_the_image_to_video_model(video, fake):
    fake.queue(*done("https://cdn.example/out.mp4"))
    result = video.generate("waves", image_url="https://example.com/a.png", reference_image_urls=["https://example.com/b.png"])
    (post,) = fake.calls("POST")
    assert post["url"].endswith("/api/v3/google/veo3.1-fast/image-to-video")
    assert post["body"] == {"prompt": "waves", "image": "https://example.com/a.png", "resolution": "720p"}
    assert result["model"] == "google/veo3.1-fast/image-to-video" and result["modality"] == "image"
    assert result["duration"] == 8
    assert result["dropped"] == ["aspect_ratio", "reference_image_urls"]


def test_video_start_image_must_be_a_url_and_the_model_must_accept_one(video, fake):
    local = video.generate("waves", image_url="data:image/png;base64,AAAA")
    assert local["success"] is False and local["error_type"] == "invalid_input"
    unsupported = video.generate("waves", model="alibaba/wan-3.0/text-to-video", image_url="https://example.com/a.png")
    assert unsupported["success"] is False and unsupported["error_type"] == "modality_unsupported"
    assert fake.calls("POST") == []


def test_video_timeout_is_an_error_result(video, fake):
    fake.queue(ok({"id": "p1", "status": "created"}), *[ok({"status": "processing"}) for _ in range(1000)])
    result = video.generate("waves")
    assert result["success"] is False and result["error_type"] == "timeout"
    assert len(fake.calls("POST")) == 1
