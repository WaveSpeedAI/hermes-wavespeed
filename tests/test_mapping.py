import pytest


@pytest.fixture
def props(client):
    return client.model_properties


@pytest.mark.parametrize("aspect, expected", [("landscape", "16:9"), ("square", "1:1"), ("portrait", "9:16")])
def test_image_aspect_uses_the_models_aspect_ratio_enum(api, props, aspect, expected):
    payload, dropped = api.image_payload(props("bytedance/seedream-v5.0-pro"), "a kite", aspect)
    assert payload == {"prompt": "a kite", "aspect_ratio": expected}
    assert dropped == []


@pytest.mark.parametrize("aspect, expected", [
    ("landscape", "1024*576"), ("square", "1024*1024"), ("portrait", "576*1024")])
def test_image_aspect_becomes_a_size_inside_the_models_bounds(api, props, aspect, expected):
    payload, dropped = api.image_payload(props("wavespeed-ai/z-image/turbo"), "a kite", aspect)
    assert payload == {"prompt": "a kite", "size": expected}
    assert dropped == []


def test_image_without_a_schema_sends_the_prompt_only(api):
    assert api.image_payload(None, "a kite", "landscape") == ({"prompt": "a kite"}, ["aspect_ratio"])
    assert api.image_payload({"prompt": {"type": "string"}}, "a kite", "portrait") == ({"prompt": "a kite"}, ["aspect_ratio"])


def test_video_maps_each_input_to_the_models_own_parameter(api, props):
    payload, dropped = api.video_payload(
        props("google/veo3.1-fast/text-to-video"), prompt="waves", duration=5, aspect_ratio="9:16",
        resolution="1080p", negative_prompt="blur", audio=False, seed=7)
    assert payload == {"prompt": "waves", "aspect_ratio": "9:16", "resolution": "1080p", "duration": 4,
                       "negative_prompt": "blur", "generate_audio": False, "seed": 7}
    assert dropped == []


def test_video_leaves_out_what_the_model_does_not_accept(api, props):
    payload, dropped = api.video_payload(
        props("kwaivgi/kling-v4.0/text-to-video"), prompt="waves", duration=40, aspect_ratio="4:3",
        resolution="480p", negative_prompt="blur", audio=True, seed=7)
    assert payload == {"prompt": "waves", "duration": 30}
    assert dropped == ["aspect_ratio", "resolution", "negative_prompt", "audio", "seed"]


def test_video_duration_is_kept_inside_a_range(api, props):
    wan = props("alibaba/wan-3.0/text-to-video")
    assert api.video_payload(wan, prompt="x", duration=1)[0]["duration"] == 2
    assert api.video_payload(wan, prompt="x", duration=12)[0]["duration"] == 12
    assert api.video_payload(wan, prompt="x", duration=99)[0]["duration"] == 30
    assert "duration" not in api.video_payload(wan, prompt="x")[0]


def test_video_size_enum_is_chosen_from_aspect_and_resolution(api, props):
    sized = props("example/sized/text-to-video")
    payload, dropped = api.video_payload(sized, prompt="x", aspect_ratio="9:16", resolution="1080p", duration=7)
    assert payload == {"prompt": "x", "size": "1080*1920", "duration": 5}
    assert dropped == []
    assert api.video_payload(sized, prompt="x", aspect_ratio="16:9", resolution="720p")[0]["size"] == "1280*720"


def test_video_start_image_goes_in_the_image_parameter(api, props):
    payload, dropped = api.video_payload(
        props("google/veo3.1-fast/image-to-video"), prompt="x", image_url="https://example.com/a.png",
        aspect_ratio="16:9", resolution="720p")
    assert payload == {"prompt": "x", "image": "https://example.com/a.png", "resolution": "720p"}
    assert dropped == ["aspect_ratio"]


def test_video_without_a_schema_sends_the_prompt_only(api):
    payload, dropped = api.video_payload(None, prompt="x", duration=5, aspect_ratio="16:9", resolution="720p")
    assert payload == {"prompt": "x"}
    assert dropped == ["duration", "aspect_ratio", "resolution"]
