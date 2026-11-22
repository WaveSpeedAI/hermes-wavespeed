import pytest
from conftest import TEST_KEY, FakeApi, Reply, http_error, ok

RESULT = "https://api.wavespeed.ai/api/v3/predictions/p1/result"


def test_submit_sends_one_post_with_bearer_key_and_json_body(client, fake):
    fake.queue(ok({"id": "p1", "status": "created", "outputs": []}))
    assert client.submit("wavespeed-ai/z-image/turbo", {"prompt": "a red kite"}) == "p1"
    (post,) = fake.calls("POST")
    assert post["url"] == "https://api.wavespeed.ai/api/v3/wavespeed-ai/z-image/turbo"
    assert post["headers"]["authorization"] == "Bearer " + TEST_KEY
    assert post["headers"]["content-type"] == "application/json"
    assert post["body"] == {"prompt": "a red kite"}


def test_wait_polls_once_a_second_until_completed(client, fake):
    fake.queue(ok({"status": "created", "outputs": []}), ok({"status": "processing", "outputs": []}),
               ok({"status": "completed", "outputs": ["https://cdn.example/out.png"]}))
    assert client.wait("p1", 300) == ["https://cdn.example/out.png"]
    gets = fake.calls("GET")
    assert [g["url"] for g in gets] == [RESULT] * 3
    assert client.clock["now"] == 2.0


def test_failed_prediction_reports_the_reason(client, fake, api):
    fake.queue(ok({"status": "failed", "error": "content rejected", "outputs": []}))
    with pytest.raises(api.WaveSpeedError) as caught:
        client.wait("p1", 300)
    assert caught.value.kind == "job_failed"
    assert "content rejected" in str(caught.value)


def test_error_envelope_on_http_200_is_an_error(client, fake, api):
    fake.queue(Reply({"code": 400, "message": "Insufficient credits"}))
    with pytest.raises(api.WaveSpeedError) as caught:
        client.submit("wavespeed-ai/z-image/turbo", {"prompt": "x"})
    assert "Insufficient credits" in str(caught.value)
    assert len(fake.calls("POST")) == 1


def test_http_errors_carry_the_api_message(client, fake, api):
    fake.queue(http_error(401, {"code": 401, "message": "Invalid API key"}))
    with pytest.raises(api.WaveSpeedError) as caught:
        client.submit("wavespeed-ai/z-image/turbo", {"prompt": "x"})
    assert caught.value.kind == "auth_error"
    assert "Invalid API key" in str(caught.value)
    assert TEST_KEY not in str(caught.value)


def test_submit_is_not_retried_when_the_connection_fails(client, fake, api):
    fake.queue(OSError("connection reset"))
    with pytest.raises(api.WaveSpeedError) as caught:
        client.submit("wavespeed-ai/z-image/turbo", {"prompt": "x"})
    assert caught.value.kind == "network_error"
    assert len(fake.calls("POST")) == 1


def test_wait_times_out_and_names_the_prediction(client, fake, api):
    fake.queue(*[ok({"status": "processing", "outputs": []}) for _ in range(10)])
    with pytest.raises(api.WaveSpeedError) as caught:
        client.wait("p1", 3)
    assert caught.value.kind == "timeout"
    assert "p1" in str(caught.value)
    assert len(fake.calls("GET")) == 4  # at 0, 1, 2 and 3 seconds
    assert fake.calls("POST") == []


def test_wait_keeps_polling_through_a_dropped_connection_and_a_502(client, fake):
    fake.queue(OSError("timed out"), http_error(502, {"message": "bad gateway"}),
               ok({"status": "completed", "outputs": ["https://cdn.example/out.mp4"]}))
    assert client.wait("p1", 300) == ["https://cdn.example/out.mp4"]


def test_wait_stops_on_a_permanent_http_error(client, fake, api):
    fake.queue(http_error(401, {"message": "Invalid API key"}))
    with pytest.raises(api.WaveSpeedError) as caught:
        client.wait("p1", 300)
    assert caught.value.kind == "auth_error"
    assert len(fake.calls("GET")) == 1


def test_missing_key_fails_before_any_request(api):
    fake = FakeApi([])
    client = api.WaveSpeedClient(lambda: "", opener=fake)
    with pytest.raises(api.WaveSpeedError) as caught:
        client.submit("wavespeed-ai/z-image/turbo", {"prompt": "x"})
    assert caught.value.kind == "missing_credentials"
    assert fake.requests == []


def test_model_list_is_read_once_and_unknown_ids_are_reported(client, fake, api):
    assert "size" in client.model_properties("wavespeed-ai/z-image/turbo")
    assert "aspect_ratio" in client.model_properties("bytedance/seedream-v5.0-pro")
    with pytest.raises(api.WaveSpeedError) as caught:
        client.model_properties("nobody/nothing")
    assert caught.value.kind == "model_not_found"
    assert sum(r["url"].endswith("/api/v3/models") for r in fake.requests) == 1


def test_unreadable_model_list_gives_none(api):
    fake = FakeApi(None).queue(http_error(503, {"message": "unavailable"}))
    client = api.WaveSpeedClient(lambda: TEST_KEY, opener=fake)
    assert client.model_properties("wavespeed-ai/z-image/turbo") is None
