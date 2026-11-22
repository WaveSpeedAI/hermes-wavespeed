# hermes-wavespeed

WaveSpeedAI image and video generation for [Hermes Agent](https://github.com/NousResearch/hermes-agent).

The plugin registers one image backend and one video backend, both with the provider id `wavespeed`. Once selected, Hermes's own `image_generate` and `video_generate` tools run on your WaveSpeedAI account. It adds no tools, hooks, commands or UI of its own.

## What it supports

- Text-to-image through `image_generate`.
- Text-to-video and image-to-video through `video_generate`.
- Any WaveSpeedAI model id, set in the Hermes configuration. The defaults are `bytedance/seedream-v5.0-pro` for images and `google/veo3.1-fast/text-to-video` for videos.

Not supported: image editing (an `image_generate` call with an input image is refused), reference images for video, video editing and video extension.

## Requirements

- Hermes Agent 0.23.0 or later.
- A WaveSpeedAI API key from <https://wavespeed.ai/accesskey>.

The plugin has no Python dependencies of its own.

## Install

```sh
hermes plugins install WaveSpeedAI/hermes-wavespeed --enable
```

Hermes asks for `WAVESPEED_API_KEY` during the install and stores it with its other secrets. To set it later, add it to the Hermes `.env` file of your profile or enter it in `hermes tools`.

Then select the backends:

```sh
hermes config set image_gen.provider wavespeed
hermes config set video_gen.provider wavespeed
```

`hermes tools` offers the same choice under Image Generation and Video Generation. Restart Hermes after installing.

## Configuration

| Setting | Where | Meaning |
|---|---|---|
| `WAVESPEED_API_KEY` | Hermes secrets (`.env`) | The API key. Required. |
| `image_gen.model` | `config.yaml` | Model id for images. |
| `video_gen.model` | `config.yaml` | Model id for videos. |
| `plugins.entries.hermes-wavespeed.settings.image_model` | `config.yaml` | Model id for images when `image_gen.model` is not set. |
| `plugins.entries.hermes-wavespeed.settings.video_model` | `config.yaml` | Model id for videos when `video_gen.model` is not set. |

A model id is the path shown on the model's page at <https://wavespeed.ai/models>, for example:

```sh
hermes config set image_gen.model google/nano-banana-pro/text-to-image
hermes config set video_gen.model kwaivgi/kling-v4.0/text-to-video
```

`image_gen.model` and `video_gen.model` are shared by all Hermes backends. If one of them still holds another backend's model name, the plugin reports that WaveSpeedAI has no such model and starts nothing; set it to a WaveSpeedAI model id or remove it.

## Usage

Ask Hermes for an image or a video as usual:

```
Generate a portrait image of a lighthouse at dawn.
Make a 6 second 16:9 video of waves breaking on a pier, with sound.
```

How the request reaches the model:

- The plugin reads the model's input schema from WaveSpeedAI's model list and sends only parameters that the model declares.
- Images: Hermes's `landscape`, `square` or `portrait` becomes the model's `aspect_ratio` (16:9, 1:1 or 9:16, or the nearest ratio the model offers) or, for models that take a pixel size, a `size` within the model's limits.
- Videos: aspect ratio, resolution, duration, negative prompt, audio and seed are sent when the model has the matching parameter. A duration the model does not offer is moved to the nearest one it does.
- Anything the selected model has no parameter for is left out, and the result lists it under `dropped`.
- Image-to-video: the start image must be a public `http(s)` URL; local files are not uploaded. If the selected model id ends in `/text-to-video` and has no image input, the plugin uses the `/image-to-video` model of the same family when there is one.

The finished file is downloaded into the Hermes media cache and that path is returned. If the download fails, the WaveSpeedAI link is returned instead. The result also carries the prediction id and the original link.

## Billing

Each generation is billed to the WaveSpeedAI account that owns the API key.

The plugin sends a generation request once and never repeats it. If the connection drops while submitting, or the result does not arrive in time (5 minutes for an image, 15 minutes for a video), the error says so and names the prediction when it is known. The run may still finish and be billed, so check the account before asking again. Hermes does not ask for approval before a generation.

## What the plugin does on your machine

- Network: HTTPS requests to `api.wavespeed.ai` (the model list, the generation request, the result), each carrying the API key, and one download of the finished file from the link WaveSpeedAI returns. Prompts, settings and the start image URL are sent to WaveSpeedAI.
- Credentials: `WAVESPEED_API_KEY` is read through Hermes for each request. The plugin does not store it, log it or read any credential file.
- Files: the downloaded result in the Hermes media cache. The model list is kept in memory for one hour.
- No telemetry, no shell commands, no background processes, no self-update.

## Development

The tests run offline against a stand-in for the API and need a Hermes Agent checkout installed in the same environment:

```sh
pytest tests
hermes plugins validate .
```

## License

MIT. See [LICENSE](LICENSE).
