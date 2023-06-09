from pathlib import Path
import fire
import requests
from PIL import Image, PngImagePlugin, ExifTags
import io
import base64

import string

from tenacity import retry, stop_after_attempt

from helpers.image import read_info_from_image, parse_generation_parameters
from helpers.video import decode_base64_image, video_to_frames, get_video_fps, frames_to_video

from rich import print, traceback
from rich.progress import track

traceback.install()

url = "http://127.0.0.1:7860"

class Styler:

  # def transform(self, input, output, model, metadata):
  #   print('Transforming data from {} to {} using model {} with metadata {}'.format(input, output, model, metadata))

  def info(self, template: str) -> dict[str, str | int]:
    template = Image.open(template)
    info, _ = read_info_from_image(template)
    info = parse_generation_parameters(info)
    print(info)

    return info
  
  def options(self, model_hash: str) -> str:
    return requests.post(url=f'{url}/sdapi/v1/options', json={"sd_model_checkpoint": model_hash}).json()

  def transform(self):
    input = Path("files/viral-guy-with-mask.mp4")
    template = r"files/template.jpeg"
    model = "cetusmix"

    print(f'Transforming data from {input} using model {model} with metadata {template}')

    # get frames
    frames = video_to_frames(str(input))
    # fps = get_video_fps(str(input))

    # get the metadata
    info = self.info(template)

    # set model and options
    self.options(model_hash=info["Model"])
    print("model set to " + info["Model"])

    output_path = input.parent / input.stem / "frames"
    output_path.mkdir(parents=True, exist_ok=True)
    existing_frames = list(output_path.glob("*.jpeg"))
    start = len(existing_frames)

    print(f"Generating {len(frames)} frames from index {start}")
    for i, frame in enumerate(track(frames[start:])):
      output_frame = self.img2img(frame, info)
      decode_base64_image(output_frame).save(output_path / f"{i}.jpeg")

    # print("Writing video to file")
    # frames_to_video(output_frames, output, fps=fps)

  @retry(stop=stop_after_attempt(2))
  def img2img(self, img: str, info):
      payload = {
          "prompt": info["Prompt"],
          "negative_prompt": info["Negative prompt"],
          "steps": int(info["Steps"]),
          "sampler_name": info["Sampler"],
          "cfg_scale": float(info["CFG scale"]),
          "seed": int(info["Seed"]),
          "width": 2 * int(info["Size-1"]),
          "height": 2 * int(info["Size-2"]),
          # "denoising_strength": info["Denoising strength"],
          # "override_settings": {'Clip skip': info["Clip skip"], 'ENSD': info["ENSD"]},
          "init_images": [img],
          "batch_size": 1,
          "alwayson_scripts": {
              "controlnet": {
                  "args": [
                      {
                          "module": "canny",
                          "model": "control_v11p_sd15_canny [d14c016b]",
                      },
                      {
                          "module": "openpose",
                          "model": "control_v11p_sd15_openpose [cab727d4]",
                      }
                  ]
              }
          }
      }
      response = requests.post(url=f'{url}/sdapi/v1/img2img', json=payload, timeout=60).json()

      if "images" not in response:
        print(response)

      return response["images"][0]

  def base64_encode(self, filepath: str) -> str:
      return base64.b64encode(Path(filepath).read_bytes()).decode("ascii")

  def test_controlnet(self):
    # Read Image in RGB order
    encoded_image = base64.b64encode(Path("files/template.jpeg").read_bytes()).decode("ascii")
    # img = cv2.imread('files/template.jpeg')

    # Encode into PNG and send to ControlNet
    # retval, bytes = cv2.imencode('.png', img)
    # encoded_image = base64.b64encode(bytes).decode('utf-8')

    # A1111 payload
    payload = {
        "init_images": [encoded_image],
        "prompt": 'a handsome man',
        "negative_prompt": "",
        "batch_size": 1,
        "steps": 20,
        "cfg_scale": 7,
        "alwayson_scripts": {
            "controlnet": {
                "args": [
                    {
                        "module": "canny",
                        "model": "control_v11p_sd15_canny [d14c016b]",
                    }
                ]
            }
        }
    }

    print("Trigger Generation")
    response = requests.post(url=f'{url}/sdapi/v1/img2img', json=payload)

    # Read results
    r = response.json()
    result = r['images'][0]
    image = Image.open(io.BytesIO(base64.b64decode(result.split(",", 1)[0])))
    image.save('output.png')
    
  def test(self):
    payload = {
        "prompt": "maltese puppy",
        "steps": 5
    }

    response = requests.post(url='http://127.0.0.1:7860/sdapi/v1/txt2img', json=payload)

    for i in response.json()['images']:
      image = Image.open(io.BytesIO(base64.b64decode(i.split(",",1)[0])))

      image.save("test.png")

if __name__ == '__main__':
  fire.Fire(Styler)