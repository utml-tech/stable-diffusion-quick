from pathlib import Path
import fire
import requests
from PIL import Image, PngImagePlugin, ExifTags
import io
import base64

url = "https://53e3e566423a8d7a39.gradio.live" # "http://127.0.0.1:7860"

class Styler:

  # def transform(self, input, output, model, metadata):
  #   print('Transforming data from {} to {} using model {} with metadata {}'.format(input, output, model, metadata))

  def transform(self):
    input = r"files/viral-guy-with-mask.mp4"
    template = r"files/template.jpeg"
    output = r"files/output.mp4"
    model = "cetusmix"

    print(f'Transforming data from {input} to {output} using model {model} with metadata {template}')

    # get the metadata
    print("getting png info")
    template = Image.open(template)
    exif = { ExifTags.TAGS[k]: v for k, v in template._getexif().items() if k in ExifTags.TAGS}
    print(exif["UserComment"].decode("utf-8"))

    # img2img
    img = base64.b64encode(Path("files/template.jpeg").read_bytes()).decode("ascii")

    payload = {
    "init_images": [str(img)],
    "denoising_strength": 0.75,
    "image_cfg_scale": 0,
    "mask_blur": 4,
    "inpainting_fill": 0,
    "inpaint_full_res": True,
    "prompt": "",
    "seed": -1,
    "subseed": -1,
    "subseed_strength": 0,
    "seed_resize_from_h": -1,
    "seed_resize_from_w": -1,
    "batch_size": 2,
    "n_iter": 1,
    "steps": 50,
    "cfg_scale": 7,
    "width": 512,
    "height": 512
    # "resize_mode": 0,
    # "denoising_strength": 0.75,
    # "image_cfg_scale": 0,
    # "mask": "",
    # "mask_blur": 4,
    # "inpainting_fill": 0,
    # "inpaint_full_res": True,
    # "inpaint_full_res_padding": 0,
    # "inpainting_mask_invert": 0,
    # "initial_noise_multiplier": 0,
    # "prompt": "maltese puppy",
    # # "styles": [
    # #   "string"
    # # ],
    # "seed": -1,
    # "subseed": -1,
    # "subseed_strength": 0,
    # "seed_resize_from_h": -1,
    # "seed_resize_from_w": -1,
    # # "sampler_name": "string",
    # "batch_size": 1,
    # "n_iter": 1,
    # "steps": 5,
    # "cfg_scale": 7,
    # "width": 512,
    # "height": 512,
    # "restore_faces": False,
    # "tiling": False,
    # "do_not_save_samples": False,
    # "do_not_save_grid": False,
    # "negative_prompt": "string",
    # "eta": 0,
    # "s_min_uncond": 0,
    # "s_churn": 0,
    # "s_tmax": 0,
    # "s_tmin": 0,
    # "s_noise": 1,
    # "override_settings": {},
    # "override_settings_restore_afterwards": True,
    # "script_args": [],
    # "sampler_index": "Euler",
    # "include_init_images": False,
    # # "script_name": "string",
    # "send_images": True,
    # "save_images": False,
    # "alwayson_scripts": {}
    }

    print("img2img")

    response = requests.post(url=f'{url}/sdapi/v1/img2img', json=payload).json()

    print("response recieved")

    if "images" not in response:
      print(response)

    for i in response["images"]:
      image = Image.open(io.BytesIO(base64.b64decode(i.split(",",1)[0])))
      image.save("transform.png")
    
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