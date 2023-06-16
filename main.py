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
        "init_images": [img],
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
                    },
                    {
                        "module": "openpose",
                        "model": "control_v11p_sd15_openpose [cab727d4]",
                    }
                ]
            }
        }
    }

    print("img2img")

    response = requests.post(url=f'{url}/sdapi/v1/img2img', json=payload).json()

    print("response recieved")

    if "images" not in response:
      print(response)

    for i, img in enumerate(response["images"]):
      image = Image.open(io.BytesIO(base64.b64decode(img.split(",",1)[0])))
      image.save(f"transform-{i}.png")

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