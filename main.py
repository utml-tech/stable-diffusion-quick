import fire
import requests
from PIL import Image
import io
import base64

class Styler:

  def transform(self, input, output, model, metadata):
    print('Transforming data from {} to {} using model {} with metadata {}'.format(input, output, model, metadata))

  def test(self):
    payload = {
        "prompt": "maltese puppy",
        "steps": 5
    }

    response = requests.post(url='http://127.0.0.1:7860/sdapi/v1/img2img', json=payload)

    for i in response.json()['images']:
      image = Image.open(io.BytesIO(base64.b64decode(i.split(",",1)[0])))

      image.save("test.png")

if __name__ == '__main__':
  fire.Fire(Styler)