import os
from pathlib import Path
import cv2
import numpy as np
from PIL import Image
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, UniPCMultistepScheduler
from controlnet_aux import OpenposeDetector
import torch
import fire
from rich.progress import Progress, track
import logging

from tqdm import tqdm

from helpers.video import VideoProcessor


class VideoConverter:

    def __init__(self):
        logging.basicConfig(level=logging.INFO)
        logging.info('Initializing VideoConverter...')
        self.generator = torch.Generator(device="cuda").manual_seed(2)

        # Initialize necessary objects like ControlNet models and pipeline
        logging.info('Loading ControlNet models...')
        model_id = "runwayml/stable-diffusion-v1-5"  # TODO: enable cetus-mix
        self.controlnet = ControlNetModel.from_pretrained("lllyasviel/sd-controlnet-canny", torch_dtype=torch.float16)

        logging.info('Initializing pipeline...')
        self.pipe = StableDiffusionControlNetPipeline.from_pretrained(
            model_id, controlnet=self.controlnet, torch_dtype=torch.float16, safety_checker=None
        )
        self.pipe.scheduler = UniPCMultistepScheduler.from_config(self.pipe.scheduler.config)
        self.pipe.enable_model_cpu_offload()
        self.pipe.enable_xformers_memory_efficient_attention()

        # logging.info('Loading OpenposeDetector...')
        # self.openpose = OpenposeDetector.from_pretrained("lllyasviel/ControlNet")  # Assuming OpenposeDetector is available

        logging.info('VideoConverter initialized.')

    def get_pose(self, image):
        return self.openpose(image)

    def get_canny(self, image):
        # Converting the image to a numpy array
        img = np.array(image)
        low_threshold = 100
        high_threshold = 200
        img = cv2.Canny(img, low_threshold, high_threshold)
        img = img[:, :, None]
        img = np.concatenate([img, img, img], axis=2)
        return Image.fromarray(img)
    
    def _process_batch(self, frame_batch, prompt):
        if not frame_batch:
            return []

        # Convert the frame batch to a PIL image batch
        frame_image_batch = [Image.fromarray(frame) for frame in frame_batch]
        # Generate canny images
        canny_image_batch = [self.get_canny(frame_image) for frame_image in frame_image_batch]
        # Generate the output image using the pipeline
        output_batch = self.pipe(
            [prompt]*len(frame_batch),  # Repeat the prompt for each frame in the batch
            image=canny_image_batch,  # Pass canny images to the pipeline
            negative_prompt=["monochrome, lowres, bad anatomy, worst quality, low quality"]*len(frame_batch),
            num_inference_steps=5,
            generator=self.generator,
        )
        # Convert the output images back to frames
        return [np.array(output_image) for output_image in output_batch.images]

    def convert(self, input_path: str, output_path: str, prompt: str):
        with VideoProcessor(input_path, output_path, batch_size=16) as video:
            for frames in tqdm(video, total=len(video)):
                converted_frames = self._process_batch(frames, prompt)
                video.write(converted_frames)

# example usage:
# python3 cli.py convert viral-guy-with-mask.mp4 test.mp4 "mr potato head, best quality, extremely detailed"
if __name__ == "__main__":
    fire.Fire(VideoConverter)
    