import os
from pathlib import Path
import cv2
import numpy as np
from PIL import Image
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, UniPCMultistepScheduler
from controlnet_aux import OpenposeDetector
import torch
import fire
from rich.progress import Progress
import logging


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

    def convert_frame(self, frame, prompt):
        # Convert the frame to a PIL image
        frame_image = Image.fromarray(frame)
        # Generate pose and canny images
        # pose_image = self.get_pose(frame_image)
        canny_image = self.get_canny(frame_image)
        # Generate the output image using the pipeline

        output = self.pipe(
            prompt,
            image=[canny_image],
            negative_prompt="monochrome, lowres, bad anatomy, worst quality, low quality",
            num_inference_steps=30,
            generator=[self.generator],
        )
        # Convert the output image back to a frame
        output_frame = np.array(output.images[0])
        return output_frame
    
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
            num_inference_steps=30,
            generator=[self.generator]*len(frame_batch),
        )
        # Convert the output images back to frames
        output_frame_batch = [np.array(output_image) for output_image in output_batch.images]
        return output_frame_batch

    def _write_batch(self, out, converted_frame_batch, task, progress):
        for converted_frame in converted_frame_batch:
            out.write(converted_frame)
            progress.update(task, advance=1)

    def convert(self, input_path: str, output_path: str, prompt: str):
        # Delete the output file if it exists
        if os.path.exists(output_path):
            os.remove(output_path)

        cap = cv2.VideoCapture(input_path)
        # Get video properties (fps, width, height) to write the output video

        out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"mp4v"), 24, (640, 480))

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        batch_size = 16
        frame_batch = []

        with Progress() as progress:
            task = progress.add_task("[cyan]Processing...", total=total_frames)

            while not progress.finished:
                ret, frame = cap.read()
                if not ret:
                    continue  # Skip the frame if it was not read correctly

                frame_batch.append(frame)
                if len(frame_batch) == batch_size:
                    converted_frame_batch = frame_batch # self._process_batch(frame_batch, prompt)
                    self._write_batch(out, converted_frame_batch, task, progress)
                    frame_batch = []  # Reset the batch
                    break

            # Process remaining frames in the batch if any
            if frame_batch: 
                converted_frame_batch = frame_batch # self._process_batch(frame_batch, prompt)
                self._write_batch(out, converted_frame_batch, task, progress)

        cap.release()
        out.release()


if __name__ == "__main__":
    fire.Fire(VideoConverter)
