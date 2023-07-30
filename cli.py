from collections import defaultdict
import os
from pathlib import Path
from typing import Any
import cv2
import numpy as np
from PIL import Image
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, UniPCMultistepScheduler
from controlnet_aux import OpenposeDetector
import torch
import fire
from rich.progress import Progress, track
import logging
import torch.nn.functional as F
import torchvision.transforms.functional as TF
import kornia


from tqdm import tqdm

from helpers.video import VideoProcessor

def resize_to_fit(tensor: torch.Tensor) -> torch.Tensor:
    """
    Resize the tensor so that both dimensions are divisible by 8.
    """
    height, width = tensor.shape[-2:]
    new_height = (height // 8) * 8
    new_width = (width // 8) * 8
    return TF.resize(tensor, size=(new_height, new_width))

def correct_offset(images: torch.Tensor, offset: int = 1) -> torch.Tensor:
    corrected_images = []
    for i, img in enumerate(images):
        shift = -i // offset  # Upward shift equal to the index
        corrected_img = torch.roll(img, shifts=shift, dims=-2)  # Shift along the height dimension
        corrected_images.append(corrected_img)
    return torch.stack(corrected_images)

class Commands:
    dtype = torch.float16

    def __init__(self):
        logging.basicConfig(level=logging.INFO)
        logging.info('Initializing VideoConverter...')

        # Initialize necessary objects like ControlNet models and pipeline
        logging.info('Loading ControlNet models...')
        self.controlnet = ControlNetModel.from_pretrained("lllyasviel/sd-controlnet-canny", torch_dtype=self.dtype)

        logging.info('Initializing pipeline...')
        model_id = "runwayml/stable-diffusion-v1-5"  # TODO: enable cetus-mix
        self.pipe = StableDiffusionControlNetPipeline.from_pretrained(model_id, controlnet=self.controlnet, torch_dtype=self.dtype, safety_checker=None)
        self.pipe.scheduler = UniPCMultistepScheduler.from_config(self.pipe.scheduler.config)
        self.pipe.enable_model_cpu_offload()
        self.pipe.enable_vae_tiling()
        self.pipe.enable_xformers_memory_efficient_attention()

        # logging.info('Loading OpenposeDetector...')
        # self.openpose = OpenposeDetector.from_pretrained("lllyasviel/ControlNet")  # Assuming OpenposeDetector is available

        self.device = self.pipe.device
        logging.info('VideoConverter initialized.')

    @property
    def generator(self):
        return torch.Generator(device=self.device).manual_seed(2)

    def get_pose(self, image: Image) -> Any:
        return self.openpose(image)
    
    def _process_frames(self, frame_batch: list[np.ndarray], prompt: str, neg_prompt: str = "monochrome, lowres, bad anatomy, worst quality, low quality") -> np.ndarray:
        if not frame_batch:
            return []

        # Convert the frame batch to tensors
        input_batch = torch.from_numpy(np.stack(frame_batch)).to(self.device).div(255).permute(0, 3, 1, 2)
        t = resize_to_fit(input_batch)
        
        t_canny = kornia.filters.canny(t, hysteresis=False)[0].expand_as(t)

        n = t.size(0)
        t = self.pipe(
            [prompt]*n,  # Repeat the prompt for each frame in the batch
            negative_prompt=[neg_prompt]*n,
            image=t_canny,  # Pass canny images to the pipeline
            num_inference_steps=30,
            generator=self.generator,
            output_type="pt"
        ).images
        t = TF.resize(t, size=input_batch.shape[-2:])

        return t.mul(255).permute(0, 2, 3, 1).to(torch.uint8).cpu().numpy()
    
    def convert(self, input_path: str, output_path: str, prompt: str) -> None:
        with VideoProcessor(input_path, output_path, batch_size=24) as video:
            for frames in track(video, total=len(video)):
                converted_frames = self._process_frames(frames, prompt)
                video.write(converted_frames)

# example usage:
# python3 cli.py convert viral-guy-with-mask.mp4 test.mp4 "mr potato head, best quality, extremely detailed"
if __name__ == "__main__":
    fire.Fire(Commands)
    