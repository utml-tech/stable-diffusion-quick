from collections import defaultdict
import numpy as np
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, UniPCMultistepScheduler
import torch
import fire
from rich.progress import track
import logging
import torchvision.transforms.functional as TF
import kornia
from controlnet_aux.processor import Processor
import math
from tqdm import tqdm

from helpers.video import VideoProcessor

def resize_to_fit(tensor: torch.Tensor) -> torch.Tensor:
    """
    Resize the tensor so that both dimensions are divisible by 8.
    """
    MUL = 8
    height, width = tensor.shape[-2:]
    new_height = math.ceil(height / MUL) * MUL
    new_width = math.ceil(width / MUL) * MUL
    return TF.resize(tensor, size=(new_height, new_width))

def correct_offset(images: torch.Tensor, offset: int = 1) -> torch.Tensor:
    corrected_images = []
    for i, img in enumerate(images):
        shift = -i // offset  # Upward shift equal to the index
        corrected_img = torch.roll(img, shifts=shift, dims=-2)  # Shift along the height dimension
        corrected_images.append(corrected_img)
    return torch.stack(corrected_images)

def to_rgb_cpu(images: torch.Tensor) -> torch.Tensor:
    return images.mul_(255).permute(0, 2, 3, 1).to(torch.uint8).cpu().numpy()

def to_rgb(images: np.ndarray) -> np.ndarray:
    # Multiply by 255
    images = images * 255
    # Ensure the datatype is uint8
    images = images.astype(np.uint8)
    # Move channels to the last dimension
    images = np.transpose(images, (0, 2, 3, 1))
    return images

class Commands:
    dtype = torch.float16

    def __init__(self):
        logging.basicConfig(level=logging.INFO)
        logging.info('Initializing VideoConverter...')
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

        logging.info('Loading Processors...')
        self.canny = Processor("canny").processor
        self.openpose = Processor("openpose_full").processor.to(self.device)
        self.depth = Processor("depth_midas").processor.to(self.device)

        # Initialize necessary objects like ControlNet models and pipeline
        logging.info('Loading ControlNet models...')
        self.controlnets = [
            ControlNetModel.from_pretrained("lllyasviel/sd-controlnet-canny", torch_dtype=self.dtype),
            # ControlNetModel.from_pretrained("lllyasviel/sd-controlnet-depth", torch_dtype=self.dtype),
            # ControlNetModel.from_pretrained("lllyasviel/sd-controlnet-openpose", torch_dtype=self.dtype),
        ]

        # logging.info('Initializing pipeline...')
        model_id = "runwayml/stable-diffusion-v1-5"  # TODO: enable cetus-mix
        self.pipe = StableDiffusionControlNetPipeline.from_pretrained(model_id, controlnet=self.controlnets, torch_dtype=self.dtype, safety_checker=None)
        self.pipe.scheduler = UniPCMultistepScheduler.from_config(self.pipe.scheduler.config)
        self.pipe.enable_model_cpu_offload()
        self.pipe.enable_vae_tiling()
        self.pipe.enable_xformers_memory_efficient_attention()

        logging.info('VideoConverter initialized.')

    @property
    def generator(self):
        return torch.Generator(device=self.device).manual_seed(2)
    
    def _process_frames(self, frame_batch: list[np.ndarray], prompt: str, neg_prompt: str = "monochrome, lowres, bad anatomy, worst quality, low quality") -> np.ndarray:
        if not frame_batch:
            return []

        t = np.stack(frame_batch)
        t = torch.from_numpy(t)
        t = t.to(self.device)
        t = t.to(self.dtype)
        t = t.div(255)
        t = t.permute(0, 3, 1, 2)
        t = resize_to_fit(t)

        # img = t[0].mul(255).movedim(0, -1).byte().cpu().numpy()
        # t_depth = self.depth(img)
        # t_depth = torch.from_numpy(t_depth).to(self.device, self.dtype).movedim(-1, 0).unsqueeze_(0)

        # t_pose = np.array(self.openpose(img))
        # t_pose = torch.from_numpy(t_pose).to(self.device, self.dtype).movedim(-1, 0).unsqueeze_(0)

        t_canny = kornia.filters.canny(t, hysteresis=False)[0].expand_as(t)
        # t_canny = TF.resize(t_canny, size=t_depth.shape[-2:], antialias=True)

        control_images = [t_canny]  # , t_depth, t_pose

        # assert t_canny.shape == t_depth.shape == t_pose.shape, "Shapes of inputs must be equal"

        n = t.size(0)
        out = self.pipe(
            [prompt] * n,  # Repeat the prompt for each frame in the batch
            negative_prompt=[neg_prompt] * n,
            image=control_images,
            num_inference_steps=5,
            generator=self.generator,
            output_type="pt"
        ).images

        out = TF.resize(out, size=t.shape[-2:], antialias=True)
        out = out.mul(255)
        out = out.permute(0, 2, 3, 1)
        out = out.to(torch.uint8)
        out = out.cpu()
        out = out.numpy()
        return out
    
    def convert(self, input_path: str, output_path: str, prompt: str) -> None:
        with VideoProcessor(input_path, output_path, batch_size=1) as video:
            for i, frames in enumerate(video):
                converted_frames = self._process_frames(frames, prompt)
                video.write(converted_frames)
                
                if i > 5:
                    break

# example usage:
# python3 cli.py convert viral-guy-with-mask.mp4 test.mp4 "mr potato head, best quality, extremely detailed"
if __name__ == "__main__":
    fire.Fire(Commands)
    