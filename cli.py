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
            ControlNetModel.from_pretrained("lllyasviel/sd-controlnet-canny", torch_dtype=self.dtype)
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
        
        img = frame_batch[0]

        # return frame_batch

        # Convert the frame batch to tensors
        # input_batch = torch.from_numpy(np.stack(frame_batch)).to(self.device, self.dtype).div(255).permute(0, 3, 1, 2)
        # shape: (frame, channel, height, width)

        # t = resize_to_fit(input_batch)
        # t_canny = kornia.filters.canny(t, hysteresis=False)[0].expand_as(t)
        # t.mul_(255).permute(0, 2, 3, 1).to(torch.uint8).cpu().numpy()

        t_canny = self.canny(img)
        # t_depth = self.depth(img)[np.newaxis, ...]
        # t_pose = np.array(self.openpose(img))[np.newaxis, ...]

        t = torch.from_numpy(t_canny).unsqueeze_(0).div_(255).movedim(-1, 0)
        # return TF.resize(t, size=img.shape[:-1], antialias=False).movedim(0, -1).numpy()

        t = self.pipe(
            prompt,  # Repeat the prompt for each frame in the batch
            negative_prompt=neg_prompt,
            image=[t],
            num_inference_steps=5,
            generator=self.generator,
            output_type="pt"
        ).images

        return TF.resize(t, size=img.shape[:-1], antialias=True).movedim(1, -1).cpu().numpy()
    
    def convert(self, input_path: str, output_path: str, prompt: str) -> None:
        with VideoProcessor(input_path, output_path, batch_size=1) as video:
            for i, frames in enumerate(video):
                converted_frames = self._process_frames(frames, prompt)
                video.write(converted_frames)

                if i > 10:
                    break

# example usage:
# python3 cli.py convert viral-guy-with-mask.mp4 test.mp4 "mr potato head, best quality, extremely detailed"
if __name__ == "__main__":
    fire.Fire(Commands)
    