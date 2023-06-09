
# Stable Diffusion Video styler

A CLI that recieves a video as input followed by style image (metadata) or prompt and SD model and converts it accordingly.

## Example

```bash
python3 main.py \
    --input input.mp4  \
    --output output.mp4 \
    --model models/512x512_diffusion_uncond_finetune_008100.pt \
    --metadata metadata/1.png
```