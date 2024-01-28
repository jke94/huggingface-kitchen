from diffusers import DiffusionPipeline, StableDiffusionXLImg2ImgPipeline
import torch
import random

model = "runwayml/stable-diffusion-v1-5"

pipe = DiffusionPipeline.from_pretrained(
    model,
    torch_dtype=torch.float16,
)
pipe.to("cuda")
pipe.load_lora_weights(
    "./carlo_project", 
    weight_name="pytorch_lora_weights.safetensors"
)

prompt = "carlo photo portrait, film noir style, monochrome, high contrast, dramatic shadows, 1940s style, mysterious, cinematic"
generator = torch.Generator("cuda").manual_seed(random.randint(1, 1000))

image = pipe(
    num_inference_steps=35,
    prompt=prompt, 
    generator=generator
    ).images[0]

image.save(f"generated_image.png")