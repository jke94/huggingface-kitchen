from diffusers import DiffusionPipeline, StableDiffusionXLImg2ImgPipeline
import torch
import random
from datetime import datetime

model = "stabilityai/stable-diffusion-2-1"

pipe = DiffusionPipeline.from_pretrained(
    model,
    torch_dtype=torch.float16,
)
pipe.to("cuda")
pipe.load_lora_weights(
    "./san_isidoro_project_A", 
    weight_name="pytorch_lora_weights.safetensors"
)

prompt = "a european man reading a book in a garden, antique painting, medieval scene, sanisidoro art style"
negative_prompt = "realistic, actual"

generator = torch.Generator("cuda").manual_seed(random.randint(1, 10000))

number_of_images = 10
for i in range(0, number_of_images):
    image = pipe(
        num_inference_steps=35,
        prompt=prompt,
        negative_prompt=negative_prompt,
        guidance_scale=7,
        generator=generator
        ).images[0]

    date = datetime.now()    
    file_name = f'generated_image_{date.strftime("%Y-%m-%d_%H-%M-%S")}_{i}.png'
    image.save(f".\generated_images\{file_name}.png")