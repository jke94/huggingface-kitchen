from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, UniPCMultistepScheduler
from diffusers.utils import load_image
from datetime import datetime
import torch
import random
import os
import cv2
import torch
import numpy as np
from PIL import Image

def show_CUDA_information():

    if(torch.cuda.is_available()):
        for i in range(torch.cuda.device_count()):
            print(torch.cuda.get_device_properties(f"cuda:{i}"))
    else:
        print('Cuda is not available.')

def generate_canny_image( image_file:str, output_canny_image_name:str, 
    save_canny_image:bool = True) -> Image: 
    
    original_image = load_image(image_file)
    image = np.array(original_image)

    # Get canny image
    image = cv2.Canny(image, 100, 200)
    image = image[:, :, None]
    image = np.concatenate([image, image, image], axis=2)
    canny_image = Image.fromarray(image)
    
    if save_canny_image:
        canny_image.save(f'{output_canny_image_name}')
    
    return canny_image

def main():
   
    canny_image = generate_canny_image(
        image_file="./pajaro_carpintero_0.jpg",
        output_canny_image_name='./canny_image_pajaro_carpintero_0.png')

    controlnet = ControlNetModel.from_pretrained(
        "lllyasviel/sd-controlnet-canny",
        torch_dtype=torch.float16
    )
    pipeline = StableDiffusionControlNetPipeline.from_pretrained(
        # "stabilityai/stable-diffusion-2-1", 
        "runwayml/stable-diffusion-v1-5", 
        controlnet=controlnet, 
        torch_dtype=torch.float16
    ).to("cuda")
    
    pipeline.scheduler = UniPCMultistepScheduler.from_config(
        pipeline.scheduler.config
    )

    prompt = "bird"
    negative_prompt = "dark style, bright colors"
    num_inference_steps = 45
    strength=0.15
    guidance_scale=15.5
    generator = torch.Generator(device='cuda').manual_seed(random.randint(0, 1000))
    
    generated_image = pipeline(
        prompt=prompt,
        negative_prompt=negative_prompt,
        num_inference_steps=num_inference_steps, 
        image=canny_image,
        strength=strength,
        guidance_scale=guidance_scale,
        generator=generator
    ).images[0]
    
    generated_image.save("./output_pajaro_carpintero_0.png")

if __name__ == "__main__":
    
    main()