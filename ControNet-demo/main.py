from diffusers import StableDiffusionControlNetPipeline
from diffusers import ControlNetModel
from diffusers import UniPCMultistepScheduler, EulerDiscreteScheduler,DPMSolverMultistepScheduler
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

def generate_canny_image( image_file:str, canny_image_name:str, 
    save_canny_image:bool = True) -> Image: 
    
    # Load image.
    
    original_image = load_image(image_file)
    image = np.array(original_image)

    # Get canny image
    
    image = cv2.Canny(image, 100, 200)
    image = image[:, :, None]
    image = np.concatenate([image, image, image], axis=2)
    canny_image = Image.fromarray(image)
    
    if save_canny_image:
        canny_image.save(f'{canny_image_name}')
    
    return canny_image

def main():
   
    image_file = "./pajaro_carpintero_0.jpg"
    canny_image_name = './canny_image_pajaro_carpintero_0.png'
    generated_image_name = "./output_pajaro_carpintero_0.png"
    
    # Generate canny image
    
    canny_image = generate_canny_image(
        image_file=image_file,
        canny_image_name=canny_image_name)
    
    # Building ControlNetModel

    controlnet = ControlNetModel.from_pretrained(
        "lllyasviel/sd-controlnet-canny",
        torch_dtype=torch.float16
    )
    
    # Building pipeline to use with ControlNet
    
    pipeline = StableDiffusionControlNetPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        controlnet=controlnet, 
        torch_dtype=torch.float16
    )
    
    pipeline.to("cuda")
    
    # pipeline.scheduler = UniPCMultistepScheduler.from_config(
    #     pipeline.scheduler.config
    # )

    # pipeline.scheduler = EulerDiscreteScheduler.from_config(
    #     pipeline.scheduler.config
    # )
    
    pipeline.scheduler = DPMSolverMultistepScheduler.from_config(
        pipeline.scheduler.config
    )
    
    # Inference arguments

    prompt = "bird"
    negative_prompt = "dark style, bright colors"
    num_inference_steps = 45
    strength=0.15
    guidance_scale=15.5
    generator = torch.Generator(device='cuda').manual_seed(random.randint(0, 1000))
    
    # Run inference.
    
    generated_image = pipeline(
        prompt=prompt,
        negative_prompt=negative_prompt,
        num_inference_steps=num_inference_steps, 
        image=canny_image,
        strength=strength,
        guidance_scale=guidance_scale,
        generator=generator
    ).images[0]
    
    # Save generated image.
    
    generated_image.save(generated_image_name)

if __name__ == "__main__":
    
    main()