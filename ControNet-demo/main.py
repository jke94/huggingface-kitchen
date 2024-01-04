from diffusers import StableDiffusionControlNetPipeline
from diffusers import ControlNetModel
from diffusers import DPMSolverMultistepScheduler
from diffusers.utils import load_image
from diffusers.utils import make_image_grid
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

def generate_canny_image(image:Image) -> Image: 
    
    # Load image.

    image = np.array(image)

    # Get canny image
    
    image = cv2.Canny(image, 100, 200)
    image = image[:, :, None]
    image = np.concatenate([image, image, image], axis=2)
    canny_image = Image.fromarray(image)
    
    return canny_image

def main(input_image_path_name:str, number_of_images:int):
    
    # Show CUDA information.
    
    show_CUDA_information()
    
    # Generate canny image
    
    original_image = load_image(input_image_path_name)
    canny_image = generate_canny_image(image=original_image)
    
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
    
    pipeline.scheduler = DPMSolverMultistepScheduler.from_config(
        pipeline.scheduler.config
    )
    
    # Inference arguments

    prompts = {
        "A" : "modern house design, wooden house, photorealistic, in the countryside",
        "B" : "modern house design, wooden house, photorealistic, in the countryside",
        "C" : "modern house design, wooden house, oil painting style, in the countryside",
        "D" : "modern house design, oil painting style, in the city",
        "E" : "modern house design, oil painting style, in the city, surrounded by buildings, sunrise",
        "F" : "modern house design, Artstyle Graffiti, in the city, surrounded by buildings, sunrise",
        "G" : "modern house design, pop art style, in the city, surrounded by buildings, summer",
        "H" : "modern house design, Artstyle Graffiti, countryside, summer, beach",
        "I" : "modern house design, Comic Book, countryside, summer, beach",
    }

    negative_prompts = {
        "A" : "dark style, dark colors",
        "B" : ": low quality, blurry, plastic, grainy, duplicate, [deformed | disfigured], poorly drawn, [bad : wrong] anatomy, [extra | missing | floating | disconnected] limb, bad proportions, (mutated hands and fingers), text, name, signature, watermark, logo, autograph, trademark, worst quality, jpeg artifacts, cut off, censored, bad anatomy, bad body, bad face, deformities, (boring, uninteresting:1.1)",
    }
    
    num_inference_steps = 35
    strength=0.35
    guidance_scale=10.5
    generator = torch.Generator(device='cuda').manual_seed(random.randint(0, 1000))
    
    # Run inference.
    
    for i in range(0, number_of_images):
    
        generated_image = pipeline(
            prompt=prompts.get('I'),
            negative_prompt=negative_prompts.get('B'),
            num_inference_steps=num_inference_steps, 
            image=canny_image,
            strength=strength,
            guidance_scale=guidance_scale,
            generator=generator
        ).images[0]
        
        # Create grid images with the images.
        
        output_image = make_image_grid([
            original_image, canny_image, generated_image], rows=1, cols=3)

        # Save generated image.

        generated_images_folder_path = "generated_images"
        
        date = datetime.now()    
        file_name = f'generated_image_{date.strftime("%Y-%m-%d_%H-%M-%S")}_{i}.png'
        file_path_name = os.path.join(generated_images_folder_path, file_name)

        if not os.path.exists(generated_images_folder_path):
            os.mkdir(generated_images_folder_path)
            
        output_image.save(file_path_name)
        
        if os.path.isfile(file_path_name):
            print(f'Image "{file_path_name}" has been saved.')
        else:
            print(f'ERROR, "{file_path_name}" has NOT been saved.')

if __name__ == "__main__":
    
    input_image_path_name = "./images/house_1.jpg"
    
    if os.path.isfile(input_image_path_name):
        main(input_image_path_name=input_image_path_name, number_of_images=3)
    else:
        print(f'ERROR, input image: "{input_image_path_name}" not exists.')