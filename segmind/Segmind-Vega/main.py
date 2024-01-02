'''
    Note: Use 'segmind-vega-env.yml' conda environment
'''

from diffusers import StableDiffusionXLPipeline
from datetime import datetime
import torch
import random
import os

def show_CUDA_information():

    if(torch.cuda.is_available()):
        for i in range(torch.cuda.device_count()):
            print(torch.cuda.get_device_properties(f"cuda:{i}"))
    else:
        print('Cuda is not available.')

def main(number_of_images:int, generated_images_folder_path:str):
    
    show_CUDA_information()
    
    # Load pipeline
    
    pipeline = StableDiffusionXLPipeline.from_pretrained(
        "segmind/Segmind-Vega", 
        torch_dtype=torch.float16, 
        use_safetensors=True, 
        variant="fp16",
        )

    pipeline.to("cuda")

    # Setup parameters for inference.

    prompt = "detailed with notes, ink sketch of medieval little village slick design clean lines blueprint perfect awesome"
    negative_prompt = "bad quality"
    num_inference_steps = 35
    generator = torch.Generator(device='cuda').manual_seed(random.randint(0, 1000))
    guidance_scale = 5

    # Run inference.
    
    for i in range(0, number_of_images):
    
        image = pipeline(
            prompt=prompt, 
            negative_prompt=negative_prompt,
            num_inference_steps=num_inference_steps,
            generator=generator,
            guidance_scale=guidance_scale
            ).images[0]

        # Save generated image

        date = datetime.now()
        file_name = f'generated_image_{date.strftime("%Y-%m-%d_%H-%M-%S")}_{i}.png'
        file_path_name = os.path.join(generated_images_folder_path, file_name)

        if not os.path.exists(generated_images_folder_path):
            os.mkdir(generated_images_folder_path)
        
        image.save(file_path_name)

if __name__ == "__main__":
    
    main(number_of_images=3, generated_images_folder_path="generated_images")