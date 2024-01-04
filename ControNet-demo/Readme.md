# Investigations about lllyasviel/sd-controlnet-canny.

Example using Stable Diffusion (v1.5) with ControlNet (with canny edge)

- [Huggingface: lllyasviel/sd-controlnet-canny](https://huggingface.co/lllyasviel/sd-controlnet-canny)

## A. Samples generated.

![generated_image_2024-01-04_20-11-59_0](https://github.com/jke94/huggingface-kitchen/assets/53972851/7d7626c6-95af-4e5b-891d-6d1512751ff8)

![generated_image_2024-01-04_20-05-19_0](https://github.com/jke94/huggingface-kitchen/assets/53972851/a93c8d0d-099e-4da7-97ad-d40b9a03b0de)

![generated_image_2024-01-04_20-05-56_2](https://github.com/jke94/huggingface-kitchen/assets/53972851/5f98b9a2-69d3-4007-9c9d-83fed19b8352)

## B. Conda development environment

### 1. How to create conda **controlnet-demo-env** environment

```
conda create -n controlnet-demo-env python=3.10
```

### 2. How to export **controlnet-demo-env**.

1. Activate env:

```
conda activate controlnet-demo-env
```

2. Export environment (the environment needs to be activated):

```
conda env export > controlnet-demo-env.yml
```

### 3. How to delete **controlnet-demo-env** environment.

```
conda remove --name controlnet-demo-env --all
```

### 4. How to create **controlnet-demo-env** from conda environment file (*.yml).

```
conda env create -f controlnet-demo-env.yml
```

#
