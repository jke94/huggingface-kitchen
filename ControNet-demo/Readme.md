# Investigations about lllyasviel/sd-controlnet-canny.

Example using Stable Diffusion (v1.5) with ControlNet (with canny edge)

- [Huggingface: lllyasviel/sd-controlnet-canny](https://huggingface.co/lllyasviel/sd-controlnet-canny)

## A. Samples generated.

TODO:

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