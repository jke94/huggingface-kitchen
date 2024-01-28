# playing-with-autotrain-advanced

## A. autotrain-advanced

- [huggingface/autotrain-advanced](https://github.com/huggingface/autotrain-advanced)

## B. Dreambooth

- [dreambooth.github.io](https://dreambooth.github.io/)

## C. Working with autotrain-advanced installed with _pip_.

- Input example 1:

```
autotrain dreambooth `
--project-name "san_isidoro_project_A" `
--model "stabilityai/stable-diffusion-2-1" `
--image-path "fine-tuning_images_san_isidoro/" `
--prompt "photo of sanisidoro art style" `
--resolution 1024 `
--batch-size 1 `
--num-steps 200 `
--fp16 `
--gradient-accumulation 4 `
--lr 1e-4
```

- Input example 2:

```
autotrain dreambooth `
--project-name "carlo_project" `
--model "runwayml/stable-diffusion-v1-5" `
--image-path "fine-tuning_images_carlo" `
--prompt "photo of carlo person" `
--resolution 512 `
--batch-size 1 `
--num-steps 200 `
--fp16 `
--gradient-accumulation 4 `
--lr 1e-4
```

## D. Conda development environment

### 1. How to create conda **playing-with-autotrain-advanced-env** environment

```
conda create -n playing-with-autotrain-advanced-env python=3.10
```

## B. Conda development environment

### 1. How to create conda **controlnet-demo-env** environment

```
conda create -n playing-with-autotrain-advanced-env python=3.10
```

### 2. How to export **controlnet-demo-env**.

1. Activate env:

```
conda activate playing-with-autotrain-advanced-env
```

2. Export environment (the environment needs to be activated):

```
conda env export > playing-with-autotrain-advanced-env.yml
```

### 3. How to delete **controlnet-demo-env** environment.

```
conda remove --name playing-with-autotrain-advanced-env --all
```

### 4. How to create **controlnet-demo-env** from conda environment file (*.yml).

```
conda env create -f playing-with-autotrain-advanced-env.yml
```
