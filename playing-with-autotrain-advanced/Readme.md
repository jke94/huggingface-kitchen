## B. Conda development environment

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