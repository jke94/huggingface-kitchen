# Investigations about segmind/Segmind-Vega AI model.

- [Huggingface: segmind/Segmind-Vega](https://huggingface.co/segmind/Segmind-Vega)


## 1. How to create conda **segmind-vega-env** environment

```
conda create -n segmind-vega-env python=3.10
```

## 2. How to export **segmind-vega-env**.

1. Activate env:

```
conda activate segmind-vega-env
```

2. Export environment (the environment needs to be activated):

```
conda env export > segmind-vega-env.yml
```

## 3. How to delete **segmind-vega-env** environment.

```
conda remove --name segmind-vega-env --all
```

## 4. How to create **segmind-vega-env** from conda environment file (*.yml).

```
conda env create -f segmind-vega-env.yml
```