# Timer_satellite

This repo forks from official code[Timer: Generative Pre-trained Transformers Are Large Time Series Models](https://arxiv.org/abs/2402.02368). [[Poster]](https://cloud.tsinghua.edu.cn/f/91da8a3d06984f209461/), [[Slides]](https://cloud.tsinghua.edu.cn/f/b766629dbc584a4e8563/). Timer_satellite modifies attention mechanism and attention mask of Timer to adapt to multivariate prediction with covariates on satellite data.

## Introduction

**Time Series Transformer** (Timer) is a Generative Pre-trained Transformer for general time series analysis.
<p align="center">
<img src="./figures/abilities.png" alt="" align=center />
</p>

## Original Pretrain Datasets

<p align="center">
<img src="./figures/utsd.png" alt="" align=center />
</p>

Unified Time Series Datasets (UTSD) is an well-curated time series to facilitate the research on large time-series models. Its dataset is released in [HuggingFace](https://huggingface.co/datasets/thuml/UTSD) and [[Tsinghua Cloud]](https://cloud.tsinghua.edu.cn/f/93868e3a9fb144fe9719/). You can also access by following method:

```bash
# huggingface-cli login
# export HF_ENDPOINT=https://hf-mirror.com 

python ./scripts/UTSD/download_dataset.py

# dataloader
python ./scripts/UTSD/utsdataset.py
```

## For Developers 

### Environment Preparation

1. Use Python 3.9 and install necessary dependencies.

```
pip install -r requirements.txt
```

2. Put downstream datasets under the folder ```./dataset/```. The dataset file format can follow ```./dataset/xw/pretrain/train.csv``` and you can put multiple time series data files under this folder. **The .csv file's first column should be timestamps with equal interval and the other columns stands for variables. If the target task is multivariate prediction, you only need to ensure the variable order. If the task is prediction with covariate, you should put prediction variable in the leading columns.** Notice that you should manually split dataset with  ```./dataset/${dataset_name}/trainval``` and ```./dataset/${dataset_name}/test``` which means trainval-subset and test-subset respectively.

3. Put the checkpoint from [Google Drive](https://drive.google.com/drive/folders/15oaiAl4OO5gFqZMJD2lOtX2fxHbpgcU8?usp=drive_link) or [Baidu Drive](https://pan.baidu.com/s/1Wj_1_qMgyLNLOSUFZK3weg?pwd=r8i1) under the folder ```./checkpoints/```.

4. Train, evaluate and use addtional funtion of the model. We provide detailed Chinese readme files ```.Timer_satellite_command_details.pdf```.

### Supported Tasks

> **Pretraining** The task aims to pretrain in satellite data for further detailed prediction tasks.

> **Forecasting**: The task aims for full- or few-shot forecasting.

> **Imputation**:  The task aims to segment-level imputation, which is more challenging than point-level imputation.

> **Anomaly Detection**: The task aims to predict normal future series and detect anomalies in advance.

> **Model Pruning**: The task aims to accelerate inference speed with least performance degradation.

> **Data Encryption/Decryption**: The task aims to transfer sensitive data with untrusted network.

> **Differentially Private Training**: The task aims to avoid model to output sensitive data in any way.

> **Interpretability**: The task aims to output visiable metrics and figures for generate physical explanation of model behavior.