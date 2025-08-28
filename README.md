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

***Tips: Owing to data confidentiality constraints, we are unable to release the satellite dataset or the associated checkpoints. Nevertheless, researchers may construct the dataset in accordance with the procedures outlined above, and initiate training from the Timer checkpoint to obtain the corresponding model weights on the satellite data.***

### Features

> **Modular Design**: The project is organized with a modular structure, making it easy to extend and maintain.  
> **Unified Argument Management**: Parameters are centrally managed through a root-level `ArgParser`, ensuring consistency across modules.  
> **Reproducible Experiments**: Example scripts are provided for running the code, enabling convenient experiment management and result reproduction.

### Project Structures
```
Timer
├── checkpoints     # Initial and trained model weights
│       ├── Timer_forecast_1.0.ckpt     # Initial weights for time-series forecasting
│       ├── Timer_imputation_1.0.ckpt     # Initial weights for data imputation
│       ├── Timer_anomaly_detection_1.0.ckpt     # Initial weights for anomaly detection
│       ├── Timer_bias_forecast_1.0.ckpt     # Forecasting with feature bias
│       ├── Timer_bias_imputation_1.0.ckpt      # Imputation with feature bias
│       └── Timer_bias_anomaly_detection_1.0.ckpt     # Anomaly detection with feature bias
├── data_provider     # Dataset-related modules
│       ├── data_factory.py     # Utility for selecting dataset by input args
│       └── data_loader.py     # Dataset class definitions
├── dataset     # Data files directory
├── exp     # Experiment classes
│    ├── exp_basic.py     # Base class for training/testing
│    ├── exp_forecast.py     # Training/testing for forecasting
│    ├── exp_imputation.py     # Training/testing for imputation
│    └── exp_anomaly_forecast.py     # Training/testing for anomaly detection
├── layers     # Model sub-layers
│     ├── Attn_Bias.py     # Bias across features
│     ├── Embed.py     # Positional encoding
│     ├── SelfAttention_Family.py     # Custom attention mechanisms
│     └── Transformer_EncDec.py     # Encoder/decoder layers
├── models     # Model definitions
│     ├── Timer_multivariate.py     # Main model with task-specific forward()
│     └── TimerBackbone_multivariate.py     # Backbone model (task-agnostic forward)
├── pretrain_data_gen/ # Pretraining data generation
│       └── general_model_data_process.py     # Pretraining data generation script
├── scripts     # Shell scripts for different tasks
├── test_results     # Output of test results
├── utils     # Utility functions
│     ├── masking.py     # Attention masks
│     └── tools.py     # Training utilities
├── draw_figure.py     # Visualization (e.g., gradient norms, epsilon)
├── enc_dec_paillier.py     # Data encryption/decryption with Paillier
├── train_utils.py     # Training-related utility functions
├── run.py     # Argument parsing & main entry for training/testing
└── requirements.txt     # Python dependencies
```

### Detailed Usage

Waiting for Implemention, you could see [Chinese ver](./Timer_satellite_command_details.pdf).

### Supported Tasks

> **Pretraining** The task aims to pretrain in satellite data for further detailed prediction tasks.

> **Forecasting**: The task aims for full- or few-shot forecasting.

> **Imputation**:  The task aims to segment-level imputation, which is more challenging than point-level imputation.

> **Anomaly Detection**: The task aims to predict normal future series and detect anomalies in advance.

> **Model Pruning**: The task aims to accelerate inference speed with least performance degradation.

> **Data Encryption/Decryption**: The task aims to transfer sensitive data with untrusted network.

> **Differentially Private Training**: The task aims to avoid model to output sensitive data in any way.

> **Interpretability**: The task aims to output visiable metrics and figures for generate physical explanation of model behavior.