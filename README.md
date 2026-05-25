# SPADE: Selective Propagation via Anchor-based Dual-seed Ensemble

[![Status: Under Review](https://img.shields.io/badge/Status-Under_Review-blue.svg)]()
[![Task: Few-Shot Fake News Detection](https://img.shields.io/badge/Task-Few--Shot_Fake_News_Detection-brightgreen.svg)]()

> This repository contains the code and processed datasets for the anonymous EMNLP submission introducing the **SPADE** framework.
> 
> All identifying information has been removed to preserve anonymity during the review process.
## 📂 Project Structure

```text
SPADE/
├── Process/
│   ├── lm_loadsplits.py        # Data splitting and processing scripts
│   └── adj_matrix_fewshot.py   # Script to construct adjacency matrices from raw social context
├── data/                       # Extracted from data.zip
│   ├── adjs/                   # Pre-processed adjacency matrices (news proximity graphs)
│   ├── news_articles_raw/      # Collected news texts and metadata
│   ├── social_context_raw/     # Processed user-news interaction records
│   ├── fang_test.csv           # Test set containing news articles and ground-truth labels
│   └── ...                     # Other dataset splits (e.g., politifact, gossipcop)
├── SPADE.py                    # Main training and evaluation script
├── run.sh                      # Shell script to reproduce experiments
├── requirements.txt            # Core dependencies
└── README.md
```

## ⚙️ Environment Setup
The code has been tested with Python 3.7 and PyTorch 1.8.0 (CUDA 11.1). Experiments can be run on a single NVIDIA GPU with at least 12GB of memory.

1. Create and activate a clean environment:

```bash
conda create -n spade python=3.7
conda activate spade
```
2. Install PyTorch compatible with CUDA 11.1:

```bash
pip install torch==1.8.0 torchvision==0.9.0 torchaudio==0.8.0 -f https://download.pytorch.org/whl/cu111/torch_stable.html
```
3. Install dependencies:

```bash
pip install -r requirements.txt
```
*(Note: The pre-trained bert-base-uncased weights will be automatically downloaded during the first run.)*

## 📥 Data Preparation
Due to file size limitations, the preprocessed graph datasets (adjacency matrices and few-shot splits) are provided as a separate compressed file (data.zip).

1. Download the provided `data.zip` file.
2. Extract it into the project root directory.

After extraction, ensure your directory structure matches the following:

```text
SPADE/
├── data/
│   └── adjs/
│       └── ... (.pkl files)
├── SPADE.py
└── ...
```

## 🚀 Running Experiments
To reproduce the main results reported in the paper, execute the provided bash script.

```bash
bash run.sh
```
Alternatively, you can run the Python script manually with customized parameters:

```bash
python SPADE.py --dataset_name politifact --n_samples 16 --u_thres 5 --n_epochs 3 --iters 20
```

## 📝 Notes on Reproducibility
**Runtime:** Due to the few-shot setting, the model trains and evaluates efficiently. A single iteration typically finishes within a few minutes, meaning the full experimental process is completed very quickly.

* **Random Seeds:** All experiments are conducted using predefined random seeds for reproducibility. Final results are averaged over multiple runs.

* **Logging** : The script automatically creates a `logs/` directory during execution to save the evaluation metrics (Accuracy, Precision, Recall, Macro-F1) for each run.
