# SPADE: Selective Propagation via Anchor-based Dual-seed Ensemble

This repository contains the code and data used in the anonymous EMNLP submission "SPADE: Selective Propagation via Anchor-based Dual-seed Ensemble".

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
2. Install PyTorch compatible with CUDA 11.1::

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

1. Download the provided `data.zip` file and extract it into the project root directory.
2. Extract it directly into the project root directory.

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
To reproduce the main results (few-shot evaluation) reported in the paper, execute the provided bash script.

```bash
bash run.sh
```
Alternatively, you can run the Python script manually with customized parameters:

```bash
python SPADE.py --dataset_name politifact --n_samples 16 --u_thres 5 --n_epochs 3 --iters 20
```

## 📝 Notes on Reproducibility
* **Runtime** : Due to the few-shot setting, the model trains and evaluates very efficiently. Each run typically finishes within a few minutes.

* **Random Seeds** : All experiments are conducted with fixed random seeds for reproducibility. Final results are averaged over multiple runs with different random seeds.

* **Logging** : The script automatically creates a `logs/` directory during execution to save the evaluation metrics (Accuracy, Precision, Recall, Macro-F1) for each run.
