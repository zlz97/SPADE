#!/bin/bash

# --- PolitiFact ---
python SPADE.py --dataset_name politifact --n_samples 16 --n_epochs 3 --iters 20
python SPADE.py --dataset_name politifact --n_samples 32 --n_epochs 3 --iters 20
python SPADE.py --dataset_name politifact --n_samples 64 --n_epochs 5 --iters 20
python SPADE.py --dataset_name politifact --n_samples 128 --n_epochs 5 --iters 20

# --- GossipCop ---
python SPADE.py --dataset_name gossipcop --n_samples 16 --n_epochs 3 --iters 20
python SPADE.py --dataset_name gossipcop --n_samples 32 --n_epochs 3 --iters 20
python SPADE.py --dataset_name gossipcop --n_samples 64 --n_epochs 5 --iters 20
python SPADE.py --dataset_name gossipcop --n_samples 128 --n_epochs 5 --iters 20

# --- FANG ---
python SPADE.py --dataset_name fang --n_samples 16 --n_epochs 3 --iters 20
python SPADE.py --dataset_name fang --n_samples 32 --n_epochs 3 --iters 20
python SPADE.py --dataset_name fang --n_samples 64 --n_epochs 5 --iters 20
python SPADE.py --dataset_name fang --n_samples 128 --n_epochs 5 --iters 20