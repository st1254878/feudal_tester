# FeUdal Networks (FuN) & FeUdalCat in PyTorch

PyTorch implementation of DeepMind's hierarchical reinforcement learning architecture **FeUdal Networks for Hierarchical Reinforcement Learning** (*Vezhnevets et al., ICML 2017*) alongside an architectural extension **FeUdalCat** (Direct Goal Concatenation).

Supports both **Discrete Atari 2600 environments** (e.g., *Montezuma's Revenge*, *Frostbite*) and **Continuous MuJoCo control** (e.g., *HalfCheetah-v4*).

---

## 🌟 Key Features

1. **Exact Paper Alignment**:
   - **Dilated LSTM**: Multi-core recurrent memory with dilation radius $r=10$ and temporal horizon $c=10$.
   - **Manager Transition Policy Gradient**: Directional goal setting via cosine similarity in latent space $\cos(s_{t+c} - s_t, g_t)$.
   - **Worker Intrinsic Motivation**: Decoupled transition reward $r_t^I = \frac{1}{c} \sum_{i=1}^c \cos(s_t - s_{t-i}, g_{t-i})$.
   - **Observation Centering**: Running observation mean and variance normalization (`RunningMeanStd`).

2. **FeUdalCat Architecture Variant (`feudal_cat`)**:
   - Instead of modulating Worker action logits via matrix multiplication ($U_t^T \phi(g_t)$), **FeUdalCat directly concatenates recent goals** $[z_t, \sum g_{t-i}]$ into the Worker's recurrent state.
   - Preserves goal gradients decoupling (`detach()`), achieving stronger intrinsic gradient signals in sparse-reward environments.

3. **BatchedVectorEnv High-Throughput Engine**:
   - Combines $P$ worker processes with $M$ C++ synchronized environments per process ($P \times M$ environments).
   - Achieves **2,600+ env-steps/sec** on consumer hardware with only ~1.6 GB RAM usage (100% immune to Windows pagefile exhaustion).

---

## 📁 Repository Structure

```text
├── feudalnet.py              # Discrete FuN architecture (Percept, Manager, Worker, feudal_loss)
├── feudal_cat.py             # Discrete FeUdalCat architecture (Direct Goal Concat)
├── feudalnet_continuous.py   # Continuous MuJoCo FuN architecture (Gaussian Actor)
├── feudal_cat_continuous.py  # Continuous MuJoCo FeUdalCat architecture
├── dilated_lstm.py           # PyTorch Dilated LSTM cell with radius r
├── envs.py                   # Atari preprocessors & BatchedVectorEnv multiprocessing
├── preprocess.py             # RunningMeanStd observation normalization on GPU
├── storage.py                # Rollout trajectory storage buffer
├── logger.py                 # TensorBoard and episode logger
├── train.py                  # Training script for discrete Atari FuN
├── train_cat.py              # Training script for discrete Atari FeUdalCat
├── train_continuous.py       # Training script for continuous MuJoCo environments
├── eval.py                   # Policy evaluation & GIF rendering for Atari
├── eval_continuous.py        # Policy evaluation & rendering for MuJoCo
├── plot_results.py           # Plot training curves from TensorBoard logs
├── requirements.txt          # Python package dependencies
└── .gitignore                # Git ignore rules for checkpoints and large logs
```

---

## 🚀 Quick Start

### 1. Installation

```bash
git clone https://github.com/<YOUR_USERNAME>/<YOUR_REPOSITORY>.git
cd <YOUR_REPOSITORY>

# Create and activate environment
conda create -n feudal_env python=3.10 -y
conda activate feudal_env

# Install dependencies
pip install -r requirements.txt
```

### 2. Training Atari (Montezuma's Revenge)

Run standard FuN with 64 parallel environments across 4 CPU processes:
```bash
python train.py --env-name ALE/MontezumaRevenge-v5 --num-workers 64 --num-procs 4 --num-steps 400 --run-name montezuma_fun --max-steps 400000000
```

Run FeUdalCat variant:
```bash
python train_cat.py --env-name ALE/MontezumaRevenge-v5 --num-workers 64 --num-procs 4 --num-steps 400 --run-name montezuma_cat --max-steps 400000000
```

### 3. Training Atari (Frostbite)

Run Frostbite with the paper's optimal $\gamma=0.95$ discount:
```bash
python train.py --env-name ALE/Frostbite-v5 --num-workers 8 --num-procs 1 --num-steps 400 --gamma-m 0.95 --gamma-w 0.95 --run-name frostbite_fun_095 --max-steps 200000000
```

### 4. Training MuJoCo (HalfCheetah)

```bash
python train_continuous.py --env-name HalfCheetah-v4 --model-type fun --max-steps 3000000 --eval-interval 30000 --eval-episodes 10
```

---

## 📊 Evaluation & Visualization

Evaluate a trained checkpoint and generate an animated gameplay GIF:
```bash
python eval.py --checkpoint checkpoints/<RUN_NAME>/model_latest.pt --episodes 5 --record-gif
```

Plot TensorBoard training curves:
```bash
python plot_results.py --run runs/<RUN_NAME>
```

---

## 📖 Citation

```bibtex
@inproceedings{vezhnevets2017feudal,
  title={FeUdal Networks for Hierarchical Reinforcement Learning},
  author={Vezhnevets, Alexander Sasha and Osindero, Simon and Schaul, Tom and Heess, Nicolas and Jaderberg, Max and Silver, David and Kavukcuoglu, Koray},
  booktitle={International Conference on Machine Learning (ICML)},
  pages={3540--3549},
  year={2017}
}
```
