# Learning When to Ask: Personalized and Context-Aware Query Timing for Human-Robot Interaction (PCAQT-HRI)

This repository contains the implementation and experimental framework for the project **“Learning When to Ask: Personalized and Context-Aware Query Timing for Human-Robot Interaction.”**  
The project is developed as part of the course **INSE 6450 – AI in Systems Engineering** at Concordia University.

---

## 📌 Project Overview

Most existing Human-Robot Interaction (HRI) and active learning research focuses on **what** questions a robot should ask or **which samples** should be labeled. However, very little attention has been given to **when** a robot should ask a human for help.

Asking at the wrong time can:

- annoy users  
- waste labeling budgets  
- reduce learning efficiency  

This project addresses this gap by developing a reinforcement learning–based system that learns to decide:

> **“Is this a good moment to ask a human for help?”**

The system models human availability and interruptibility using contextual information such as:

- user personality traits  
- demographic features  
- activity type  
- social and spatial context  
- number of people/animals around  
- time of day  

A **Double Deep Q-Network (DDQN)** is trained to learn an optimal policy for deciding whether to query a human in real-world scenarios.

---

## Repository Structure

```
PCAQT_HRI/
├── whisper_dataset/
│ └── user_responses/
│ ├── <user_id_1>/
│ │ ├── hri_activity_with_responses_user_<user_id_1>.csv
│ │ └── best_dqn_user_<user_id_1><timestamp>.pt
│ ├── <user_id_2>/
│ └── ...
├── dqn_oracle.py # DQN oracle training utilities (user-wise)
└── README.md
```

---

## Dataset

This project uses a synthetic dataset named **WHISPER-HRI (When Humans in Situated Perception Expect Robot Help)**.

The dataset includes:

- simulated user profiles  
- contextual activity logs  
- human response labels:
  - `HELP`
  - `NO_HELP`
  - `NOISY_HELP`

The data is generated using OpenAI’s GPT-4.1-mini model and is designed to mimic real-world human interruptibility behavior.

---

## ⚙️ Requirements

To run the project, install the required dependencies:

```bash
- Python >= 3.9
- PyTorch (CUDA recommended)
- torchvision
- numpy, pandas, scikit-learn, matplotlib
- tqdm
- OpenAI

> GPU: Experiments in the paper were run on **NVIDIA GeForce RTX 5060**.

---

## Installation

```bash
# 1) Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate        # (Linux/macOS)
# .venv\Scripts\activate         # (Windows)

# 2) Install dependencies
pip install --upgrade pip
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install numpy pandas scikit-learn matplotlib tqdm seaborn openai
```

---

## Usage
1. Train User-Wise DQN Query-Timing Oracles

    This step learns one DQN per user and stores model checkpoints inside each user folder.
```bash
python dqn_oracle.py
```
will automatically save the best DDQN model for each user in:

```bash
whisper_dataset/user_responses/<user_id>/best_dqn_user_<user_id>_%Y%m%d_%H%M%S.pth
```
These models will be used by the oracle and experiment modules for evaluation.

---

## License

This project is licensed under the [MIT License](LICENSE). See the LICENSE file for details.
