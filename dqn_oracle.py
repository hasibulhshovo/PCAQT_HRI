import os
import json
import random
from collections import deque
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim

from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split

# --------------------------
# Repro
# --------------------------
def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
seed_everything(42)


# --------------------------
# Filesystem helpers
# --------------------------
def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def save_plot(values, title, ylabel, save_path):
    plt.figure(figsize=(10, 6))
    plt.plot(values)
    plt.title(title)
    plt.xlabel("Epoch")
    plt.ylabel(ylabel)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


# --------------------------
# Reward Policy
# --------------------------
REWARD_MAP = {
    1: {
        'help': 2.0,
        'no_help': -2.0,
        'noisy_help': -1.0,
    },
    0: {
        'help': -3.0,
        'no_help': 2.0,
        'noisy_help': 1.0,
    }
}

# --------------------------
# Environment
# --------------------------
class RL_Env:
    """
    State: encoded feature vector
    Action: 0 = no_ask, 1 = ask
    Reward:
        If action == ask:
            help -> +2.0
            no_help -> -2.0
            noisy_help -> -1.0
        If action == no_ask:
            help -> -3.0
            no_help -> +2.0
            noisy_help -> +1.0
    Transitions: iterate through dataset in order; one episode = one pass.
    """
    def __init__(self, X, y):
        self.X = np.asarray(X, dtype=np.float32)
        self.y = np.asarray(y, dtype=object)
        self.n_actions = 2
        self.n_states = len(self.X)
        self.state_idx = 0
        self.max_steps = self.n_states
        self.steps_taken = 0

    def reset(self):
        # Start from a random index to reduce determinism
        self.state_idx = random.randint(0, max(0, self.n_states - 1))
        self.steps_taken = 0
        return self.X[self.state_idx].copy()

    def step(self, action: int):
        actual = self.y[self.state_idx]
        reward = REWARD_MAP[action][actual]

        self.state_idx = (self.state_idx + 1) % max(1, self.n_states)
        self.steps_taken += 1
        next_state = self.X[self.state_idx].copy()
        done = self.steps_taken >= self.max_steps
        return next_state, float(reward), bool(done)


# --------------------------
# DDQN
# --------------------------
class DDQNModel(nn.Module):
    def __init__(self, input_dim, num_actions):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Dropout(p=0.4),  # add dropout here too
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(p=0.4),  # another dropout
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Dropout(p=0.4),
            nn.Linear(64, num_actions)
        )

    def forward(self, x):
        return self.net(x)


def soft_update(target: nn.Module, source: nn.Module, tau: float = 0.01):
    with torch.no_grad():
        for tp, sp in zip(target.parameters(), source.parameters()):
            tp.data.copy_(tau * sp.data + (1.0 - tau) * tp.data)


# --------------------------
# Replay Buffer
# --------------------------
class ReplayBuffer:
    def __init__(self, capacity=100_000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        # store as numpy arrays for compactness
        self.buffer.append((state.astype(np.float32), int(action), float(reward), next_state.astype(np.float32), bool(done)))

    def sample(self, batch_size=256):
        batch = random.sample(self.buffer, batch_size)
        s, a, r, ns, d = zip(*batch)
        return (
            np.stack(s, axis=0).astype(np.float32),
            np.array(a, dtype=np.int64),
            np.array(r, dtype=np.float32),
            np.stack(ns, axis=0).astype(np.float32),
            np.array(d, dtype=np.float32),
        )

    def __len__(self):
        return len(self.buffer)


# --------------------------
# Preprocessing
# --------------------------
CAT_POSSIBLE = [
    "user_id",
    "room_type",
    "spatial_context",
    "activity",
    "social_context",
    "group_radius",
    "distance_of_people",
    "time_range",
]

def preprocess_df(df: pd.DataFrame):
    df = df.copy()

    # Replace capped counts & scale them
    if "num_people_around" in df.columns:
        df["num_people_around"] = df["num_people_around"].replace("6+", 7)
        df["num_people_around"] = pd.to_numeric(df["num_people_around"], errors="coerce")
    if "num_animals_around" in df.columns:
        df["num_animals_around"] = df["num_animals_around"].replace("2+", 3)
        df["num_animals_around"] = pd.to_numeric(df["num_animals_around"], errors="coerce")

    scalers = {}

    # Scale only these two columns to stabilize gradients
    for col in ["num_people_around", "num_animals_around"]:
        if col in df.columns:
            scaler = StandardScaler()
            df[[col]] = scaler.fit_transform(df[[col]])
            scalers[col] = scaler

    # Ensure target exists
    if "human_response" not in df.columns:
        raise ValueError("Column 'human_response' is missing.")
    # Keep all 3 labels (help, no_help, noisy_help)
    y = df["human_response"].astype(str).values

    # Time features → time_range category if present
    X_raw = df.drop(columns=["human_response"]).reset_index(drop=True)
    if "start_time" in X_raw.columns and "end_time" in X_raw.columns:
        X_raw["time_range"] = X_raw["start_time"].astype(str) + "-" + X_raw["end_time"].astype(str)
        X_raw.drop(columns=["start_time", "end_time"], inplace=True)

    # Figure categorical cols that exist
    categorical_cols = [c for c in CAT_POSSIBLE if c in X_raw.columns]
    numeric_cols = [c for c in X_raw.columns if c not in categorical_cols]

    # One-hot
    if categorical_cols:
        ohe = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
        cat = ohe.fit_transform(X_raw[categorical_cols])
    else:
        cat = np.zeros((len(X_raw), 0), dtype=np.float32)

    num = X_raw[numeric_cols].fillna(0).to_numpy(dtype=np.float32)
    X = np.concatenate([num, cat.astype(np.float32)], axis=1)
    X = np.ascontiguousarray(X, dtype=np.float32)

    meta = {
        "categorical_cols": categorical_cols,
        "numeric_cols": numeric_cols,
        "n_features": X.shape[1],
        "ohe": ohe,
        "scalers": scalers
    }
    return X, y, meta


# --------------------------
# Data loading (user-wise)
# --------------------------
def load_user_datasets(user_output_folder: str):
    """
    Returns dict: user_id -> (X, y, meta)
    Only loads users for which the expected CSV exists.
    """
    datasets = {}
    for user_dir in sorted(os.listdir(user_output_folder)):
        user_dir_path = os.path.join(user_output_folder, user_dir)
        if not os.path.isdir(user_dir_path):
            continue
        file_name = f"hri_activity_with_responses_user_{user_dir}.csv"
        file_path = os.path.join(user_dir_path, file_name)
        if not os.path.exists(file_path):
            print(f"[WARN] Missing {file_name}, skipping {user_dir}")
            continue

        df = pd.read_csv(file_path)
        X, y, meta = preprocess_df(df)
        datasets[user_dir] = (X, y, meta, user_dir_path)
    return datasets


# --------------------------
# Evaluation helpers
# --------------------------
def expected_action_from_label(lbl: str) -> int:
    # "help" => ask(1); "no_help" or "noisy_help" => no_ask(0)
    return 1 if lbl == "help" else 0


@torch.no_grad()
def greedy_actions(model: nn.Module, device, X: np.ndarray) -> np.ndarray:
    model.eval()
    acts = []
    bs = 1024
    for i in range(0, len(X), bs):
        chunk = torch.from_numpy(X[i:i+bs]).float().to(device)
        q = model(chunk)
        a = torch.argmax(q, dim=1).detach().cpu().numpy()
        acts.append(a)
    return np.concatenate(acts, axis=0)


def dqn_proxy_accuracy(model: nn.Module, device, X: np.ndarray, y: np.ndarray) -> float:
    """
    Accuracy proxy: compares greedy action with expected_action(y)
    (This is only for monitoring; final evaluation in your paper can ignore it.)
    """
    acts = greedy_actions(model, device, X)
    y_expected = np.array([expected_action_from_label(t) for t in y], dtype=np.int64)
    return float((acts == y_expected).mean())


# --------------------------
# Training (Double DQN)
# --------------------------
def train_dqn_with_replay(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    results_dir: str,
    user_tag: str,
    train_epochs: int = 400,
    gamma: float = 0.99,
    lr: float = 1e-3,
    weight_decay: float = 1e-5,
    buffer_cap: int = 100_000,
    batch_size: int = 256,
    epsilon_start: float = 1.0,
    epsilon_min: float = 0.05,
    epsilon_decay: float = 0.995,
    tau: float = 0.01,             # soft update factor
    early_stop_threshold: float = 0.96,
    early_stop_patience: int = 5,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ensure_dir(results_dir)

    state_dim = X_train.shape[1]
    n_actions = 2

    policy = DDQNModel(state_dim, n_actions).to(device)
    target = DDQNModel(state_dim, n_actions).to(device)
    target.load_state_dict(policy.state_dict())
    target.eval()

    optimizer = optim.Adam(policy.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.MSELoss()
    buffer = ReplayBuffer(capacity=buffer_cap)

    # Build envs that iterate once per epoch
    train_env = RL_Env(X_train, y_train)

    epsilon = epsilon_start
    train_acc_hist, test_acc_hist, reward_hist = [], [], []

    best_test_acc = -1.0
    patience = 0

    for epoch in range(1, train_epochs + 1):
        policy.train()
        state = train_env.reset().astype(np.float32)
        done = False
        total_reward = 0.0

        while not done:
            # epsilon-greedy
            if np.random.rand() < epsilon:
                action = random.randrange(n_actions)
            else:
                s = torch.from_numpy(state[None, :]).float().to(device)
                with torch.no_grad():
                    q = policy(s)
                    action = int(torch.argmax(q).item())

            next_state, reward, done = train_env.step(action)
            buffer.push(state, action, reward, next_state, done)

            state = next_state.astype(np.float32)
            total_reward += reward

            # learn
            if len(buffer) >= batch_size:
                s_mb, a_mb, r_mb, ns_mb, d_mb = buffer.sample(batch_size)
                s_t = torch.from_numpy(s_mb).float().to(device)
                ns_t = torch.from_numpy(ns_mb).float().to(device)
                a_t = torch.from_numpy(a_mb).long().to(device)
                r_t = torch.from_numpy(r_mb).float().to(device)
                d_t = torch.from_numpy(d_mb).float().to(device)

                # Double DQN target
                with torch.no_grad():
                    next_q_policy = policy(ns_t)                      # argmax over policy
                    next_actions = torch.argmax(next_q_policy, dim=1)
                    next_q_target = target(ns_t).gather(1, next_actions.unsqueeze(1)).squeeze(1)
                    y_target = r_t + gamma * next_q_target * (1.0 - d_t)

                q_curr = policy(s_t).gather(1, a_t.unsqueeze(1)).squeeze(1)
                loss = loss_fn(q_curr, y_target)

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(policy.parameters(), max_norm=2.0)  # stabilize
                optimizer.step()

                # Soft update target
                soft_update(target, policy, tau=tau)

        # decay epsilon per episode
        epsilon = max(epsilon * epsilon_decay, epsilon_min)

        # Monitoring (proxy classification accuracy)
        train_acc = dqn_proxy_accuracy(policy, device, X_train, y_train)
        test_acc = dqn_proxy_accuracy(policy, device, X_test, y_test)
        train_acc_hist.append(train_acc)
        test_acc_hist.append(test_acc)
        reward_hist.append(total_reward)

        if epoch % 25 == 0 or epoch == 1:
            print(f"[{user_tag}] Epoch {epoch:03d}/{train_epochs} | eps={epsilon:.3f} | "
                  f"reward={total_reward:.2f} | train_acc={train_acc:.4f} | test_acc={test_acc:.4f}")

            # early stopping on test_acc plateau
            if test_acc > best_test_acc:
                best_test_acc = test_acc
                patience = 0
                # Save best policy
                torch.save(policy.state_dict(), os.path.join(results_dir, f"best_dqn_{user_tag}.pt"))
            else:
                patience += 1

            if best_test_acc >= early_stop_threshold and patience >= early_stop_patience:
                print(f"[{user_tag}] Early stopping at epoch {epoch} (best_test_acc={best_test_acc:.4f})")
                break

    print(f'Model saved to {results_dir}/best_dqn_{user_tag}.pt')

    summary = {
        "user_tag": user_tag,
        "epochs_run": len(reward_hist),
        "best_test_acc_proxy": float(best_test_acc),
        "final_test_acc_proxy": float(test_acc_hist[-1] if test_acc_hist else 0.0),
        "avg_reward": float(np.mean(reward_hist)) if reward_hist else 0.0,
        "reward_history_len": len(reward_hist),
    }
    return summary


# --------------------------
# Approach: User-wise
# --------------------------
def run_userwise(user_output_folder: str, train_epochs: int = 400):
    datasets = load_user_datasets(user_output_folder)
    if not datasets:
        print("[ERROR] No user datasets found.")
        return {}

    all_results = {}
    for user_id, (X, y, meta, user_dir_path) in datasets.items():
        # stratified split if possible
        if len(np.unique(y)) > 1 and len(y) >= 2:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )
        else:
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        user_results_dir = user_dir_path  # save beside CSV
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")  # e.g., 20251103_142530
        tag = f"user_{user_id}_{timestamp}"

        summary = train_dqn_with_replay(
            X_train, y_train, X_test, y_test,
            results_dir=user_results_dir,
            user_tag=tag,
            train_epochs=train_epochs
        )
        all_results[user_id] = summary

    # Aggregate
    agg = {
        "avg_best_test_acc_proxy": float(np.mean([r["best_test_acc_proxy"] for r in all_results.values()])),
        "avg_final_test_acc_proxy": float(np.mean([r["final_test_acc_proxy"] for r in all_results.values()])),
        "num_users": len(all_results),
    }
    print("\n[User-wise] Aggregate:")
    print(json.dumps(agg, indent=2))

    return {"users": all_results, "aggregate": agg}

# --------------------------
# Main Function
# --------------------------
def main(user_output_folder: str = "whisper_dataset/user_responses", train_epochs_userwise: int = 500):
    print("=== REWARDS ===")
    print(REWARD_MAP)
    print("=== Running Approach: User-wise ===")
    _ = run_userwise(user_output_folder=user_output_folder, train_epochs=train_epochs_userwise)
    print("\nDone.")


if __name__ == "__main__":
    main(
        user_output_folder="whisper_dataset/user_responses",
        train_epochs_userwise=300
    )
