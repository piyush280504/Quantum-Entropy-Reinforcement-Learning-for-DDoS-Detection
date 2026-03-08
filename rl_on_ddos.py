import streamlit as st

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from qutip import Qobj, entropy_vn
import matplotlib.pyplot as plt
from collections import defaultdict

st.title("Quantum Entropy RL DDoS Detection")

st.write("""
This demo trains a Reinforcement Learning agent to detect attacks
using entropy derived from PCA-transformed network flows.
""")

run_button = st.button("Start RL Training")

if run_button:

    st.write("Loading dataset...")

    train_path = r"C:\Users\piyus\Downloads\DDoS\MSSQL-training.parquet"
    test_path = r"C:\Users\piyus\Downloads\DDoS\MSSQL-testing.parquet"

    train_df = pd.read_parquet(train_path)
    test_df = pd.read_parquet(test_path)

    st.success("Dataset loaded successfully")

    numeric_cols = train_df.select_dtypes(include=[np.number]).columns.tolist()
    train_data = train_df[numeric_cols].dropna()
    test_data = test_df[numeric_cols].dropna()

    train_labels = train_df['Label'].iloc[train_data.index]
    test_labels = test_df['Label'].iloc[test_data.index]

    scaler = StandardScaler()
    train_scaled = scaler.fit_transform(train_data)
    test_scaled = scaler.transform(test_data)

    st.write("Running PCA...")

    pca = PCA(n_components=5)
    train_pca = pca.fit_transform(train_scaled)
    test_pca = pca.transform(test_scaled)

    def to_density_matrix(vec):
        vec = vec / np.linalg.norm(vec)
        return Qobj(np.outer(vec, vec.conj()))

    st.write("Computing quantum entropy...")

    train_entropies = np.array([entropy_vn(to_density_matrix(v)) for v in train_pca])
    test_entropies = np.array([entropy_vn(to_density_matrix(v)) for v in test_pca])

    train_binary_labels = train_labels.apply(lambda x: 0 if 'BENIGN' in str(x).upper() else 1)

    def discretize(states, bins=10):
        discretized = []
        for i in range(states.shape[1]):
            _, bin_edges = np.histogram(states[:, i], bins=bins)
            digitized = np.digitize(states[:, i], bins=bin_edges[1:-1], right=False)
            discretized.append(digitized)
        return np.stack(discretized, axis=1)

    train_states = np.column_stack([train_entropies, train_pca[:, 0]])
    state_bins = discretize(train_states, bins=10)

    class QLearningAgent:
        def __init__(self, n_actions=2, alpha=0.1, gamma=0.95, epsilon=0.1):
            self.q_table = defaultdict(lambda: np.zeros(n_actions))
            self.alpha = alpha
            self.gamma = gamma
            self.epsilon = epsilon
            self.n_actions = n_actions

        def get_state_key(self, state):
            return tuple(state)

        def choose_action(self, state):
            key = self.get_state_key(state)
            if np.random.rand() < self.epsilon:
                return np.random.choice(self.n_actions)
            return np.argmax(self.q_table[key])

        def update(self, state, action, reward, next_state):
            key = self.get_state_key(state)
            next_key = self.get_state_key(next_state)
            predict = self.q_table[key][action]
            target = reward + self.gamma * np.max(self.q_table[next_key])
            self.q_table[key][action] += self.alpha * (target - predict)

    class SimpleQuantumEnv:
        def __init__(self, states, labels, entropies):
            self.states = states
            self.labels = labels.reset_index(drop=True)
            self.entropies = entropies
            self.current_index = 0
            self.n = len(states)

        def reset(self):
            self.current_index = 0
            return self.states[self.current_index]

        def step(self, action):
            true_label = self.labels[self.current_index]
            entropy_penalty = self.entropies[self.current_index]
            reward = 1 if action == true_label else -1
            reward -= 0.05 * entropy_penalty

            self.current_index += 1
            done = self.current_index >= self.n
            next_state = self.states[self.current_index % self.n] if not done else None

            return next_state, reward, done, {}

    st.write("Initializing RL agent...")

    agent = QLearningAgent()
    env = SimpleQuantumEnv(state_bins, train_binary_labels, train_entropies)

    episodes = 10
    reward_history = []

    progress = st.progress(0)

    st.write("Starting training...")

    for ep in range(episodes):
        state = env.reset()
        total_reward = 0

        while True:
            action = agent.choose_action(state)
            next_state, reward, done, _ = env.step(action)

            if next_state is not None:
                agent.update(state, action, reward, next_state)

            state = next_state
            total_reward += reward

            if done:
                break

        st.write(f"Episode {ep+1} — Total Reward: {total_reward}")
        reward_history.append(total_reward)

        progress.progress((ep + 1) / episodes)

    st.success("Training complete!")

    fig, ax = plt.subplots()
    ax.plot(range(1, episodes+1), reward_history, marker='o')
    ax.set_title("Total Reward per Episode (Q-learning with Quantum Entropy)")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Total Reward")
    ax.grid(True)

    st.pyplot(fig)
