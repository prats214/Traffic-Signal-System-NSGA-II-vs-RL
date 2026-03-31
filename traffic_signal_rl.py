"""
traffic_signal_rl.py
====================

Reinforcement Learning (DQN) for adaptive traffic signal control using SUMO.

Optimizes:
- Average Delay ↓
- Queue Length ↓
- CO2 Emissions ↓
- Throughput ↑

Requirements:
    pip install torch numpy traci

NOTE:
    Replace "your_config.sumocfg" with the actual SUMO config file before running.
"""

import traci
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque
import time


# ─────────────────────────────────────────────────────────────
# 1. DQN MODEL
# ─────────────────────────────────────────────────────────────

class DQN(nn.Module):
    def __init__(self, state_size, action_size):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_size, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, action_size)
        )

    def forward(self, x):
        return self.net(x)


class Agent:
    def __init__(self, state_size, action_size):
        self.model = DQN(state_size, action_size)
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)

        self.memory = deque(maxlen=2000)
        self.gamma = 0.95

        self.epsilon = 1.0
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.05

        self.action_size = action_size

    def act(self, state):
        if random.random() < self.epsilon:
            return random.randrange(self.action_size)

        state = torch.FloatTensor(state)
        q_values = self.model(state)
        return torch.argmax(q_values).item()

    def remember(self, s, a, r, ns, done):
        self.memory.append((s, a, r, ns, done))

    def replay(self, batch_size=32):
        if len(self.memory) < batch_size:
            return

        batch = random.sample(self.memory, batch_size)

        for s, a, r, ns, done in batch:
            target = r
            if not done:
                target += self.gamma * torch.max(self.model(torch.FloatTensor(ns))).item()

            q_vals = self.model(torch.FloatTensor(s)).detach().clone()
            q_vals[a] = target

            prediction = self.model(torch.FloatTensor(s))
            loss = nn.MSELoss()(prediction, q_vals)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay


# ─────────────────────────────────────────────────────────────
# 2. SUMO ENVIRONMENT
# ─────────────────────────────────────────────────────────────

class SumoEnv:
    def __init__(self, sumo_cmd, tls_id="0"):
        self.sumo_cmd = sumo_cmd
        self.tls_id = tls_id

    def start(self):
        traci.start(self.sumo_cmd)

    def reset(self):
        traci.load(self.sumo_cmd[1:])
        return self.get_state()

    def step(self, action):
        self.apply_action(action)

        for _ in range(10):
            traci.simulationStep()

        next_state = self.get_state()
        reward = self.compute_reward()

        done = traci.simulation.getMinExpectedNumber() == 0
        return next_state, reward, done

    def get_state(self):
        lanes = traci.trafficlight.getControlledLanes(self.tls_id)

        return np.array([
            traci.lane.getLastStepHaltingNumber(l)
            for l in lanes
        ])

    def apply_action(self, action):
        traci.trafficlight.setPhase(self.tls_id, action)

    def compute_metrics(self):
        lanes = traci.trafficlight.getControlledLanes(self.tls_id)

        delay = sum(traci.lane.getWaitingTime(l) for l in lanes)
        queue = sum(traci.lane.getLastStepHaltingNumber(l) for l in lanes)
        co2 = sum(traci.lane.getCO2Emission(l) for l in lanes)

        throughput = traci.simulation.getArrivedNumber()

        return delay, queue, co2, throughput

    def compute_reward(self):
        delay, queue, co2, throughput = self.compute_metrics()

        # Balanced reward function
        return - (delay + queue + 0.001 * co2) + (2 * throughput)

    def close(self):
        traci.close()


# ─────────────────────────────────────────────────────────────
# 3. TRAINING LOOP
# ─────────────────────────────────────────────────────────────

def train_rl():
    # ⚠️ CHANGE THIS BEFORE RUNNING
    sumo_cmd = ["sumo-gui", "-c", "your_config.sumocfg"]

    env = SumoEnv(sumo_cmd)
    env.start()

    state = env.get_state()
    state_size = len(state)

    action_size = 4  # adjust if different number of phases
    agent = Agent(state_size, action_size)

    episodes = 30

    print("\n" + "="*60)
    print("  RL · Traffic Signal Control (DQN)")
    print("="*60)

    start_time = time.time()

    for ep in range(episodes):
        state = env.reset()
        total_reward = 0
        done = False

        while not done:
            action = agent.act(state)

            next_state, reward, done = env.step(action)

            agent.remember(state, action, reward, next_state, done)
            agent.replay()

            state = next_state
            total_reward += reward

        delay, queue, co2, throughput = env.compute_metrics()

        print(f"Episode {ep+1}: Reward={total_reward:.2f} | "
              f"Delay={delay:.2f}, Queue={queue:.2f}, CO2={co2:.2f}, Throughput={throughput}")

    env.close()

    elapsed = time.time() - start_time

    print("\n" + "="*60)
    print(f"Training complete in {elapsed:.2f} seconds")
    print("="*60)


# ─────────────────────────────────────────────────────────────
# 4. ENTRY POINT
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    train_rl()
