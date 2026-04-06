import csv
import os
import traci
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque
import time
# ✅ Set seeds (IMPORTANT)


# ===============================
# DQN MODEL
# ===============================
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


# ===============================
# AGENT
# ===============================
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


# ===============================
# SUMO ENVIRONMENT
# ===============================
class SumoEnv:
    def __init__(self, sumo_cmd, tls_id="center"):
        self.sumo_cmd = sumo_cmd
        self.tls_id = tls_id
        self.edges = ["n2c", "s2c", "e2c", "w2c"]

    def start(self):
        traci.start(self.sumo_cmd)

    def reset(self):
        traci.load(self.sumo_cmd[1:])
        return self.get_state()

    def step(self, action):
        traci.trafficlight.setPhase(self.tls_id, action)

        for _ in range(20):
            traci.simulationStep()

        next_state = self.get_state()
        reward = self.compute_reward()

        done = traci.simulation.getMinExpectedNumber() == 0
        return next_state, reward, done

    def get_state(self):
        return np.array([
            traci.edge.getLastStepVehicleNumber(e)
            for e in self.edges
        ])

    def compute_metrics(self):
        delay = 0
        queue = 0
        co2 = 0

        for e in self.edges:
            delay += traci.edge.getWaitingTime(e)
            queue += traci.edge.getLastStepVehicleNumber(e)
            co2 += traci.edge.getCO2Emission(e)

        throughput = traci.simulation.getArrivedNumber()

        return delay, queue, co2, throughput

    def compute_reward(self):
        delay, queue, co2, throughput = self.compute_metrics()
        return - (delay + queue + 0.001 * co2) + (2 * throughput)  # stable reward

    def close(self):
        traci.close()


# ===============================
# TRAINING LOOP
# ===============================
def train_rl():
    sumo_cmd = ["sumo", "-c", "config.sumocfg"]  # use sumo for speed

    env = SumoEnv(sumo_cmd)
    env.start()

    state = env.get_state()
    state_size = len(state)

    action_size = 4
    agent = Agent(state_size, action_size)

    episodes = 50  # change to 50 for final

    print("\n" + "="*60)
    print("  RL · Traffic Signal Control (DQN)")
    print("="*60)

    start_time = time.time()

    for ep in range(episodes):
        state = env.reset()
        total_reward = 0
        done = False

        total_delay = 0
        total_queue = 0
        total_co2 = 0
        total_throughput = 0
        steps = 0

        while not done:
            action = agent.act(state)

            next_state, reward, done = env.step(action)

            d, q, c, t = env.compute_metrics()

            total_delay += d
            total_queue += q
            total_co2 += c
            total_throughput += t
            steps += 1

            agent.remember(state, action, reward, next_state, done)
            agent.replay()

            state = next_state
            total_reward += reward

        print(f"Episode {ep+1}: Reward={total_reward:.2f} | "
              f"Delay={total_delay/steps:.2f}, "
              f"Queue={total_queue/steps:.2f}, "
              f"CO2={total_co2/steps:.2f}, "
              f"Throughput={total_throughput}")
        # after training loop
       
    env.close()
    save_results("roboust", "RL", total_delay/steps, total_queue/steps, total_co2/steps, total_throughput)


    elapsed = time.time() - start_time

    print("\n" + "="*60)
    print(f"Training complete in {elapsed:.2f} seconds")
    print("="*60)
def save_results(scenario, algorithm, delay, queue, co2, throughput):
    file = "results.csv"
    file_exists = os.path.isfile(file)

    with open(file, "a", newline="") as f:
        writer = csv.writer(f)

        # write header only once
        if not file_exists:
            writer.writerow(["Scenario", "Algorithm", "Delay", "Queue", "CO2", "Throughput"])

        writer.writerow([scenario, algorithm, delay, queue, co2, throughput])

# ===============================
# MAIN
# ===============================
if __name__ == "__main__":
    train_rl()