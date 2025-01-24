import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque

# Definizione della rete neurale feed-and-forward per DQN
class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, output_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Definizione della classe che rappresenta l'agente basato su DQN
class DQNAgent:
    def __init__(self, state_dim, action_dim, lr, gamma, epsilon, epsilon_decay, buffer_size):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.memory = deque(maxlen=buffer_size)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = DQN(state_dim, action_dim).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return np.random.choice(self.action_dim)
        q_values = self.model(torch.tensor(state, dtype=torch.float32))
        return torch.argmax(q_values).item()

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def replay(self, batch_size):
        if len(self.memory) < batch_size:
            return
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            state = torch.tensor(state, dtype=torch.float32).to(self.device)  
            next_state = torch.tensor(next_state, dtype=torch.float32).to(self.device) 
            target = reward
            if not done:
                target = reward + self.gamma * torch.max(self.model(next_state)).item()
            
            target_f = self.model(state).detach().cpu().numpy() 
            target_f[action] = target
            
            target_f_tensor = torch.tensor(target_f, dtype=torch.float32).to(self.device)
            
            self.optimizer.zero_grad()
            loss = nn.MSELoss()(target_f_tensor, self.model(state))
            loss.backward()
            self.optimizer.step()

        if self.epsilon > 0.01:
            self.epsilon *= self.epsilon_decay