import torch
import torch.nn as nn
import torch.nn.functional as F

class ExpertNet(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))
        return x

class MoEGate(nn.Module):
    def __init__(self, input_dim, num_experts):
        super().__init__()
        self.fc = nn.Linear(input_dim, num_experts)

    def forward(self, x):
        # Produce gating weights for each expert
        return F.softmax(self.fc(x), dim=1)

class MoEModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_experts=3):
        super().__init__()
        self.experts = nn.ModuleList([ExpertNet(input_dim, hidden_dim) for _ in range(num_experts)])
        self.gate = MoEGate(input_dim, num_experts)
        self.num_experts = num_experts

    def forward(self, x):
        # x shape: (batch_size, input_dim)
        gate_weights = self.gate(x)  # shape: (batch_size, num_experts)
        expert_outputs = []
        for i, expert in enumerate(self.experts):
            expert_outputs.append(expert(x))  # shape: (batch_size, 1)

        # Concatenate outputs: shape => (batch_size, num_experts)
        expert_outputs = torch.cat(expert_outputs, dim=1)

        # Weighted sum by gate weights
        out = torch.sum(gate_weights * expert_outputs, dim=1, keepdim=True)
        return out  # shape: (batch_size, 1)
