"""
R1 - FYP
COMP1682 – Final Year Project
Author: Omith Chowdhury

PPO Pruning Script
-------------------
Loads a PPO model and prunes low-magnitude weights
in all linear layers (MLP policy network).

Outputs:
- Logs parameter count before and after
- Saves the pruned model as ./models/PPO_Optuna_Stage4_pruned.zip
-------------------
"""

import torch
import torch.nn.utils.prune as prune
from stable_baselines3 import PPO
from R1_Training_Environment import BasicObstacleEnv
import os

# Loads the original trained model
model_path = "./models/PPO_Optuna_Stage4_20250412_195425.zip" # I set it hard coded here but if  want to change just just do it here
model = PPO.load(model_path)

# Get the policy network (assumes MlpPolicy)
policy = model.policy

# Tracking the total parameters before pruning
def count_parameters(module):
    return sum(p.numel() for p in module.parameters() if p.requires_grad)

original_params = count_parameters(policy)
print(f"Original parameter count: {original_params}")

# Prune all nn.Linear layers in the MLP policy and value networks
prune_amount = 0.2  # 20% of weights will be zeroed    , here we set how much to prune , usually between 20 to 40% is best, I put 20

for name, module in policy.mlp_extractor.policy_net.named_modules():
    if isinstance(module, torch.nn.Linear):
        prune.l1_unstructured(module, name='weight', amount=prune_amount)

for name, module in policy.mlp_extractor.value_net.named_modules():
    if isinstance(module, torch.nn.Linear):
        prune.l1_unstructured(module, name='weight', amount=prune_amount)

# Count parameters after pruning
pruned_params = count_parameters(policy)
print(f"Pruned parameter count: {pruned_params}")

# Location and folder in which save the pruned model
save_path = "./models/PPO_Optuna_Stage4_pruned.zip"
model.save(save_path)
print(f"Pruned model saved to: {save_path}")
