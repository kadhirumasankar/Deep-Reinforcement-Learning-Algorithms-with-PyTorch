from torch.utils.tensorboard import SummaryWriter, writer
import torch
from pathlib import Path


def load_dict(file):
    return torch.load(
        Path(
            f"/home/kadhir/research/Deep-Reinforcement-Learning-Algorithms-with-PyTorch/results/PredatorPrey/{file}.pt"
        )
    )


tensorboard_writer = SummaryWriter()
network_list = [
    "critic_target",
    "critic_target_2",
    "critic_local",
    "critic_local_2",
    "actor_local",
]
for network in network_list:
    for i in range(50, 4451, 50):
        dict = load_dict(f"{network}_{i}")
        for key, value in dict.items():
            tensorboard_writer.add_scalar(f"{network}_{key}", torch.mean(value), i)
