import torch
import torch.nn.functional as functional
from torch import optim
from agents.Base_Agent import Base_Agent
from utilities.data_structures.Replay_Buffer import Replay_Buffer
from exploration_strategies.OU_Noise_Exploration import OU_Noise_Exploration
import numpy as np
import dill

TRAINING_EPISODES_PER_MODEL_SAVE = 5000


class DDPG(Base_Agent):
    """A DDPG Agent"""

    agent_name = "DDPG"

    def __init__(self, config):
        Base_Agent.__init__(self, config)
        self.hyperparameters = config.hyperparameters
        if self.config.load_model:
            self.critic_local = torch.load(
                f"{self.config.model_dir}/critic_local_{self.config.load_model_episode}.pt"
            )
            self.critic_target = torch.load(
                f"{self.config.model_dir}/critic_target_{self.config.load_model_episode}.pt"
            )
            Base_Agent.copy_model_over(self.critic_local, self.critic_target)
            self.memory = Replay_Buffer(
                self.hyperparameters["Critic"]["buffer_size"],
                self.hyperparameters["batch_size"],
                self.config.seed,
                device=self.device,
                load_memory_path=f"{self.config.model_dir}/memory_{self.config.load_model_episode}.pkl",
            )
            self.actor_local = torch.load(
                f"{self.config.model_dir}/actor_local_{self.config.load_model_episode}.pt"
            )
            self.actor_target = torch.load(
                f"{self.config.model_dir}/actor_target_{self.config.load_model_episode}.pt"
            )
            Base_Agent.copy_model_over(self.actor_local, self.actor_target)
        else:
            self.critic_local = self.create_NN(
                input_dim=self.state_size + self.action_size * self.config.nagents,
                output_dim=self.config.nagents,
                key_to_use="Critic",
            )
            self.critic_target = self.create_NN(
                input_dim=self.state_size + self.action_size * self.config.nagents,
                output_dim=self.config.nagents,
                key_to_use="Critic",
            )
            Base_Agent.copy_model_over(self.critic_local, self.critic_target)
            self.memory = Replay_Buffer(
                self.hyperparameters["Critic"]["buffer_size"],
                self.hyperparameters["batch_size"],
                self.config.seed,
            )
            self.actor_local = self.create_NN(
                input_dim=self.state_size,
                output_dim=self.action_size * self.config.nagents,
                key_to_use="Actor",
            )
            self.actor_target = self.create_NN(
                input_dim=self.state_size,
                output_dim=self.action_size * self.config.nagents,
                key_to_use="Actor",
            )
            Base_Agent.copy_model_over(self.actor_local, self.actor_target)

        self.critic_optimizer = optim.Adam(
            self.critic_local.parameters(),
            lr=self.hyperparameters["Critic"]["learning_rate"],
            eps=1e-4,
        )
        self.actor_optimizer = optim.Adam(
            self.actor_local.parameters(),
            lr=self.hyperparameters["Actor"]["learning_rate"],
            eps=1e-4,
        )
        self.exploration_strategy = OU_Noise_Exploration(self.config)

        if self.config.load_model:
            print(
                f"Episode {self.config.load_model_episode} models will be loaded from {self.model_dir}"
            )
        if self.config.save_model:
            print(f"Models will be saved to {self.model_dir}")

    def step(self):
        """Runs a step in the game"""
        self.episode_step_number_val = 0
        while not self.done:
            self.episode_step_number_val += 1
            # print("State ", self.state.shape)
            self.action = self.pick_action()
            # np.argmax returns the index of the action with the highest score
            self.conduct_action(self.action.argmax(axis=1))
            if not self.config.evaluate_policy:
                if self.time_for_critic_and_actor_to_learn():
                    for _ in range(
                        self.hyperparameters["learning_updates_per_learning_session"]
                    ):
                        states, actions, rewards, next_states, dones = (
                            self.sample_experiences()
                        )
                        self.critic_learn(states, actions, rewards, next_states, dones)
                        self.actor_learn(states)
                self.save_experience()
            self.state = (
                self.next_state
            )  # this is to set the state for the next iteration
            self.global_step_number += 1
        if (
            self.episode_number % TRAINING_EPISODES_PER_MODEL_SAVE == 0
            and self.config.save_model
        ):
            self.save_models()
        self.episode_number += 1

    def sample_experiences(self):
        return self.memory.sample()

    def pick_action(self, state=None):
        """Picks an action using the actor network and then adds some noise to it to ensure exploration"""
        if state is None:
            state = torch.from_numpy(self.state).float().unsqueeze(0).to(self.device)
        self.actor_local.eval()
        with torch.no_grad():
            action = (
                self.actor_local(torch.reshape(state, [-1, self.state_size]))
                .cpu()
                .data.numpy()
            )
        self.actor_local.train()
        action = self.exploration_strategy.perturb_action_for_exploration_purposes(
            {"action": action}
        )
        return np.reshape(action.squeeze(0), (self.config.nagents, self.action_size))

    def critic_learn(self, states, actions, rewards, next_states, dones):
        """Runs a learning iteration for the critic"""
        loss = self.compute_loss(states, next_states, rewards, actions, dones)
        self.take_optimisation_step(
            self.critic_optimizer,
            self.critic_local,
            loss,
            self.hyperparameters["Critic"]["gradient_clipping_norm"],
        )
        self.soft_update_of_target_network(
            self.critic_local, self.critic_target, self.hyperparameters["Critic"]["tau"]
        )

    def compute_loss(self, states, next_states, rewards, actions, dones):
        """Computes the loss for the critic"""
        with torch.no_grad():
            critic_targets = self.compute_critic_targets(next_states, rewards, dones)
        critic_expected = self.compute_expected_critic_values(states, actions)
        loss = functional.mse_loss(critic_expected, critic_targets)
        return loss

    def compute_critic_targets(self, next_states, rewards, dones):
        """Computes the critic target values to be used in the loss for the critic"""
        critic_targets_next = self.compute_critic_values_for_next_states(next_states)
        critic_targets = self.compute_critic_values_for_current_states(
            rewards, critic_targets_next, dones
        )
        return critic_targets

    def compute_critic_values_for_next_states(self, next_states):
        """Computes the critic values for next states to be used in the loss for the critic"""
        with torch.no_grad():
            actions_next = self.actor_target(next_states)
            critic_targets_next = self.critic_target(
                torch.cat((next_states, actions_next), 1)
            )
        return critic_targets_next

    def compute_critic_values_for_current_states(
        self, rewards, critic_targets_next, dones
    ):
        """Computes the critic values for current states to be used in the loss for the critic"""
        critic_targets_current = rewards + (
            self.hyperparameters["discount_rate"] * critic_targets_next * (1.0 - dones)
        )
        return critic_targets_current

    def compute_expected_critic_values(self, states, actions):
        """Computes the expected critic values to be used in the loss for the critic"""
        critic_expected = self.critic_local(torch.cat((states, actions), 1))
        return critic_expected

    def time_for_critic_and_actor_to_learn(self):
        """Returns boolean indicating whether there are enough experiences to learn from and it is time to learn for the
        actor and critic"""
        return (
            self.enough_experiences_to_learn_from()
            and self.global_step_number % self.hyperparameters["update_every_n_steps"]
            == 0
        )

    def actor_learn(self, states):
        """Runs a learning iteration for the actor"""
        if self.done:  # we only update the learning rate at end of each episode
            self.update_learning_rate(
                self.hyperparameters["Actor"]["learning_rate"], self.actor_optimizer
            )
        actor_loss = self.calculate_actor_loss(states)
        self.take_optimisation_step(
            self.actor_optimizer,
            self.actor_local,
            actor_loss,
            self.hyperparameters["Actor"]["gradient_clipping_norm"],
        )
        self.soft_update_of_target_network(
            self.actor_local, self.actor_target, self.hyperparameters["Actor"]["tau"]
        )

    def calculate_actor_loss(self, states):
        """Calculates the loss for the actor"""
        actions_pred = self.actor_local(torch.reshape(states, [-1, self.state_size]))
        actor_loss = -self.critic_local(
            torch.cat((torch.reshape(states, [-1, self.state_size]), actions_pred), 1)
        ).mean()
        return actor_loss

    def save_models(self):
        torch.save(
            self.critic_local,
            f"{self.model_dir}/critic_local_{len(self.game_full_episode_scores)}.pt",
        )
        torch.save(
            self.critic_target,
            f"{self.model_dir}/critic_target_{len(self.game_full_episode_scores)}.pt",
        )
        torch.save(
            self.actor_local,
            f"{self.model_dir}/actor_local_{len(self.game_full_episode_scores)}.pt",
        )
        torch.save(
            self.actor_target,
            f"{self.model_dir}/actor_target_{len(self.game_full_episode_scores)}.pt",
        )
        torch.save(
            self.critic_local_2,
            f"{self.model_dir}/critic_local_2_{len(self.game_full_episode_scores)}.pt",
        )
        torch.save(
            self.critic_target_2,
            f"{self.model_dir}/critic_target_2_{len(self.game_full_episode_scores)}.pt",
        )
        with open(
            f"{self.model_dir}/memory_{len(self.game_full_episode_scores)}.pkl", "wb"
        ) as f:
            dill.dump(self.memory, f)
        print(f"SAVED {self.episode_number} models to {self.model_dir}")
