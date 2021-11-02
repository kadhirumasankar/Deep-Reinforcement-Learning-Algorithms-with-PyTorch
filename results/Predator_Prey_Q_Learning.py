from numpy.core.numeric import roll
from torch.utils.tensorboard import SummaryWriter, writer
import random
import matplotlib.pyplot as plt
from time import sleep
from ic3net_envs.predator_prey_env import PredatorPreyEnv
import argparse
import numpy as np
import time


def encode(predator_loc, prey_loc):
    # 3, 3, 3, 3
    i = predator_loc[0]
    i *= 3
    i += predator_loc[1]
    i *= 3
    i += prey_loc[0]
    i *= 3
    i += prey_loc[1]
    return i


def render(environment):
    time.sleep(0.1)
    environment.render()


tensorboard_writer = SummaryWriter()

parser = argparse.ArgumentParser("Example GCCNet environment random agent")
parser.add_argument("--nagents", type=int, default=1, help="Number of agents")
parser.add_argument(
    "--display", action="store_true", default=False, help="Use to display environment"
)
parser.add_argument("--nepisodes", type=int, default=50000, help="Number of episodes")

env = PredatorPreyEnv()
# env.init_curses()
env.init_args(parser)
args = parser.parse_args()
env.multi_agent_init(args)

q_table = np.zeros([3 * 3 * 3 * 3, env.action_space.nvec.item()])


# Hyperparameters
alpha = 0.1
gamma = 0.6
epsilon = 0.1

# For plotting metrics
all_epochs = []
all_penalties = []
rolling_scores = np.array([])

for i in range(1, 100001):
    state = env.reset()
    q_state = encode(env.predator_loc[0], env.prey_loc[0])

    epochs, penalties, reward, = 0, 0, 0
    total_reward = 0
    done = False

    while not done:
        if random.uniform(0, 1) < epsilon:
            action = env.action_space.sample()  # Explore action space
        else:
            action = np.argmax(q_table[q_state])  # Exploit learned values

        next_state, reward, done, info = env.step(action)

        current_row = env.predator_loc[0][0]
        current_col = env.predator_loc[0][1]
        if action == 0 and current_row != 0:
            next_predator_loc = np.array([current_row - 1, current_col])
        elif action == 1 and current_col != 2:
            next_predator_loc = np.array([current_row, current_col + 1])
        elif action == 2 and current_row != 2:
            next_predator_loc = np.array([current_row + 1, current_col])
        elif action == 3 and current_col != 0:
            next_predator_loc = np.array([current_row, current_col - 1])
        else:
            next_predator_loc = np.array([current_row, current_col])

        next_q_state = encode(next_predator_loc, env.prey_loc[0])

        old_value = q_table[q_state, action]
        next_max = np.max(q_table[next_q_state])

        new_value = (1 - alpha) * old_value + alpha * (reward + gamma * next_max)
        q_table[q_state, action] = new_value

        if reward == -10:
            penalties += 1

        q_state = next_q_state
        epochs += 1

        total_reward += reward

    if rolling_scores.size < 100:
        rolling_scores = np.append(rolling_scores, total_reward)
    else:
        rolling_scores = np.append(rolling_scores, total_reward)
        rolling_scores = rolling_scores[1:]
    tensorboard_writer.add_scalar("Score", total_reward, i)
    tensorboard_writer.add_scalar("RollingScore", np.mean(rolling_scores), i)
    tensorboard_writer.add_scalar("Steps", total_reward / -0.05, i)

    if i % 100 == 0:
        print(f"Episode {i}: rolling score {np.mean(rolling_scores)}")


print("Training finished.\n")
