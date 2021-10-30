from ic3net_envs.predator_prey_env import PredatorPreyEnv
import argparse
import sys
from os.path import dirname, abspath
sys.path.append(dirname(dirname(abspath(__file__))))
import pathlib

from agents.actor_critic_agents.SAC_Discrete import SAC_Discrete
from agents.DQN_agents.DQN import DQN
from agents.Trainer import Trainer
from utilities.data_structures.Config import Config

class PredatorPreyAgent(object):
    def __init__(self, action_space):
        self.action_space = action_space
    
    def act(self):
        return self.action_space.sample()

if __name__ == '__main__':
    parser = argparse.ArgumentParser('Example GCCNet environment random agent')
    parser.add_argument('--nagents', type=int, default=1, help="Number of agents")
    parser.add_argument('--display', action="store_true", default=False,
                        help="Use to display environment")
    parser.add_argument('--nepisodes', type=int, default=50000, help="Number of episodes")

    env = PredatorPreyEnv()
    # env.init_curses()
    env.init_args(parser)
    args = parser.parse_args()
    env.multi_agent_init(args)
    
    config = Config()
    config.seed = 1
    config.environment = env
    config.num_episodes_to_run = args.nepisodes
    config.file_to_save_data_results = "results/data_and_graphs/Predator_Prey_Results_Data.pkl"
    config.file_to_save_results_graph = "results/data_and_graphs/Predator_Prey_Results_Graph.png"
    config.show_solution_score = False
    config.visualise_individual_results = False
    config.visualise_overall_agent_results = True
    config.standard_deviation_results = 1.0
    config.runs_per_agent = 1
    config.use_GPU = False
    config.overwrite_existing_results_file = False
    config.randomise_random_seed = True
    config.save_model = True
    # config.model_dir = None
    config.model_dir = pathlib.Path("results/PredatorPreyModels/")
    # config.load_model = True
    config.load_model = False
    # config.load_model_episode = 26000
    config.load_model_episode = 0
    config.render_env = False

    if config.save_model and config.model_dir == None:
        raise Exception("config.save_model is set to True but no model_dir provided")
    # elif not config.save_model and not config.model_dir == None:
        # raise Exception("config.save_model is set to False but a model_dir was provided")
    
    if config.load_model and config.load_model_episode == 0:
        raise Exception("config.load_model is set to True but no episode number provided")
    elif not config.load_model and config.load_model_episode != 0:
        raise Exception("config.load_model is set to False but load_model_episode is not set to 0")

    config.hyperparameters = {
        "Actor_Critic_Agents":  {

            "linear_hidden_units": [20, 10],
            "final_layer_activation": ["SOFTMAX", None],
            "gradient_clipping_norm": 5.0,
            "discount_rate": 0.99,
            "epsilon_decay_rate_denominator": 1.0,
            "normalise_rewards": True,
            "exploration_worker_difference": 2.0,
            "clip_rewards": False,

            "Actor": {
                "learning_rate": 0.00003,  # KU: lowered LR by an order of 10
                "linear_hidden_units": [64, 64],
                "final_layer_activation": "Softmax",
                "batch_norm": False,
                "tau": 0.005,
                "gradient_clipping_norm": 5,
                "initialiser": "Xavier"
            },

            "Critic": {
                "learning_rate": 0.00003,  # KU: lowered LR by an order of 10
                "linear_hidden_units": [64, 64],
                "final_layer_activation": None,
                "batch_norm": False,
                "buffer_size": 1000000,
                "tau": 0.005,
                "gradient_clipping_norm": 5,
                "initialiser": "Xavier"
            },

            "min_steps_before_learning": 400,
            "batch_size": 256,
            "discount_rate": 0.99,
            "mu": 0.0, #for O-H noise
            "theta": 0.15, #for O-H noise
            "sigma": 0.25, #for O-H noise
            "action_noise_std": 0.2,  # for TD3
            "action_noise_clipping_range": 0.5,  # for TD3
            "update_every_n_steps": 1,
            "learning_updates_per_learning_session": 1,
            "automatically_tune_entropy_hyperparameter": True,
            "entropy_term_weight": None,
            "add_extra_noise": False,
            "do_evaluation_iterations": True
        },
        "DQN_Agents": {
            "learning_rate": 0.005,
            "batch_size": 64,
            "buffer_size": 40000,
            "epsilon": 0.1,
            "epsilon_decay_rate_denominator": 200,
            "discount_rate": 0.99,
            "tau": 0.1,
            "alpha_prioritised_replay": 0.6,
            "beta_prioritised_replay": 0.4,
            "incremental_td_error": 1e-8,
            "update_every_n_steps": 3,
            "linear_hidden_units": [20, 20, 20],
            "final_layer_activation": "None",
            "batch_norm": False,
            "gradient_clipping_norm": 5,
            "HER_sample_proportion": 0.8,
            "clip_rewards": False,
            "learning_iterations": 1
        }
    }

    # AGENTS = [SAC_Discrete]
    AGENTS = [DQN]
    trainer = Trainer(config, AGENTS)
    trainer.run_games_for_agents()