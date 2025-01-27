# a2c - no noise
# /opt/anaconda3/lib/python3.12
import sys
sys.path.insert(0, '/opt/anaconda3/lib/python3.12/site-packages')
import gymnasium as gym
import numpy as np
#import wandb
import logging
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.evaluation import evaluate_policy
# from wand.integration.sb3 import WandbCallback
from stable_baselines3.common.logger import configure
# import pdb
from stable_baselines3 import A2C
from stable_baselines3.common.vec_env import DummyVecEnv, VecVideoRecorder

import ale_py
gym.register_envs(ale_py)

# Setting the seed for reproducibility
seed = 42
np.random.seed(seed)

def make_env():
    env = gym.make("ALE/Breakout-v5", render_mode="rgb_array")
    env = gym.wrappers.RecordEpisodeStatistics(env)  # record stats such as returns
    env.reset(seed=seed) 
    env.action_space.seed(seed)
    return env

# run = wandb.init(
#     project="atari_a2c_breakout",
#     config = {"policy": "A2C_MlpPolicy", "learning_rate": 0.000827, "gamma":0.96979, "gae_lambda":0.92680, "ent_coef":.0000199},
#     monitor_gym=True,
#     sync_tensorboard=True,
#     save_code=True

# )

env = DummyVecEnv([make_env])
env = VecVideoRecorder(env, f"videos/test1", record_video_trigger = lambda x: x % 2000 == 0, video_length=200) # f"videos/{run.id}"

# Instantiate the agent without noise
#tmp_path = "/tmp/sb3_log/"
#new_logger = configure(tmp_path, ["stdout", "csv", "tensorboard"])
model = A2C("MlpPolicy", env, learning_rate= 0.000827, gamma = 0.96979, gae_lambda= 0.92680, ent_coef= 0.0000199, verbose=1, tensorboard_log=f"runs/a2c_breakout")
#model.set_logger(new_logger)

# Train the agent
model.learn(total_timesteps= 300000,
            #callback=WandbCallback(gradient_save_freq=200, model_save_path=f"models/{run.id}", verbose=2)
            callback=EvalCallback(env, best_model_save_path=f"models/best/test1", log_path="./evalLogs/", eval_freq=1000)
)

# Save the agent
model.save("atari_a2c_breakout")
vec_env = model.get_env()

del model # remove to demonstrate saving and loading

# Load the agent
model = A2C.load("atari_a2c_breakout")

# Evaluate the agent
mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=10)
#wandb.log({"mean_reward": mean_reward, "std_reward": std_reward})

print(f"mean reward: {mean_reward} +/- {std_reward}")
obs = vec_env.reset()

reward_total = 0
for _ in range(1000): 
    action, _states = model.predict(obs)
    obs, rewards, dones, info = vec_env.step(action)
    reward_total += rewards
    #wandb.log({"returns" : reward_total, "reward": rewards})
    # print(rewards)
    if dones:
        reward_total = 0
        obs = vec_env.step.reset()

print(reward_total)
env.close()