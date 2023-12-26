import os
import cv2
import numpy as np
import gymnasium as gym
from stable_baselines3 import PPO, A2C
from gymnasium.wrappers import GrayScaleObservation,FrameStack, NormalizeReward



### Pasta onde vamos guardar os modelos (models_dir) e as informações sobre os modelos (logs) ###
models_dir = "models/PPO/(Nome da Pasta)"
log_dir= "logs"
if not os.path.exists(models_dir):
	os.makedirs(models_dir)
if not os.path.exists(log_dir):
	os.makedirs(log_dir)
     


### Reward c/ Penalty quando vai para a relva ###
class PenaltyReward(gym.Wrapper):
    def __init__(self, env):
        super(PenaltyReward, self).__init__(env)

    def step(self, action):
        results = self.env.step(action)
        state, reward, done, info = results[:4]
        # Admitindo a relva como uma cor representada por np.array([102,204,102])
        grass = cv2.inRange(state,np.array([102,204,102]),np.array([102,204,102]))
        grass_pixels = cv2.findNonZero(grass)
        if grass_pixels is not None:
            reward = -100
            done = True
        return state, reward, done, info, {}
    
### Clipping Reward
from typing import SupportsFloat

class ClipReward(RewardWrapper):
    def __init__(self, env, min_reward, max_reward):
        super().__init__(env)
        self.min_reward = min_reward
        self.max_reward = max_reward
        self.reward_range = (min_reward,max_reward)

    def reward(self, r: SupportsFloat) -> SupportsFloat:
        return np.clip(r,self.min_reward,self.max_reward)


### Melhor Gamma 
gamma=0.9 #ou 0.99 ou 0.95

initial_env = gym.make("CarRacing-v2", render_mode="human", continuous=False)  
gray_env = GrayScaleObservation(initial_env, keep_dim=True) #keep_dim mantém as dimensões do initial_env
#frame_env = FrameStack(initial_env,3)
#penalty_env = PenaltyReward(initial_env)
#clip_env = ClipReward(initial_env, -1.0,1.0)
normalize_env = NormalizeReward(gray_env,gamma)
env = normalize_env  #ou gray_env ou frame_env ou penalty_env ou normalize_env


### Para Guardar o Modelo ###
model = PPO('CnnPolicy', env, verbose=1, tensorboard_log=log_dir)
TIMESTEPS = 10000
iteract = 0
for i in range(1000000):
    iteract+=1
    model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False, tb_log_name="Nome da Pasta")
    model.save(f"{models_dir}/{TIMESTEPS*i}")


### Para Ler um Modelo Guardado ###
model_path = f"{models_dir}/(Ficheiro .zip)"
model = PPO.load(model_path, env=env)

episodes = 5
for ep in range(episodes):
    obs, info = env.reset()
    done = False
    while not done:
        action, _states = model.predict(obs)
        obs, rewards, trunc, done, info = env.step(action)
        env.render()