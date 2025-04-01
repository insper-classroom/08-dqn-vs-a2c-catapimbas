from stable_baselines3 import A2C, DQN
import gymnasium as gym
from stable_baselines3.common.atari_wrappers import AtariWrapper


model = DQN.load("models/ale_breakout_v5_dqn_model_run_0.zip")

# Create environment with proper wrappers and render mode
env = AtariWrapper(gym.make("ALE/Breakout-v5", render_mode="human"))
obs, _ = env.reset()
done = False
total_reward = 0

# Run the game
while not done:
    action, _ = model.predict(obs)
    obs, reward, terminated, truncated, info = env.step(action)
    total_reward += reward
    done = terminated or truncated
    
print(f"Total reward: {total_reward}")
env.close()