from stable_baselines3 import PPO, DQN, A2C
import gymnasium as gym
from stable_baselines3.common.logger import configure
from stable_baselines3.common.evaluation import evaluate_policy

# Configure logger
tmp_path = "./results/"
new_logger = configure(tmp_path, ["stdout", "csv", "log", "json"])

# Create environment
env = gym.make("CartPole-v1")
model = A2C(policy="MlpPolicy", env=env)

# Set logger and train the model
model.set_logger(new_logger)
model.learn(total_timesteps=100_000)

# Evaluate the model
mean_reward, std_reward = evaluate_policy(model, model.get_env(), n_eval_episodes=10)
print(f'Mean reward: {mean_reward} +/- {std_reward:.2f}')

print('modelo treinado')

# Render the trained model
env = gym.make("CartPole-v1", render_mode='human')
obs, info = env.reset()  # Unpack both obs and info
for i in range(1000):
    action, _state = model.predict(obs, deterministic=True)
    obs, reward, done, truncated, info = env.step(action)
    env.render()
    if done or truncated:
        obs, info = env.reset()  # Unpack both obs and info when resetting