import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import torch
import random
import os
import argparse
import csv
from stable_baselines3 import A2C, DQN
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.logger import configure
from stable_baselines3.common.vec_env import DummyVecEnv

def plot_learning_curves(rewards_a2c, rewards_dqn, runs):
    """
    Plot learning curves for A2C and DQN
    
    Parameters:
        rewards_a2c (list): Rewards for A2C across runs
        rewards_dqn (list): Rewards for DQN across runs
        runs (int): Number of runs
    """
    plt.figure(figsize=(15, 10))
    
    # A2C subplot
    plt.subplot(2, 1, 1)
    rewards_a2c = np.array(rewards_a2c)
    
    # Compute moving average
    window = min(50, rewards_a2c.shape[1] // 5)
    rewards_a2c_ma = np.zeros((rewards_a2c.shape[0], rewards_a2c.shape[1] - window + 1))
    
    for i in range(rewards_a2c.shape[0]):
        rewards_a2c_ma[i] = np.convolve(rewards_a2c[i], np.ones(window)/window, mode='valid')
    
    # Plot raw rewards and moving average
    plt.plot(rewards_a2c.T, alpha=0.1, color='blue')
    plt.plot(np.mean(rewards_a2c_ma, axis=0), color='red', linewidth=2, 
             label=f'Moving Average (window={window})')
    plt.title(f'A2C Training Rewards - {runs} Runs')
    plt.xlabel('Episodes')
    plt.ylabel('Reward')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # DQN subplot
    plt.subplot(2, 1, 2)
    rewards_dqn = np.array(rewards_dqn)
    
    # Compute moving average
    window = min(50, rewards_dqn.shape[1] // 5)
    rewards_dqn_ma = np.zeros((rewards_dqn.shape[0], rewards_dqn.shape[1] - window + 1))
    
    for i in range(rewards_dqn.shape[0]):
        rewards_dqn_ma[i] = np.convolve(rewards_dqn[i], np.ones(window)/window, mode='valid')
    
    # Plot raw rewards and moving average
    plt.plot(rewards_dqn.T, alpha=0.1, color='green')
    plt.plot(np.mean(rewards_dqn_ma, axis=0), color='red', linewidth=2, 
             label=f'Moving Average (window={window})')
    plt.title(f'DQN Training Rewards - {runs} Runs')
    plt.xlabel('Episodes')
    plt.ylabel('Reward')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('results/learning_curves_comparison.png', dpi=300)
    plt.close()

def train_agent(agent_type, run_id, max_episodes=1000, seed=42):
    """
    Train a Stable Baselines agent on the Lunar Lander environment
    
    Parameters:
        agent_type (str): Type of agent ('A2C' or 'DQN')
        run_id (int): Unique run identifier
        max_episodes (int): Maximum number of training episodes
        seed (int): Random seed for reproducibility
    
    Returns:
        tuple: Trained model, training results, and training time
    """
    import time
    
    # Set random seeds
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    
    # Create environment
    env = gym.make('LunarLander-v3')
    vec_env = DummyVecEnv([lambda: env])
    
    # Configure logger
    log_path = f'results/{agent_type.lower()}_logs_run_{run_id}/'
    os.makedirs(log_path, exist_ok=True)
    logger = configure(log_path, ["csv"])
    
    # Select and initialize agent
    if agent_type == 'A2C':
        model = A2C("MlpPolicy", vec_env, seed=seed, verbose=1)
    elif agent_type == 'DQN':
        model = DQN("MlpPolicy", vec_env, seed=seed, verbose=1)
    else:
        raise ValueError(f"Unsupported agent type: {agent_type}")
    
    # Set logger
    model.set_logger(logger)
    
    # Train the agent and track rewards
    start_time = time.time()
    
    # Custom training loop to track rewards
    rewards_per_episode = []
    for episode in range(max_episodes):
        # Reset environment
        state, _ = env.reset()
        done = False
        total_reward = 0
        
        while not done:
            # Select and take action
            action, _ = model.predict(state, deterministic=False)
            next_state, reward, terminated, truncated, _ = env.step(action)
            
            total_reward += reward
            state = next_state
            
            if terminated or truncated:
                done = True
        
        rewards_per_episode.append(total_reward)
        
        # Learn from collected experiences if needed
        model.learn(total_timesteps=1000)
    
    training_time = time.time() - start_time
    
    # Save the model
    model_path = f'data/{agent_type.lower()}_lunar_lander_model_run_{run_id}.zip'
    model.save(model_path)
    
    # Evaluate training performance
    mean_reward, std_reward = evaluate_policy(model, vec_env, n_eval_episodes=100)
    
    # Close environment
    env.close()
    
    return model, {
        'mean_reward': mean_reward, 
        'std_reward': std_reward,
        'rewards_per_episode': rewards_per_episode
    }, training_time

def main():
    parser = argparse.ArgumentParser(description='Stable Baselines A2C and DQN Comparison')
    parser.add_argument('--runs', type=int, default=1, help='Number of runs (default: 3)')
    parser.add_argument('--episodes', type=int, default=1000, help='Maximum training episodes (default: 1000)')
    parser.add_argument('--seed', type=int, default=42, help='Random seed (default: 42)')
    parser.add_argument('--render', action='store_true', help='Render test episodes')
    
    args = parser.parse_args()
    
    # Ensure results and data directories exist
    os.makedirs('results', exist_ok=True)
    os.makedirs('data', exist_ok=True)
    
    # Comparative results dictionary
    comparative_results = {}
    
    # Agents to compare
    agents = ['A2C', 'DQN']
    
    for agent in agents:
        print(f"\nRunning {agent} experiments...")
        results_per_run = []
        rewards_across_runs = []
        
        for run in range(args.runs):
            print(f"Run {run+1}/{args.runs}")
            
            # Train agent
            _, training_results, training_time = train_agent(
                agent_type=agent, 
                run_id=run, 
                max_episodes=args.episodes, 
                seed=args.seed
            )
            
            rewards_across_runs.append(training_results['rewards_per_episode'])
            
            results_per_run.append({
                'training_results': training_results,
                'training_time': training_time
            })
        
        # Average results across runs
        comparative_results[agent] = {
            'training_results': {
                'mean_reward': np.mean([r['training_results']['mean_reward'] for r in results_per_run]),
                'std_reward': np.mean([r['training_results']['std_reward'] for r in results_per_run])
            },
            'training_time': np.mean([r['training_time'] for r in results_per_run]),
            'rewards_across_runs': rewards_across_runs
        }
        
        print(f"{agent} Results:")
        print(f"Training Mean Reward: {comparative_results[agent]['training_results']['mean_reward']:.2f}")
    
    # Plot learning curves
    plot_learning_curves(
        comparative_results['A2C']['rewards_across_runs'], 
        comparative_results['DQN']['rewards_across_runs'], 
        args.runs
    )
    
    # Save comparative results to CSV
    with open('results/comparative_results.csv', 'w', newline='') as csvfile:
        fieldnames = ['Agent', 'Mean Reward', 'Std Reward', 'Training Time']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        writer.writeheader()
        for agent, results in comparative_results.items():
            writer.writerow({
                'Agent': agent,
                'Mean Reward': results['training_results']['mean_reward'],
                'Std Reward': results['training_results']['std_reward'],
                'Training Time': results['training_time']
            })
    
    # Print final comparative summary
    print("\nComparative Summary:")
    for agent, results in comparative_results.items():
        print(f"\n{agent}:")
        print(f"  Training Mean Reward: {results['training_results']['mean_reward']:.2f}")
        print(f"  Training Time: {results['training_time']:.2f}s")

if __name__ == "__main__":
    main()