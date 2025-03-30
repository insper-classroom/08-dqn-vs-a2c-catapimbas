import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import torch
import random
import os
import argparse
import csv
from tqdm import tqdm
from stable_baselines3 import A2C, DQN
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.logger import configure
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.atari_wrappers import AtariWrapper
import ale_py

def plot_learning_curves(rewards_a2c, rewards_dqn, runs, env_name):
    """
    Plot learning curves for A2C and DQN
    
    Parameters:
        rewards_a2c (list): Rewards for A2C across runs
        rewards_dqn (list): Rewards for DQN across runs
        runs (int): Number of runs
        env_name (str): Name of the environment
    """
    plt.figure(figsize=(15, 10))
    
    # Convert to numpy arrays
    rewards_a2c = np.array(rewards_a2c)
    rewards_dqn = np.array(rewards_dqn)
    
    # Compute moving average
    a2c_window = min(50, rewards_a2c.shape[1] // 5)
    dqn_window = min(50, rewards_dqn.shape[1] // 5)
    
    rewards_a2c_ma = np.zeros((rewards_a2c.shape[0], rewards_a2c.shape[1] - a2c_window + 1))
    rewards_dqn_ma = np.zeros((rewards_dqn.shape[0], rewards_dqn.shape[1] - dqn_window + 1))
    
    for i in range(rewards_a2c.shape[0]):
        rewards_a2c_ma[i] = np.convolve(rewards_a2c[i], np.ones(a2c_window)/a2c_window, mode='valid')
    
    for i in range(rewards_dqn.shape[0]):
        rewards_dqn_ma[i] = np.convolve(rewards_dqn[i], np.ones(dqn_window)/dqn_window, mode='valid')
    
    # Plot raw rewards with transparency
    plt.plot(rewards_a2c.T, alpha=0.25, color='blue', label='A2C Raw Rewards')
    plt.plot(rewards_dqn.T, alpha=0.25, color='green', label='DQN Raw Rewards')
    
    # Plot moving averages
    plt.plot(np.mean(rewards_a2c_ma, axis=0), color='darkblue', linewidth=2, 
             label=f'A2C Moving Average (window={a2c_window})')
    plt.plot(np.mean(rewards_dqn_ma, axis=0), color='darkgreen', linewidth=2, 
             label=f'DQN Moving Average (window={dqn_window})')
    
    plt.title(f'{env_name}: A2C and DQN Training Rewards - {runs} Runs')
    plt.xlabel('Episodes')
    plt.ylabel('Reward')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Ensure directories exist
    os.makedirs('results', exist_ok=True)
    
    # Save figure with sanitized filename
    safe_env_name = env_name.lower().replace("-", "_").replace("/", "_").replace("\\", "_")
    plt.savefig(f'results/{safe_env_name}_learning_curves_comparison.png', dpi=300)
    plt.close()

def train_agent(agent_type, env_name, run_id, max_episodes=1000, seed=42):
    """
    Train a Stable Baselines agent on specified environment
    
    Parameters:
        agent_type (str): Type of agent ('A2C' or 'DQN')
        env_name (str): Name of the environment
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
    
    # Configure for Atari vs standard environments
    is_atari = 'ALE/' in env_name
    
    # Create environment with appropriate wrappers
    if is_atari:
        # For Atari, use AtariWrapper and CnnPolicy
        env = AtariWrapper(gym.make(env_name))
        policy = "CnnPolicy"
        # Reduced learning rate for Atari
        learning_rate = 2.5e-4
    else:
        # For standard environments, use normal gym and MlpPolicy
        env = gym.make(env_name)
        policy = "MlpPolicy"
        learning_rate = 1e-3
    
    # Wrap with Monitor and VecEnv
    env = Monitor(env)
    vec_env = DummyVecEnv([lambda: env])
    
    # Configure logger
    # Sanitize environment name for file paths
    safe_env_name = env_name.lower().replace("-", "_").replace("/", "_").replace("\\", "_")
    log_path = f'results/{safe_env_name}_{agent_type.lower()}_logs_run_{run_id}/'
    os.makedirs(log_path, exist_ok=True)
    logger = configure(log_path, ["csv"])
    
    # Hyperparameters
    gamma = 0.99  # Discount Factor
    batch_size = 64  # Batch Size for DQN
    
    # Buffer sizes - reduced for Atari
    if is_atari:
        buffer_size = 10000  # Reduced buffer size for Atari
    else:
        buffer_size = 50000  # Original buffer size
    
    # Select and initialize agent with specified hyperparameters
    if agent_type == 'A2C':
        model = A2C(
            policy, 
            vec_env, 
            seed=seed, 
            verbose=1,
            gamma=gamma,
            learning_rate=learning_rate
        )
    elif agent_type == 'DQN':
        model = DQN(
            policy, 
            vec_env, 
            seed=seed, 
            verbose=1,
            gamma=gamma,
            batch_size=batch_size,
            learning_rate=learning_rate,
            buffer_size=buffer_size,
            exploration_initial_eps=1.0,
            exploration_final_eps=0.01,
            exploration_fraction=0.995
        )
    else:
        raise ValueError(f"Unsupported agent type: {agent_type}")
    
    # Set logger
    model.set_logger(logger)
    
    # Train the agent and track rewards
    start_time = time.time()
    
    # Custom training loop to track rewards with progress bar
    rewards_per_episode = []
    
    # Use tqdm for progress tracking
    with tqdm(total=max_episodes, desc=f'{agent_type} {env_name} Training', unit='episode') as pbar:
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
            
            # Update progress bar with current episode reward
            pbar.set_postfix({'Episode Reward': f'{total_reward:.2f}'})
            pbar.update(1)
            
            # Learn from collected experiences if needed
            model.learn(total_timesteps=1000)
    
    training_time = time.time() - start_time
    
    # Save the model
    # Ensure data directory exists
    os.makedirs('data', exist_ok=True)
    
    # Save with sanitized filename
    model_path = f'data/{safe_env_name}_{agent_type.lower()}_model_run_{run_id}.zip'
    model.save(model_path)
    
    # Evaluate training performance with fewer episodes for Atari
    n_eval_episodes = 10 if is_atari else 100
    mean_reward, std_reward = evaluate_policy(model, vec_env, n_eval_episodes=n_eval_episodes)
    
    # Close environment
    env.close()
    
    return model, {
        'mean_reward': mean_reward, 
        'std_reward': std_reward,
        'rewards_per_episode': rewards_per_episode
    }, training_time

def main():
    parser = argparse.ArgumentParser(description='Multi-Environment RL Agent Comparison')
    parser.add_argument('--runs', type=int, default=1, help='Number of runs (default: 3)')
    parser.add_argument('--episodes', type=int, default=2000, help='Maximum training episodes (default: 1000)')
    parser.add_argument('--seed', type=int, default=42, help='Random seed (default: 42)')
    parser.add_argument('--environments', nargs='+', 
                        default=['CartPole-v1', 'LunarLander-v3', 'ALE/Breakout-v5'], 
                        help='Environments to train on')
    
    args = parser.parse_args()
    
    # Ensure results and data directories exist
    os.makedirs('results', exist_ok=True)
    os.makedirs('data', exist_ok=True)
    
    # Agents to compare
    agents = ['A2C', 'DQN']
    
    # Comparative results for all environments
    all_comparative_results = {}
    
    # Iterate through environments
    for env_name in args.environments:
        print(f"\nTraining on {env_name} environment...")
        
        # Sanitize environment name for file paths
        safe_env_name = env_name.lower().replace("-", "_").replace("/", "_").replace("\\", "_")
        
        # Comparative results dictionary for this environment
        comparative_results = {}
        
        for agent in agents:
            print(f"\nRunning {agent} experiments...")
            results_per_run = []
            rewards_across_runs = []
            
            for run in range(args.runs):
                print(f"Run {run+1}/{args.runs}")
                
                # Train agent
                _, training_results, training_time = train_agent(
                    agent_type=agent, 
                    env_name=env_name,
                    run_id=run, 
                    max_episodes=args.episodes, 
                    seed=args.seed + run  # Different seed for each run
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
        
        # Plot learning curves for this environment
        plot_learning_curves(
            comparative_results['A2C']['rewards_across_runs'], 
            comparative_results['DQN']['rewards_across_runs'], 
            args.runs,
            env_name
        )
        
        # Save results for this environment
        all_comparative_results[env_name] = comparative_results
        
        # Save comparative results to CSV
        with open(f'results/{safe_env_name}_comparative_results.csv', 'w', newline='') as csvfile:
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
    for env_name, results in all_comparative_results.items():
        print(f"\n{env_name}:")
        for agent, agent_results in results.items():
            print(f"  {agent}:")
            print(f"    Training Mean Reward: {agent_results['training_results']['mean_reward']:.2f}")
            print(f"    Training Time: {agent_results['training_time']:.2f}s")

if __name__ == "__main__":
    main()