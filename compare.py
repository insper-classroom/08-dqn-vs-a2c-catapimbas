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
    logger = configure(log_path, [ "csv", "log"])
    
    # Select and initialize agent
    if agent_type == 'A2C':
        model = A2C("MlpPolicy", vec_env, seed=seed, verbose=1)
    elif agent_type == 'DQN':
        model = DQN("MlpPolicy", vec_env, seed=seed, verbose=1)
    else:
        raise ValueError(f"Unsupported agent type: {agent_type}")
    
    # Set logger
    model.set_logger(logger)
    
    # Train the agent
    start_time = time.time()
    model.learn(total_timesteps=max_episodes * 1000)  # Convert episodes to timesteps
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
        'std_reward': std_reward
    }, training_time

def test_agent(agent_type, run_id, max_episodes=10, render=False, seed=42):
    """
    Test a trained Stable Baselines agent on the Lunar Lander environment
    
    Parameters:
        agent_type (str): Type of agent ('A2C' or 'DQN')
        run_id (int): Unique run identifier
        max_episodes (int): Number of test episodes
        render (bool): Whether to render the environment
        seed (int): Random seed for reproducibility
    
    Returns:
        dict: Test results including rewards and statistics
    """
    # Set random seeds
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    
    # Create environment
    render_mode = "human" if render else None
    env = gym.make('LunarLander-v3', render_mode=render_mode)
    
    # Load the trained model
    model_path = f'data/{agent_type.lower()}_lunar_lander_model_run_{run_id}.zip'
    try:
        if agent_type == 'A2C':
            model = A2C.load(model_path, env=env)
        elif agent_type == 'DQN':
            model = DQN.load(model_path, env=env)
        else:
            raise ValueError(f"Unsupported agent type: {agent_type}")
    except FileNotFoundError:
        print(f"Error: Could not find {agent_type} model for run {run_id}")
        return {}
    
    # Test the agent
    rewards = []
    for episode in range(max_episodes):
        state, _ = env.reset()
        total_reward = 0
        done = False
        
        while not done:
            action, _ = model.predict(state, deterministic=True)
            state, reward, terminated, truncated, _ = env.step(action)
            
            total_reward += reward
            
            if terminated or truncated:
                done = True
        
        print(f"{agent_type} - Episode {episode+1}: Reward = {total_reward:.2f}")
        rewards.append(total_reward)
    
    env.close()
    
    # Compute statistics
    return {
        'rewards': rewards,
        'mean_reward': np.mean(rewards),
        'std_reward': np.std(rewards),
        'min_reward': np.min(rewards),
        'max_reward': np.max(rewards)
    }

def plot_test_results(test_results_a2c, test_results_dqn, runs):
    """
    Plot comparative test results for A2C and DQN
    
    Parameters:
        test_results_a2c (dict): A2C test results
        test_results_dqn (dict): DQN test results
        runs (int): Number of runs
    """
    plt.figure(figsize=(12, 6))
    
    # A2C results
    plt.subplot(1, 2, 1)
    plt.title(f'A2C Test Results (Runs: {runs})')
    plt.plot(test_results_a2c['rewards'], marker='o', linestyle='-', alpha=0.6, label='Episode Rewards')
    plt.axhline(y=test_results_a2c['mean_reward'], color='r', linestyle='--', 
                label=f'Mean: {test_results_a2c["mean_reward"]:.2f}')
    plt.xlabel('Test Episodes')
    plt.ylabel('Reward')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # DQN results
    plt.subplot(1, 2, 2)
    plt.title(f'DQN Test Results (Runs: {runs})')
    plt.plot(test_results_dqn['rewards'], marker='o', linestyle='-', alpha=0.6, label='Episode Rewards')
    plt.axhline(y=test_results_dqn['mean_reward'], color='r', linestyle='--', 
                label=f'Mean: {test_results_dqn["mean_reward"]:.2f}')
    plt.xlabel('Test Episodes')
    plt.ylabel('Reward')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('results/comparative_test_results.png', dpi=300)
    plt.close()

def save_comparative_results(results_dict):
    """
    Save comparative results to a CSV file
    
    Parameters:
        results_dict (dict): Dictionary containing results for A2C and DQN
    """
    with open('results/comparative_results.csv', 'w', newline='') as csvfile:
        fieldnames = ['Agent', 'Mean Reward', 'Std Reward', 'Min Reward', 'Max Reward', 'Training Time']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        writer.writeheader()
        for agent, results in results_dict.items():
            writer.writerow({
                'Agent': agent,
                'Mean Reward': results['test_results']['mean_reward'],
                'Std Reward': results['test_results']['std_reward'],
                'Min Reward': results['test_results']['min_reward'],
                'Max Reward': results['test_results']['max_reward'],
                'Training Time': results['training_time']
            })

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
        
        for run in range(args.runs):
            print(f"Run {run+1}/{args.runs}")
            
            # Train agent
            _, training_results, training_time = train_agent(
                agent_type=agent, 
                run_id=run, 
                max_episodes=args.episodes, 
                seed=args.seed
            )
            
            # Test agent
            test_results = test_agent(
                agent_type=agent, 
                run_id=run, 
                max_episodes=10, 
                render=args.render, 
                seed=args.seed
            )
            
            results_per_run.append({
                'training_results': training_results,
                'test_results': test_results,
                'training_time': training_time
            })
        
        # Average results across runs
        comparative_results[agent] = {
            'training_results': {
                'mean_reward': np.mean([r['training_results']['mean_reward'] for r in results_per_run]),
                'std_reward': np.mean([r['training_results']['std_reward'] for r in results_per_run])
            },
            'test_results': {
                'mean_reward': np.mean([r['test_results']['mean_reward'] for r in results_per_run]),
                'std_reward': np.mean([r['test_results']['std_reward'] for r in results_per_run]),
                'min_reward': np.min([r['test_results']['min_reward'] for r in results_per_run]),
                'max_reward': np.max([r['test_results']['max_reward'] for r in results_per_run]),
                'rewards': np.concatenate([r['test_results']['rewards'] for r in results_per_run])
            },
            'training_time': np.mean([r['training_time'] for r in results_per_run])
        }
        
        print(f"{agent} Results:")
        print(f"Training Mean Reward: {comparative_results[agent]['training_results']['mean_reward']:.2f}")
        print(f"Test Mean Reward: {comparative_results[agent]['test_results']['mean_reward']:.2f}")
    
    # Plot and save results
    plot_test_results(
        comparative_results['A2C']['test_results'], 
        comparative_results['DQN']['test_results'], 
        args.runs
    )
    save_comparative_results(comparative_results)
    
    # Print final comparative summary
    print("\nComparative Summary:")
    for agent, results in comparative_results.items():
        print(f"\n{agent}:")
        print(f"  Training Mean Reward: {results['training_results']['mean_reward']:.2f}")
        print(f"  Test Mean Reward: {results['test_results']['mean_reward']:.2f}")
        print(f"  Training Time: {results['training_time']:.2f}s")

if __name__ == "__main__":
    main()