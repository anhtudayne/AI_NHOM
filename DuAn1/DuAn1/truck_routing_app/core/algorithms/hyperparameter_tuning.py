"""
Module for hyperparameter tuning of reinforcement learning algorithms
using Optuna for the truck routing problem.
"""

import os
import time
import numpy as np
import optuna
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler
import json
from datetime import datetime

from core.map import Map
from core.rl_environment import TruckRoutingEnv
from core.algorithms.rl_DQNAgent import DQNAgentTrainer

def create_env(map_object, initial_fuel=70.0, initial_money=1500.0, fuel_per_move=0.3):
    """
    Create a TruckRoutingEnv environment with specified parameters.
    
    Args:
        map_object: Map object
        initial_fuel: Initial fuel amount
        initial_money: Initial money amount
        fuel_per_move: Fuel consumed per move
        
    Returns:
        TruckRoutingEnv: The environment
    """
    return TruckRoutingEnv(
        map_object=map_object,
        initial_fuel=initial_fuel,
        initial_money=initial_money,
        fuel_per_move=fuel_per_move,
        max_steps_per_episode=2 * map_object.size * map_object.size
    )

def sample_dqn_params(trial):
    """
    Sample DQN hyperparameters to test.
    
    Args:
        trial: Optuna trial object
        
    Returns:
        dict: Dictionary of sampled hyperparameters
    """
    # Learning rate with log scale
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True)
    
    # Buffer size for experience replay
    buffer_size = trial.suggest_int("buffer_size", 10000, 100000, step=10000)
    
    # Batch size for training
    batch_size = trial.suggest_categorical("batch_size", [32, 64, 128, 256])
    
    # Discount factor
    gamma = trial.suggest_float("gamma", 0.9, 0.999, step=0.01)
    
    # Target network update rate (tau)
    tau = trial.suggest_float("tau", 0.001, 1.0, log=True)
    
    # Exploration parameters
    exploration_fraction = trial.suggest_float("exploration_fraction", 0.05, 0.5)
    exploration_final_eps = trial.suggest_float("exploration_final_eps", 0.01, 0.2)
    
    # Training frequency and gradient steps
    train_freq = trial.suggest_categorical("train_freq", [1, 4, 8])
    gradient_steps = trial.suggest_categorical("gradient_steps", [1, 2, 4])
    
    # Neural network architecture
    net_arch_size = trial.suggest_categorical("net_arch_size", ["small", "medium", "large"])
    if net_arch_size == "small":
        net_arch = [64, 64]
    elif net_arch_size == "medium":
        net_arch = [128, 128]
    else:  # large
        net_arch = [256, 256]
    
    # Advanced DQN features
    use_double_dqn = trial.suggest_categorical("double_dqn", [True, False])
    use_dueling_dqn = trial.suggest_categorical("dueling_dqn", [True, False])
    use_prioritized_replay = trial.suggest_categorical("prioritized_replay", [True, False])
    
    if use_prioritized_replay:
        prioritized_replay_alpha = trial.suggest_float("prioritized_replay_alpha", 0.2, 0.8)
        prioritized_replay_beta0 = trial.suggest_float("prioritized_replay_beta0", 0.4, 1.0)
    else:
        prioritized_replay_alpha = 0.6
        prioritized_replay_beta0 = 0.4
    
    return {
        "learning_rate": learning_rate,
        "buffer_size": buffer_size,
        "batch_size": batch_size,
        "gamma": gamma,
        "tau": tau,
        "exploration_fraction": exploration_fraction,
        "exploration_final_eps": exploration_final_eps,
        "train_freq": train_freq,
        "gradient_steps": gradient_steps,
        "policy_kwargs": dict(net_arch=net_arch),
        "double_q": use_double_dqn,
        "dueling_net": use_dueling_dqn,
        "prioritized_replay": use_prioritized_replay,
        "prioritized_replay_alpha": prioritized_replay_alpha,
        "prioritized_replay_beta0": prioritized_replay_beta0,
    }

def evaluate_model(model, eval_env, n_eval_episodes=10):
    """
    Evaluate a trained model.
    
    Args:
        model: Trained DQN agent
        eval_env: Evaluation environment
        n_eval_episodes: Number of episodes for evaluation
        
    Returns:
        dict: Performance metrics
    """
    return model.evaluate(n_eval_episodes=n_eval_episodes)

def objective(trial, train_maps, eval_maps, n_timesteps=10000, n_eval_episodes=5):
    """
    Objective function for hyperparameter optimization.
    
    Args:
        trial: Optuna trial
        train_maps: List of maps for training
        eval_maps: List of maps for evaluation
        n_timesteps: Number of timesteps for training
        n_eval_episodes: Number of episodes for evaluation
        
    Returns:
        float: Mean reward (objective metric to maximize)
    """
    # Sample hyperparameters
    dqn_params = sample_dqn_params(trial)
    
    # Setup training and evaluation environments
    train_map = np.random.choice(train_maps)
    train_env = create_env(train_map)
    
    eval_map = np.random.choice(eval_maps)
    eval_env = create_env(eval_map)
    
    # Create model with sampled parameters
    log_dir = f"./rl_models_logs/optuna_trial_{trial.number}"
    model = DQNAgentTrainer(train_env, log_dir=log_dir)
    
    # Update model parameters
    from stable_baselines3 import DQN
    model.model = DQN(
        "MultiInputPolicy",
        train_env,
        verbose=0,
        tensorboard_log=log_dir,
        **dqn_params
    )
    
    try:
        # Train the agent
        model.train(total_timesteps=n_timesteps)
        
        # Evaluate on multiple maps
        rewards = []
        success_rates = []
        
        for eval_map in eval_maps:
            eval_env = create_env(eval_map)
            model.env = eval_env  # Update the environment
            
            metrics = evaluate_model(model, eval_env, n_eval_episodes)
            rewards.append(metrics["avg_reward"])
            success_rates.append(metrics["success_rate"])
        
        # Use mean reward and success rate as the optimization metric
        mean_reward = np.mean(rewards)
        mean_success_rate = np.mean(success_rates)
        
        # Combined metric: weight success rate more heavily
        combined_metric = 0.3 * mean_reward + 0.7 * mean_success_rate * 200
        
        # Store additional info
        trial.set_user_attr("mean_reward", mean_reward)
        trial.set_user_attr("mean_success_rate", mean_success_rate)
        
        return combined_metric
    
    except Exception as e:
        print(f"Error in trial {trial.number}: {e}")
        return float('-inf')

def optimize_hyperparameters(train_maps_dir, eval_maps_dir, n_trials=50, n_timesteps=25000, 
                             n_eval_episodes=5, n_jobs=1, study_name="dqn_optimization"):
    """
    Run hyperparameter optimization using Optuna.
    
    Args:
        train_maps_dir: Directory containing training maps
        eval_maps_dir: Directory containing evaluation maps
        n_trials: Number of optimization trials
        n_timesteps: Number of timesteps for each training
        n_eval_episodes: Number of episodes for evaluation
        n_jobs: Number of parallel jobs
        study_name: Name of the study
        
    Returns:
        dict: Best hyperparameters
    """
    # Load maps
    train_maps = []
    for filename in os.listdir(train_maps_dir):
        if filename.endswith(".json"):
            map_path = os.path.join(train_maps_dir, filename)
            train_maps.append(Map.load(map_path))
    
    eval_maps = []
    for filename in os.listdir(eval_maps_dir):
        if filename.endswith(".json"):
            map_path = os.path.join(eval_maps_dir, filename)
            eval_maps.append(Map.load(map_path))
    
    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = f"./hyperparameter_tuning_results/{timestamp}"
    os.makedirs(results_dir, exist_ok=True)
    
    # Create Optuna study
    sampler = TPESampler(n_startup_trials=5)
    pruner = MedianPruner(n_startup_trials=5, n_warmup_steps=10)
    
    study = optuna.create_study(
        study_name=study_name,
        direction="maximize",
        sampler=sampler,
        pruner=pruner,
    )
    
    try:
        study.optimize(
            lambda trial: objective(
                trial, train_maps, eval_maps, n_timesteps, n_eval_episodes
            ),
            n_trials=n_trials,
            n_jobs=n_jobs,
            show_progress_bar=True,
        )
    except KeyboardInterrupt:
        print("Optimization stopped early by user.")
    
    # Save study results
    study_results = {
        "best_params": study.best_params,
        "best_value": study.best_value,
        "best_trial": study.best_trial.number,
        "n_trials": len(study.trials),
        "datetime": timestamp
    }
    
    with open(os.path.join(results_dir, "study_results.json"), "w") as f:
        json.dump(study_results, f, indent=2)
    
    # Create a model with the best parameters
    best_dqn_params = sample_dqn_params(study.best_trial)
    best_params_path = os.path.join(results_dir, "best_params.json")
    
    with open(best_params_path, "w") as f:
        json.dump(best_dqn_params, f, indent=2)
    
    print(f"Best parameters saved to {best_params_path}")
    print(f"Best parameters: {best_dqn_params}")
    print(f"Best value (combined metric): {study.best_value}")
    print(f"Best trial: {study.best_trial.number}")
    
    # Print mean reward and success rate
    mean_reward = study.best_trial.user_attrs["mean_reward"]
    mean_success_rate = study.best_trial.user_attrs["mean_success_rate"]
    print(f"Mean reward: {mean_reward}")
    print(f"Mean success rate: {mean_success_rate}")
    
    return best_dqn_params

def train_agent_with_best_params(best_params, train_map, n_timesteps=50000, save_path=None):
    """
    Train an agent with the best parameters found.
    
    Args:
        best_params: Dictionary of best hyperparameters
        train_map: Map to train on
        n_timesteps: Number of timesteps for training
        save_path: Path to save the trained model
        
    Returns:
        DQNAgentTrainer: Trained agent
    """
    # Create environment
    env = create_env(train_map)
    
    # Create log directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = f"./rl_models_logs/best_agent_{timestamp}"
    os.makedirs(log_dir, exist_ok=True)
    
    # Create agent
    agent = DQNAgentTrainer(env, log_dir=log_dir)
    
    # Set best parameters
    from stable_baselines3 import DQN
    agent.model = DQN(
        "MultiInputPolicy",
        env,
        verbose=1,
        tensorboard_log=log_dir,
        **best_params
    )
    
    # Train the agent
    print(f"Training agent with best parameters for {n_timesteps} timesteps...")
    agent.train(total_timesteps=n_timesteps)
    
    # Save the model if path is provided
    if save_path:
        agent.save_model(save_path)
        print(f"Model saved to {save_path}")
    
    return agent

# Example usage
if __name__ == "__main__":
    # Define directories
    train_maps_dir = "./maps/train"
    eval_maps_dir = "./maps/eval"
    
    # Make sure directories exist
    os.makedirs(train_maps_dir, exist_ok=True)
    os.makedirs(eval_maps_dir, exist_ok=True)
    
    # If no maps exist, create some
    if len(os.listdir(train_maps_dir)) == 0:
        print("No training maps found. Creating some...")
        for i in range(5):
            map_obj = Map.generate_random(size=10, toll_ratio=0.05, gas_ratio=0.05, brick_ratio=0.2)
            map_obj.save(os.path.join(train_maps_dir, f"train_map_{i}.json"))
    
    if len(os.listdir(eval_maps_dir)) == 0:
        print("No evaluation maps found. Creating some...")
        for i in range(3):
            map_obj = Map.generate_random(size=10, toll_ratio=0.05, gas_ratio=0.05, brick_ratio=0.2)
            map_obj.save(os.path.join(eval_maps_dir, f"eval_map_{i}.json"))
    
    # Run hyperparameter optimization (reduced n_trials for demonstration)
    best_params = optimize_hyperparameters(
        train_maps_dir=train_maps_dir,
        eval_maps_dir=eval_maps_dir,
        n_trials=10,  # Set higher for better results, e.g., 50-100
        n_timesteps=10000,  # Set higher for better results, e.g., 50000
        n_eval_episodes=3,
        n_jobs=1  # Increase for parallel optimization if multiple CPUs available
    )
    
    # Train with best parameters
    # Load a map for final training
    map_obj = Map.load(os.path.join(eval_maps_dir, os.listdir(eval_maps_dir)[0]))
    
    # Train and save the best model
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_path = f"./saved_models/best_dqn_agent_{timestamp}"
    
    train_agent_with_best_params(
        best_params=best_params,
        train_map=map_obj,
        n_timesteps=25000,  # Set higher for better results, e.g., 100000
        save_path=save_path
    ) 