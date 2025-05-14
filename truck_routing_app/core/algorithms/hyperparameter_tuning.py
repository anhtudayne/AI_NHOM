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
from typing import Any

from core.map import Map
from core.rl_environment import TruckRoutingEnv
from core.constants import CellType, MovementCosts, StationCosts
from core.algorithms.rl_DQNAgent import DQNAgentTrainer
from stable_baselines3.common.callbacks import BaseCallback

def create_env(map_object: Map, initial_fuel: float = 70.0, initial_money: float = 1500.0, fuel_per_move: float = 0.3) -> TruckRoutingEnv:
    """
    Create a truck routing environment.
    
    Args:
        map_object: The map object
        initial_fuel: Initial fuel level
        initial_money: Initial money
        fuel_per_move: Fuel consumption per move
        
    Returns:
        TruckRoutingEnv: The environment
    """
    # Set default values in MovementCosts
    from ..constants import MovementCosts, StationCosts
    
    # Cập nhật các hằng số trong constants.py
    MovementCosts.MAX_FUEL = initial_fuel
    MovementCosts.FUEL_PER_MOVE = fuel_per_move
    
    # Khởi tạo môi trường
    from ..rl_environment import TruckRoutingEnv
    return TruckRoutingEnv(
        map_object=map_object
    )

def sample_dqn_params(trial: optuna.trial.Trial) -> dict[str, Any]:
    """
    Sample DQN hyperparameters to test.
    
    Args:
        trial: Optuna trial object
        
    Returns:
        dict: Dictionary of sampled hyperparameters
    """
    # Learning rate with log scale - mở rộng phạm vi
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-2, log=True)
    
    # Buffer size for experience replay - tăng kích thước tối đa
    buffer_size = trial.suggest_int("buffer_size", 20000, 500000, step=20000)
    
    # Batch size for training - thêm kích thước lớn hơn
    batch_size = trial.suggest_categorical("batch_size", [32, 64, 128, 256, 512])
    
    # Discount factor - mở rộng phạm vi để cân nhắc cả ngắn và dài hạn
    gamma = trial.suggest_float("gamma", 0.8, 0.999, step=0.01)
    
    # Target network update rate (tau) - mở rộng phạm vi
    tau = trial.suggest_float("tau", 0.0005, 0.2, log=True)
    
    # Exploration parameters - mở rộng phạm vi exploration để đảm bảo thoát lặp
    exploration_fraction = trial.suggest_float("exploration_fraction", 0.15, 0.8)
    exploration_final_eps = trial.suggest_float("exploration_final_eps", 0.01, 0.3)
    exploration_initial_eps = trial.suggest_float("exploration_initial_eps", 0.8, 1.0)
    
    # Training frequency and gradient steps
    train_freq = trial.suggest_categorical("train_freq", [1, 2, 4, 8])
    gradient_steps = trial.suggest_categorical("gradient_steps", [1, 2, 4, 8])
    
    # Learning starts - đảm bảo đủ exploration ban đầu
    learning_starts = trial.suggest_int("learning_starts", 1000, 50000, log=True)
    
    # Neural network architecture - thêm mạng rộng và sâu hơn
    net_arch_size = trial.suggest_categorical("net_arch_size", ["small", "medium", "large", "xlarge"])
    if net_arch_size == "small":
        net_arch = [64, 64]
    elif net_arch_size == "medium":
        net_arch = [128, 128]
    elif net_arch_size == "large":
        net_arch = [256, 256]
    else:  # xlarge
        net_arch = [512, 256, 128]
    
    # Advanced DQN features - mặc định bật Double DQN vì hiệu quả
    use_double_dqn = trial.suggest_categorical("use_double_dqn", [True])
    use_dueling_network = trial.suggest_categorical("use_dueling_network", [True, False])
    use_prioritized_replay = trial.suggest_categorical("use_prioritized_replay", [True, False])
    
    # Thêm cấu hình chi tiết cho PER nếu được sử dụng
    if use_prioritized_replay:
        prioritized_replay_alpha = trial.suggest_float("prioritized_replay_alpha", 0.3, 0.9)
        prioritized_replay_beta0 = trial.suggest_float("prioritized_replay_beta0", 0.3, 1.0)
        prioritized_replay_eps = trial.suggest_float("prioritized_replay_eps", 1e-8, 1e-5, log=True)
    else:
        prioritized_replay_alpha = 0.6
        prioritized_replay_beta0 = 0.4
        prioritized_replay_eps = 1e-6
    
    # Thêm các tham số điều chỉnh gradient
    max_grad_norm = trial.suggest_float("max_grad_norm", 5, 20)
    
    return {
        "learning_rate": learning_rate,
        "buffer_size": buffer_size,
        "batch_size": batch_size,
        "gamma": gamma,
        "tau": tau,
        "exploration_fraction": exploration_fraction,
        "exploration_initial_eps": exploration_initial_eps,
        "exploration_final_eps": exploration_final_eps,
        "train_freq": train_freq,
        "gradient_steps": gradient_steps,
        "learning_starts": learning_starts,
        "policy_kwargs": dict(net_arch=net_arch),
        "use_double_dqn": use_double_dqn,
        "use_dueling_network": use_dueling_network,
        "use_prioritized_replay": use_prioritized_replay,
        "prioritized_replay_alpha": prioritized_replay_alpha,
        "prioritized_replay_beta0": prioritized_replay_beta0,
        "prioritized_replay_eps": prioritized_replay_eps,
        "max_grad_norm": max_grad_norm,
    }

def evaluate_model(model: DQNAgentTrainer, eval_env: TruckRoutingEnv, n_eval_episodes: int = 10) -> dict[str, Any]:
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

def objective(trial: optuna.trial.Trial, train_maps: list[Map], eval_maps: list[Map], n_timesteps: int = 10000, n_eval_episodes: int = 5) -> float:
    """
    Objective function for hyperparameter optimization.
    
    Args:
        trial: Optuna trial object
        train_maps: List of maps for training
        eval_maps: List of maps for evaluation
        n_timesteps: Number of timesteps for training
        n_eval_episodes: Number of episodes for evaluation
        
    Returns:
        float: Negative score (to minimize, since Optuna minimizes by default)
    """
    # Sample hyperparameters
    dqn_params = sample_dqn_params(trial)
    
    # Setup training and evaluation environments
    # Lấy ngẫu nhiên 2 maps để tăng tính tổng quát khi train
    train_maps_sample = np.random.choice(train_maps, size=min(2, len(train_maps)), replace=False)
    
    # training metrics cho mỗi map
    train_results = []
    
    # Train trên nhiều map để đảm bảo khả năng thích ứng
    for train_map in train_maps_sample:
        train_env = create_env(train_map)
        # Create model with sampled parameters
        log_dir = f"./rl_models_logs/optuna_trial_{trial.number}_map_{train_map.size}x{train_map.size}"
        model = DQNAgentTrainer(train_env, log_dir=log_dir)
        
        try:
            # Use create_model method to properly initialize the DQN agent
            model.create_model(
                learning_rate=dqn_params["learning_rate"],
                buffer_size=dqn_params["buffer_size"],
                batch_size=dqn_params["batch_size"],
                gamma=dqn_params["gamma"],
                tau=dqn_params["tau"],
                train_freq=dqn_params["train_freq"],
                gradient_steps=dqn_params["gradient_steps"],
                exploration_fraction=dqn_params["exploration_fraction"],
                exploration_initial_eps=dqn_params["exploration_initial_eps"],
                exploration_final_eps=dqn_params["exploration_final_eps"],
                learning_starts=dqn_params.get("learning_starts", 1000),
                policy_kwargs=dqn_params["policy_kwargs"],
                use_double_dqn=dqn_params["use_double_dqn"],
                use_dueling_network=dqn_params["use_dueling_network"],
                use_prioritized_replay=dqn_params["use_prioritized_replay"],
                prioritized_replay_alpha=dqn_params["prioritized_replay_alpha"],
                prioritized_replay_beta0=dqn_params["prioritized_replay_beta0"],
                max_grad_norm=dqn_params.get("max_grad_norm", 10),
                verbose=0
            )
            
            # Increase training timesteps to ensure model has enough time to learn
            effective_train_timesteps = max(n_timesteps, 20000)  # At least 20K timesteps
            
            # Train the agent
            model.train(total_timesteps=effective_train_timesteps)
            
            # Save model metrics for this map
            if hasattr(model, 'training_metrics') and model.training_metrics is not None:
                # Extract key metrics for early assessment
                metrics = model.training_metrics
                if 'episode_rewards' in metrics and len(metrics['episode_rewards']) > 0:
                    final_rewards = metrics['episode_rewards'][-10:]  # last 10 episodes
                    mean_final_reward = np.mean(final_rewards)
                    train_results.append({
                        'map_size': train_map.size,
                        'mean_final_reward': mean_final_reward,
                        'model': model  # Keep reference to continue evaluating
                    })
            
        except Exception as e:
            print(f"Error training on map size {train_map.size}: {e}")
            import traceback
            traceback.print_exc()
            # Give a very bad score for failures
            return float('-inf')
    
    # If no successful training, return a bad score
    if not train_results:
        return float('-inf')
    
    # Use multiple maps for evaluation to ensure generalization
    # Limit to max 5 maps for evaluation to save time
    eval_maps_to_use = np.random.choice(eval_maps, size=min(5, len(eval_maps)), replace=False)
    
    # Track comprehensive metrics across all evaluations
    all_rewards = []
    all_success_rates = []
    all_path_lengths = []
    all_timeouts = []
    all_stuck_rates = []  # Track rate of getting stuck
    
    # Evaluate each trained model on each evaluation map
    for train_result in train_results:
        model = train_result['model']
        
        for eval_map in eval_maps_to_use:
            eval_env = create_env(eval_map)
            model.env = eval_env  # Update the environment
            
            # Track metrics for this specific (model, map) combo
            map_rewards = []
            map_success = 0
            map_path_lengths = []
            map_timeouts = 0
            map_stuck_count = 0  # Count stuck episodes
            
            for _ in range(n_eval_episodes):
                observation, _ = eval_env.reset()
                done = truncated = False
                episode_reward = 0
                path_length = 0
                episode_actions = []  # Track actions to detect repetition
                position_counts = {}  # Track position counts within episode
                
                # Main episode loop
                while not (done or truncated):
                    action, _ = model.model.predict(observation, deterministic=True)
                    episode_actions.append(action)
                    
                    # Track current position for stuck detection
                    pos_key = str(tuple(observation['agent_pos']))
                    position_counts[pos_key] = position_counts.get(pos_key, 0) + 1
                    
                    # Take action
                    observation, reward, done, truncated, info = eval_env.step(action)
                    episode_reward += reward
                    path_length += 1
                    
                    # Check termination conditions
                    if done:
                        # Success if reached goal
                        if info.get("termination_reason") == "den_dich":
                            map_success += 1
                            map_path_lengths.append(path_length)
                        # Stuck detection via termination reason
                        elif info.get("termination_reason") in ["lap_qua_nhieu", "khong_tien_trien"]:
                            map_stuck_count += 1
                    elif truncated:
                        map_timeouts += 1
                
                # Additional stuck detection via action repetition patterns
                if len(episode_actions) > 20:  # Only check if episode was long enough
                    # Check for excessive position revisits
                    max_position_visits = max(position_counts.values()) if position_counts else 0
                    if max_position_visits > 10:  # Threshold for being considered stuck
                        map_stuck_count += 1
                
                map_rewards.append(episode_reward)
            
            # Calculate metrics for this map
            all_rewards.extend(map_rewards)
            map_success_rate = map_success / n_eval_episodes
            all_success_rates.append(map_success_rate)
            
            if map_path_lengths:
                all_path_lengths.extend(map_path_lengths)
            
            map_timeout_rate = map_timeouts / n_eval_episodes
            all_timeouts.append(map_timeout_rate)
            
            # Calculate stuck rate for this map
            map_stuck_rate = map_stuck_count / n_eval_episodes
            all_stuck_rates.append(map_stuck_rate)
    
    # Calculate overall metrics
    mean_reward = np.mean(all_rewards) if all_rewards else -1000.0
    mean_success_rate = np.mean(all_success_rates)
    mean_timeout_rate = np.mean(all_timeouts)
    mean_stuck_rate = np.mean(all_stuck_rates)  # Average stuck rate
    mean_path_length = np.mean(all_path_lengths) if all_path_lengths else float('inf')
    
    # Get average optimal path length estimate
    avg_map_size = np.mean([m.size for m in eval_maps_to_use])
    optimal_path_estimate = avg_map_size * 1.5  # Better estimate (less conservative)
    
    # Path efficiency (only for successful paths)
    path_efficiency = optimal_path_estimate / mean_path_length if mean_path_length > 0 and mean_path_length < float('inf') else 0.0
    
    # Anti-stuck score - rewards agents that don't get stuck
    anti_stuck_score = 1.0 - mean_stuck_rate  # Higher is better
    
    # IMPROVED SCORING SYSTEM: Better handling of success vs. no success cases
    if mean_success_rate > 0:
        # When we have successful episodes, prioritize based on:
        # 1. Success rate (most important)
        # 2. Path efficiency (reward shorter paths)
        # 3. Anti-stuck capability (avoid repeating positions)
        # 4. Reward (as a secondary metric)
        score = (
            mean_success_rate * 500 +         # Success is most important (50-500 points)
            path_efficiency * 100 +           # Efficient paths (0-100 points)
            anti_stuck_score * 100 +          # Not getting stuck (0-100 points)
            (mean_reward / 100) * 50          # Raw reward normalized (0-50 points)
        )
    else:
        # When no successes, prioritize models that make progress toward goals:
        # 1. Anti-stuck behavior (most important when no success)
        # 2. Normalized reward (should be higher for models closer to goals)
        # 3. Small bonus for low timeout rate (completing episodes)
        normalized_reward = (mean_reward + 150) / 300  # Convert from typically [-150, +150] to [0, 1]
        normalized_reward = max(0, min(1, normalized_reward))  # Clamp to [0, 1]
        
        score = (
            anti_stuck_score * 30 +           # Not getting stuck (0-30 points)
            normalized_reward * 15 +          # Reward normalized (0-15 points)
            (1.0 - mean_timeout_rate) * 5     # Not timing out (0-5 points)
        )
        
        # Ensure any success is better than no success
        # The max score with no success should be < min score with any success
        score = min(score, 40)  # Cap at 40, less than the min ~50 for even 0.1 success rate
    
    # Report metrics for pruning and visualization
    trial.set_user_attr("mean_reward", mean_reward)
    trial.set_user_attr("mean_success_rate", mean_success_rate)
    trial.set_user_attr("mean_path_length", float(mean_path_length) if mean_path_length < float('inf') else -1)
    trial.set_user_attr("mean_timeout_rate", mean_timeout_rate)
    trial.set_user_attr("mean_stuck_rate", mean_stuck_rate)
    trial.set_user_attr("anti_stuck_score", anti_stuck_score)
    trial.set_user_attr("score", score)
    
    print(f"Trial {trial.number}: Success={mean_success_rate:.2f}, Reward={mean_reward:.2f}, " +
          f"Anti-stuck={anti_stuck_score:.2f}, Score={score:.2f}")
        
    # Optuna minimizes, so return negative of the score
    return -score

def optimize_hyperparameters(train_maps_dir: str, eval_maps_dir: str, n_trials: int = 50, n_timesteps: int = 25000, 
                             n_eval_episodes: int = 5, n_jobs: int = 1, study_name: str = "dqn_optimization") -> dict[str, Any]:
    """
    Run hyperparameter optimization using Optuna.
    
    Args:
        train_maps_dir: Directory containing training maps
        eval_maps_dir: Directory containing evaluation maps
        n_trials: Number of optimization trials
        n_timesteps: Number of timesteps for training
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
    
    # Đảm bảo có đủ maps để train và evaluate
    if len(train_maps) < 2:
        print("Warning: Less than 2 training maps found. Creating additional maps...")
        while len(train_maps) < 2:
            size = np.random.randint(8, 13)  # Random sizes between 8x8 and 12x12
            map_obj = Map.generate_random(size=size, num_tolls=2, num_gas=3, num_obstacles=10)
            map_path = os.path.join(train_maps_dir, f"gen_train_map_{len(train_maps)}.json")
            map_obj.save(map_path)
            train_maps.append(map_obj)
    
    if len(eval_maps) < 3:
        print("Warning: Less than 3 evaluation maps found. Creating additional maps...")
        while len(eval_maps) < 3:
            size = np.random.randint(8, 13)  # Random sizes between 8x8 and 12x12
            map_obj = Map.generate_random(size=size, num_tolls=2, num_gas=3, num_obstacles=10)
            map_path = os.path.join(eval_maps_dir, f"gen_eval_map_{len(eval_maps)}.json")
            map_obj.save(map_path)
            eval_maps.append(map_obj)
    
    # Create output directory with more detailed timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = f"./hyperparameter_tuning_results/{timestamp}"
    os.makedirs(results_dir, exist_ok=True)
    
    # Save information about the maps being used
    maps_info = {
        "train_maps": [{"filename": os.path.basename(m.filename) if hasattr(m, 'filename') else f"map_{i}", 
                        "size": m.size} for i, m in enumerate(train_maps)],
        "eval_maps": [{"filename": os.path.basename(m.filename) if hasattr(m, 'filename') else f"map_{i}", 
                       "size": m.size} for i, m in enumerate(eval_maps)]
    }
    
    with open(os.path.join(results_dir, "maps_info.json"), "w") as f:
        json.dump(maps_info, f, indent=2)
    
    # Create Optuna study with enhanced configuration
    sampler = TPESampler(n_startup_trials=10, seed=42)  # Increase startup trials and set seed
    
    # Improved pruner configuration for better efficiency
    pruner = MedianPruner(
        n_startup_trials=10,  # More trials before pruning
        n_warmup_steps=5000,   # More steps before pruning
        interval_steps=2000,   # Check more frequently
        n_min_trials=3         # Need at least 3 trials to compare
    )
    
    study = optuna.create_study(
        study_name=study_name,
        direction="maximize",
        sampler=sampler,
        pruner=pruner,
        storage=f"sqlite:///{results_dir}/optuna.db",  # Save database for later inspection
        load_if_exists=False
    )
    
    # Create pruning callback for earlier termination of bad trials
    pruning_callback = optuna.integration.TFPruningCallback(
        study=study,
        interval_steps=1000
    )
    
    # Save study configuration
    study_config = {
        "n_trials": n_trials,
        "n_timesteps": n_timesteps,
        "n_eval_episodes": n_eval_episodes,
        "n_jobs": n_jobs,
        "study_name": study_name,
        "timestamp": timestamp
    }
    
    with open(os.path.join(results_dir, "study_config.json"), "w") as f:
        json.dump(study_config, f, indent=2)
    
    try:
        # Add progress output for long optimization runs
        print(f"Starting hyperparameter optimization with {n_trials} trials...")
        print(f"Training on {len(train_maps)} maps, evaluating on {len(eval_maps)} maps")
        print(f"Each trial uses {n_timesteps} timesteps and evaluates on {n_eval_episodes} episodes per map")
        
        # Start timer to track optimization time
        start_time = time.time()
        
        study.optimize(
            lambda trial: objective(
                trial, train_maps, eval_maps, n_timesteps, n_eval_episodes
            ),
            n_trials=n_trials,
            n_jobs=n_jobs,
            show_progress_bar=True,
            callbacks=[pruning_callback]
        )
        
        # Calculate optimization time
        optimization_time = time.time() - start_time
        hours, remainder = divmod(optimization_time, 3600)
        minutes, seconds = divmod(remainder, 60)
        time_str = f"{int(hours)}h {int(minutes)}m {int(seconds)}s"
        
    except KeyboardInterrupt:
        print("\nOptimization stopped early by user.")
        optimization_time = time.time() - start_time
        hours, remainder = divmod(optimization_time, 3600)
        minutes, seconds = divmod(remainder, 60)
        time_str = f"{int(hours)}h {int(minutes)}m {int(seconds)}s"
    
    # Get all trials data for analysis
    trials_data = []
    for trial in study.trials:
        if trial.state == optuna.trial.TrialState.COMPLETE:
            trial_data = {
                "number": trial.number,
                "value": trial.value,
                "params": trial.params
            }
            # Add user attributes if available
            for key in ["mean_reward", "mean_success_rate", "mean_path_length", 
                       "mean_timeout_rate", "mean_stuck_rate", "anti_stuck_score"]:
                if key in trial.user_attrs:
                    trial_data[key] = trial.user_attrs[key]
            trials_data.append(trial_data)
    
    # Sort trials by performance
    sorted_trials = sorted(trials_data, key=lambda x: x["value"] if x["value"] is not None else float('-inf'), reverse=True)
    
    # Save all trials data
    with open(os.path.join(results_dir, "all_trials.json"), "w") as f:
        json.dump(sorted_trials, f, indent=2)
    
    # Save top 5 trials for comparison
    top_trials = sorted_trials[:5] if len(sorted_trials) >= 5 else sorted_trials
    with open(os.path.join(results_dir, "top_trials.json"), "w") as f:
        json.dump(top_trials, f, indent=2)
    
    # Save comprehensive study results
    study_results = {
        "best_params": study.best_params,
        "best_value": study.best_value,
        "best_trial": study.best_trial.number,
        "n_trials_completed": len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]),
        "n_trials_pruned": len([t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]),
        "total_trials": len(study.trials),
        "optimization_time": time_str,
        "datetime": timestamp
    }
    
    # Add best trial metrics
    for key in ["mean_reward", "mean_success_rate", "mean_path_length", 
               "mean_timeout_rate", "mean_stuck_rate", "anti_stuck_score"]:
        if key in study.best_trial.user_attrs:
            study_results[key] = study.best_trial.user_attrs[key]
    
    with open(os.path.join(results_dir, "study_results.json"), "w") as f:
        json.dump(study_results, f, indent=2)
    
    # Get the best hyperparameters and create a formatted version for display/save
    best_dqn_params = sample_dqn_params(study.best_trial)
    best_params_path = os.path.join(results_dir, "best_params.json")
    
    with open(best_params_path, "w") as f:
        json.dump(best_dqn_params, f, indent=2)
    
    # Print optimization summary 
    print("\n" + "="*60)
    print(f"Hyperparameter optimization completed in {time_str}")
    print(f"Total trials: {len(study.trials)}")
    print(f"Completed trials: {len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])}")
    print(f"Pruned trials: {len([t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED])}")
    print(f"Best trial: #{study.best_trial.number}")
    print(f"Best combined metric value: {study.best_value:.4f}")
    
    # Print best metrics
    if "mean_success_rate" in study.best_trial.user_attrs:
        print(f"Success rate: {study.best_trial.user_attrs['mean_success_rate']:.2f}")
    if "mean_reward" in study.best_trial.user_attrs:
        print(f"Mean reward: {study.best_trial.user_attrs['mean_reward']:.2f}")
    if "anti_stuck_score" in study.best_trial.user_attrs:
        print(f"Anti-stuck score: {study.best_trial.user_attrs['anti_stuck_score']:.2f}")
    
    print("\nBest parameters:")
    for param, value in best_dqn_params.items():
        # Format for better readability
        if isinstance(value, float):
            print(f"  {param}: {value:.6f}")
        else:
            print(f"  {param}: {value}")
    
    print(f"\nResults saved to: {results_dir}")
    print("="*60)
    
    return best_dqn_params

def train_agent_with_best_params(best_params: dict[str, Any], train_map: Map, n_timesteps: int = 100000, save_path: str | None = None, 
                             enable_callback_saving: bool = True, verbose: int = 1) -> DQNAgentTrainer:
    """
    Train an agent with the best parameters found.
    
    Args:
        best_params: Dictionary of best hyperparameters
        train_map: Map to train on
        n_timesteps: Number of timesteps for training (increased default)
        save_path: Path to save the trained model
        enable_callback_saving: Whether to save best models during training
        verbose: Verbosity level
        
    Returns:
        DQNAgentTrainer: Trained agent
    """
    # Create environment
    env = create_env(train_map)
    
    # Create log directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = f"./rl_models_logs/best_agent_{timestamp}"
    os.makedirs(log_dir, exist_ok=True)
    
    # Save hyperparameters being used
    params_copy = best_params.copy()
    # Convert any numpy values to Python native types for JSON serialization
    for k, v in params_copy.items():
        if hasattr(v, 'tolist'):
            params_copy[k] = v.tolist()
    
    with open(os.path.join(log_dir, "training_params.json"), "w") as f:
        json.dump(params_copy, f, indent=2)
    
    # Save map information
    map_info = {
        "size": train_map.size,
        "filename": train_map.filename if hasattr(train_map, 'filename') else "unknown",
        "n_obstacles": sum(1 for y in range(train_map.size) for x in range(train_map.size) 
                         if train_map.get_cell_type((x, y)) == CellType.OBSTACLE),
        "n_gas_stations": sum(1 for y in range(train_map.size) for x in range(train_map.size) 
                           if train_map.get_cell_type((x, y)) == CellType.GAS),
        "n_toll_stations": sum(1 for y in range(train_map.size) for x in range(train_map.size) 
                            if train_map.get_cell_type((x, y)) == CellType.TOLL),
        "start_pos": list(train_map.start_pos),
        "end_pos": list(train_map.end_pos)
    }
    
    with open(os.path.join(log_dir, "map_info.json"), "w") as f:
        json.dump(map_info, f, indent=2)
    
    # Create agent
    agent = DQNAgentTrainer(env, log_dir=log_dir)
    
    # Extract key parameters, using default values if not provided
    # This handles cases where the best params might be missing some keys
    policy_kwargs = best_params.get("policy_kwargs", {"net_arch": [256, 256]})
    use_double_dqn = best_params.get("use_double_dqn", True)  # Default to True
    use_dueling_network = best_params.get("use_dueling_network", False)
    use_prioritized_replay = best_params.get("use_prioritized_replay", False)
    
    # Set best parameters using create_model method
    agent.create_model(
        learning_rate=best_params.get("learning_rate", 0.0001),
        buffer_size=best_params.get("buffer_size", 100000),
        batch_size=best_params.get("batch_size", 128),
        gamma=best_params.get("gamma", 0.99),
        tau=best_params.get("tau", 0.005),
        train_freq=best_params.get("train_freq", 4),
        gradient_steps=best_params.get("gradient_steps", 1),
        learning_starts=best_params.get("learning_starts", 5000),
        exploration_fraction=best_params.get("exploration_fraction", 0.2),
        exploration_initial_eps=best_params.get("exploration_initial_eps", 1.0),
        exploration_final_eps=best_params.get("exploration_final_eps", 0.05),
        max_grad_norm=best_params.get("max_grad_norm", 10),
        policy_kwargs=policy_kwargs,
        use_double_dqn=use_double_dqn,
        use_dueling_network=use_dueling_network,
        use_prioritized_replay=use_prioritized_replay,
        prioritized_replay_alpha=best_params.get("prioritized_replay_alpha", 0.6),
        prioritized_replay_beta0=best_params.get("prioritized_replay_beta0", 0.4),
        verbose=verbose
    )
    
    print(f"\nStarting training with best parameters for {n_timesteps} timesteps...\n")
    
    # Train the agent (with periodic checkpoint saves if enabled)
    if enable_callback_saving:
        # Add checkpoints for better tracking
        checkpoint_dir = os.path.join(log_dir, "checkpoints")
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        # Define a callback that saves models at multiple points
        class CheckpointCallback(BaseCallback):
            def __init__(self, save_freq=50000, save_path=checkpoint_dir, verbose=1):
                super(CheckpointCallback, self).__init__(verbose)
                self.save_freq = save_freq
                self.save_path = save_path
                self.best_mean_reward = -np.inf
                
            def _on_step(self):
                if self.n_calls % self.save_freq == 0:
                    checkpoint_path = os.path.join(self.save_path, f"model_step_{self.n_calls}")
                    self.model.save(checkpoint_path)
                    if self.verbose > 0:
                        print(f"Saved checkpoint at {self.n_calls} steps to {checkpoint_path}")
                    
                # Also save if there's a significant reward improvement
                if len(self.model.ep_info_buffer) > 0:
                    mean_reward = np.mean([ep_info['r'] for ep_info in self.model.ep_info_buffer])
                    if mean_reward > self.best_mean_reward * 1.2:  # 20% improvement
                        self.best_mean_reward = mean_reward
                        best_model_path = os.path.join(self.save_path, f"best_reward_model_{self.n_calls}")
                        self.model.save(best_model_path)
                        if self.verbose > 0:
                            print(f"New best reward {mean_reward:.2f} at {self.n_calls} steps, saved to {best_model_path}")
                
                return True
        
        # Create and use checkpoint callback
        checkpoint_callback = CheckpointCallback(save_freq=max(10000, n_timesteps // 10))
        agent.train(total_timesteps=n_timesteps, callback=checkpoint_callback)
    else:
        # Standard training without checkpoints
        agent.train(total_timesteps=n_timesteps)
    
    # Save the final trained model
    if save_path:
        agent.save_model(save_path)
        print(f"Final model saved to {save_path}")
    else:
        # If no specific save path provided, still save to the log directory
        final_save_path = os.path.join(log_dir, "final_model")
        agent.save_model(final_save_path)
        print(f"Final model saved to {final_save_path}")
    
    # Comprehensive evaluation on the training map
    print("\nPerforming final evaluation...")
    evaluation_results = agent.evaluate(n_eval_episodes=20)  # More episodes for better statistics
    
    # Save evaluation results
    with open(os.path.join(log_dir, "evaluation_results.json"), "w") as f:
        # Filter out non-serializable items
        serializable_results = {k: v for k, v in evaluation_results.items() 
                              if isinstance(v, (int, float, bool, str, list, dict))}
        json.dump(serializable_results, f, indent=2)
    
    # Print summary of evaluation
    print("\nEvaluation results:")
    print(f"Success rate: {evaluation_results.get('success_rate', 'N/A'):.2f}")
    print(f"Average reward: {evaluation_results.get('avg_reward', 'N/A'):.2f}")
    print(f"Average episode length: {evaluation_results.get('avg_episode_length', 'N/A'):.2f}")
    
    if 'termination_reasons' in evaluation_results:
        print("\nTermination reasons:")
        for reason, count in evaluation_results['termination_reasons'].items():
            print(f"  {reason}: {count}")
    
    print(f"\nTraining logs and results saved to: {log_dir}")
    
    return agent

# Example usage
if __name__ == "__main__":
    # Define directories
    train_maps_dir = "./maps/train"
    eval_maps_dir = "./maps/eval"
    
    # Make sure directories exist
    os.makedirs(train_maps_dir, exist_ok=True)
    os.makedirs(eval_maps_dir, exist_ok=True)
    
    # If no maps exist, create some test maps
    if len(os.listdir(train_maps_dir)) == 0:
        print("No training maps found. Creating some...")
        for i in range(5):
            map_obj = Map.generate_random(size=10, num_tolls=2, num_gas=3, num_obstacles=10)
            map_obj.save(os.path.join(train_maps_dir, f"train_map_{i}.json"))
    
    if len(os.listdir(eval_maps_dir)) == 0:
        print("No evaluation maps found. Creating some...")
        for i in range(3):
            map_obj = Map.generate_random(size=10, num_tolls=2, num_gas=3, num_obstacles=10)
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