"""
Script để huấn luyện agent DQN cho bài toán định tuyến xe tải.
"""

import os
import time
import argparse
import numpy as np
from datetime import datetime

from core.map import Map
from core.rl_environment import TruckRoutingEnv
from core.algorithms.rl_DQNAgent import DQNAgentTrainer

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Train a DQN agent for truck routing')
    
    parser.add_argument('--map_size', type=int, default=8,
                        help='Size of the map for training (default: 8)')
    parser.add_argument('--toll_ratio', type=float, default=0.05,
                        help='Ratio of toll stations in random maps (default: 0.05)')
    parser.add_argument('--gas_ratio', type=float, default=0.05,
                        help='Ratio of gas stations in random maps (default: 0.05)')
    parser.add_argument('--brick_ratio', type=float, default=0.1,
                        help='Ratio of obstacles in random maps (default: 0.1)')
    
    parser.add_argument('--initial_fuel', type=float, default=70.0,
                        help='Initial fuel for the truck (default: 70.0)')
    parser.add_argument('--initial_money', type=float, default=1500.0,
                        help='Initial money for the truck (default: 1500.0)')
    parser.add_argument('--fuel_per_move', type=float, default=0.3,
                        help='Fuel consumption per move (default: 0.3)')
    
    parser.add_argument('--total_timesteps', type=int, default=100000,
                        help='Total timesteps for training (default: 100000)')
    parser.add_argument('--eval_freq', type=int, default=10000,
                        help='Frequency of evaluation during training (default: 10000)')
    
    parser.add_argument('--log_dir', type=str, default='./rl_models_logs',
                        help='Directory for TensorBoard logs (default: ./rl_models_logs)')
    parser.add_argument('--model_dir', type=str, default='./saved_models',
                        help='Directory for saving models (default: ./saved_models)')
    parser.add_argument('--map_dir', type=str, default='./maps',
                        help='Directory for saving/loading maps (default: ./maps)')
    
    parser.add_argument('--save_freq', type=int, default=25000,
                        help='Frequency of saving models during training (default: 25000)')
    
    parser.add_argument('--use_demo_map', action='store_true',
                        help='Use a demo map instead of generating random maps')
    parser.add_argument('--load_map_file', type=str, default=None,
                        help='Load a specific map file (if not specified, will generate or use demo)')
    
    return parser.parse_args()

def setup_dirs(args):
    """Create necessary directories for logs, models, and maps."""
    os.makedirs(args.log_dir, exist_ok=True)
    os.makedirs(args.model_dir, exist_ok=True)
    os.makedirs(args.map_dir, exist_ok=True)

def create_or_load_map(args):
    """Create a new random map or load an existing one."""
    if args.load_map_file:
        # Load specific map file
        map_path = os.path.join(args.map_dir, args.load_map_file)
        if os.path.exists(map_path):
            return Map.load(map_path)
        else:
            print(f"Warning: Map file {map_path} not found, generating random map instead.")
    
    if args.use_demo_map:
        # Use demo map
        return Map.create_demo_map(args.map_size)
    else:
        # Generate random map
        return Map.generate_random(args.map_size, args.toll_ratio, args.gas_ratio, args.brick_ratio)

def train(args):
    """Train the DQN agent."""
    # Create or load map
    map_object = create_or_load_map(args)
    
    # Save map for later testing
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    map_filename = f"training_map_{timestamp}.json"
    map_path = os.path.join(args.map_dir, map_filename)
    map_object.save(map_path)
    print(f"Training map saved to {map_path}")
    
    # Initialize environment
    env = TruckRoutingEnv(
        map_object=map_object,
        initial_fuel_config=args.initial_fuel,
        initial_money_config=args.initial_money,
        fuel_per_move_config=args.fuel_per_move,
        max_steps_per_episode=2 * map_object.size * map_object.size
    )
    
    # Create unique run name for logging
    run_name = f"DQN_truck_routing_{timestamp}"
    log_dir = os.path.join(args.log_dir, run_name)
    
    # Initialize trainer
    trainer = DQNAgentTrainer(env, log_dir=log_dir)
    
    # Training loop with periodic evaluation and model saving
    total_timesteps = args.total_timesteps
    eval_freq = args.eval_freq
    save_freq = args.save_freq
    
    print(f"Starting training for {total_timesteps} timesteps...")
    start_time = time.time()
    
    # Train in increments of eval_freq timesteps
    for i in range(0, total_timesteps, eval_freq):
        current_timesteps = min(eval_freq, total_timesteps - i)
        if current_timesteps <= 0:
            break
            
        print(f"\nTraining for {current_timesteps} timesteps ({i}/{total_timesteps})...")
        trainer.train(total_timesteps=current_timesteps)
        
        # Evaluate the agent periodically
        print("\nEvaluating current model...")
        metrics = trainer.evaluate(n_episodes=10)
        print(f"Evaluation metrics:")
        print(f"  Success Rate: {metrics['success_rate']:.2f}")
        print(f"  Average Reward: {metrics['avg_reward']:.2f}")
        print(f"  Average Path Length: {metrics['avg_path_length']:.2f}")
        print(f"  Average Remaining Fuel: {metrics['avg_remaining_fuel']:.2f}")
        print(f"  Average Remaining Money: {metrics['avg_remaining_money']:.2f}")
        
        # Save the model periodically
        if (i + current_timesteps) % save_freq == 0 or (i + current_timesteps) >= total_timesteps:
            checkpoint_filename = f"{run_name}_steps_{i + current_timesteps}"
            checkpoint_path = os.path.join(args.model_dir, checkpoint_filename)
            trainer.save_model(checkpoint_path)
            print(f"Model checkpoint saved to {checkpoint_path}.zip")
    
    # Save final model
    final_model_path = os.path.join(args.model_dir, f"{run_name}_final")
    trainer.save_model(final_model_path)
    print(f"Final model saved to {final_model_path}.zip")
    
    # Training summary
    total_time = time.time() - start_time
    print(f"\nTraining completed in {total_time:.2f} seconds.")
    
    # Final evaluation
    print("\nFinal evaluation:")
    metrics = trainer.evaluate(n_episodes=20)
    print(f"Final evaluation metrics:")
    print(f"  Success Rate: {metrics['success_rate']:.2f}")
    print(f"  Average Reward: {metrics['avg_reward']:.2f}")
    print(f"  Average Path Length: {metrics['avg_path_length']:.2f}")
    print(f"  Average Remaining Fuel: {metrics['avg_remaining_fuel']:.2f}")
    print(f"  Average Remaining Money: {metrics['avg_remaining_money']:.2f}")
    
    # Close environment
    env.close()
    return final_model_path, map_path

def main():
    """Main function."""
    args = parse_args()
    setup_dirs(args)
    model_path, map_path = train(args)
    print(f"Training completed. Final model: {model_path}.zip, Map: {map_path}")

if __name__ == "__main__":
    main() 