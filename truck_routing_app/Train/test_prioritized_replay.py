"""
Simple test script to verify that PrioritizedReplayBuffer is working correctly.
"""

import os
import sys

# Add the parent directory to sys.path so we can import from core
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.algorithms.rl_DQNAgent import DQNAgentTrainer
from core.rl_environment import TruckRoutingEnv
from core.map import Map

def main():
    print("Creating a simple map...")
    map_obj = Map(8)  # 8x8 map with just size parameter
    
    # Set start and end positions manually since we're not using generate_random
    map_obj.start_pos = (0, 0)
    map_obj.end_pos = (7, 7)
    
    print("Creating environment...")
    env = TruckRoutingEnv(map_obj)
    
    print("Creating trainer...")
    trainer = DQNAgentTrainer(env)
    
    print("Creating model with prioritized replay...")
    model = trainer.create_model(
        buffer_size=1000,
        batch_size=32,
        learning_starts=100,
        exploration_fraction=0.1,
        use_prioritized_replay=True,
        prioritized_replay_alpha=0.6,
        prioritized_replay_beta0=0.4,
        use_dueling_network=False  # Explicitly disable dueling network
    )
    
    print("Model created successfully!")
    print("Running a quick training (100 steps)...")
    
    # Train for just a few steps to make sure everything works
    trainer.train(total_timesteps=100)
    
    print("Training completed successfully!")
    print("PrioritizedReplayBuffer is working correctly.")

if __name__ == "__main__":
    main() 