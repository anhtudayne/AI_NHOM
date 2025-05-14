"""
Custom implementation of PrioritizedReplayBuffer for Truck Routing environment.
This provides a simplified version that can be used when the sb3_contrib implementation
is not available.
"""

import numpy as np
import torch as th
from stable_baselines3.common.buffers import ReplayBuffer, DictReplayBuffer
from typing import Dict, List, Any, Union, Optional, Tuple
from gymnasium import spaces

def create_prioritized_buffer(
    buffer_size: int,
    observation_space: spaces.Space,
    action_space: spaces.Space,
    device: Union[th.device, str] = "auto",
    n_envs: int = 1,
    optimize_memory_usage: bool = False,
    alpha: float = 0.6,
    beta: float = 0.4,
    eps: float = 1e-6,
):
    """
    Factory function to create the appropriate type of prioritized buffer
    based on the observation space.
    """
    # For Dict observation spaces, use PrioritizedDictReplayBuffer
    if isinstance(observation_space, spaces.Dict):
        print("Using DictReplayBuffer for dictionary observation space")
        return PrioritizedDictReplayBuffer(
            buffer_size=buffer_size,
            observation_space=observation_space,
            action_space=action_space,
            device=device,
            n_envs=n_envs,
            optimize_memory_usage=optimize_memory_usage,
            alpha=alpha,
            beta=beta,
            eps=eps
        )
    
    # For regular observation spaces, use PrioritizedReplayBuffer
    return PrioritizedReplayBuffer(
        buffer_size=buffer_size,
        observation_space=observation_space,
        action_space=action_space,
        device=device,
        n_envs=n_envs,
        optimize_memory_usage=optimize_memory_usage,
        alpha=alpha,
        beta=beta,
        eps=eps
    )

class PrioritizedDictReplayBuffer(DictReplayBuffer):
    """Prioritized version of the DictReplayBuffer."""
    
    def __init__(
        self,
        buffer_size: int,
        observation_space: spaces.Dict,
        action_space: spaces.Space,
        device: Union[th.device, str] = "auto",
        n_envs: int = 1,
        optimize_memory_usage: bool = False,
        alpha: float = 0.6,
        beta: float = 0.4,
        eps: float = 1e-6,
    ):
        super().__init__(
            buffer_size=buffer_size,
            observation_space=observation_space,
            action_space=action_space,
            device=device,
            n_envs=n_envs,
            optimize_memory_usage=optimize_memory_usage
        )
        self.alpha = alpha
        self.beta = beta 
        self.eps = eps
        self.priorities = np.ones(buffer_size, dtype=np.float32)
    
    def add(self, *args, **kwargs):
        idx = self.pos
        # Check if size is a method or property
        if callable(self.size):
            current_size = self.size()
        else:
            current_size = self.size
        max_priority = self.priorities.max() if current_size > 0 else 1.0
        super().add(*args, **kwargs)
        self.priorities[idx] = max_priority
    
    def sample(self, batch_size, env=None):
        if callable(self.size):
            current_size = self.size()
        else:
            current_size = self.size
        
        if current_size == 0:
            raise RuntimeError("Cannot sample from an empty buffer")
        
        if current_size < self.buffer_size:
            priorities = self.priorities[:current_size]
        else:
            priorities = self.priorities
        
        probabilities = priorities ** self.alpha
        probabilities /= probabilities.sum()
        
        indices = np.random.choice(
            len(probabilities), size=batch_size, replace=True, p=probabilities
        )
        
        weights = (current_size * probabilities[indices]) ** (-self.beta)
        weights /= weights.max()
        weights_tensor = th.FloatTensor(weights).to(self.device)
        
        # Get the samples from the replay buffer
        data = self._get_samples(indices, env)
        
        # Store the indices and weights for later priority updates
        # Use a separate attribute to store data needed for priority updates
        self._last_sampled_indices = indices
        
        # Return samples in the format expected by the DQN algorithm
        # We don't modify data directly since it's a namedtuple
        # Instead, we store weights separately and will retrieve during training
        self._last_weights = weights_tensor
        
        # Return data as is, without trying to add weights field
        return data
    
    def update_priorities(self, indices, priorities):
        priorities = np.abs(priorities) + self.eps
        for idx, priority in zip(indices, priorities):
            self.priorities[idx] = priority
            
    # Add a method to get the last weights
    def get_last_weights(self):
        if hasattr(self, '_last_weights'):
            return self._last_weights
        return None
        
    # Add a method to get the last sampled indices
    def get_last_indices(self):
        if hasattr(self, '_last_sampled_indices'):
            return self._last_sampled_indices
        return None

class PrioritizedReplayBuffer(ReplayBuffer):
    """
    Prioritized Replay Buffer implementation for Deep Q-Network (DQN).
    
    This implementation provides a simplified prioritized replay mechanism
    that gives higher probability to samples with higher TD error.
    
    Attributes:
        alpha (float): How much prioritization is used (0: no prioritization, 1: full prioritization)
        beta (float): To what degree to use importance weights (0: no correction, 1: full correction)
        eps (float): Small positive constant to ensure all priorities are non-zero
    """
    
    def __init__(
        self,
        buffer_size: int,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        device: Union[th.device, str] = "auto",
        n_envs: int = 1,
        optimize_memory_usage: bool = False,
        alpha: float = 0.6,
        beta: float = 0.4,
        eps: float = 1e-6,
    ):
        """
        Initialize PrioritizedReplayBuffer.
        
        Args:
            buffer_size: Max number of elements in the buffer
            observation_space: Observation space
            action_space: Action space
            device: PyTorch device
            n_envs: Number of parallel environments
            optimize_memory_usage: Enable memory optimization
            alpha: Prioritization exponent (higher = more prioritization)
            beta: Importance sampling correction exponent (higher = more correction)
            eps: Small positive constant to ensure non-zero priorities
        """
        super().__init__(buffer_size, observation_space, action_space, device, n_envs, optimize_memory_usage)
        
        # Prioritization parameters
        self.alpha = alpha
        self.beta = beta
        self.eps = eps
        
        # Initialize priorities with ones (will be updated during training)
        self.priorities = np.ones(buffer_size, dtype=np.float32)
    
    def add(
        self,
        obs: np.ndarray,
        next_obs: np.ndarray,
        action: np.ndarray,
        reward: np.ndarray,
        done: np.ndarray,
        infos: List[Dict[str, Any]],
    ) -> None:
        """
        Add a new experience to the buffer with maximum priority.
        
        Args:
            obs: Observation
            next_obs: Next observation
            action: Action
            reward: Reward
            done: Done flag
            infos: Additional information
        """
        # Add with maximum priority on new experiences
        # Check if size is a method or property
        if callable(self.size):
            current_size = self.size()
        else:
            current_size = self.size
        max_priority = self.priorities.max() if current_size > 0 else 1.0
        
        # Call parent class add method
        idx = self.pos
        super().add(obs, next_obs, action, reward, done, infos)
        
        # Set priority for new experience
        self.priorities[idx] = max_priority
    
    def sample(self, batch_size: int, env: Optional[Any] = None) -> Dict[str, Any]:
        """
        Sample a batch of experiences with prioritization.
        
        Args:
            batch_size: Number of samples to draw
            env: Can be ignored for this implementation
            
        Returns:
            Dictionary with sampled observations, actions, rewards, etc.
        """
        # Check if size is a method or property
        if callable(self.size):
            current_size = self.size()
        else:
            current_size = self.size
            
        if current_size == 0:
            raise RuntimeError("Cannot sample from an empty buffer")
        
        # Calculate sampling probabilities based on priorities
        if current_size < self.buffer_size:
            priorities = self.priorities[:current_size]
        else:
            priorities = self.priorities
        
        # Apply prioritization (alpha parameter)
        probabilities = priorities ** self.alpha
        probabilities /= probabilities.sum()
        
        # Sample indices based on priorities
        indices = np.random.choice(
            len(probabilities), size=batch_size, replace=True, p=probabilities
        )
        
        # Calculate importance sampling weights
        weights = (current_size * probabilities[indices]) ** (-self.beta)
        weights /= weights.max()
        weights_tensor = th.FloatTensor(weights).to(self.device)
        
        # Get the samples from parent class method
        data = self._get_samples(indices, env)
        
        # Store indices and weights for later use
        self._last_sampled_indices = indices
        self._last_weights = weights_tensor
        
        # Return data without modifying it
        return data
    
    def update_priorities(self, indices: np.ndarray, priorities: np.ndarray) -> None:
        """
        Update priorities for sampled transitions.
        
        Args:
            indices: Indices of the sampled transitions
            priorities: TD errors (or other priority measure) for each transition
        """
        # Ensure positive priorities
        priorities = np.abs(priorities) + self.eps
        
        # Update priorities
        for idx, priority in zip(indices, priorities):
            self.priorities[idx] = priority
            
    # Add a method to get the last weights
    def get_last_weights(self):
        if hasattr(self, '_last_weights'):
            return self._last_weights
        return None
        
    # Add a method to get the last sampled indices
    def get_last_indices(self):
        if hasattr(self, '_last_sampled_indices'):
            return self._last_sampled_indices
        return None 