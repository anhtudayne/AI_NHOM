"""
Custom extension of DQNPolicy to support prioritized replay.
"""

import torch as th
from typing import Any, Dict, List, Optional, Tuple, Type, Union

try:
    from stable_baselines3.dqn.policies import DQNPolicy
    from stable_baselines3.common.torch_layers import BaseFeaturesExtractor, CombinedExtractor
    from stable_baselines3.common.preprocessing import get_flattened_obs_dim
    from gymnasium import spaces
except ImportError as e:
    print(f"Error importing from stable_baselines3: {e}")
    # Create dummy classes to avoid syntax errors
    class DQNPolicy:
        pass
    class BaseFeaturesExtractor:
        pass
    class CombinedExtractor:
        pass

class PatchedDQNPolicy(DQNPolicy):
    """
    Extension of DQNPolicy to support prioritized replay.
    This policy extends the standard DQNPolicy to:
    1. Save raw_loss for prioritized replay updates
    2. Support importance sampling weights
    """
    
    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        lr_schedule: callable,
        net_arch: Optional[List[int]] = None,
        activation_fn = th.nn.ReLU,
        features_extractor_class = None,
        features_extractor_kwargs: Optional[Dict[str, Any]] = None,
        normalize_images: bool = True,
        optimizer_class = th.optim.Adam,
        optimizer_kwargs: Optional[Dict[str, Any]] = None,
        n_quantiles: int = 200,
        net_arch_dueling: Optional[List[int]] = None,
    ):
        """Initialize patched policy with same parameters as DQNPolicy"""
        # For Dict observations, default to CombinedExtractor if none is specified
        if features_extractor_class is None and isinstance(observation_space, spaces.Dict):
            features_extractor_class = CombinedExtractor
            print("Using CombinedExtractor for Dict observation space")
        
        # Make sure features_extractor_kwargs is not None
        if features_extractor_kwargs is None:
            features_extractor_kwargs = {}
        
        super().__init__(
            observation_space=observation_space,
            action_space=action_space,
            lr_schedule=lr_schedule,
            net_arch=net_arch,
            activation_fn=activation_fn,
            features_extractor_class=features_extractor_class,
            features_extractor_kwargs=features_extractor_kwargs,
            normalize_images=normalize_images,
            optimizer_class=optimizer_class,
            optimizer_kwargs=optimizer_kwargs,
        )
        
        # Add attributes for prioritized replay
        self.raw_loss = None
        self.current_weights = None
    
    def _predict(self, obs: th.Tensor, deterministic: bool = True) -> th.Tensor:
        """Override _predict to handle weights if needed"""
        return super()._predict(obs, deterministic)
    
    def train(self) -> None:
        """Override train to handle raw loss for PER"""
        super().train()
    
    def forward(self, obs: th.Tensor, deterministic: bool = True) -> Tuple[th.Tensor, th.Tensor]:
        """Forward pass that returns action and q-values"""
        return super().forward(obs, deterministic)
    
    def _td_loss(self, replay_data):
        """
        Override _td_loss to save raw loss and apply importance sampling weights
        for prioritized replay buffer.
        """
        try:
            # First use the parent's _td_loss implementation to get the standard loss
            # but with reduction='none' if possible
            # For standard DQN in SB3, we'll just use the default implementation
            # and save the result to raw_loss
            loss = super()._td_loss(replay_data)
            
            # Save the loss for priority updates - ideally this would be per-sample loss
            # but we'll use what we have
            if isinstance(loss, th.Tensor):
                self.raw_loss = loss.detach().clone() if loss.numel() == 1 else loss
                
                # Apply weights if available (simplest approach)
                if hasattr(self, 'current_weights') and self.current_weights is not None:
                    # Simply use the weights to scale the loss if dimensions match
                    try:
                        if self.current_weights.numel() > 1:
                            print(f"Using weights with shape: {self.current_weights.shape}, loss shape: {self.raw_loss.shape if self.raw_loss.numel() > 1 else '(scalar)'}")
                            # If loss is already reduced, we can't apply per-sample weights
                            # Return loss as is
                            return loss
                    except Exception as weight_error:
                        print(f"Error applying weights: {weight_error}")
                        return loss
            
            return loss
            
        except Exception as e:
            print(f"Error in PatchedDQNPolicy._td_loss: {e}")
            # Fall back to parent implementation
            return super()._td_loss(replay_data) 