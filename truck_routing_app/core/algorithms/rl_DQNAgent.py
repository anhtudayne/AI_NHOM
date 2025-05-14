"""
DQN Agent Trainer for Truck Routing RL Environment
"""

from stable_baselines3 import DQN
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.buffers import ReplayBuffer
# from stable_baselines3.common.buffers import PrioritizedReplayBuffer # Will be imported conditionally
import os
import numpy as np
import json
from stable_baselines3.common.callbacks import BaseCallback
import datetime
from pathlib import Path
from gymnasium import spaces
import sys  # For debug printing
import time
import pickle
import torch as th
import matplotlib.pyplot as plt
from typing import Any
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.logger import configure # Import configure for logger
import threading

# Debug info to understand Python environment
print(f"Python version: {sys.version}")
print(f"Python executable: {sys.executable}")
print(f"Python path: {sys.path}")

# Try to check if sb3_contrib is available first as a direct import
try:
    import sb3_contrib
    print(f"sb3_contrib version: {sb3_contrib.__version__}")
    SB3_CONTRIB_AVAILABLE = True
except ImportError as e:
    print(f"Error importing sb3_contrib: {e}")
    SB3_CONTRIB_AVAILABLE = False

PRIORITIZED_REPLAY_AVAILABLE = False

# First, try our custom implementation
try:
    from .custom_buffers import create_prioritized_buffer
    PRIORITIZED_REPLAY_AVAILABLE = True
    print("Successfully imported PrioritizedReplayBuffer from custom implementation")
except ImportError as e:
    print(f"Error importing custom PrioritizedReplayBuffer: {e}")
    
    # If custom fails, try sb3_contrib
    try:
        import sb3_contrib
        print(f"sb3_contrib version: {sb3_contrib.__version__}")
        
        # Try finding PrioritizedReplayBuffer in various locations
        try:
            # In newer versions it might be in different locations
            locations_to_try = [
                "sb3_contrib.common.buffers",
                "sb3_contrib.common.maskable.buffers",
                "sb3_contrib.common.recurrent.buffers"
            ]
            
            PrioritizedReplayBuffer = None
            for location in locations_to_try:
                try:
                    PrioritizedReplayBuffer = __import__(location, fromlist=["PrioritizedReplayBuffer"]).PrioritizedReplayBuffer
                    print(f"Successfully imported PrioritizedReplayBuffer from {location}")
                    PRIORITIZED_REPLAY_AVAILABLE = True
                    break
                except (ImportError, AttributeError) as e:
                    print(f"Could not import from {location}: {e}")
            
            if not PRIORITIZED_REPLAY_AVAILABLE:
                # Fall back to trying stable_baselines3 (older versions might have it here)
                from stable_baselines3.common.buffers import PrioritizedReplayBuffer
                PRIORITIZED_REPLAY_AVAILABLE = True
                print("Successfully imported PrioritizedReplayBuffer from stable_baselines3")
        except ImportError as e:
            print(f"Error importing PrioritizedReplayBuffer from all known locations: {e}")
    except ImportError:
        print("sb3_contrib not found")

if not PRIORITIZED_REPLAY_AVAILABLE:
    print("Warning: PrioritizedReplayBuffer could not be imported. Using custom implementation.")
    print("Creating a basic implementation...")
    
    # Create a minimal fallback implementation directly in this file
    class PrioritizedReplayBuffer(ReplayBuffer):
        """Minimal fallback implementation of PrioritizedReplayBuffer"""
        def __init__(self, *args, alpha=0.6, beta=0.4, eps=1e-6, **kwargs):
            super().__init__(*args, **kwargs)
            self.alpha = alpha
            self.beta = beta
            self.eps = eps
            print("Using minimal fallback PrioritizedReplayBuffer implementation")
            # This implementation will behave like a regular ReplayBuffer
            # but at least allows the code to run
            
        def update_priorities(self, indices, priorities):
            # This is a dummy method that would normally update priorities
            pass
    
    PRIORITIZED_REPLAY_AVAILABLE = True
    print("Using fallback PrioritizedReplayBuffer implementation")

class MetricsCallback(BaseCallback):
    """
    Callback để theo dõi và lưu các metrics trong quá trình huấn luyện
    """
    def __init__(self, log_dir: str | None = None, verbose: int = 0):
        super().__init__(verbose)
        self.log_dir = log_dir
        if log_dir is not None:
            os.makedirs(log_dir, exist_ok=True)
            self.metrics_path = os.path.join(log_dir, "training_metrics.json")
        else:
            self.metrics_path = None
            
        # Khởi tạo các metrics
        self.metrics: dict[str, list] = {
            "episode_rewards": [],
            "episode_lengths": [],
            "success_rates": [],
            "learning_rates": [],
            "exploration_rates": [],
            "losses": [],
            "timesteps": [],
            "termination_reasons": {}
        }
        
        # Theo dõi thời gian
        self.start_time = datetime.datetime.now()
        
        # Tracking for success rate calculation
        self.episode_success = []
        self.success_window_size = 100
        
    def _on_step(self):
        # Ghi lại thông tin loss từ model
        if len(self.model.ep_info_buffer) > 0 and len(self.model.ep_info_buffer[-1]) > 0:
            if "r" in self.model.ep_info_buffer[-1]:
                self.metrics["episode_rewards"].append(self.model.ep_info_buffer[-1]["r"])
                # Thêm log reward trung bình mỗi 1000 bước
                if self.num_timesteps % 1000 == 0 and len(self.metrics["episode_rewards"]) >= 10:
                    mean_reward = float(np.mean(self.metrics["episode_rewards"][-10:]))
                    print(f"[MetricsCallback] Step {self.num_timesteps}: Mean reward (last 10 episodes): {mean_reward:.2f}")
                    
                    # Thêm phát hiện lặp để điều chỉnh exploration
                    if len(self.metrics["episode_rewards"]) >= 20:
                        # Kiểm tra nếu reward không cải thiện trong 10 episode gần nhất
                        recent_reward = np.mean(self.metrics["episode_rewards"][-10:])
                        previous_reward = np.mean(self.metrics["episode_rewards"][-20:-10])
                        
                        if recent_reward <= previous_reward:
                            # Có thể agent đang bị mắc kẹt, tăng exploration tạm thời
                            current_eps = self.model.exploration_rate
                            # Tăng epsilon lên ít nhất 0.3 nếu nó đang nhỏ hơn
                            if current_eps < 0.3: # You suggested possibly increasing this boost to 0.5
                                new_eps = max(current_eps * 2, 0.3) # Consider 0.5 here
                                self.model.exploration_rate = new_eps
                                print(f"[MetricsCallback] Phát hiện mắc kẹt. Tăng exploration từ {current_eps:.3f} lên {new_eps:.3f}")
            
            if "l" in self.model.ep_info_buffer[-1]:
                self.metrics["episode_lengths"].append(self.model.ep_info_buffer[-1]["l"])
            
            if "termination_reason" in self.model.ep_info_buffer[-1]:
                reason = self.model.ep_info_buffer[-1]["termination_reason"]
                if reason not in self.metrics["termination_reasons"]:
                    self.metrics["termination_reasons"][reason] = 0
                self.metrics["termination_reasons"][reason] += 1
                
                # Log việc đâm vào vật cản và các lỗi thường gặp
                if reason in ["va_cham_vat_can_nhung_khong_ket_thuc", "het_nhien_lieu", "het_tien"] and self.num_timesteps % 2000 == 0:
                    counts = {k: v for k, v in self.metrics["termination_reasons"].items()}
                    print(f"  Termination stats at step {self.num_timesteps}: {counts}")
        
        # Ghi lại learning rate và exploration rate
        if hasattr(self.model, "learning_rate"):
            if callable(self.model.learning_rate):
                lr = self.model.learning_rate(self.num_timesteps)
            else:
                lr = self.model.learning_rate
            self.metrics["learning_rates"].append(lr)
        
        if hasattr(self.model, "exploration_schedule"):
            eps = self.model.exploration_schedule(self.num_timesteps)
            self.metrics["exploration_rates"].append(eps)
            
            # Log tỷ lệ khám phá
            if self.num_timesteps % 5000 == 0:
                print(f"[MetricsCallback] Step {self.num_timesteps}: Exploration rate: {eps:.4f}")
        
        # Ghi lại loss
        if hasattr(self.model.policy, "raw_loss"): # Check if raw_loss attribute exists
            current_loss = float(self.model.policy.raw_loss)
            self.metrics["losses"].append(current_loss)
            
            # Log loss định kỳ
            if self.num_timesteps % 5000 == 0 and len(self.metrics["losses"]) > 0:
                recent_losses = self.metrics["losses"][-100:] if len(self.metrics["losses"]) > 100 else self.metrics["losses"]
                avg_loss = sum(recent_losses) / len(recent_losses)
                print(f"[MetricsCallback] Step {self.num_timesteps}: Average loss (recent): {avg_loss:.4f}")
        
        # Ghi lại timestep và thời gian huấn luyện
        self.metrics["timesteps"].append(self.num_timesteps)
        elapsed_time = (datetime.datetime.now() - self.start_time).total_seconds()
        steps_per_second = self.num_timesteps / elapsed_time if elapsed_time > 0 else 0
        
        # Log tiến độ và hiệu suất huấn luyện
        if self.num_timesteps % 10000 == 0:
            print(f"[MetricsCallback] Progress: {self.num_timesteps} steps, {elapsed_time:.1f} seconds ({steps_per_second:.1f} steps/s)")
        
        # Lưu metrics định kỳ
        if self.num_timesteps % 1000 == 0 and self.metrics_path is not None:
            self.save_metrics()
        
        return True
    
    def save_metrics(self):
        """Lưu metrics vào file JSON"""
        if self.metrics_path is not None:
            # Thêm thông tin thời gian huấn luyện
            elapsed_time = (datetime.datetime.now() - self.start_time).total_seconds()
            self.metrics["elapsed_time"] = elapsed_time
            self.metrics["last_update"] = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            
            with open(self.metrics_path, 'w') as f:
                json.dump(self.metrics, f, indent=2)
    
    def get_metrics(self) -> dict[str, Any]:
        """Trả về metrics đã thu thập"""
        return self.metrics

class DQNAgentTrainer:
    """
    Lớp huấn luyện DQN Agent cho bài toán truck routing.
    Sử dụng thư viện stable-baselines3.
    """
    
    def __init__(self, env: Any, log_dir: str | None = None):
        """
        Khởi tạo DQNAgent.
        
        Args:
            env: Môi trường OpenAI Gym
            log_dir: Thư mục lưu log và model
        """
        self.env = env
        self.model = None
        self.map_memory = {}  # Khởi tạo map_memory ngay từ đầu
        self.log_dir = log_dir
        self.logger = None # Initialize logger attribute
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
            self.tensorboard_log = os.path.join(log_dir, "tensorboard_logs")
             # Configure logger
            self.logger = configure(folder=log_dir, format_strings=["stdout", "csv", "tensorboard"])
        else:
            self.tensorboard_log = None
            # If no log_dir, use a basic logger that prints to stdout
            self.logger = configure(folder=None, format_strings=["stdout"])

        self.training_metrics: dict = {} 
        
        # Khởi tạo với tham số mặc định (can be overridden by create_model)
        self.exploration_fraction = 0.2
        self.exploration_initial_eps = 1.0
        self.exploration_final_eps = 0.05
        self.max_grad_norm = 10
        self.learning_rate = 0.0003 # Default, can be overridden
        self.model_config: dict = {}  # Cấu hình mô hình được sử dụng gần đây nhất
        
        # Thêm biến để theo dõi kết quả reward và phát hiện early stopping
        self.best_mean_reward = -np.inf
        self.no_improvement_count = 0
        self.early_stopping_patience = 10  # Số lần kiểm tra không cải thiện trước khi dừng
        self.early_stopping_threshold = 0.1  # Ngưỡng tối thiểu để coi là cải thiện
        self.check_freq = 5000  # Kiểm tra mỗi 5000 bước
        
    def create_model(self, 
                     learning_rate=0.0001,
                     buffer_size=10000,
                     batch_size=64,
                     gamma=0.99,
                     tau=0.001,
                     train_freq=1, # Can be a tuple (frequency, unit) e.g., (1, "episode") or int (steps)
                     gradient_steps=1,
                     learning_starts=1000,
                     target_update_interval=1000, # Added
                     exploration_fraction=0.1,
                     exploration_initial_eps=1.0,
                     exploration_final_eps=0.05,
                     max_grad_norm=10,
                     policy_kwargs: dict | None = None,
                     use_double_dqn=True, # Added, SB3 DQN usually has double_q=True by default
                     use_dueling_network=False, # Added
                     use_prioritized_replay=False, # Added
                     prioritized_replay_alpha=0.6,
                     prioritized_replay_beta0=0.4,
                     prioritized_replay_eps=1e-6,
                     verbose=1,
                     device="auto",
                     tensorboard_log: str | None = None): # tensorboard_log already handled in __init__
        """Khởi tạo mô hình DQN với các hyperparameters được chỉ định."""
        
        try:
            if policy_kwargs is None:
                policy_kwargs = {
                    "net_arch": [128, 128], 
                    "features_extractor_kwargs": {
                        "cnn_output_dim": 128,
                    }
                }
            
            # Add dueling to policy_kwargs if enabled
            final_policy_kwargs = policy_kwargs.copy()
            if use_dueling_network:
                # Dueling network requires special structure in network architecture
                # Instead of setting a 'dueling' flag, configure the proper network for dueling
                if self.logger: self.logger.info("Configuring network architecture for Dueling DQN")
                # We do not modify the policy_kwargs directly but will pass the dueling flag to DQN
                # which will handle the dueling network architecture internally
            
            if hasattr(self, 'model') and self.model is not None:
                del self.model # Clean up existing model
            
            replay_buffer_class = None
            replay_buffer_kwargs = {}
            use_custom_buffer = False
            
            # Mặc định, sử dụng MultiInputPolicy
            policy_class = "MultiInputPolicy"
            
            if use_prioritized_replay:
                # Remember if we should use a custom buffer
                use_custom_buffer = 'create_prioritized_buffer' in globals()
                
                # Try to use our custom policy for prioritized replay
                try:
                    # Import custom policy to support prioritized replay
                    from .custom_policy import PatchedDQNPolicy
                    print("Using PatchedDQNPolicy for prioritized replay")
                    # Override policy class
                    policy_class = PatchedDQNPolicy
                except ImportError as policy_error:
                    print(f"Error importing PatchedDQNPolicy: {policy_error}")
                    # Fall back to standard MultiInputPolicy - already set by default
                
                if not use_custom_buffer and PRIORITIZED_REPLAY_AVAILABLE:
                    # Classic way - use PrioritizedReplayBuffer class
                    replay_buffer_class = PrioritizedReplayBuffer
                    replay_buffer_kwargs['alpha'] = prioritized_replay_alpha
                    replay_buffer_kwargs['beta'] = prioritized_replay_beta0
                    replay_buffer_kwargs['eps'] = prioritized_replay_eps
                    if self.logger: self.logger.info("Using standard PrioritizedReplayBuffer class")
                elif not PRIORITIZED_REPLAY_AVAILABLE:
                    if self.logger: self.logger.info("Prioritized Replay was requested but is not available. Using standard Replay Buffer.")
                else:
                    if self.logger: self.logger.info("Will use custom PrioritizedReplayBuffer implementation after model creation")
            
            # Check which version of stable-baselines3 we're using to determine the correct parameter name
            import inspect
            dqn_params = inspect.signature(DQN.__init__).parameters
            
            # Create a dict of kwargs that will be filtered based on accepted parameters
            dqn_kwargs = {
                "learning_rate": learning_rate,
                "buffer_size": buffer_size,
                "batch_size": batch_size,
                "gamma": gamma, 
                "tau": tau,
                "train_freq": train_freq,
                "gradient_steps": gradient_steps,
                "learning_starts": learning_starts,
                "exploration_fraction": exploration_fraction,
                "exploration_initial_eps": exploration_initial_eps,
                "exploration_final_eps": exploration_final_eps,
                "target_update_interval": target_update_interval,
                "policy_kwargs": final_policy_kwargs,
                "tensorboard_log": self.tensorboard_log,
                "verbose": verbose,
                "device": device,
                "max_grad_norm": max_grad_norm
            }
            
            # Only add replay_buffer parameters if we're using the standard way (not custom factory)
            if replay_buffer_class is not None:
                dqn_kwargs["replay_buffer_class"] = replay_buffer_class
                dqn_kwargs["replay_buffer_kwargs"] = replay_buffer_kwargs
            
            # Add double_q parameter with the correct name
            if "double_q" in dqn_params:
                dqn_kwargs["double_q"] = use_double_dqn
            elif "use_double_q" in dqn_params:
                dqn_kwargs["use_double_q"] = use_double_dqn
            else:
                print(f"Warning: Neither 'double_q' nor 'use_double_q' found in DQN parameters. Double DQN flag might be ignored.")
            
            # Process dueling network configuration
            if use_dueling_network:
                # Check if the DQN implementation accepts 'dueling' as a parameter
                if "dueling" in dqn_params:
                    dqn_kwargs["dueling"] = True
                else:
                    # If not, we need to use policy_kwargs to enable dueling
                    # This depends on the version of Stable Baselines3 being used
                    try:
                        # Try to configure dueling in a way compatible with the current SB3 version
                        from stable_baselines3.dqn.policies import DQNPolicy
                        if hasattr(DQNPolicy, "dueling"):
                            # Some versions support dueling via policy config
                            final_policy_kwargs["features_extractor_kwargs"] = final_policy_kwargs.get("features_extractor_kwargs", {})
                            final_policy_kwargs["features_extractor_kwargs"]["dueling"] = True
                            if self.logger: self.logger.info("Configured dueling via features_extractor_kwargs")
                        else:
                            # Newer versions might use a different approach or not need explicit configuration
                            if self.logger: self.logger.info("Dueling network requested but not explicitly configured. SB3 may handle this automatically.")
                    except ImportError:
                        if self.logger: self.logger.info("Could not import DQNPolicy to configure dueling network")
            
            # Filter out any kwargs that aren't accepted by the current DQN implementation
            accepted_kwargs = {k: v for k, v in dqn_kwargs.items() if k in dqn_params}
            
            self.model = DQN(
                policy_class if use_prioritized_replay else "MultiInputPolicy",
                self.env,
                **accepted_kwargs
            )
            
            # If we're using our custom buffer implementation, replace the replay buffer after creation
            if use_prioritized_replay and use_custom_buffer:
                if self.logger: self.logger.info("Creating custom prioritized replay buffer")
                # Import locally to ensure the name is available
                from .custom_buffers import create_prioritized_buffer
                
                # Get relevant parameters from the model
                buffer_params = {
                    "buffer_size": buffer_size,
                    "observation_space": self.env.observation_space,
                    "action_space": self.env.action_space,
                    "device": self.model.device,
                    "n_envs": 1,  # Assuming single env
                    "optimize_memory_usage": False,
                    "alpha": prioritized_replay_alpha,
                    "beta": prioritized_replay_beta0,
                    "eps": prioritized_replay_eps
                }
                
                # Create and set the custom buffer
                try:
                    custom_buffer = create_prioritized_buffer(**buffer_params)
                    # Replace the model's replay buffer
                    self.model.replay_buffer = custom_buffer
                    if self.logger: self.logger.info("Successfully created and set custom prioritized replay buffer")
                    
                    # Add patching for the train method to handle prioritized replay correctly
                    self._patch_dqn_for_prioritized_replay()
                    
                except Exception as buffer_error:
                    if self.logger: self.logger.error(f"Failed to create custom buffer: {buffer_error}")
                    print(f"Error creating custom prioritized replay buffer: {buffer_error}")
            
            if self.logger:
                self.logger.info(f"Created DQN model with: learning_rate={learning_rate}, buffer_size={buffer_size}, batch_size={batch_size}, target_update_interval={target_update_interval}")
                self.logger.info(f"Double DQN: {use_double_dqn}, Dueling DQN hint: {use_dueling_network}, Prioritized Replay: {use_prioritized_replay and PRIORITIZED_REPLAY_AVAILABLE}")
                self.logger.info(f"Network architecture: {final_policy_kwargs.get('net_arch', 'default')}")
            
            self.current_config = {
                "learning_rate": learning_rate,
                "buffer_size": buffer_size,
                "batch_size": batch_size,
                "gamma": gamma,
                "tau": tau,
                "train_freq": train_freq,
                "gradient_steps": gradient_steps,
                "learning_starts": learning_starts,
                "target_update_interval": target_update_interval,
                "exploration_fraction": exploration_fraction,
                "exploration_initial_eps": exploration_initial_eps,
                "exploration_final_eps": exploration_final_eps,
                "max_grad_norm": max_grad_norm,
                "policy_kwargs": final_policy_kwargs,
                "use_double_dqn": use_double_dqn,
                "use_dueling_network": use_dueling_network, 
                "use_prioritized_replay": use_prioritized_replay and PRIORITIZED_REPLAY_AVAILABLE,
                "prioritized_replay_alpha": prioritized_replay_alpha if use_prioritized_replay else None,
                "prioritized_replay_beta0": prioritized_replay_beta0 if use_prioritized_replay else None
            }
            
            return self.model
            
        except Exception as e:
            if self.logger:
                self.logger.error(f"Error creating DQN model: {str(e)}")
            else:
                print(f"Error creating DQN model: {str(e)}")
            import traceback
            traceback.print_exc()
            raise e
        
    def _patch_dqn_for_prioritized_replay(self):
        """
        Patch the DQN algorithm's train method to correctly use our custom prioritized replay buffer.
        This captures the train method to extract weights and indices since they're not part of the sample anymore.
        """
        if not hasattr(self, 'model') or self.model is None:
            print("Cannot patch DQN: model not created yet")
            return
            
        # Ensure torch is available
        import torch as th
            
        # Store original train method
        original_train = self.model.train
        
        # Create a patched version
        def patched_train(*args, **kwargs):
            try:
                # Call original train method
                loss = original_train(*args, **kwargs)
                
                # After training, get the last_sampled_indices and update priorities if available
                if hasattr(self.model.replay_buffer, 'get_last_indices') and hasattr(self.model.replay_buffer, 'update_priorities'):
                    # Get indices that were used in last sampling
                    indices = self.model.replay_buffer.get_last_indices()
                    
                    if indices is not None:
                        # Default priorities - use 1.0 if we can't get raw_loss
                        priorities = np.ones(len(indices), dtype=np.float32)
                        
                        # Try to get priorities from policy's raw_loss if available
                        try:
                            if hasattr(self.model.policy, 'raw_loss'):
                                raw_loss = self.model.policy.raw_loss
                                if isinstance(raw_loss, th.Tensor):
                                    # If it's a scalar or wrong shape, use a default value
                                    if raw_loss.numel() == 1:
                                        print("Warning: raw_loss is a scalar, using default priorities")
                                    else:
                                        # Try to convert to priorities
                                        priorities = raw_loss.abs().detach().cpu().numpy()
                                        # Ensure correct shape
                                        if len(priorities.shape) > 1:
                                            priorities = priorities.mean(axis=1)
                                        # Match the length of indices
                                        priorities = priorities[:len(indices)]
                        except Exception as e:
                            print(f"Error extracting priorities from raw_loss: {e}")
                            
                        # Update priorities in replay buffer
                        self.model.replay_buffer.update_priorities(indices, priorities)
                
                return loss
            except Exception as e:
                print(f"Error in patched_train: {e}")
                # Try to fall back to original method
                return original_train(*args, **kwargs)
            
        # Replace the train method
        self.model.train = patched_train
        
        # Also patch the _train_step method to handle weights
        if hasattr(self.model, '_train_step'):
            original_train_step = self.model._train_step
            
            def patched_train_step(*args, **kwargs):
                try:
                    # Check if we need to get weights from the buffer
                    if hasattr(self.model.replay_buffer, 'get_last_weights'):
                        weights = self.model.replay_buffer.get_last_weights()
                        if weights is not None and hasattr(self.model.policy, 'current_weights'):
                            # Make sure weights are available to policy
                            self.model.policy.current_weights = weights
                    
                    return original_train_step(*args, **kwargs)
                except Exception as e:
                    print(f"Error in patched_train_step: {e}")
                    # Fall back to original
                    return original_train_step(*args, **kwargs)
                
            self.model._train_step = patched_train_step
    
    def train(self, total_timesteps=100000, callback: BaseCallback | None = None):
        """Huấn luyện agent trong môi trường."""
        try:
            # Sử dụng một cách tiếp cận đơn giản hơn cho các callbacks
            # Tạo early stopping callback
            early_stopping_callback = self._create_early_stopping_callback(total_timesteps)
            
            # Thay vì sử dụng danh sách callbacks, chỉ sử dụng callback đã cung cấp hoặc early stopping callback
            # Điều này giúp tránh vấn đề unhashable type với danh sách callback
            
            final_callback = None
            
            if callback is not None:
                # Nếu callback đã được cung cấp, sử dụng nó
                # Kiểm tra xem callback có phải là BaseCallback không
                from stable_baselines3.common.callbacks import BaseCallback
                if isinstance(callback, BaseCallback):
                    final_callback = callback
                else:
                    # Nếu không phải BaseCallback, vẫn sử dụng callback đó nhưng cảnh báo
                    print("Warning: Provided callback is not a BaseCallback instance. This might cause issues.")
                    final_callback = callback
            else:
                # Nếu không có callback được chỉ định, sử dụng early stopping callback
                final_callback = early_stopping_callback
            
            # Bắt đầu huấn luyện
            print(f"Starting training for {total_timesteps} timesteps with callback...")
            
            # Sử dụng callback đơn lẻ thay vì list và đảm bảo callback không phải là list
            self.model.learn(
                total_timesteps=total_timesteps,
                callback=final_callback
            )
            
            # Tự quản lý lưu trữ kinh nghiệm tốt sau khi huấn luyện xong
            self._save_experience_after_training()
            
            # Lưu metrics nếu là MetricsCallback
            if hasattr(final_callback, 'get_metrics') and callable(getattr(final_callback, 'get_metrics')):
                self.training_metrics = final_callback.get_metrics()
            else:
                self.training_metrics = None
            
            # Sau khi huấn luyện, đánh giá model
            print("Training complete! Evaluating agent...")
            
            # Đánh giá model trên 10 episode
            n_eval_episodes = 10
            mean_reward, success_rate = self._evaluate_quick(n_eval_episodes)
            
            # Lưu thông tin đánh giá
            print(f"Evaluation results: Mean Reward: {mean_reward:.2f}, Success Rate: {success_rate:.2%}")
            
            # Lưu checkpoint cuối cùng
            if self.log_dir:
                final_model_path = os.path.join(self.log_dir, "final_model")
                self.model.save(final_model_path)
                print(f"Saved final model to {final_model_path}")
            
            return mean_reward, success_rate
            
        except Exception as e:
            print(f"Error during training: {e}")
            import traceback
            traceback.print_exc()
            raise
    
    def _create_early_stopping_callback(self, total_timesteps):
        """Tạo callback giám sát và dừng sớm nếu không có cải thiện"""
        
        class EarlyStoppingCallback(BaseCallback):
            def __init__(self, trainer_instance, verbose=0):
                super().__init__(verbose)
                self.trainer = trainer_instance
                self.best_mean_reward = -np.inf
                self.no_improvement_count = 0
                self.check_freq = self.trainer.check_freq
                self.early_stopping_patience = self.trainer.early_stopping_patience
                self.early_stopping_threshold = self.trainer.early_stopping_threshold
                
            def _on_step(self):
                if self.n_calls % self.check_freq == 0:
                    # Tính reward trung bình từ 10 episode gần nhất
                    if len(self.model.ep_info_buffer) > 0:
                        # Sử dụng buffer thông tin episode để tính toán reward trung bình
                        mean_reward = np.mean([ep_info['r'] for ep_info in self.model.ep_info_buffer])
                        
                        # Kiểm tra xem có cải thiện đáng kể không
                        if mean_reward - self.best_mean_reward > self.early_stopping_threshold:
                            print(f"Reward improved from {self.best_mean_reward:.2f} to {mean_reward:.2f}")
                            self.best_mean_reward = mean_reward
                            self.no_improvement_count = 0
                            
                            # Lưu model tốt nhất nếu có log_dir
                            if self.trainer.log_dir:
                                best_model_path = os.path.join(self.trainer.log_dir, "best_model")
                                self.model.save(best_model_path)
                                print(f"New best model saved to {best_model_path}")
                        else:
                            self.no_improvement_count += 1
                            print(f"No significant improvement for {self.no_improvement_count} checks. Best: {self.best_mean_reward:.2f}, Current: {mean_reward:.2f}")
                            
                            # Early stopping sau nhiều lần không cải thiện
                            if self.no_improvement_count >= self.early_stopping_patience:
                                print(f"Early stopping triggered after {self.no_improvement_count * self.check_freq} steps without improvement")
                                # Đánh dấu training đã hoàn thành
                                if self.no_improvement_count >= self.early_stopping_patience + 5:  # Thêm buffer để đảm bảo đã quá ngưỡng
                                    return False  # Dừng training
                
                return True  # Tiếp tục training
                
        return EarlyStoppingCallback(self)
    
    def _save_experience_after_training(self):
        """
        Lưu trữ kinh nghiệm của agent sau khi huấn luyện, thay vì dùng callback (gây lỗi)
        """
        try:
            # Khởi tạo map_memory nếu chưa có
            if not hasattr(self, 'map_memory'):
                self.map_memory = {}
            
            # Xác định map_id an toàn
            map_id = "current_map"
            if hasattr(self.env, 'map_object'):
                map_id = str(id(self.env.map_object))
            elif hasattr(self.env, 'unwrapped') and hasattr(self.env.unwrapped, 'map_object'):
                map_id = str(id(self.env.unwrapped.map_object))
            
            # Thực hiện đánh giá để lấy thông tin path
            print("Evaluating model to collect successful paths...")
            observation, _ = self.env.reset()
            done = False
            path = []
            total_reward = 0
            
            # Thêm vị trí đầu tiên vào path
            if isinstance(observation, dict) and 'agent_pos' in observation:
                path.append(tuple(observation['agent_pos']))
            
            # Chạy một episode để lấy thông tin
            while not done:
                action, _ = self.model.predict(observation, deterministic=True)
                observation, reward, terminated, truncated, info = self.env.step(action)
                done = terminated or truncated
                total_reward += reward
                
                # Thêm vị trí hiện tại vào path
                if isinstance(observation, dict) and 'agent_pos' in observation:
                    path.append(tuple(observation['agent_pos']))
                
                # Kiểm tra nếu đến đích
                if done and 'termination_reason' in info and info['termination_reason'] == 'den_dich':
                    # Lưu path thành công
                    if map_id not in self.map_memory:
                        self.map_memory[map_id] = []
                    
                    # Tạo thông tin path
                    trajectory_data = {
                        'positions': path,
                        'total_reward': float(total_reward),
                        'timestamp': time.time()
                    }
                    
                    # Thêm vào bộ nhớ
                    self.map_memory[map_id].append(trajectory_data)
                    
                    # Chỉ giữ 5 đường đi tốt nhất
                    if len(self.map_memory[map_id]) > 5:
                        self.map_memory[map_id].sort(key=lambda x: x['total_reward'], reverse=True)
                        self.map_memory[map_id] = self.map_memory[map_id][:5]
                    
                    print(f"Saved successful path with reward {total_reward}")
            
        except Exception as e:
            print(f"Error saving experience after training: {e}")
            import traceback
            traceback.print_exc()
    
    def predict_action(self, observation):
        """Dự đoán hành động dựa trên observation."""
        if self.model is None:
            raise ValueError("Model chưa được khởi tạo. Vui lòng gọi create_model() hoặc load_model() trước.")
        
        # Thử sử dụng đường đi đã biết nếu có thể
        current_position = None
        map_id = None
        
        # Cố gắng trích xuất vị trí hiện tại và ID bản đồ một cách an toàn
        try:
            if hasattr(self.env, 'map_object'):
                map_id = str(id(self.env.map_object))
            elif hasattr(self.env, 'unwrapped') and hasattr(self.env.unwrapped, 'map_object'):
                map_id = str(id(self.env.unwrapped.map_object))
            else:
                map_id = "default_map"
                
            # Trích xuất vị trí hiện tại từ observation
            if isinstance(observation, dict) and 'agent_pos' in observation:
                current_position = tuple(observation['agent_pos'])
        except Exception as e:
            print(f"Warning: Could not extract position from observation: {e}")
        
        # Khởi tạo map_memory nếu chưa tồn tại
        if not hasattr(self, 'map_memory'):
            self.map_memory = {}
        
        # Kiểm tra và sử dụng đường đi đã biết nếu có
        if current_position is not None and map_id is not None and map_id in self.map_memory and len(self.map_memory[map_id]) > 0:
            try:
                # Sử dụng đường đi có reward cao nhất (đã được sắp xếp)
                best_path_data = self.map_memory[map_id][0]
                best_path = best_path_data['positions']
                
                # Tìm vị trí hiện tại trong đường đi đã biết
                for i, pos in enumerate(best_path):
                    if pos == current_position and i < len(best_path) - 1:
                        # Tìm vị trí tiếp theo trong đường đi
                        next_pos = best_path[i + 1]
                        
                        # Xác định hướng di chuyển
                        dx = next_pos[0] - current_position[0]
                        dy = next_pos[1] - current_position[1]
                        
                        # Ánh xạ hướng thành action
                        if dx == 0 and dy == -1:  # Lên
                            return 0
                        elif dx == 1 and dy == 0:  # Phải
                            return 1
                        elif dx == 0 and dy == 1:  # Xuống
                            return 2
                        elif dx == -1 and dy == 0:  # Trái
                            return 3
                        # Nếu là trạm xăng và cần đổ xăng
                        elif dx == 0 and dy == 0 and hasattr(self.env, 'map_object'):
                            try:
                                # Kiểm tra nếu vị trí hiện tại là trạm xăng
                                from core.constants import CellType
                                grid = self.env.map_object.grid
                                if grid[current_position[1], current_position[0]] == CellType.GAS:
                                    return 4  # Action đổ xăng
                            except Exception as cell_error:
                                # Bỏ qua lỗi và dùng action từ model
                                pass
                        
                        # Nếu không xác định được hành động, dùng model
                        break
            except Exception as path_error:
                print(f"Error using saved path: {path_error}, falling back to model")
        
        # Nếu không thể sử dụng đường đi đã biết, dùng model dự đoán
        try:
            action, _states = self.model.predict(observation, deterministic=True)
            return action
        except Exception as predict_error:
            print(f"Error in model.predict: {predict_error}")
            # Fallback: Trả về action ngẫu nhiên nếu dự đoán lỗi
            return np.random.randint(0, 5)  # 0-4: Lên, Phải, Xuống, Trái, Đổ xăng
    
    def _evaluate_quick(self, n_episodes=10):
        """
        Đánh giá nhanh model trên n_episodes.
            
        Returns:
            mean_reward: Phần thưởng trung bình
            success_rate: Tỷ lệ thành công
        """
        all_rewards = []
        success_count = 0
        
        # Đảm bảo model đã được khởi tạo
        if self.model is None:
            print("Model chưa được khởi tạo, không thể đánh giá")
            return 0.0, 0.0
        
        for episode in range(n_episodes):
            try:
                episode_reward = 0
                done = False
                obs, _ = self.env.reset()
                
                while not done:
                    # Predict best action
                    action, _ = self.model.predict(obs, deterministic=True)
                    
                    # Step environment
                    obs, reward, terminated, truncated, info = self.env.step(action)
                    done = terminated or truncated
                    episode_reward += reward
                
                    # Kiểm tra nếu thành công (đến đích)
                    if done and isinstance(info, dict) and 'termination_reason' in info:
                        if info['termination_reason'] == 'den_dich':
                            success_count += 1
            
                all_rewards.append(episode_reward)
                
            except Exception as e:
                print(f"Lỗi trong khi đánh giá episode {episode}: {e}")
                import traceback
                traceback.print_exc()
                
                # Tiếp tục với episode tiếp theo
                continue
        
        # Tính toán kết quả cuối cùng
        mean_reward = np.mean(all_rewards) if all_rewards else 0.0
        success_rate = success_count / n_episodes if n_episodes > 0 else 0.0
        
        return float(mean_reward), float(success_rate)
    
    def save_model(self, path: str):
        """Lưu model DQN và dữ liệu map_memory."""
        if self.model is None:
            raise ValueError("Không có model để lưu. Vui lòng tạo hoặc tải model trước.")
        
        # Tạo thư mục nếu chưa tồn tại
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        # Lưu model DQN
        self.model.save(path)
        print(f"Đã lưu model DQN tại: {path}")
        
        # Lưu map_memory vào file riêng
        try:
            memory_path = f"{path}_map_memory.pkl"
            
            # Chuyển đổi các đối tượng trong map_memory sang dạng có thể serializable
            serializable_memory = {}
            for map_id, paths in self.map_memory.items():
                serializable_memory[map_id] = []
                for path_data in paths:
                    # Đảm bảo tất cả dữ liệu đều có thể serialize
                    serializable_path = {
                        'positions': [tuple(pos) for pos in path_data['positions']],
                        'total_reward': float(path_data['total_reward']),
                        'timestamp': path_data['timestamp']
                    }
                    serializable_memory[map_id].append(serializable_path)
            
            # Lưu dữ liệu đã chuyển đổi
            with open(memory_path, 'wb') as f:
                pickle.dump(serializable_memory, f)
            print(f"Đã lưu map_memory tại: {memory_path}")
            
            # Lưu cấu hình model
            config_path = f"{path}_config.json"
            with open(config_path, 'w') as f:
                json.dump(self.model_config, f, indent=2)
            print(f"Đã lưu config tại: {config_path}")
            
        except Exception as e:
            print(f"Lỗi khi lưu map_memory: {e}")
            import traceback
            traceback.print_exc()
    
    def load_model(self, path: str, device: str = "auto"):
        """Tải model DQN và dữ liệu map_memory nếu có."""
        try:
            # Tải model chính
            self.model = DQN.load(path, env=self.env, device=device)
            print(f"Đã tải model DQN từ: {path}")
            
            # Cố gắng tải map_memory từ file riêng
            memory_path = f"{path}_map_memory.pkl"
            if os.path.exists(memory_path):
                try:
                    with open(memory_path, 'rb') as f:
                        self.map_memory = pickle.load(f)
                    print(f"Đã tải map_memory từ: {memory_path}")
                except Exception as memory_error:
                    print(f"Lỗi khi tải map_memory: {memory_error}, khởi tạo map_memory mới")
                    self.map_memory = {}
            else:
                print(f"Không tìm thấy file map_memory tại: {memory_path}, khởi tạo mới")
                self.map_memory = {}
                
            # Cố gắng tải cấu hình model
            config_path = f"{path}_config.json"
            if os.path.exists(config_path):
                try:
                    with open(config_path, 'r') as f:
                        self.model_config = json.load(f)
                    print(f"Đã tải cấu hình model từ: {config_path}")
                except Exception as config_error:
                    print(f"Lỗi khi tải cấu hình model: {config_error}")
                    self.model_config = {}
            else:
                print(f"Không tìm thấy file cấu hình tại: {config_path}")
                self.model_config = {}
                
            return self.model
            
        except Exception as e:
            print(f"Lỗi khi tải model: {e}")
            import traceback
            traceback.print_exc()
            raise
    
    def evaluate(self, n_episodes=10):
        """
        Đánh giá agent trên môi trường.
        
        Args:
            n_episodes: Số lượng episode đánh giá
            
        Returns:
            results: Dictionary chứa kết quả đánh giá
        """
        rewards = []
        episode_lengths = []
        success_count = 0
        fuel_remaining = []
        money_remaining = []
        visited_cells = []
        paths = []
        termination_reasons = {}
        
        for i in range(n_episodes):
            obs, _ = self.env.reset()
            done = False
            truncated = False
            episode_reward = 0
            step_count = 0
            path = []
            
            if hasattr(self.env, 'current_pos'):
                path.append(self.env.current_pos)
            
            while not (done or truncated):
                action = self.predict_action(obs)
                obs, reward, done, truncated, info = self.env.step(action)
                episode_reward += reward
                step_count += 1
                
                # Lưu vị trí vào path nếu có
                if hasattr(self.env, 'current_pos'):
                    if self.env.current_pos not in path:  # Tránh trùng lặp
                        path.append(self.env.current_pos)
            
            rewards.append(episode_reward)
            episode_lengths.append(step_count)
            
            # Ghi nhận thông tin kết thúc episode
            termination_reason = info.get("termination_reason", "unknown")
            termination_reasons[termination_reason] = termination_reasons.get(termination_reason, 0) + 1
            
            # Ghi nhận thành công/thất bại
            if termination_reason == "den_dich":
                success_count += 1
                if hasattr(self.env, 'current_fuel'):
                    fuel_remaining.append(self.env.current_fuel)
                elif "truck_state" in info and "fuel" in info["truck_state"]:
                    fuel_remaining.append(info["truck_state"]["fuel"])
                
                if hasattr(self.env, 'current_money'):
                    money_remaining.append(self.env.current_money)
                elif "truck_state" in info and "money" in info["truck_state"]:
                    money_remaining.append(info["truck_state"]["money"])
            
            # Lưu số ô đã thăm
            visited_count = len(set(path)) if path else 0
            visited_cells.append(visited_count)
            
            # Lưu đường đi
            paths.append(path)
            
            # In thông tin tiến độ đánh giá
            print(f"Episode {i+1}/{n_episodes}: Reward={episode_reward:.2f}, Steps={step_count}, Success={'Yes' if termination_reason == 'den_dich' else 'No'}")
            
        # Tính toán các chỉ số
        success_rate = success_count / n_episodes
        avg_reward = sum(rewards) / n_episodes
        avg_episode_length = sum(episode_lengths) / n_episodes
        avg_visited_cells = sum(visited_cells) / n_episodes
        
        results = {
            "success_rate": success_rate,
            "avg_reward": avg_reward,
            "avg_episode_length": avg_episode_length,
            "avg_visited_cells": avg_visited_cells,
            "n_episodes": n_episodes,
            "termination_reasons": termination_reasons,
            "rewards": rewards,
            "episode_lengths": episode_lengths,
            "paths": paths
        }
        
        # Thêm thông tin về nhiên liệu và tiền nếu có
        if fuel_remaining:
            avg_fuel_remaining = sum(fuel_remaining) / len(fuel_remaining)
            results["avg_remaining_fuel"] = avg_fuel_remaining
            results["fuel_remaining"] = fuel_remaining
            
        if money_remaining:
            avg_money_remaining = sum(money_remaining) / len(money_remaining)
            results["avg_remaining_money"] = avg_money_remaining
            results["money_remaining"] = money_remaining
        
        # Lưu thông tin đánh giá nếu có log_dir
        if self.log_dir:
            eval_path = os.path.join(self.log_dir, "evaluation_results.json")
            try:
                with open(eval_path, 'w') as f:
                    # Chuyển đổi paths để có thể lưu dưới dạng JSON
                    serializable_results = results.copy()
                    serializable_results["paths"] = [[list(pos) if hasattr(pos, "__iter__") else pos for pos in path] for path in paths]
                    json.dump(serializable_results, f, indent=2)
                print(f"Evaluation results saved to {eval_path}")
            except Exception as e:
                print(f"Error saving evaluation results: {e}")
        
        # In tóm tắt kết quả đánh giá
        print(f"\nEvaluation Summary (n={n_episodes}):")
        print(f"Success Rate: {success_rate:.2f}")
        print(f"Average Reward: {avg_reward:.2f}")
        print(f"Average Episode Length: {avg_episode_length:.2f}")
        print(f"Average Visited Cells: {avg_visited_cells:.2f}")
        if "avg_remaining_fuel" in results:
            print(f"Average Remaining Fuel: {results['avg_remaining_fuel']:.2f}")
        if "avg_remaining_money" in results:
            print(f"Average Remaining Money: {results['avg_remaining_money']:.2f}")
        print(f"Termination Reasons: {termination_reasons}")
            
        return results 
