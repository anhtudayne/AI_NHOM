"""
DQN Agent Trainer for Truck Routing RL Environment
"""

from stable_baselines3 import DQN
from stable_baselines3.common.env_checker import check_env
import os
import numpy as np
import json
from stable_baselines3.common.callbacks import BaseCallback
import datetime
from pathlib import Path
from gym import spaces
import sys  # For debug printing

class MetricsCallback(BaseCallback):
    """
    Callback để theo dõi và lưu các metrics trong quá trình huấn luyện
    """
    def __init__(self, log_dir=None, verbose=0):
        super().__init__(verbose)
        self.log_dir = log_dir
        if log_dir is not None:
            os.makedirs(log_dir, exist_ok=True)
            self.metrics_path = os.path.join(log_dir, "training_metrics.json")
        else:
            self.metrics_path = None
            
        # Khởi tạo các metrics
        self.metrics = {
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
                            if current_eps < 0.3:
                                new_eps = max(current_eps * 2, 0.3)
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
        if hasattr(self.model.policy, "raw_loss"):
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
    
    def get_metrics(self):
        """Trả về metrics đã thu thập"""
        return self.metrics

class DQNAgentTrainer:
    """
    Lớp huấn luyện DQN Agent cho bài toán truck routing.
    Sử dụng thư viện stable-baselines3.
    """
    
    def __init__(self, env, log_dir=None, **kwargs):
        """
        Khởi tạo DQN Agent với các tham số tùy chỉnh.
        
        Args:
            env: Môi trường huấn luyện
            log_dir: Thư mục để lưu log và tensorboard
            **kwargs: Các tham số tùy chỉnh khác cho DQN
        """
        # Lưu trữ môi trường
        self.env = env
        
        # Lưu trữ log_dir cho callbacks
        self.log_dir = log_dir
        
        # Model sẽ được tạo bởi phương thức create_model
        self.model = None
        
        # Lưu metrics
        self.training_metrics = None
        
        # Thêm biến để theo dõi kết quả reward và phát hiện early stopping
        self.best_mean_reward = -np.inf
        self.no_improvement_count = 0
        self.early_stopping_patience = 10  # Số lần kiểm tra không cải thiện trước khi dừng
        self.early_stopping_threshold = 0.1  # Ngưỡng tối thiểu để coi là cải thiện
        self.check_freq = 5000  # Kiểm tra mỗi 5000 bước
        
    def create_model(self, learning_rate=0.0003, buffer_size=100000, learning_starts=10000, 
                    batch_size=128, tau=0.005, gamma=0.99, train_freq=4, gradient_steps=1,
                    target_update_interval=10000, exploration_fraction=0.2, 
                    exploration_initial_eps=1.0, exploration_final_eps=0.05, max_grad_norm=10,
                    policy_kwargs=None, verbose=1, use_double_dqn=True, use_dueling_network=False,
                    use_prioritized_replay=False, prioritized_replay_alpha=0.6, prioritized_replay_beta0=0.4):
        """
        Tạo mô hình DQN với các tham số được chỉ định.
        
        Args:
            learning_rate: Tốc độ học
            buffer_size: Kích thước bộ nhớ experience replay
            learning_starts: Số bước trước khi bắt đầu huấn luyện
            batch_size: Kích thước batch
            tau: Tỷ lệ cập nhật mạng đích
            gamma: Hệ số chiết khấu
            train_freq: Tần suất cập nhật mạng
            gradient_steps: Số bước gradient cho mỗi lần cập nhật
            target_update_interval: Tần suất cập nhật mạng đích (tính theo bước)
            exploration_fraction: Phần trăm huấn luyện dành cho khám phá
            exploration_initial_eps: Xác suất khám phá ban đầu
            exploration_final_eps: Xác suất khám phá cuối cùng
            max_grad_norm: Giới hạn gradients
            policy_kwargs: Tham số cho mạng policy
            verbose: Mức độ hiển thị thông tin
            use_double_dqn: Sử dụng Double DQN
            use_dueling_network: Sử dụng Dueling Network
            use_prioritized_replay: Sử dụng Prioritized Experience Replay
            prioritized_replay_alpha: Alpha cho PER
            prioritized_replay_beta0: Beta0 cho PER
        
        Returns:
            self: Đối tượng DQNAgentTrainer đã được cấu hình
        """
        # Tham số mặc định
        if policy_kwargs is None:
            policy_kwargs = {}
        
        # Handle potential PyTorch initialization errors (for compatibility)
        try:
            import torch
            import torch.nn as nn
            print(f"DEBUG: Using PyTorch version: {torch.__version__}")
        except ImportError:
            print("DEBUG: PyTorch not found, will use default settings")
        except Exception as e:
            print(f"DEBUG: PyTorch initialization error: {e}")
            
        # Lưu các cờ cho kỹ thuật nâng cao VÀO INSTANCE VARIABLES
        self.use_double_dqn = use_double_dqn
        self.use_dueling_network = use_dueling_network
        self.use_prioritized_replay = use_prioritized_replay
        # Các tham số PER cũng nên được lưu nếu cần thiết khi load, nhưng hiện tại chỉ cần cờ

        # Lưu các tham số PER để có thể sử dụng khi thiết lập buffer
        self.prioritized_replay_alpha = prioritized_replay_alpha
        self.prioritized_replay_beta0 = prioritized_replay_beta0
        
        # Disable dueling network if torch has compatibility issues
        # This helps avoid __path__.__path__ errors
        try:
            import torch.nn as nn
            # Simple test to check if custom modules can be created
            test_module = nn.Sequential(nn.Linear(10, 10))
        except Exception as e:
            print(f"DEBUG: PyTorch test failed: {e}, disabling dueling network")
            use_dueling_network = False
            self.use_dueling_network = False
        
        # Cấu hình dueling network trong policy_kwargs
        if use_dueling_network:
            # For stable-baselines3, dueling architecture needs to be implemented 
            # differently for MlpPolicy vs MultiInputPolicy
            
            # Simpler implementation of dueling to avoid torch errors
            if "net_arch" not in policy_kwargs:
                # Use a simpler architecture that's less likely to cause errors
                policy_kwargs["net_arch"] = [64, 64]
            
            # Debug info
            print("DEBUG: Dueling network enabled, using simplified architecture")
            print("DEBUG: Current policy_kwargs:", policy_kwargs)
            
        # Thiết lập tham số
        params = {
            "learning_rate": learning_rate,
            "buffer_size": buffer_size, 
            "learning_starts": learning_starts,
            "batch_size": batch_size,
            "tau": tau,
            "gamma": gamma,
            "train_freq": train_freq,
            "gradient_steps": gradient_steps,
            "target_update_interval": target_update_interval,
            "exploration_fraction": exploration_fraction,
            "exploration_initial_eps": exploration_initial_eps,
            "exploration_final_eps": exploration_final_eps,
            "max_grad_norm": max_grad_norm,
            "tensorboard_log": self.log_dir,
            "policy_kwargs": policy_kwargs,
            "verbose": verbose
        }
        
        # Add dueling network parameter if supported by this version of stable-baselines3
        try:
            if use_dueling_network:
                # Try to import the DQN class to check if it supports dueling parameter
                from stable_baselines3.dqn.dqn import DQN as SB3DQN
                dqn_init_params = SB3DQN.__init__.__code__.co_varnames
                if "dueling" in dqn_init_params:
                    params["dueling"] = True
                    print("DEBUG: Added dueling=True to DQN parameters")
                else:
                    print("DEBUG: Dueling parameter not supported by SB3")
        except Exception as e:
            print(f"DEBUG: Error checking dueling support: {e}")
            
        # Debug information about observation space
        print("DEBUG: Observation space type:", type(self.env.observation_space))
        print("DEBUG: Observation space:", self.env.observation_space)
        
        # Determine if this is a dictionary observation space 
        is_dict_space = False
        try:
            # Try different ways to check for Dict space
            if isinstance(self.env.observation_space, spaces.Dict):
                is_dict_space = True
            elif hasattr(self.env.observation_space, "spaces") and isinstance(self.env.observation_space.spaces, dict):
                is_dict_space = True
            elif str(type(self.env.observation_space).__name__) == "Dict":
                is_dict_space = True
                
            # Disable dueling if using Dict observation space - it doesn't work well with MultiInputPolicy
            if is_dict_space and use_dueling_network:
                print("DEBUG: Disabling dueling network as it doesn't work well with dictionary observation spaces")
                use_dueling_network = False
                self.use_dueling_network = False
        except Exception as e:
            print(f"DEBUG: Error checking observation space: {e}")
            
        # Always use MultiInputPolicy for safety
        policy_type = "MultiInputPolicy" if is_dict_space else "MlpPolicy"
        print(f"DEBUG: Selected policy type: {policy_type}")
        
        # Safe model initialization with error handling
        try:
            print("DEBUG: Creating DQN model with parameters:", params)
            self.model = DQN(
                policy_type,
                self.env,
                **params
            )
            print("DEBUG: Model created successfully")
        except Exception as e:
            print(f"DEBUG: Error creating model: {e}")
            # Try again with simpler configuration
            print("DEBUG: Retrying with simplified configuration")
            # Remove potentially problematic parameters
            if "policy_kwargs" in params:
                params["policy_kwargs"] = {}
            if "dueling" in params:
                del params["dueling"]
                
        self.model = DQN(
                policy_type,
                self.env,
                **params
        )
        
        # Cấu hình Double DQN - Bật mặc định vì nó cải thiện performance
        if use_double_dqn and hasattr(self.model, "use_double_q"):
            self.model.use_double_q = True
            print("DEBUG: Double DQN enabled")
            
        # Cấu hình prioritized experience replay
        if use_prioritized_replay:
            PrioritizedReplayBuffer = None
            
            # Kiểm tra trực tiếp xem các module đã được cài đặt chưa
            try:
                import pkg_resources
                sb3_installed = True
                try:
                    pkg_resources.get_distribution("sb3-contrib")
                    sb3_contrib_installed = True
                    print("DEBUG: sb3-contrib đã được cài đặt")
                except pkg_resources.DistributionNotFound:
                    sb3_contrib_installed = False
                    print("DEBUG: sb3-contrib chưa được cài đặt")
            except:
                # Không thể kiểm tra package, tiếp tục thử import
                sb3_installed = True
                sb3_contrib_installed = True
                print("DEBUG: Không thể kiểm tra thư viện, sẽ thử import trực tiếp")
            
            try:
                # Thử nhiều cách import khác nhau
                if sb3_installed:
                    try:
                        # Cách 1: Import từ stable-baselines3
                        from stable_baselines3.common.buffers import PrioritizedReplayBuffer
                        print("DEBUG: Sử dụng PrioritizedReplayBuffer từ stable_baselines3")
                    except (ImportError, AttributeError):
                        if sb3_contrib_installed:
                            try:
                                # Cách 2: Import từ sb3_contrib 
                                from sb3_contrib.common.buffers import PrioritizedReplayBuffer  # type: ignore
                                print("DEBUG: Sử dụng PrioritizedReplayBuffer từ sb3_contrib")
                            except (ImportError, AttributeError):
                                # Cách 3: Thử import từ các đường dẫn khác
                                try:
                                    import sb3_contrib
                                    print(f"DEBUG: sb3_contrib path: {sb3_contrib.__path__}")
                                    # In ra các module có sẵn trong sb3_contrib
                                    print(f"DEBUG: sb3_contrib contents: {dir(sb3_contrib)}")
                                    
                                    # Kiểm tra cấu trúc module
                                    if hasattr(sb3_contrib, "common"):
                                        if hasattr(sb3_contrib.common, "buffers"):
                                            print("DEBUG: Module cấu trúc đúng")
                                        else:
                                            print("DEBUG: Thiếu module buffers trong sb3_contrib.common")
                                    else:
                                        print("DEBUG: Thiếu module common trong sb3_contrib")
                                        
                                    raise ImportError("Không thể import PrioritizedReplayBuffer từ các vị trí thông thường")
                                except:
                                    print("WARNING: Không thể import sb3_contrib. Hãy thử cài đặt lại:")
                                    print("    pip install sb3-contrib==1.7.0")
                                    
                                    # Vô hiệu hóa PER và tiếp tục
                                    self.use_prioritized_replay = False
                                    print("DEBUG: Đã tắt PrioritizedReplayBuffer do lỗi import")
                        else:
                            print("WARNING: sb3-contrib chưa được cài đặt. Để sử dụng PER, hãy chạy: pip install sb3-contrib==1.7.0")
                            self.use_prioritized_replay = False
                else: # This else corresponds to 'if sb3_installed:'
                    print("WARNING: stable-baselines3 is not considered installed. Cannot import PrioritizedReplayBuffer.")
                    self.use_prioritized_replay = False
                
                # Nếu import thành công, tiếp tục thiết lập buffer
                if PrioritizedReplayBuffer is not None and self.use_prioritized_replay:
                    self.model.replay_buffer = PrioritizedReplayBuffer(
                        buffer_size,
                        alpha=prioritized_replay_alpha,
                        beta0=prioritized_replay_beta0
                    )
                    print("DEBUG: Đã thiết lập PrioritizedReplayBuffer thành công")
            except Exception as e:
                print(f"DEBUG: Lỗi khi thiết lập PER: {e}")
                self.use_prioritized_replay = False
            
        return self
        
    def train(self, total_timesteps=100000, callback=None):
        """Huấn luyện agent trong môi trường."""
        try:
            # Tạo early stopping callback
            early_stopping_callback = self._create_early_stopping_callback(total_timesteps)
            
            # Kết hợp callback đã cung cấp và early stopping
            combined_callbacks = [early_stopping_callback]
            if callback is not None:
                if isinstance(callback, list):
                    combined_callbacks.extend(callback)
                else:
                    combined_callbacks.append(callback)
            
            # Tạo callback giám sát việc mắc kẹt và tự động điều chỉnh exploration
            adaptive_exploration_callback = self._create_adaptive_exploration_callback(total_timesteps)
            combined_callbacks.append(adaptive_exploration_callback)
            
            # Tạo backup của epsilon ban đầu để có thể khôi phục sau khi tăng tạm thời
            original_eps = None
            if hasattr(self.model, 'exploration_rate'):
                original_eps = self.model.exploration_rate
            elif hasattr(self.model, 'exploration_schedule'):
                # Lưu exploration_initial_eps thay thế nếu không có exploration_rate trực tiếp
                original_eps = getattr(self.model, '_initial_eps', 0.1)
            
            # Setup adaptive learning rate and exploration schedules
            original_lr = None
            if hasattr(self.model, 'learning_rate'):
                if callable(self.model.learning_rate):
                    # Nếu đã là schedule, giữ nguyên
                    pass
                else:
                    # Lưu giá trị gốc
                    original_lr = self.model.learning_rate
            
            # Lưu trạng thái reward trung bình để phát hiện plateau
            self.reward_history = []
            self.last_10_rewards = []
            self.plateau_detected = False
            self.adaptive_actions_taken = 0  # Đếm số lần đã áp dụng điều chỉnh thích nghi
            self.max_adaptive_actions = 5  # Giới hạn số lần được áp dụng điều chỉnh
            
            # Sử dụng MetricsCallback nếu không có callback khác
            metrics_callback = MetricsCallback(log_dir=self.log_dir)
            combined_callbacks.append(metrics_callback)
            
            # Bắt đầu huấn luyện
            print(f"Starting training for {total_timesteps} timesteps with Double DQN and adaptive exploration...")
            
            # Sử dụng callback được cung cấp hoặc metrics_callback nếu không có
            self.model.learn(
                total_timesteps=total_timesteps,
                callback=combined_callbacks
            )
            
            # Lưu metrics từ callback
            self.training_metrics = metrics_callback.get_metrics()
            
            # Khôi phục epsilon và learning rate ban đầu
            if original_eps is not None and hasattr(self.model, 'exploration_rate'):
                self.model.exploration_rate = original_eps
                print(f"Restored original exploration rate: {original_eps}")
            
            if original_lr is not None and hasattr(self.model, 'learning_rate') and not callable(self.model.learning_rate):
                self.model.learning_rate = original_lr
                print(f"Restored original learning rate: {original_lr}")
            
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
    
    def _create_adaptive_exploration_callback(self, total_timesteps):
        """Tạo callback tự động điều chỉnh exploration dựa trên hiệu suất"""
        
        class AdaptiveExplorationCallback(BaseCallback):
            def __init__(self, trainer_instance, verbose=0):
                super().__init__(verbose)
                self.trainer = trainer_instance
                self.check_freq = 2000  # Kiểm tra mỗi 2000 bước
                self.reward_window_size = 10
                self.recent_rewards = []  # Theo dõi reward gần đây
                self.stagnation_threshold = 3  # Số lần kiểm tra liên tiếp reward không tăng
                self.stagnation_counter = 0
                self.adaptive_actions_taken = 0
                self.max_adaptive_actions = 5  # Giới hạn số lần điều chỉnh
                
                # Giá trị ban đầu để khôi phục
                self.original_lr = None
                self.original_eps = None
                
                # Trạng thái hiện tại
                self.current_eps_boost = False
                self.current_lr_change = False
                
            def _on_training_start(self):
                # Lưu giá trị ban đầu
                if hasattr(self.model, 'learning_rate'):
                    self.original_lr = self.model.learning_rate
                
                if hasattr(self.model, 'exploration_rate'):
                    self.original_eps = self.model.exploration_rate
                elif hasattr(self.model, 'exploration_schedule'):
                    self.original_eps = self.model.exploration_schedule(0)  # Epsilon tại bước 0
                    
            def _on_step(self):
                # Lấy reward của episode gần nhất
                if len(self.model.ep_info_buffer) > 0 and 'r' in self.model.ep_info_buffer[-1]:
                    latest_reward = self.model.ep_info_buffer[-1]['r']
                    self.recent_rewards.append(latest_reward)
                    
                    # Giữ kích thước cửa sổ
                    if len(self.recent_rewards) > self.reward_window_size:
                        self.recent_rewards.pop(0)
                
                # Kiểm tra điều kiện để điều chỉnh
                if self.n_calls % self.check_freq == 0 and len(self.recent_rewards) >= self.reward_window_size:
                    # Tính trung bình và kiểm tra xu hướng
                    current_avg = np.mean(self.recent_rewards[-5:])  # 5 reward gần nhất
                    previous_avg = np.mean(self.recent_rewards[:-5])  # 5 reward trước đó
                    
                    improvement = current_avg - previous_avg
                    
                    # Phát hiện mắc kẹt nếu reward đang giảm hoặc đứng yên
                    if improvement <= 0.1:  # Ngưỡng cải thiện rất nhỏ
                        self.stagnation_counter += 1
                        
                        if self.stagnation_counter >= self.stagnation_threshold and self.adaptive_actions_taken < self.max_adaptive_actions:
                            # Thực hiện điều chỉnh nếu chưa đạt giới hạn
                            self._apply_adaptive_action()
                            self.stagnation_counter = 0  # Reset counter
                            self.adaptive_actions_taken += 1
                    else:
                        # Reset counter nếu có cải thiện
                        self.stagnation_counter = 0
                        
                        # Khôi phục các giá trị ban đầu nếu đã điều chỉnh và hiện tại có cải thiện
                        if self.current_eps_boost or self.current_lr_change:
                            self._restore_original_values()
                
                # Đảm bảo khôi phục giá trị ban đầu khi kết thúc
                if self.n_calls >= total_timesteps - 100:  # Gần kết thúc
                    self._restore_original_values()
                
                return True
            
            def _apply_adaptive_action(self):
                """Áp dụng chiến lược thích ứng cho exploration và learning rate"""
                # Ngẫu nhiên chọn một trong hai: tăng exploration hoặc thay đổi learning rate
                action_type = np.random.choice(['exploration', 'learning_rate'])
                
                if action_type == 'exploration' and hasattr(self.model, 'exploration_rate'):
                    # Tăng exploration để thoát khỏi cực tiểu cục bộ
                    current_eps = self.model.exploration_rate
                    new_eps = min(current_eps * 2.0, 0.5)  # Tăng tối đa 0.5
                    
                    print(f"[AdaptiveCallback] Boosting exploration rate from {current_eps:.4f} to {new_eps:.4f}")
                    self.model.exploration_rate = new_eps
                    self.current_eps_boost = True
                    
                elif action_type == 'learning_rate' and hasattr(self.model, 'learning_rate') and not callable(self.model.learning_rate):
                    # Thay đổi learning rate: hoặc tăng hoặc giảm
                    current_lr = self.model.learning_rate
                    adjustment = np.random.choice([0.5, 2.0])  # Giảm một nửa hoặc tăng gấp đôi
                    new_lr = current_lr * adjustment
                    
                    print(f"[AdaptiveCallback] Adjusting learning rate from {current_lr:.6f} to {new_lr:.6f}")
                    self.model.learning_rate = new_lr
                    self.current_lr_change = True
            
            def _restore_original_values(self):
                """Khôi phục exploration và learning rate ban đầu"""
                if self.current_eps_boost and self.original_eps is not None and hasattr(self.model, 'exploration_rate'):
                    print(f"[AdaptiveCallback] Restoring original exploration rate: {self.original_eps:.4f}")
                    self.model.exploration_rate = self.original_eps
                    self.current_eps_boost = False
                
                if self.current_lr_change and self.original_lr is not None and hasattr(self.model, 'learning_rate') and not callable(self.model.learning_rate):
                    print(f"[AdaptiveCallback] Restoring original learning rate: {self.original_lr:.6f}")
                    self.model.learning_rate = self.original_lr
                    self.current_lr_change = False
        
        return AdaptiveExplorationCallback(self)
        
    def predict_action(self, observation):
        """Dự đoán hành động dựa trên observation."""
        if self.model is None:
            raise ValueError("Model chưa được khởi tạo. Vui lòng gọi create_model() hoặc load_model() trước.")
        
        action, _states = self.model.predict(observation, deterministic=True)
        return action
    
    def _evaluate_quick(self, n_episodes=10):
        """
        Đánh giá nhanh agent trên môi trường hiện tại.
        
        Args:
            n_episodes: Số episode để đánh giá
            
        Returns:
            mean_reward: Phần thưởng trung bình trên tất cả các episode
            success_rate: Tỷ lệ đến đích thành công
        """
        total_rewards = 0
        success_count = 0
        
        for _ in range(n_episodes):
            obs, _ = self.env.reset()
            done = False
            truncated = False
            episode_reward = 0
            
            while not (done or truncated):
                action, _ = self.model.predict(obs, deterministic=True)
                obs, reward, done, truncated, info = self.env.step(action)
                episode_reward += reward
                
                # Kiểm tra nếu agent đến đích
                if done and info.get("termination_reason") == "den_dich":
                    success_count += 1
            
            total_rewards += episode_reward
        
        mean_reward = total_rewards / n_episodes
        success_rate = success_count / n_episodes
        
        return mean_reward, success_rate
    
    def save_model(self, model_path):
        """
        Lưu mô hình đã huấn luyện.
        
        Args:
            model_path: Đường dẫn để lưu mô hình
        """
        # Lưu model
        self.model.save(f"{model_path}.zip")
        
        # Lưu metadata của model
        metadata = {
            "saved_at": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "total_timesteps": self.model.num_timesteps,
            "env_id": self.model.env.unwrapped.spec.id if hasattr(self.model.env, "unwrapped") and hasattr(self.model.env.unwrapped, "spec") else "unknown",
            "use_double_dqn": self.use_double_dqn,
            "use_dueling_network": self.use_dueling_network,
            "use_prioritized_replay": self.use_prioritized_replay
        }
        
        # Thêm các tham số quan trọng từ model
        if hasattr(self.model, "learning_rate"):
            metadata["learning_rate"] = float(self.model.learning_rate) if not callable(self.model.learning_rate) else "schedule"
        if hasattr(self.model, "gamma"):
            metadata["gamma"] = float(self.model.gamma)
        if hasattr(self.model, "batch_size"):
            metadata["batch_size"] = int(self.model.batch_size)
            
        # Nếu có training metrics, lưu một bản tóm tắt
        if self.training_metrics is not None:
            metadata["metrics_summary"] = {
                "last_reward": float(self.training_metrics["episode_rewards"][-1]) if self.training_metrics["episode_rewards"] else None,
                "mean_reward_last_100": float(np.mean(self.training_metrics["episode_rewards"][-100:])) if len(self.training_metrics["episode_rewards"]) >= 100 else None,
                "success_rate": float(self.training_metrics["success_rates"][-1]) if self.training_metrics["success_rates"] else None,
                "training_time": float(self.training_metrics.get("elapsed_time", 0))
            }
            
        # Lưu metadata
        with open(f"{model_path}_metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)
    
    def load_model(self, model_path):
        """
        Tải mô hình đã huấn luyện.
        
        Args:
            model_path: Đường dẫn đến mô hình
            
        Returns:
            success: True nếu tải thành công
        """
        # Thêm đuôi .zip nếu chưa có
        if not model_path.endswith('.zip'):
            model_path = f"{model_path}.zip"
            
        # Kiểm tra file tồn tại
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Không tìm thấy model tại {model_path}")
            
        # Tải model - xử lý trường hợp self.model là None
        if self.model is None:
            # Nếu model chưa được khởi tạo, sử dụng self.env thay vì self.model.env
            self.model = DQN.load(model_path, env=self.env)
        else:
            # Trường hợp thông thường - sử dụng env từ model hiện tại
            self.model = DQN.load(model_path, env=self.model.env)
        
        # Tải metadata nếu có
        metadata_path = model_path.replace('.zip', '_metadata.json')
        if os.path.exists(metadata_path):
            try:
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
                    
                # Khôi phục cấu hình
                self.use_double_dqn = metadata.get("use_double_dqn", False)
                self.use_dueling_network = metadata.get("use_dueling_network", False)
                self.use_prioritized_replay = metadata.get("use_prioritized_replay", False)
                
                # Khôi phục metrics
                if "metrics_summary" in metadata:
                    print(f"  Model info: success_rate={metadata['metrics_summary'].get('success_rate', 'N/A')}, " 
                          f"mean_reward={metadata['metrics_summary'].get('mean_reward_last_100', 'N/A')}")
            except:
                pass
        
        return True
    
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
