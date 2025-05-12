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
            "timesteps": []
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
            
            if "l" in self.model.ep_info_buffer[-1]:
                self.metrics["episode_lengths"].append(self.model.ep_info_buffer[-1]["l"])
            
            if "success" in self.model.ep_info_buffer[-1]:
                success = 1 if self.model.ep_info_buffer[-1]["success"] else 0
                self.episode_success.append(success)
                
                # Tính success rate trên cửa sổ trượt
                if len(self.episode_success) > self.success_window_size:
                    self.episode_success.pop(0)
                
                success_rate = sum(self.episode_success) / len(self.episode_success)
                self.metrics["success_rates"].append(success_rate)
        
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
        
        # Ghi lại loss
        if hasattr(self.model.policy, "raw_loss"):
            self.metrics["losses"].append(float(self.model.policy.raw_loss))
        
        # Ghi lại timestep
        self.metrics["timesteps"].append(self.num_timesteps)
        
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
        
    def create_model(self, learning_rate=0.0003, buffer_size=100000, learning_starts=10000, 
                    batch_size=128, tau=0.005, gamma=0.99, train_freq=4, gradient_steps=1,
                    target_update_interval=10000, exploration_fraction=0.2, 
                    exploration_initial_eps=1.0, exploration_final_eps=0.05, max_grad_norm=10,
                    policy_kwargs=None, verbose=1, use_double_dqn=False, use_dueling_network=False,
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
        
        # Cấu hình double DQN
        if use_double_dqn and hasattr(self.model, "use_double_q"):
            self.model.use_double_q = True
            
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
        
    def train(self, total_timesteps, callback=None):
        """
        Huấn luyện agent.
        
        Args:
            total_timesteps: Tổng số bước huấn luyện
            callback: Callback tùy chỉnh để sử dụng trong quá trình huấn luyện
                      Có thể là một đối tượng BaseCallback hoặc một hàm callback đơn giản
        """
        # Tạo metrics callback
        metrics_callback = MetricsCallback(log_dir=self.log_dir)
        
        # Wrap function callback in a proper BaseCallback object if needed
        from stable_baselines3.common.callbacks import BaseCallback
        
        if callback is not None:
            if callable(callback) and not isinstance(callback, BaseCallback):
                # It's a function callback, we need to wrap it
                print("DEBUG: Wrapping function callback in FunctionCallback")
                
                class FunctionCallback(BaseCallback):
                    def __init__(self, callback_fn, verbose=0):
                        super().__init__(verbose)
                        self.callback_fn = callback_fn
                        
                    def _on_step(self):
                        # Call the function with local and global variables
                        # as stable-baselines would do in older versions
                        locals_dict = {
                            "self": self,
                            "num_timesteps": self.num_timesteps,
                            "step": self.num_timesteps,
                            "model": self.model,
                            "_locals": {},
                            "_globals": {}
                        }
                        result = self.callback_fn(locals_dict, {})
                        # Function should return True to continue training
                        return False if result is False else True
                
                callback = FunctionCallback(callback)
                
            # Now combine with metrics callback
            from stable_baselines3.common.callbacks import CallbackList
            callbacks = CallbackList([metrics_callback, callback])
        else:
            callbacks = metrics_callback
            
        # Huấn luyện mô hình
        self.model.learn(total_timesteps=total_timesteps, callback=callbacks)
        
        # Lưu metrics sau khi huấn luyện
        self.training_metrics = metrics_callback.get_metrics()
        metrics_callback.save_metrics()
        
    def predict_action(self, observation):
        """
        Dự đoán hành động dựa trên quan sát.
        
        Args:
            observation: Trạng thái quan sát hiện tại
            
        Returns:
            action: Hành động được dự đoán
        """
        action, _ = self.model.predict(observation, deterministic=True)
        return action
    
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
    
    def evaluate(self, env, n_episodes=10):
        """
        Đánh giá agent trên môi trường.
        
        Args:
            env: Môi trường đánh giá
            n_episodes: Số lượng episode đánh giá
            
        Returns:
            results: Dictionary chứa kết quả đánh giá
        """
        rewards = []
        episode_lengths = []
        success_count = 0
        fuel_remaining = []
        money_remaining = []
        
        for i in range(n_episodes):
            obs, _ = env.reset()
            done = False
            truncated = False
            episode_reward = 0
            step_count = 0
            
            while not (done or truncated):
                action = self.predict_action(obs)
                obs, reward, done, truncated, info = env.step(action)
                episode_reward += reward
                step_count += 1
            
            rewards.append(episode_reward)
            episode_lengths.append(step_count)
            
            # Ghi nhận thành công/thất bại
            if "termination_reason" in info and info["termination_reason"] == "den_dich":
                success_count += 1
                if "truck_state" in info:
                    fuel_remaining.append(info["truck_state"]["fuel"])
                    money_remaining.append(info["truck_state"]["money"])
            
        # Tính toán các chỉ số
        success_rate = success_count / n_episodes
        avg_reward = sum(rewards) / n_episodes
        avg_episode_length = sum(episode_lengths) / n_episodes
        
        results = {
            "success_rate": success_rate,
            "avg_reward": avg_reward,
            "avg_episode_length": avg_episode_length,
            "n_episodes": n_episodes
        }
        
        # Thêm thông tin về nhiên liệu và tiền nếu có
        if fuel_remaining:
            results["avg_fuel_remaining"] = sum(fuel_remaining) / len(fuel_remaining)
        if money_remaining:
            results["avg_money_remaining"] = sum(money_remaining) / len(money_remaining)
            
        return results 
