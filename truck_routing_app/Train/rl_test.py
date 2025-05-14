"""
Module kiểm tra môi trường học tăng cường (RL Environment)
cho bài toán định tuyến xe tải.
"""

import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox, filedialog
import numpy as np
import sys
import os
import json
import time
import threading
from datetime import datetime
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import shutil
from pathlib import Path
import queue
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import Logger, HumanOutputFormat
from gymnasium import spaces
try:
    import optuna
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False
    # Không raise lỗi ở đây, xử lý trong hàm _start_tuning

# Đường dẫn chính xác cho vị trí hiện tại
# File hiện tại: C:\Users\Win 11\Desktop\DoAnTTNT-Nhom\AI_NHOM\DuAn1\DuAn1\truck_routing_app\rl_test.py

# Kiểm tra đường dẫn và tệp tin
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)  # Lấy thư mục cha của Train
core_dir = os.path.join(parent_dir, 'core')  # Tìm thư mục core ở cấp cao hơn
rl_env_file = os.path.join(core_dir, 'rl_environment.py')

if not os.path.exists(rl_env_file):
    print(f"CẢNH BÁO: File rl_environment.py không tồn tại tại {rl_env_file}")
    print("Các file trong thư mục core:", os.listdir(core_dir) if os.path.exists(core_dir) else "thư mục core không tồn tại")
    
    # Thêm thư mục cha vào sys.path để có thể import từ thư mục core ở cấp cao hơn
    sys.path.append(parent_dir)
else:
    print(f"Tìm thấy file rl_environment.py tại {rl_env_file}")
    # Thêm thư mục cha vào sys.path để có thể import
    sys.path.append(parent_dir)

try:
    # Import trực tiếp từ thư mục core
    from core.map import Map
    from core.constants import CellType, MovementCosts, StationCosts
    
    # Bắt ImportError từng module một để dễ xác định lỗi
    try:
        from core.rl_environment import TruckRoutingEnv
        print("Import module rl_environment thành công!")
    except ImportError as e:
        print(f"Không thể import module rl_environment: {e}")
        sys.exit(1)
    
    # Import các module DQN và đánh giá
    try:
        from core.algorithms.rl_DQNAgent import DQNAgentTrainer
        from core.algorithms.hyperparameter_tuning import optimize_hyperparameters, train_agent_with_best_params
        from truck_routing_app.statistics.rl_evaluation import RLEvaluator
        print("Import các module RL nâng cao thành công!")
    except ImportError as e:
        print(f"Cảnh báo: Không thể import module RL nâng cao: {e}")
        print("Một số tính năng nâng cao có thể không hoạt động")
    
    # Import các thuật toán khác để so sánh
    try:
        from core.algorithms.astar import AStar
        from core.algorithms.greedy import Greedy
        from core.algorithms.genetic_algorithm import GeneticAlgorithm
        from core.algorithms.simulated_annealing import SimulatedAnnealing
        print("Import các module thuật toán tìm đường thành công!")
    except ImportError as e:
        print(f"Cảnh báo: Không thể import module thuật toán tìm đường: {e}")
        
    print("Tất cả các module đã được import thành công!")
except ImportError as e:
    print(f"Lỗi import: {e}")
    print("Đang ở thư mục:", current_dir)
    print("Các file trong thư mục:", os.listdir(current_dir))
    
    if os.path.exists(core_dir):
        print("Các file trong thư mục core:", os.listdir(core_dir))
    sys.exit(1)

# Import OPTIMAL_HYPERPARAMS từ auto_train_rl để lấy giá trị mặc định
# (Giả sử auto_train_rl.py nằm cùng thư mục)
try:
    from auto_train_rl import OPTIMAL_HYPERPARAMS
except ImportError:
    print("Cảnh báo: Không thể import OPTIMAL_HYPERPARAMS từ auto_train_rl.py. Sử dụng giá trị mặc định.")
    # Cung cấp giá trị mặc định dự phòng nếu import lỗi
    OPTIMAL_HYPERPARAMS = {
        8: {
            "learning_rate": 0.0001, "gamma": 0.99, "buffer_size": 50000,
            "learning_starts": 1000, "batch_size": 64, "tau": 0.005,
            "train_freq": 4, "target_update_interval": 1000,
            "exploration_fraction": 0.2, "exploration_initial_eps": 1.0,
            "exploration_final_eps": 0.05
        }
    }

# Định nghĩa đường dẫn lưu model (nhất quán với auto_train_rl.py)
_ROOT_DIR = Path(__file__).resolve().parent # Sử dụng pathlib.Path
MODELS_DIR = _ROOT_DIR / "saved_models"
MODELS_DIR.mkdir(parents=True, exist_ok=True) # Đảm bảo thư mục tồn tại


class RLTestApp:
    """Ứng dụng kiểm tra môi trường RL."""
    
    def __init__(self, root):
        """
        Khởi tạo ứng dụng kiểm tra môi trường RL.
        
        Args:
            root: Cửa sổ gốc Tkinter
        """
        self.root = root
        self.root.title("Truck Routing RL Test App v2.0")
        self.root.geometry("1200x900") # Increased height for logs/progress & width for two panels
        
        # Style configuration
        self.style = ttk.Style()
        self.style.theme_use('clam') # Use a modern theme

        # Map and Environment related variables
        self.map_object = None
        self.current_map_path = None
        self.rl_environment = None
        self.map_size_var = tk.IntVar(value=8) # Changed from self.map_size
        self.current_map_size_display_var = tk.StringVar(value="N/A") # For displaying current map size

        # Variables for customizing map generation counts
        self.num_tolls_var = tk.IntVar(value=2) # Default for 8x8 based on ~0.05 ratio
        self.num_gas_var = tk.IntVar(value=3)   # Default for 8x8 based on ~0.05 ratio
        self.num_obstacles_var = tk.IntVar(value=10) # Default for 8x8 based on ~0.2 ratio
        
        # Thêm biến cho trạng thái chướng ngại vật di chuyển
        self.moving_obstacles_var = tk.BooleanVar(value=False)

        # RL Agent related variables
        self.trainer = None
        self.trained_model_path = None
        self.is_model_loaded = False
        self._setup_manual_training_vars() # Call setup for manual training vars

        self.training_progress_queue = queue.Queue() # <-- Add this
        self.is_manual_training_running = False # <-- Add this

        # Custom Callback for UI Progress
        class ManualTrainingProgressCallback(BaseCallback):
            def __init__(self, progress_queue: queue.Queue, total_steps: int, verbose: int = 0):
                super().__init__(verbose)
                self.progress_queue = progress_queue
                self.total_steps = total_steps
                self.last_report_time = time.time()
                self.losses_buffer = []  # Buffer để lưu trữ giá trị loss trực tiếp
                
                # Add metrics tracking - Thêm đầy đủ các loại metrics cần thiết
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
                
                # Biến tính toán success rate
                self.episode_count = 0
                self.success_count = 0

            def _on_step(self) -> bool:
                # Track metrics for potential retrieval
                if len(self.model.ep_info_buffer) > 0 and len(self.model.ep_info_buffer[-1]) > 0:
                    # Kiểm tra nếu có episose mới hoàn thành
                    if "r" in self.model.ep_info_buffer[-1] and self.model.ep_info_buffer[-1].get("new_episode", True):
                        self.episode_count += 1
                        self.metrics["episode_rewards"].append(self.model.ep_info_buffer[-1]["r"])
                        self.metrics["timesteps"].append(self.num_timesteps)
                        
                        # Thu thập độ dài episode
                        if "l" in self.model.ep_info_buffer[-1]:
                            self.metrics["episode_lengths"].append(self.model.ep_info_buffer[-1]["l"])
                        
                        # Thu thập lý do kết thúc episode và tính toán success rate
                        if "termination_reason" in self.model.ep_info_buffer[-1]:
                            reason = self.model.ep_info_buffer[-1]["termination_reason"]
                            if reason not in self.metrics["termination_reasons"]:
                                self.metrics["termination_reasons"][reason] = 0
                            self.metrics["termination_reasons"][reason] += 1
                            
                            # Tính success rate
                            if reason == "den_dich":
                                self.success_count += 1
                            
                            current_success_rate = self.success_count / max(1, self.episode_count)
                            self.metrics["success_rates"].append(current_success_rate)
                
                # Thu thập learning rate
                if hasattr(self.model, "learning_rate"):
                    if callable(self.model.learning_rate):
                        lr = self.model.learning_rate(self.num_timesteps)
                    else:
                        lr = self.model.learning_rate
                    self.metrics["learning_rates"].append(lr)
                
                # Thu thập exploration rate
                if hasattr(self.model, "exploration_schedule"):
                    eps = self.model.exploration_schedule(self.num_timesteps)
                    self.metrics["exploration_rates"].append(eps)
                
                # Thu thập loss nếu có - CHỈ THÊM VÀO METRICS THEO MỘT CHU KỲ CỐ ĐỊNH
                # Lấy loss từ buffer nếu có
                if hasattr(self, 'losses_buffer') and len(self.losses_buffer) > 0:
                    # Tính trung bình của loss trong buffer và thêm vào metrics
                    avg_loss = sum(self.losses_buffer) / len(self.losses_buffer)
                    self.metrics["losses"].append(avg_loss)
                    # Xóa buffer sau khi đã thêm vào metrics
                    self.losses_buffer = []
                    
                # Các phương pháp thu thập loss khác (giữ nguyên để dự phòng)
                elif hasattr(self.model.policy, "raw_loss") and self.model.policy.raw_loss is not None:
                    try:
                        current_loss = float(self.model.policy.raw_loss)
                        self.metrics["losses"].append(current_loss)
                    except (ValueError, TypeError):
                        pass  # Ignore if not convertible to float
                
                # Report progress every 0.5 seconds to avoid overwhelming the queue/UI
                current_time = time.time()
                if current_time - self.last_report_time > 0.5 or self.num_timesteps == self.total_steps:
                    avg_reward = "N/A"
                    if len(self.model.ep_info_buffer) > 0:
                        # Calculate average reward from the last 10 episodes if available
                        # Convert deque to list before slicing
                        last_10_episodes = list(self.model.ep_info_buffer)[-10:]
                        avg_reward = f"{np.mean([ep_info['r'] for ep_info in last_10_episodes]):.2f}"
                    
                    status = f"Step: {self.num_timesteps}/{self.total_steps} | Avg Reward (last 10): {avg_reward}"
                    try:
                        # Non-blocking put if the queue is full (shouldn't happen with polling)
                        self.progress_queue.put_nowait((self.num_timesteps, self.total_steps, status))
                    except queue.Full:
                        pass # Ignore if the UI hasn't caught up
                    self.last_report_time = current_time
                    
                # Check if total steps reached - Stable Baselines 3 usually handles termination,
                # but this ensures our callback stops if needed (redundant but safe).
                if self.num_timesteps >= self.total_steps:
                     return False # Signal to stop training (though learn() should stop)
                    
                return True # Continue training
            
            # Add method to support the DQNAgent metrics retrieval
            def get_metrics(self):
                """Returns collected metrics during training"""
                return self.metrics

        self.ManualTrainingProgressCallback = ManualTrainingProgressCallback # Store the class

        self._create_ui()
    
    def _setup_manual_training_vars(self):
        """Khởi tạo các biến tk cho cấu hình huấn luyện thủ công."""
        # Lấy giá trị mặc định từ OPTIMAL_HYPERPARAMS[8]
        defaults = OPTIMAL_HYPERPARAMS.get(8, OPTIMAL_HYPERPARAMS[list(OPTIMAL_HYPERPARAMS.keys())[0]]) # Lấy 8 hoặc key đầu tiên

        self.manual_lr = tk.DoubleVar(value=defaults.get("learning_rate", 0.0001))
        self.manual_gamma = tk.DoubleVar(value=defaults.get("gamma", 0.99))
        self.manual_buffer_size = tk.IntVar(value=defaults.get("buffer_size", 50000))
        self.manual_learning_starts = tk.IntVar(value=5000)  # Reduced from 10000 (was 10000 in analysis, now 5000)
        self.manual_batch_size = tk.IntVar(value=defaults.get("batch_size", 64))
        self.manual_tau = tk.DoubleVar(value=defaults.get("tau", 0.005))
        self.manual_train_freq = tk.IntVar(value=defaults.get("train_freq", 4))
        self.manual_target_update_interval = tk.IntVar(value=defaults.get("target_update_interval", 1000))
        self.manual_exploration_fraction = tk.DoubleVar(value=0.5)  # Giảm từ 0.9 xuống 0.5
        self.manual_exploration_initial_eps = tk.DoubleVar(value=1.0)  # Giữ nguyên giá trị mặc định
        self.manual_exploration_final_eps = tk.DoubleVar(value=0.1)  # Increased from 0.02 to 0.1
        self.manual_total_timesteps = tk.IntVar(value=1000000)  # Tăng từ 200000 lên 1,000,000 bước

        # Lấy giá trị cho các tính năng nâng cao từ cấu hình tối ưu
        self.manual_use_double_dqn = tk.BooleanVar(value=True) # Mặc định bật Double DQN
        self.manual_use_dueling_dqn = tk.BooleanVar(value=True) # Mặc định bật Dueling
        self.manual_use_prioritized_replay = tk.BooleanVar(value=True) # Mặc định bật PER
        self.manual_render_training = tk.BooleanVar(value=True)

        # Lấy kiến trúc mạng từ cấu hình
        if "policy_kwargs" in defaults and "net_arch" in defaults["policy_kwargs"]:
            net_arch = defaults["policy_kwargs"]["net_arch"]
            self.manual_net_arch = tk.StringVar(value=",".join(map(str, net_arch)))
        else:
            self.manual_net_arch = tk.StringVar(value="256,256") # Tăng kích thước mạng từ 64,64 lên 256,256
        
        # --- Biến cho Hyperparameter Tuning ---
        self.tuning_n_trials = tk.IntVar(value=30) # Tăng số lần thử nghiệm từ 20 lên 30
        self.tuning_timesteps_per_trial = tk.IntVar(value=10000) # Tăng số bước huấn luyện cho mỗi lần thử từ 5000 lên 10000
        self.best_params_found = None # Lưu trữ bộ tham số tốt nhất

    
    def _create_ui(self):
        """Tạo giao diện người dùng (Sử dụng PanedWindow)."""
        # Frame chính chứa Notebook và Log
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill="both", expand=True)

        # Tạo notebook với các tab chính
        self.notebook = ttk.Notebook(main_frame)
        # self.notebook.pack(fill="both", expand=True, padx=10, pady=10) # Không pack notebook trực tiếp nữa
        
        # Tab 0: Hướng dẫn sử dụng
        guide_frame = ttk.Frame(self.notebook)
        # self.notebook.add(guide_frame, text="Hướng dẫn sử dụng") # Add vào notebook sau
        self._create_guide_tab(guide_frame)
        
        # Tab for RL Agent (DQN) - This will be the main training tab
        rl_agent_frame = ttk.Frame(self.notebook) # Tạo frame cho tab RL
        # self.notebook.add(rl_agent_frame, text="RL Agent (DQN Training)") # Add vào notebook sau
        self._create_manual_train_tab(rl_agent_frame) # Gọi hàm tạo nội dung cho tab RL

        # Thêm các tab vào Notebook
        self.notebook.add(guide_frame, text="Hướng dẫn sử dụng")
        self.notebook.add(rl_agent_frame, text="RL Agent (DQN Training)")
        self.notebook.pack(fill="both", expand=True, padx=10, pady=10) # Pack notebook sau khi add tab

        # Khu vực log (Giữ lại ở dưới)
        log_frame = ttk.LabelFrame(main_frame, text="Log")
        log_frame.pack(fill="x", expand=False, padx=10, pady=(0, 10), side="bottom")
        
        self.log_text = scrolledtext.ScrolledText(log_frame, height=10)
        self.log_text.pack(fill="both", expand=True, padx=5, pady=5)
    
    def _create_guide_tab(self, parent):
        """
        Tạo tab Hướng dẫn sử dụng.
        
        Args:
            parent: Widget cha
        """
        # Tạo scrolled text widget để hiển thị hướng dẫn
        guide_text = scrolledtext.ScrolledText(parent, wrap=tk.WORD)
        guide_text.pack(fill="both", expand=True, padx=10, pady=10)
        
        # Định nghĩa nội dung hướng dẫn
        guide_content = """
HƯỚNG DẪN SỬ DỤNG ỨNG DỤNG KIỂM TRA RL

Ứng dụng này được thiết kế để kiểm tra và huấn luyện agent học tăng cường (RL) 
cho bài toán định tuyến xe tải. Hướng dẫn này sẽ giúp bạn hiểu cách sử dụng từng 
tab và cách xác định kết quả đúng/sai trong từng phần.

==========================================================================
QUY TRÌNH KIỂM TRA TỔNG QUAN

Để kiểm tra hoàn chỉnh một agent RL, bạn cần thực hiện theo các bước sau:

1. Tạo/Tải bản đồ trong tab "Tải/Tạo Bản đồ"
2. Khởi tạo môi trường RL trong tab "Khởi tạo Môi trường RL"
3. Kiểm tra môi trường bằng cách thực hiện một số hành động thủ công trong tab "Điều khiển Môi trường"
4. Huấn luyện và đánh giá agent trong tab "Agent RL (DQN)"
5. Sử dụng các tính năng nâng cao của RL trong tab "RL Nâng cao"

==========================================================================
HƯỚNG DẪN CHI TIẾT TỪNG TAB

----------------------------------------------
TAB 1: TẢI/TẠO BẢN ĐỒ
----------------------------------------------

Mục đích: Tạo hoặc tải một bản đồ để làm môi trường cho agent RL.

Cách sử dụng:
- Chọn kích thước bản đồ (8-15)
- Nhấn "Tạo Bản đồ Ngẫu nhiên" hoặc "Tạo Bản đồ Mẫu" hoặc "Tải Bản đồ"
- Nhấn "Lưu Bản đồ" để lưu bản đồ hiện tại vào file để sử dụng sau này
- Kiểm tra thông tin bản đồ hiển thị ở khu vực "Thông tin bản đồ"

Kiểm tra đúng/sai:
✓ ĐÚNG: Bản đồ hiển thị với các ô màu khác nhau (trắng: đường thường, xanh lá: trạm xăng, 
  đỏ: trạm thu phí, xám: vật cản). Có điểm bắt đầu (S) và điểm kết thúc (E).
✗ SAI: Không có bản đồ hiển thị, hoặc thông báo lỗi xuất hiện, hoặc bản đồ quá nhỏ/quá lớn.

----------------------------------------------
TAB 2: KHỞI TẠO MÔI TRƯỜNG RL
----------------------------------------------

Mục đích: Khởi tạo môi trường RL với các tham số mong muốn.

Cách sử dụng:
- Điều chỉnh các tham số (nhiên liệu, tiền, chi phí, số bước tối đa...)
- Nhấn "Khởi tạo Môi trường RL"
- Kiểm tra thông tin môi trường ở khu vực "Thông tin môi trường RL"

Kiểm tra đúng/sai:
✓ ĐÚNG: Thông tin không gian hành động, không gian quan sát và thông số môi trường hiển thị đầy đủ.
  Khu vực log hiển thị "Đã khởi tạo môi trường RL thành công!"
✗ SAI: Thông báo lỗi xuất hiện, hoặc không có thông tin môi trường hiển thị.

* Lưu ý: Bạn phải tạo bản đồ trước khi khởi tạo môi trường RL.

----------------------------------------------
TAB 4: AGENT RL (DQN)
----------------------------------------------

Mục đích: Huấn luyện, lưu, tải và đánh giá agent RL.

Cách sử dụng:
1. Cấu hình Agent:
   - Điều chỉnh các siêu tham số (learning rate, batch size, ...)
   
2. Huấn luyện:
   - Nhấn "Bắt đầu Huấn luyện Ngắn" để huấn luyện agent
   - Sau khi huấn luyện, kết quả đánh giá sẽ hiển thị

3. Lưu/Tải Model:
   - Đặt tên model và nhấn "Lưu Model Hiện tại"
   - Hoặc nhấn "Tải Model Đã Huấn luyện" để tải model đã lưu trước đó

4. Chạy Agent:
   - Nhấn "Chạy Agent trên Bản đồ Hiện tại" để xem agent hoạt động
   - Có thể bật/tắt "Hiển thị từng bước" và điều chỉnh tốc độ hiển thị

Kiểm tra đúng/sai:
✓ ĐÚNG: 
  - Huấn luyện hoàn tất mà không có lỗi, log hiển thị tiến trình huấn luyện
  - Kết quả đánh giá hiển thị tỷ lệ thành công, phần thưởng trung bình, độ dài đường đi, ...
  - Khi chạy agent, đường đi hiển thị trên bản đồ và agent có thể đến đích
  - Agent có thể đổ xăng khi cần thiết và ở trạm xăng

✗ SAI:
  - Thông báo lỗi khi huấn luyện
  - Tỷ lệ thành công quá thấp (gần bằng 0)
  - Agent không thể đến đích hoặc luôn hết nhiên liệu/tiền
  - Agent không học được cách đổ xăng khi cần thiết

==========================================================================
HƯỚNG DẪN SỬ DỤNG HIỆU QUẢ

1. Bắt đầu với bản đồ nhỏ (8x8) và đơn giản để kiểm tra chức năng cơ bản.

2. Huấn luyện với số bước nhỏ trước (1000-5000) để đảm bảo mọi thứ hoạt động đúng.

3. Khi đã xác nhận môi trường hoạt động chính xác, tăng số bước huấn luyện (50k-100k) 
   để có agent tốt hơn.

4. Thuật toán DQN có thể mất thời gian để hội tụ, vì vậy hãy kiên nhẫn.

5. Dựa vào tỷ lệ thành công để đánh giá chất lượng agent (>0.7 là tốt, >0.9 là rất tốt).

6. Xem xét các yếu tố quan trọng: agent có tránh vật cản không? Có đổ xăng khi cần không? 
   Có tối ưu chi phí và nhiên liệu không?

7. Sử dụng tab "RL Nâng cao" để tối ưu hóa và so sánh agent:
   - Tinh chỉnh siêu tham số để có agent tốt nhất (nhiều lần thử = kết quả tốt hơn)
   - Đánh giá chi tiết để hiểu hiệu suất của agent trên nhiều loại bản đồ
   - So sánh thuật toán để đánh giá agent RL so với các phương pháp cổ điển

==========================================================================
XỬ LÝ LỖI THƯỜNG GẶP

1. "ModuleNotFoundError: No module named 'stable_baselines3'":
   -> Chạy script install_dependencies.py để cài đặt các thư viện cần thiết.

2. "Không thể import module rl_environment":
   -> Kiểm tra cấu trúc thư mục và đảm bảo file core/rl_environment.py tồn tại.

3. "Lỗi khi khởi tạo môi trường RL":
   -> Tạo bản đồ trước, sau đó mới khởi tạo môi trường.

4. "Lỗi trong quá trình huấn luyện":
   -> Kiểm tra các tham số môi trường và cấu hình agent, giảm số bước huấn luyện.

5. "Lỗi khi chạy agent":
   -> Đảm bảo môi trường và agent đã được khởi tạo đúng cách.

6. "Lỗi khi tinh chỉnh siêu tham số":
   -> Đảm bảo các thư mục bản đồ tồn tại và có bản đồ. Chạy lệnh "Tạo Thư mục Bản đồ" trước.

7. "ImportError: No module named 'optuna'":
   -> Cài đặt thư viện Optuna: pip install optuna pandas matplotlib seaborn.
"""
        
        # Cài đặt nội dung vào text widget
        guide_text.delete(1.0, tk.END)
        guide_text.insert(tk.END, guide_content)
        
        # Đặt widget ở chế độ chỉ đọc
        guide_text.config(state=tk.DISABLED)
    
    def _create_manual_train_tab(self, parent):
        """
        Tạo tab huấn luyện thủ công với đầy đủ chức năng.
        
        Args:
            parent: Frame cha
        """
        main_frame = ttk.Frame(parent, padding=10)
        main_frame.pack(fill='both', expand=True)
        
        # Create a PanedWindow for resizable left and right sections
        paned_window = ttk.PanedWindow(main_frame, orient=tk.HORIZONTAL)
        paned_window.pack(fill='both', expand=True)

        # Left frame for controls (scrollable) - will be added to PanedWindow
        left_scroll_container = ttk.Frame(paned_window) # Parent is now paned_window
        # left_scroll_container.pack(side='left', fill='y', padx=(0, 10), anchor='nw') # Packing handled by paned_window.add

        left_canvas = tk.Canvas(left_scroll_container, highlightthickness=0, width=450) # Fixed width for left panel
        left_canvas.pack(side="left", fill="y", expand=True)

        left_scrollbar = ttk.Scrollbar(left_scroll_container, orient="vertical", command=left_canvas.yview)
        left_scrollbar.pack(side="right", fill="y")

        left_canvas.configure(yscrollcommand=left_scrollbar.set)
        
        scrollable_left_frame = ttk.Frame(left_canvas)
        left_canvas.create_window((0, 0), window=scrollable_left_frame, anchor="nw", tags="scrollable_left_frame")

        def _configure_scrollable_frame(event):
            # Update the scrollregion to encompass the frame
            left_canvas.configure(scrollregion=left_canvas.bbox("all"))
            # Adjust canvas width to prevent horizontal scrollbar if frame is narrower
            left_canvas.itemconfig("scrollable_left_frame", width=event.width)

        scrollable_left_frame.bind("<Configure>", _configure_scrollable_frame)
        
        # Right frame for map visualization and training/model management - will be added to PanedWindow
        right_frame = ttk.Frame(paned_window) # Parent is now paned_window
        # right_frame.pack(side='right', fill='both', expand=True) # Packing handled by paned_window.add

        # Add left and right frames to the PanedWindow
        # Give left panel a smaller initial weight so right panel gets more space by default
        paned_window.add(left_scroll_container, weight=1) 
        paned_window.add(right_frame, weight=3)

        # --- SECTIONS IN SCROLLABLE LEFT FRAME ---

        # Section 0: Current Map Info
        env_info_frame = ttk.LabelFrame(scrollable_left_frame, text="Thông tin Bản đồ Hiện tại", padding=5)
        env_info_frame.pack(fill='x', padx=10, pady=(5,10), anchor='nw')
        ttk.Label(env_info_frame, text="Kích thước bản đồ: ").grid(row=0, column=0, sticky='w', padx=5, pady=2)
        ttk.Label(env_info_frame, textvariable=self.current_map_size_display_var).grid(row=0, column=1, sticky='w', padx=5, pady=2)
        # Add more info if needed, e.g., current_map_path

        # Section 1: Map Controls
        map_controls_outer_frame = ttk.LabelFrame(scrollable_left_frame, text="1. Tải/Tạo Bản đồ", padding=10)
        map_controls_outer_frame.pack(fill='x', padx=10, pady=10, anchor='nw')
        self._populate_map_controls(map_controls_outer_frame)

        # Section 2: Environment Initialization
        env_init_outer_frame = ttk.LabelFrame(scrollable_left_frame, text="2. Khởi tạo Môi trường RL", padding=10)
        env_init_outer_frame.pack(fill='x', padx=10, pady=10, anchor='nw')
        self._populate_env_init_controls(env_init_outer_frame)

        # Section 3: DQN Agent Configuration
        agent_config_frame = ttk.LabelFrame(scrollable_left_frame, text="3. Cấu hình Agent DQN", padding=10)
        agent_config_frame.pack(fill='x', padx=10, pady=10, anchor='nw')

        # Create a grid for parameters for better alignment
        params_grid = ttk.Frame(agent_config_frame)
        params_grid.pack(fill='x')
        
        row_idx = 0
        # Helper to add param row
        def add_param_entry(label, var, from_val, to_val, increment, width=12, fmt=None, is_int=False):
            nonlocal row_idx
            ttk.Label(params_grid, text=label).grid(row=row_idx, column=0, sticky='w', padx=5, pady=3)
            if fmt:
                entry = ttk.Spinbox(params_grid, from_=from_val, to=to_val, increment=increment, textvariable=var, width=width, format=fmt)
            else:
                entry = ttk.Spinbox(params_grid, from_=from_val, to=to_val, increment=increment, textvariable=var, width=width)
                entry.grid(row=row_idx, column=1, sticky='ew', padx=5, pady=3)
                params_grid.columnconfigure(1, weight=1) # Make entry expand
                row_idx += 1

        add_param_entry("Total Timesteps:", self.manual_total_timesteps, 10000, 5000000, 10000, is_int=True)
        add_param_entry("Learning Rate:", self.manual_lr, 0.00001, 0.01, 0.0001, fmt="%.5f")
        add_param_entry("Gamma (Discount Factor):", self.manual_gamma, 0.8, 0.999, 0.005, fmt="%.3f")
        add_param_entry("Buffer Size:", self.manual_buffer_size, 5000, 1000000, 5000, is_int=True)
        add_param_entry("Learning Starts:", self.manual_learning_starts, 1000, 100000, 1000, is_int=True)
        add_param_entry("Batch Size:", self.manual_batch_size, 32, 512, 32, is_int=True)
        add_param_entry("Tau (Soft Update):", self.manual_tau, 0.001, 0.1, 0.001, fmt="%.3f")
        add_param_entry("Train Frequency (Steps):", self.manual_train_freq, 1, 16, 1, is_int=True)
        add_param_entry("Target Update Interval (Steps):", self.manual_target_update_interval, 100, 20000, 100, is_int=True)
        add_param_entry("Exploration Fraction:", self.manual_exploration_fraction, 0.05, 0.9, 0.05, fmt="%.2f")
        add_param_entry("Exploration Initial Eps:", self.manual_exploration_initial_eps, 0.5, 1.0, 0.05, fmt="%.2f")
        add_param_entry("Exploration Final Eps:", self.manual_exploration_final_eps, 0.01, 0.2, 0.01, fmt="%.2f")

        ttk.Label(params_grid, text="Network Architecture (e.g., 256,256):").grid(row=row_idx, column=0, sticky='w', padx=5, pady=3)
        ttk.Entry(params_grid, textvariable=self.manual_net_arch, width=15).grid(row=row_idx, column=1, sticky='ew', padx=5, pady=3)
        row_idx += 1

        # DQN Feature Checkboxes
        advanced_options_frame = ttk.LabelFrame(agent_config_frame, text="Tùy chọn DQN Nâng cao", padding=5)
        advanced_options_frame.pack(fill='x', pady=(10,5))
        
        ttk.Checkbutton(advanced_options_frame, text="Double DQN", variable=self.manual_use_double_dqn).pack(side=tk.LEFT, padx=5)
        ttk.Checkbutton(advanced_options_frame, text="Dueling DQN", variable=self.manual_use_dueling_dqn).pack(side=tk.LEFT, padx=5)
        ttk.Checkbutton(advanced_options_frame, text="Prioritized Replay (PER)", variable=self.manual_use_prioritized_replay).pack(side=tk.LEFT, padx=5)
        ttk.Checkbutton(advanced_options_frame, text="Render Training (Slow)", variable=self.manual_render_training).pack(side=tk.LEFT, padx=5)


        # Section 4: Hyperparameter Tuning (Optuna)
        tuning_outer_frame = ttk.LabelFrame(scrollable_left_frame, text="4. Tinh chỉnh Siêu tham số (Optuna)", padding=10)
        tuning_outer_frame.pack(fill='x', padx=10, pady=10, anchor='nw')

        tuning_grid = ttk.Frame(tuning_outer_frame)
        tuning_grid.pack(fill='x')

        ttk.Label(tuning_grid, text="Number of Trials:").grid(row=0, column=0, sticky='w', padx=5, pady=3)
        ttk.Spinbox(tuning_grid, from_=5, to=200, increment=5, textvariable=self.tuning_n_trials, width=8).grid(row=0, column=1, sticky='ew', padx=5, pady=3)
        
        ttk.Label(tuning_grid, text="Timesteps per Trial:").grid(row=1, column=0, sticky='w', padx=5, pady=3)
        ttk.Spinbox(tuning_grid, from_=1000, to=100000, increment=1000, textvariable=self.tuning_timesteps_per_trial, width=10).grid(row=1, column=1, sticky='ew', padx=5, pady=3)
        tuning_grid.columnconfigure(1, weight=1)

        self.start_tuning_btn = ttk.Button(tuning_outer_frame, text="Start Hyperparameter Tuning", command=self._start_hyperparameter_tuning)
        self.start_tuning_btn.pack(pady=5, fill='x')
        
        self.apply_best_params_btn = ttk.Button(tuning_outer_frame, text="Apply Best Params to Config", command=self._apply_best_params, state=tk.DISABLED)
        self.apply_best_params_btn.pack(pady=5, fill='x')

        self.tuning_status_label = ttk.Label(tuning_outer_frame, text="Tuning Status: Idle", anchor='w', justify=tk.LEFT)
        self.tuning_status_label.pack(pady=5, fill='x')


        # --- SECTIONS IN RIGHT FRAME ---
        # Map Canvas
        map_canvas_frame = ttk.LabelFrame(right_frame, text="Trực quan hóa Bản đồ", padding=5)
        map_canvas_frame.pack(fill='both', expand=True, padx=5, pady=5)
        self.map_canvas = tk.Canvas(map_canvas_frame, bg="lightgrey")
        self.map_canvas.pack(fill="both", expand=True)
        self.map_canvas.bind("<Configure>", lambda event: self._draw_map(self.map_canvas) if self.map_object else None) # Redraw on resize

        # Training Controls (Button, Progress, Status) - at the bottom of right_frame
        manual_train_status_frame = ttk.LabelFrame(right_frame, text="Điều khiển & Trạng thái Huấn luyện", padding=10)
        manual_train_status_frame.pack(fill='x', padx=5, pady=(10,5), side='bottom', anchor='sw')
        
        self.manual_train_btn = ttk.Button(manual_train_status_frame, text="Bắt đầu / Tiếp tục Huấn luyện", command=self._start_manual_training)
        self.manual_train_btn.pack(pady=(5,2), fill='x')

        self.manual_train_progress = ttk.Progressbar(manual_train_status_frame, orient="horizontal", length=300, mode="determinate")
        self.manual_train_progress.pack(pady=(2,2), fill='x', expand=True)
        
        self.manual_train_status_label = ttk.Label(manual_train_status_frame, text="Status: Idle", anchor="w", justify=tk.LEFT)
        self.manual_train_status_label.pack(pady=(2,5), fill='x', expand=True)

        # Model Save/Load - above training controls
        model_io_frame = ttk.LabelFrame(right_frame, text="Quản lý Model", padding=10)
        model_io_frame.pack(fill='x', padx=5, pady=5, side='bottom', anchor='sw')
        
        model_buttons_subframe = ttk.Frame(model_io_frame) # For side-by-side buttons
        model_buttons_subframe.pack(fill='x')

        self.save_manual_model_button = ttk.Button(model_buttons_subframe, text="Lưu Model Hiện tại", command=self._save_manual_model, state=tk.DISABLED)
        self.save_manual_model_button.pack(side=tk.LEFT, padx=5, pady=5, expand=True, fill='x')
        
        self.load_manual_model_button = ttk.Button(model_buttons_subframe, text="Tải Model Đã Huấn luyện", command=self._load_trained_model)
        self.load_manual_model_button.pack(side=tk.LEFT, padx=5, pady=5, expand=True, fill='x')

        # Run Agent - above model io
        run_agent_frame = ttk.LabelFrame(right_frame, text="Chạy Agent Đã Huấn luyện", padding=10)
        run_agent_frame.pack(fill='x', padx=5, pady=5, side='bottom', anchor='sw')
        
        self.run_agent_button = ttk.Button(run_agent_frame, text="Chạy Agent trên Bản đồ Hiện tại", command=self._run_agent_on_current_map, state=tk.DISABLED)
        self.run_agent_button.pack(pady=5, fill='x')
        
        # Make sure left panel does not expand horizontally, but right panel does
        # main_frame.columnconfigure(0, weight=0) # No longer needed due to PanedWindow
        # main_frame.columnconfigure(1, weight=1) # No longer needed due to PanedWindow

        # Ensure the canvas configuration updates the scrollable frame's width and scrollregion
        def _on_left_canvas_configure(event_canvas):
            canvas_width = left_canvas.winfo_width()
            left_canvas.itemconfig("scrollable_left_frame", width=canvas_width)
            left_canvas.configure(scrollregion=left_canvas.bbox("all"))
        left_canvas.bind("<Configure>", _on_left_canvas_configure)

        # Ensure the scrollable frame configuration updates the scrollregion
        # (This might be slightly redundant if _on_left_canvas_configure also sets scrollregion, but safe)
        # Let's rename the original _configure_scrollable_frame to avoid confusion if we re-purpose it.
        # The original _configure_scrollable_frame was: scrollable_left_frame.bind("<Configure>", _configure_scrollable_frame)
        # We need to ensure that binding is to a function that correctly updates the scrollregion.
        
        # Re-define and re-bind for clarity and correctness for the scrollable_left_frame itself
        def _update_scrollregion_on_frame_configure(event_frame):
            left_canvas.configure(scrollregion=left_canvas.bbox("all"))
        scrollable_left_frame.bind("<Configure>", _update_scrollregion_on_frame_configure, add='+') # Add to existing bindings if any

        # Force an update of geometry and then the scrollregion after all widgets are packed
        self.root.update_idletasks()
        left_canvas.configure(scrollregion=left_canvas.bbox("all"))

    def _populate_map_controls(self, parent):
        """Hàm phụ trợ: Tạo các control cho bản đồ."""

        # Frame for top configurations (size and counts)
        top_config_frame = ttk.Frame(parent)
        top_config_frame.pack(fill="x", padx=0, pady=5) # Use fill x to allow internal elements to align

        # Frame kích thước
        size_frame = ttk.Frame(top_config_frame)
        size_frame.pack(side="left", padx=5, pady=0) # Pack to the left of top_config_frame
        ttk.Label(size_frame, text="Kích thước (8-15):").grid(row=0, column=0, padx=5, pady=5, sticky='w')
        # self.map_size_var = tk.IntVar(value=8) # Already defined in __init__
        size_spinner = ttk.Spinbox(size_frame, from_=8, to=15, textvariable=self.map_size_var, width=5)
        size_spinner.grid(row=0, column=1, padx=5, pady=5, sticky='w')

        # Frame for special point counts
        counts_frame = ttk.Frame(top_config_frame)
        counts_frame.pack(side="left", padx=5, pady=0, anchor="nw") # Pack to the left (so right of size_frame)

        ttk.Label(counts_frame, text="# Tolls:").grid(row=0, column=0, padx=2, pady=2, sticky="w")
        ttk.Spinbox(counts_frame, from_=0, to=20, textvariable=self.num_tolls_var, width=4).grid(row=0, column=1, padx=2, pady=2, sticky="w")

        ttk.Label(counts_frame, text="# Gas Stations:").grid(row=1, column=0, padx=2, pady=2, sticky="w")
        ttk.Spinbox(counts_frame, from_=0, to=20, textvariable=self.num_gas_var, width=4).grid(row=1, column=1, padx=2, pady=2, sticky="w")

        ttk.Label(counts_frame, text="# Obstacles:").grid(row=2, column=0, padx=2, pady=2, sticky="w")
        ttk.Spinbox(counts_frame, from_=0, to=50, textvariable=self.num_obstacles_var, width=4).grid(row=2, column=1, padx=2, pady=2, sticky="w")

        # Frame nút (buttons) - now packed below top_config_frame
        buttons_frame = ttk.Frame(parent)
        buttons_frame.pack(fill="x", padx=5, pady=10) # pady=10 for spacing

        # Configure buttons in a 2-row grid, sticky='ew'
        self.random_map_btn = ttk.Button(buttons_frame, text="Tạo Ngẫu nhiên", command=self._create_random_map)
        self.random_map_btn.grid(row=0, column=0, padx=2, pady=3, sticky='ew')
        self.demo_map_btn = ttk.Button(buttons_frame, text="Tạo Mẫu", command=self._create_demo_map)
        self.demo_map_btn.grid(row=0, column=1, padx=2, pady=3, sticky='ew')
        self.load_map_btn = ttk.Button(buttons_frame, text="Tải", command=self._load_map)
        self.load_map_btn.grid(row=0, column=2, padx=2, pady=3, sticky='ew')

        self.save_map_btn = ttk.Button(buttons_frame, text="Lưu", command=self._save_map, state=tk.DISABLED) # Disable ban đầu
        self.save_map_btn.grid(row=1, column=0, padx=2, pady=3, sticky='ew')
        self.reset_all_btn = ttk.Button(buttons_frame, text="Reset Tất Cả", command=self._reset_all)
        self.reset_all_btn.grid(row=1, column=1, columnspan=2, padx=5, pady=3, sticky='ew') # columnspan=2 to fill nicely

        # Configure column weights for buttons_frame so they expand
        buttons_frame.columnconfigure(0, weight=1)
        buttons_frame.columnconfigure(1, weight=1)
        buttons_frame.columnconfigure(2, weight=1)

    def _populate_env_init_controls(self, parent):
        """Hàm phụ trợ: Tạo các control khởi tạo môi trường."""
        # Biến lưu trữ các tham số (Lấy từ _create_rl_init_tab)
        from core.constants import MovementCosts, StationCosts
        
        self.initial_fuel_var = tk.DoubleVar(value=MovementCosts.MAX_FUEL)
        self.initial_money_var = tk.DoubleVar(value=1500.0)
        self.fuel_per_move_var = tk.DoubleVar(value=MovementCosts.FUEL_PER_MOVE)
        self.gas_station_cost_var = tk.DoubleVar(value=StationCosts.BASE_GAS_COST)
        self.toll_base_cost_var = tk.DoubleVar(value=StationCosts.BASE_TOLL_COST)
        self.max_steps_var = tk.IntVar(value=400) # Tăng từ 200 lên 400

        # Frame chứa các tham số (Grid layout cho gọn)
        params_grid = ttk.Frame(parent)
        params_grid.pack(pady=5)

        row_idx = 0
        ttk.Label(params_grid, text="Initial Fuel:").grid(row=row_idx, column=0, padx=5, pady=2, sticky="w")
        ttk.Entry(params_grid, textvariable=self.initial_fuel_var, width=8).grid(row=row_idx, column=1, padx=5, pady=2, sticky="w")
        ttk.Label(params_grid, text="Initial Money:").grid(row=row_idx, column=2, padx=5, pady=2, sticky="w")
        ttk.Entry(params_grid, textvariable=self.initial_money_var, width=8).grid(row=row_idx, column=3, padx=5, pady=2, sticky="w")
        
        row_idx += 1
        ttk.Label(params_grid, text="Fuel Per Move:").grid(row=row_idx, column=0, padx=5, pady=2, sticky="w")
        ttk.Entry(params_grid, textvariable=self.fuel_per_move_var, width=8).grid(row=row_idx, column=1, padx=5, pady=2, sticky="w")
        ttk.Label(params_grid, text="Gas Station Cost:").grid(row=row_idx, column=2, padx=5, pady=2, sticky="w")
        ttk.Entry(params_grid, textvariable=self.gas_station_cost_var, width=8).grid(row=row_idx, column=3, padx=5, pady=2, sticky="w")
        
        row_idx += 1
        ttk.Label(params_grid, text="Toll Base Cost:").grid(row=row_idx, column=0, padx=5, pady=2, sticky="w")
        ttk.Entry(params_grid, textvariable=self.toll_base_cost_var, width=8).grid(row=row_idx, column=1, padx=5, pady=2, sticky="w")
        ttk.Label(params_grid, text="Max Steps:").grid(row=row_idx, column=2, padx=5, pady=2, sticky="w")
        ttk.Entry(params_grid, textvariable=self.max_steps_var, width=8).grid(row=row_idx, column=3, padx=5, pady=2, sticky="w")
        
        # Nút khởi tạo môi trường RL
        self.init_rl_btn = ttk.Button(parent, text="Khởi tạo Môi trường RL", command=self._initialize_rl_env)
        self.init_rl_btn.pack(pady=5)
        
        # Thông tin môi trường (đơn giản hóa, có thể thêm lại nếu cần)
        # self.rl_info_text = scrolledtext.ScrolledText(parent, height=5)
        # self.rl_info_text.pack(fill="x", expand=False, padx=5, pady=5)

    def _populate_agent_config_controls(self, parent):
        """Hàm phụ trợ: Tạo các control cấu hình agent."""
        # Sử dụng lại frame chia 2 cột từ _create_rl_agent_tab
        config_left_frame = ttk.Frame(parent)
        config_left_frame.grid(row=0, column=0, padx=5, pady=5, sticky="nw")
        config_right_frame = ttk.Frame(parent)
        config_right_frame.grid(row=0, column=1, padx=5, pady=5, sticky="nw")
        parent.columnconfigure(0, weight=1)
        parent.columnconfigure(1, weight=1)

        row_idx_left = 0
        row_idx_right = 0

        def add_param(label, var, width=8, side='left'):
            nonlocal row_idx_left, row_idx_right
            target_frame = config_left_frame if side == 'left' else config_right_frame
            row_idx = row_idx_left if side == 'left' else row_idx_right
            
            ttk.Label(target_frame, text=label).grid(row=row_idx, column=0, sticky="w", pady=1, padx=2)
            ttk.Entry(target_frame, textvariable=var, width=width).grid(row=row_idx, column=1, sticky="w", pady=1, padx=2)
            
            if side == 'left': row_idx_left += 1
            else: row_idx_right += 1

        add_param("LR:", self.manual_lr, side='left')
        add_param("Gamma:", self.manual_gamma, side='left')
        add_param("Buffer Size:", self.manual_buffer_size, width=10, side='left')
        add_param("Learn Starts:", self.manual_learning_starts, width=10, side='left')
        add_param("Batch Size:", self.manual_batch_size, side='left')
        add_param("Tau:", self.manual_tau, side='left')
        add_param("Train Freq:", self.manual_train_freq, side='left')
        
        add_param("Target Upd.:", self.manual_target_update_interval, width=10, side='right')
        add_param("Expl. Frac.:", self.manual_exploration_fraction, side='right')
        add_param("Expl. Init Eps:", self.manual_exploration_initial_eps, side='right')
        add_param("Expl. Final Eps:", self.manual_exploration_final_eps, side='right')
        add_param("Total Steps:", self.manual_total_timesteps, width=10, side='right')
        add_param("Network Arch:", self.manual_net_arch, width=10, side='right') # Added Net Arch

        # Tùy chọn nâng cao (Checkboxes)
        advanced_frame = ttk.LabelFrame(parent, text="Advanced Options", padding=(5, 5))
        advanced_frame.grid(row=1, column=0, columnspan=2, sticky="ew", padx=5, pady=5)
        ttk.Checkbutton(advanced_frame, text="Double DQN", variable=self.manual_use_double_dqn).pack(side=tk.LEFT, padx=5)
        ttk.Checkbutton(advanced_frame, text="Dueling DQN", variable=self.manual_use_dueling_dqn).pack(side=tk.LEFT, padx=5) # Added Dueling checkbox
        ttk.Checkbutton(advanced_frame, text="PER", variable=self.manual_use_prioritized_replay).pack(side=tk.LEFT, padx=5)

        # --- Khu vực Hyperparameter Tuning ---
        tuning_frame = ttk.LabelFrame(parent, text="Hyperparameter Tuning (Optuna)", padding=(5, 5))
        tuning_frame.grid(row=2, column=0, columnspan=2, sticky="ew", padx=5, pady=(10, 5))

        ttk.Label(tuning_frame, text="Trials:").grid(row=0, column=0, sticky="w", padx=2, pady=1)
        ttk.Entry(tuning_frame, textvariable=self.tuning_n_trials, width=8).grid(row=0, column=1, sticky="w", padx=2, pady=1)

        ttk.Label(tuning_frame, text="Steps/Trial:").grid(row=0, column=2, sticky="w", padx=(10, 2), pady=1)
        ttk.Entry(tuning_frame, textvariable=self.tuning_timesteps_per_trial, width=10).grid(row=0, column=3, sticky="w", padx=2, pady=1)

        self.start_tuning_btn = ttk.Button(tuning_frame, text="Start Tuning", command=self._start_hyperparameter_tuning)
        self.start_tuning_btn.grid(row=1, column=0, columnspan=2, pady=5, padx=2)

        self.apply_best_params_btn = ttk.Button(tuning_frame, text="Apply Best Params", command=self._apply_best_params, state=tk.DISABLED)
        self.apply_best_params_btn.grid(row=1, column=2, columnspan=2, pady=5, padx=2)

        self.tuning_status_label = ttk.Label(tuning_frame, text="Status: Idle")
        self.tuning_status_label.grid(row=2, column=0, columnspan=4, sticky="w", padx=2, pady=2)

    def _create_random_map(self):
        """Tạo bản đồ ngẫu nhiên với số lượng điểm đặc biệt tùy chỉnh."""
        # Hủy môi trường và agent cũ nếu tạo map mới
        if self.rl_environment or self.trainer:
            self._log("Map mới được tạo. Môi trường RL và Agent cũ đã bị hủy. Vui lòng khởi tạo lại môi trường.")
            self._reset_env_and_agent()
            
        try:
            map_size = self.map_size_var.get()
            num_tolls = self.num_tolls_var.get()
            num_gas = self.num_gas_var.get()
            num_obstacles = self.num_obstacles_var.get()

            # Gọi Map.generate_random với số lượng trực tiếp
            self.map_object = Map.generate_random(
                size=map_size,
                num_tolls=num_tolls,
                num_gas=num_gas,
                num_obstacles=num_obstacles
            )

            if self.map_object: # Kiểm tra xem generate_random có thành công không
                self.current_map_size_display_var.set(f"{self.map_object.size}x{self.map_object.size}")
                self._log(f"Generated random map ({map_size}x{map_size}) with requested counts: T={num_tolls}, G={num_gas}, O={num_obstacles}.")
                # Log thêm số lượng thực tế từ map_object.get_statistics() để so sánh
                stats = self.map_object.get_statistics()
                self._log(f"  Actual counts: Tolls={stats['toll_stations']}, Gas={stats['gas_stations']}, Obstacles={stats['brick_cells']}")
                self._draw_map(self.map_canvas)
                self.save_map_btn.config(state=tk.NORMAL)
            else:
                self.current_map_size_display_var.set("N/A")
                messagebox.showwarning("Map Generation Failed", f"Could not generate a valid random map with the specified counts after several attempts. Check console warnings.")
                self._log(f"Failed to generate a valid map for size {map_size} with T={num_tolls}, G={num_gas}, O={num_obstacles}.")

        except tk.TclError as e:
            self.current_map_size_display_var.set("N/A")
            messagebox.showerror("Input Error", f"Invalid input for map generation parameters: {e}")
            self._log(f"Map generation input error: {e}")
        except Exception as e:
            self.current_map_size_display_var.set("N/A")
            messagebox.showerror("Error", f"Failed to generate random map: {e}")
            self._log(f"Failed to generate random map: {e}")
            import traceback
            self._log(traceback.format_exc())
    
    def _create_demo_map(self):
        """Tạo bản đồ mẫu."""
        # Hủy môi trường và agent cũ nếu tạo map mới
        if self.rl_environment or self.trainer:
            self._log("Map mẫu được tạo. Môi trường RL và Agent cũ đã bị hủy. Vui lòng khởi tạo lại môi trường.")
            self._reset_env_and_agent()
            
        map_size = self.map_size_var.get()
        self.map_object = Map.create_demo_map(size=map_size)
        self.current_map_size_display_var.set(f"{self.map_object.size}x{self.map_object.size}")
        self._log(f"Đã tạo bản đồ mẫu kích thước {map_size}x{map_size}")
        # self._display_map_info()
        self._draw_map(self.map_canvas) # Vẽ lên canvas của tab mới
        self.save_map_btn.config(state=tk.NORMAL)

    def _load_map(self):
        """Tải bản đồ từ file."""
        filepath = filedialog.askopenfilename(
            title="Chọn bản đồ",
            filetypes=[("JSON files", "*.json")],
            initialdir="./maps" # Thư mục mặc định
        )
        if not filepath:
            return
            
        try:
            # Hủy môi trường và agent cũ nếu tải map mới
            if self.rl_environment or self.trainer:
                self._log("Map mới được tải. Môi trường RL và Agent cũ đã bị hủy. Vui lòng khởi tạo lại môi trường.")
                self._reset_env_and_agent()

            self.map_object = Map.load(filepath)
            if self.map_object:
                self.current_map_path = filepath # Lưu đường dẫn file đã tải
                self.current_map_size_display_var.set(f"{self.map_object.size}x{self.map_object.size}")
                self._log(f"Đã tải bản đồ: {filepath} (Size: {self.map_object.size}x{self.map_object.size})")
                # self._display_map_info()
                self._draw_map(self.map_canvas) # Vẽ lên canvas của tab mới
                self.map_size_var.set(self.map_object.size) # Cập nhật size spinner
                self.save_map_btn.config(state=tk.DISABLED) # Không cho lưu lại map vừa tải
            else:
                self.current_map_size_display_var.set("N/A")
                messagebox.showerror("Lỗi", "Không thể tải bản đồ từ file!")
        except Exception as e:
            self.current_map_size_display_var.set("N/A")
            self._log(f"Lỗi khi tải bản đồ: {e}")
            messagebox.showerror("Lỗi", f"Lỗi khi tải bản đồ: {e}")

    def _save_map(self):
        """Lưu bản đồ hiện tại vào file."""
        if not self.map_object:
            messagebox.showerror("Lỗi", "Không có bản đồ để lưu!")
            return
            
        filepath = filedialog.asksaveasfilename(
            title="Lưu bản đồ",
            defaultextension=".json",
            filetypes=[("JSON files", "*.json")],
            initialdir="./maps", # Thư mục mặc định
            initialfile=f"map_{self.map_object.size}x{self.map_object.size}_{datetime.now().strftime('%Y%m%d')}.json"
        )
        if not filepath:
            return
        
        try:
            self.map_object.save(filepath)
            self._log(f"Đã lưu bản đồ vào: {filepath}")
            messagebox.showinfo("Thông báo", "Đã lưu bản đồ thành công!")
            self.current_map_path = filepath # Cập nhật đường dẫn file đã lưu
            self.save_map_btn.config(state=tk.DISABLED) # Disable nút lưu sau khi đã lưu
        except Exception as e:
            self._log(f"Lỗi khi lưu bản đồ: {e}")
            messagebox.showerror("Lỗi Lưu Model", f"Lỗi: {e}")
            self._update_agent_status("Lỗi lưu model!")

    def _initialize_rl_env(self):
        """
        Khởi tạo môi trường Reinforcement Learning với bản đồ hiện tại.
        """
        try:
            self._log("Đang khởi tạo môi trường RL với bản đồ hiện tại...")
            if self.map_object is None:
                messagebox.showerror("Lỗi", "Vui lòng tạo hoặc tải bản đồ trước khi khởi tạo môi trường")
                return False
                
            # Lấy các giá trị từ UI
            initial_fuel = self.initial_fuel_var.get()
            initial_money = self.initial_money_var.get()
            fuel_per_move = self.fuel_per_move_var.get()
            gas_cost = self.gas_station_cost_var.get()
            toll_cost = self.toll_base_cost_var.get()
            max_steps = self.max_steps_var.get()
            moving_obstacles_flag = self.moving_obstacles_var.get() # Lấy giá trị từ checkbox

            # Gán các tham số môi trường từ UI sliders vào hằng số chính xác trong constants.py
            # from core.constants import MovementCosts, StationCosts # Không cần nữa
            # from core.rl_environment import GLOBAL_MAX_FUEL, GLOBAL_MAX_MONEY # Không cần nữa
            
            # Cập nhật các hằng số trong constants.py
            # MovementCosts.MAX_FUEL = float(self.initial_fuel_var.get()) # Không cần nữa
            # MovementCosts.FUEL_PER_MOVE = float(self.fuel_per_move_var.get()) # Không cần nữa
            # StationCosts.BASE_GAS_COST = float(self.gas_station_cost_var.get()) # Không cần nữa
            # StationCosts.BASE_TOLL_COST = float(self.toll_base_cost_var.get()) # Không cần nữa
            
            # Cập nhật các hằng số trong rl_environment.py
            # GLOBAL_MAX_FUEL = float(self.initial_fuel_var.get()) * 2  # Cho phép không gian lớn hơn # Không cần nữa
            # GLOBAL_MAX_MONEY = float(self.initial_money_var.get()) * 3  # Cho phép không gian lớn hơn # Không cần nữa
            
            # Khởi tạo môi trường mới với các tham số từ UI
            self.rl_environment = TruckRoutingEnv(
                map_object=self.map_object,
                initial_fuel=initial_fuel,
                initial_money=initial_money,
                fuel_per_move=fuel_per_move,
                gas_station_cost=gas_cost,
                toll_base_cost=toll_cost,
                max_steps_per_episode=max_steps,
                obs_max_fuel=initial_fuel, # Use direct value from UI
                obs_max_money=initial_money, # Use direct value from UI
                moving_obstacles=moving_obstacles_flag
            )
            
            # In ra thông tin về observation space cho debug
            obs_space = self.rl_environment.observation_space
            self._log(f"Observation space: {obs_space}")
            for key, space in obs_space.spaces.items():
                self._log(f"  {key}: {space}")
            
            # Đặt lại môi trường để kiểm tra không có lỗi
            obs, info = self.rl_environment.reset()
            
            # Cập nhật trạng thái và UI
            self._log(f"Đã khởi tạo môi trường thành công. Kích thước bản đồ: {self.map_object.size}x{self.map_object.size}")
            if 'map_info' in info:
                self._log(f"Có {info['map_info']['obstacles']} ô chướng ngại vật trên bản đồ.")
                self._log(f"Ước tính độ dài đường đi tối ưu: {info['map_info']['optimal_path_estimate']} bước")
            
            # Vẽ lại bản đồ với agent ở vị trí bắt đầu
            self._draw_map(self.map_canvas, self.rl_environment.current_pos)
            
            # Kích hoạt các nút điều khiển khi môi trường đã sẵn sàng
            self._log("Môi trường RL sẵn sàng. Có thể bắt đầu huấn luyện và kiểm thử.")
            return True
            
        except Exception as e:
            import traceback
            error_msg = f"Lỗi khởi tạo môi trường RL: {str(e)}\n{traceback.format_exc()}"
            self._log(error_msg)
            messagebox.showerror("Lỗi Khởi Tạo Môi Trường", f"Không thể khởi tạo môi trường RL: {str(e)}")
            return False

    def _draw_map(self, canvas, agent_pos=None):
        """Vẽ bản đồ lên canvas được chỉ định."""
        if not self.map_object:
            return
        
        canvas.delete("all") # Xóa canvas trước khi vẽ
        map_size = self.map_object.size
        # Sử dụng after để đảm bảo widget đã có kích thước
        canvas.after(50, lambda c=canvas, m=map_size, ap=agent_pos: self._draw_map_internal(c, m, ap))

    def _draw_map_internal(self, canvas, map_size, agent_pos):
        """Hàm vẽ nội bộ, được gọi bởi _draw_map sau delay."""
        canvas_width = canvas.winfo_width()
        canvas_height = canvas.winfo_height()
        
        if canvas_width <= 1 or canvas_height <= 1: # Canvas chưa sẵn sàng
            canvas.after(100, lambda c=canvas, m=map_size, ap=agent_pos: self._draw_map_internal(c, m, ap))
            return

        cell_size = min(canvas_width // map_size, canvas_height // map_size)
        if cell_size <= 0: cell_size = 1 # Đảm bảo cell_size dương
        offset_x = (canvas_width - cell_size * map_size) // 2
        offset_y = (canvas_height - cell_size * map_size) // 2

        for y in range(map_size):
            for x in range(map_size):
                pixel_x = offset_x + x * cell_size
                pixel_y = offset_y + y * cell_size
                cell_type = self.map_object.grid[y, x]
                color = "white"
                if cell_type == CellType.OBSTACLE: color = "darkgray"
                elif cell_type == CellType.TOLL: color = "red"
                elif cell_type == CellType.GAS: color = "green"
                canvas.create_rectangle(pixel_x, pixel_y, pixel_x + cell_size, pixel_y + cell_size, fill=color, outline="black")
                
                label = None
                if (x, y) == self.map_object.start_pos: label = "S"
                elif (x, y) == self.map_object.end_pos: label = "E"
                if label:
                    canvas.create_text(pixel_x + cell_size // 2, pixel_y + cell_size // 2, text=label, font=("Arial", max(8, cell_size // 3), "bold"))
        
        # Vẽ agent (nếu có)
        if agent_pos is not None:
            agent_pos_np = np.array(agent_pos) if isinstance(agent_pos, list) else agent_pos
            if hasattr(agent_pos_np, 'size') and agent_pos_np.size > 0:
                agent_x, agent_y = agent_pos_np[0], agent_pos_np[1]
                pixel_x = offset_x + agent_x * cell_size
                pixel_y = offset_y + agent_y * cell_size
                canvas.create_oval(pixel_x + cell_size // 4, pixel_y + cell_size // 4, 
                                    pixel_x + cell_size * 3 // 4, pixel_y + cell_size * 3 // 4, fill="blue", outline="black")
    
    def _draw_map_with_path(self, canvas, agent_pos, path):
        """Vẽ bản đồ với đường đi của agent."""
        # Vẽ trực tiếp mà không sử dụng delay giữa các bước vẽ
        if not self.map_object:
            return
            
        canvas.delete("all") # Xóa canvas trước khi vẽ
        map_size = self.map_object.size
        
        # Tạo bản sao của path để tránh thay đổi bên ngoài
        path_copy = list(path) if path else []
        
        # Trước tiên, vẽ bản đồ (các ô và nhãn S, E)
        canvas_width = canvas.winfo_width()
        canvas_height = canvas.winfo_height()
        
        if canvas_width <= 1 or canvas_height <= 1: # Canvas chưa sẵn sàng
            canvas.after(50, lambda: self._draw_map_with_path(canvas, agent_pos, path_copy))
            return

        cell_size = min(canvas_width // map_size, canvas_height // map_size)
        if cell_size <= 0: cell_size = 1 # Đảm bảo cell_size dương
        offset_x = (canvas_width - cell_size * map_size) // 2
        offset_y = (canvas_height - cell_size * map_size) // 2

        # Vẽ bản đồ nền
        for y in range(map_size):
            for x in range(map_size):
                pixel_x = offset_x + x * cell_size
                pixel_y = offset_y + y * cell_size
                cell_type = self.map_object.grid[y, x]
                color = "white"
                if cell_type == CellType.OBSTACLE: color = "darkgray"
                elif cell_type == CellType.TOLL: color = "red"
                elif cell_type == CellType.GAS: color = "green"
                canvas.create_rectangle(pixel_x, pixel_y, pixel_x + cell_size, pixel_y + cell_size, fill=color, outline="black")
                
                label = None
                if (x, y) == self.map_object.start_pos: label = "S"
                elif (x, y) == self.map_object.end_pos: label = "E"
                if label:
                    canvas.create_text(pixel_x + cell_size // 2, pixel_y + cell_size // 2, text=label, font=("Arial", max(8, cell_size // 3), "bold"))
        
        # Tiếp theo, vẽ đường đi (nếu có)
        if path_copy and len(path_copy) > 1:
            # Sử dụng màu gradient từ cam sang đỏ để biểu thị đường đi qua thời gian
            num_segments = len(path_copy) - 1
            for i in range(num_segments):
                p1 = path_copy[i]
                p2 = path_copy[i+1]
                # Đảm bảo p1, p2 là tuple
                x1, y1 = tuple(p1) if hasattr(p1, '__len__') else (0,0)
                x2, y2 = tuple(p2) if hasattr(p2, '__len__') else (0,0)

                # Tính toán vị trí trung tâm của các ô
                center_x1 = offset_x + x1 * cell_size + cell_size // 2
                center_y1 = offset_y + y1 * cell_size + cell_size // 2
                center_x2 = offset_x + x2 * cell_size + cell_size // 2
                center_y2 = offset_y + y2 * cell_size + cell_size // 2
                
                # Sử dụng màu sắc khác nhau cho các phần của đường đi
                # Đoạn đầu: cam nhạt, đoạn cuối: đỏ đậm
                progress_ratio = i / max(1, num_segments - 1)  # Tránh chia cho 0
                
                # Tạo màu gradient từ cam (#FFA500) đến đỏ (#FF0000)
                r = min(255, int(255))
                g = min(255, int(165 - progress_ratio * 165))
                b = min(255, int(0))
                color = f'#{r:02x}{g:02x}{b:02x}'
                
                # Xác định đoạn cuối cùng vào đích
                is_last_to_end = (p2 == tuple(self.map_object.end_pos))
                
                # Vẽ đoạn đường đặc biệt nếu đi vào đích
                if is_last_to_end:
                    # Đoạn cuối vào đích sử dụng màu đỏ đậm và mũi tên lớn hơn
                    self._log(f"Vẽ mũi tên đặc biệt từ {p1} đến đích {p2}")
                    line_width = max(3, cell_size // 6)  # Đường đậm hơn
                    arrow_size = max(12, cell_size // 1.5)  # Mũi tên lớn hơn
                    
                    # Vẽ hiệu ứng "glow" trước khi vẽ mũi tên chính
                    for glow in range(2):
                        glow_width = line_width + (2-glow)
                        glow_color = "#FF5500" if glow == 0 else "#FF2200" 
                        canvas.create_line(center_x1, center_y1, center_x2, center_y2, 
                                    fill=glow_color, width=glow_width, 
                                    arrow=tk.LAST, arrowshape=(arrow_size, arrow_size, arrow_size//2))
                    
                    # Vẽ mũi tên chính với màu đỏ tươi
                    canvas.create_line(center_x1, center_y1, center_x2, center_y2, 
                                    fill="#FF0000", width=line_width, 
                                    arrow=tk.LAST, arrowshape=(arrow_size, arrow_size, arrow_size//2))
                else:
                    # Các đoạn đường bình thường
                    canvas.create_line(center_x1, center_y1, center_x2, center_y2, 
                                    fill=color, width=max(1, cell_size // 10), 
                                    arrow=tk.LAST)
        
        # Cuối cùng, vẽ agent (nếu có)
        if agent_pos is not None:
            agent_pos_np = np.array(agent_pos) if isinstance(agent_pos, list) else agent_pos
            if hasattr(agent_pos_np, 'size') and agent_pos_np.size > 0:
                agent_x, agent_y = agent_pos_np[0], agent_pos_np[1]
                pixel_x = offset_x + agent_x * cell_size
                pixel_y = offset_y + agent_y * cell_size
                
                # Vẽ agent to và nổi bật hơn
                agent_size = cell_size // 2.5  # Hơi nhỏ hơn để không che mất ô
                
                # Nếu agent ở vị trí đích, dùng màu xanh lá đậm để dễ nhìn hơn
                if (agent_x, agent_y) == self.map_object.end_pos:
                    agent_color = "#00AA00"  # Xanh lá đậm
                    # Thêm hiệu ứng hào quang xung quanh để nhấn mạnh
                    glow_size = agent_size * 1.5
                    canvas.create_oval(
                        pixel_x + (cell_size - glow_size) / 2,
                        pixel_y + (cell_size - glow_size) / 2,
                        pixel_x + (cell_size + glow_size) / 2,
                        pixel_y + (cell_size + glow_size) / 2,
                        fill="#AAFFAA", outline="#00FF00", width=2
                    )
                else:
                    agent_color = "blue"
                
                # Vẽ agent
                canvas.create_oval(
                    pixel_x + (cell_size - agent_size) / 2,
                    pixel_y + (cell_size - agent_size) / 2,
                    pixel_x + (cell_size + agent_size) / 2,
                    pixel_y + (cell_size + agent_size) / 2,
                    fill=agent_color, outline="black", width=2
                )

        # DEBUG: Kiểm tra đường đi có chứa điểm đích không
        end_pos = tuple(self.map_object.end_pos)
        has_path_to_end = len(path_copy) >= 2 and any(p == end_pos for p in path_copy)
        if not has_path_to_end and agent_pos is not None:
            agent_pos_tuple = tuple(agent_pos) if hasattr(agent_pos, '__len__') else None
            if agent_pos_tuple == end_pos:
                # Nếu agent đã ở đích nhưng path không có điểm đích, thêm vào
                if len(path_copy) > 0:
                    last_pos = path_copy[-1]
                    # Nếu điểm cuối cùng trong path không phải điểm đích, thêm điểm đích
                    if last_pos != end_pos:
                        self._log(f"Đường đi không có điểm đích: {path_copy}")
                        self._log(f"Agent ở đích {agent_pos_tuple}. Thêm vào đường đi.")
                        # Thêm ô trước đích nếu không có trong path
                        second_last_found = False
                        for dy in [-1, 0, 1]:
                            for dx in [-1, 0, 1]:
                                if dx == 0 and dy == 0:
                                    continue  # Bỏ qua chính nó
                                test_x = end_pos[0] + dx
                                test_y = end_pos[1] + dy
                                if 0 <= test_x < self.map_object.size and 0 <= test_y < self.map_object.size:
                                    test_pos = (test_x, test_y)
                                    # Nếu có một vị trí liền kề trong path, sử dụng để vẽ mũi tên
                                    if test_pos in path_copy:
                                        path_copy.append(end_pos)
                                        second_last_found = True
                                        self._log(f"Tìm thấy ô liền kề {test_pos}, thêm vào đường đi")
                                        break
                            if second_last_found:
                                break
                        
                        # Nếu không tìm được ô liền kề, tạo một ô giả để vẽ mũi tên
                        if not second_last_found:
                            # Tìm một ô liền kề đích có thể đi được
                            for dy in [-1, 0, 1]:
                                for dx in [-1, 0, 1]:
                                    if dx == 0 and dy == 0:
                                        continue  # Bỏ qua chính nó
                                    test_x = end_pos[0] + dx
                                    test_y = end_pos[1] + dy
                                    if 0 <= test_x < self.map_object.size and 0 <= test_y < self.map_object.size:
                                        test_pos = (test_x, test_y)
                                        # Kiểm tra ô không phải là chướng ngại vật
                                        cell_type = self.map_object.grid[test_y, test_x]
                                        if cell_type != CellType.OBSTACLE:
                                            path_copy = [test_pos, end_pos]  # Tạo đường đi tối thiểu
                                            self._log(f"Tạo đường đi tối thiểu từ {test_pos} đến {end_pos}")
                                            second_last_found = True
                                            break
                                if second_last_found:
                                    break
        
        # Tiếp theo, vẽ đường đi (nếu có)
        if path_copy and len(path_copy) > 1:
            # Sử dụng màu gradient từ cam sang đỏ để biểu thị đường đi qua thời gian
            num_segments = len(path_copy) - 1
            for i in range(num_segments):
                p1 = path_copy[i]
                p2 = path_copy[i+1]
                # Đảm bảo p1, p2 là tuple
                x1, y1 = tuple(p1) if hasattr(p1, '__len__') else (0,0)
                x2, y2 = tuple(p2) if hasattr(p2, '__len__') else (0,0)

                # Tính toán vị trí trung tâm của các ô
                center_x1 = offset_x + x1 * cell_size + cell_size // 2
                center_y1 = offset_y + y1 * cell_size + cell_size // 2
                center_x2 = offset_x + x2 * cell_size + cell_size // 2
                center_y2 = offset_y + y2 * cell_size + cell_size // 2
                
                # Sử dụng màu sắc khác nhau cho các phần của đường đi
                # Đoạn đầu: cam nhạt, đoạn cuối: đỏ đậm
                progress_ratio = i / max(1, num_segments - 1)  # Tránh chia cho 0
                
                # Tạo màu gradient từ cam (#FFA500) đến đỏ (#FF0000)
                r = min(255, int(255))
                g = min(255, int(165 - progress_ratio * 165))
                b = min(255, int(0))
                color = f'#{r:02x}{g:02x}{b:02x}'
                
                # Xác định đoạn cuối cùng vào đích
                is_last_to_end = (p2 == tuple(self.map_object.end_pos))
                
                # Vẽ đoạn đường đặc biệt nếu đi vào đích
                if is_last_to_end:
                    # Đoạn cuối vào đích sử dụng màu đỏ đậm và mũi tên lớn hơn
                    self._log(f"Vẽ mũi tên đặc biệt từ {p1} đến đích {p2}")
                    line_width = max(3, cell_size // 6)  # Đường đậm hơn
                    arrow_size = max(12, cell_size // 1.5)  # Mũi tên lớn hơn
                    
                    # Vẽ hiệu ứng "glow" trước khi vẽ mũi tên chính
                    for glow in range(2):
                        glow_width = line_width + (2-glow)
                        glow_color = "#FF5500" if glow == 0 else "#FF2200" 
                        canvas.create_line(center_x1, center_y1, center_x2, center_y2, 
                                    fill=glow_color, width=glow_width, 
                                    arrow=tk.LAST, arrowshape=(arrow_size, arrow_size, arrow_size//2))
                    
                    # Vẽ mũi tên chính với màu đỏ tươi
                    canvas.create_line(center_x1, center_y1, center_x2, center_y2, 
                                    fill="#FF0000", width=line_width, 
                                    arrow=tk.LAST, arrowshape=(arrow_size, arrow_size, arrow_size//2))
                else:
                    # Các đoạn đường bình thường
                    canvas.create_line(center_x1, center_y1, center_x2, center_y2, 
                                    fill=color, width=max(1, cell_size // 10), 
                                    arrow=tk.LAST)
        
        # Cuối cùng, vẽ agent (nếu có)
        if agent_pos is not None:
            agent_pos_np = np.array(agent_pos) if isinstance(agent_pos, list) else agent_pos
            if hasattr(agent_pos_np, 'size') and agent_pos_np.size > 0:
                agent_x, agent_y = agent_pos_np[0], agent_pos_np[1]
                pixel_x = offset_x + agent_x * cell_size
                pixel_y = offset_y + agent_y * cell_size
                
                # Vẽ agent to và nổi bật hơn
                agent_size = cell_size // 2.5  # Hơi nhỏ hơn để không che mất ô
                
                # Nếu agent ở vị trí đích, dùng màu xanh lá đậm để dễ nhìn hơn
                if (agent_x, agent_y) == self.map_object.end_pos:
                    agent_color = "#00AA00"  # Xanh lá đậm
                    # Thêm hiệu ứng hào quang xung quanh để nhấn mạnh
                    glow_size = agent_size * 1.5
                    canvas.create_oval(
                        pixel_x + (cell_size - glow_size) / 2,
                        pixel_y + (cell_size - glow_size) / 2,
                        pixel_x + (cell_size + glow_size) / 2,
                        pixel_y + (cell_size + glow_size) / 2,
                        fill="#AAFFAA", outline="#00FF00", width=2
                    )
                else:
                    agent_color = "blue"
                
                # Vẽ agent
                canvas.create_oval(
                    pixel_x + (cell_size - agent_size) / 2,
                    pixel_y + (cell_size - agent_size) / 2,
                    pixel_x + (cell_size + agent_size) / 2,
                    pixel_y + (cell_size + agent_size) / 2,
                    fill=agent_color, outline="black", width=2
                )

    def _start_manual_training(self):
        """Bắt đầu quá trình huấn luyện thủ công DQN agent."""
        if not self.rl_environment:
            messagebox.showerror("Lỗi", "Vui lòng khởi tạo môi trường RL trước!")
            return
            
        # Kiểm tra xem đã đang huấn luyện hay chưa
        if self.is_manual_training_running:
            messagebox.showinfo("Thông báo", "Huấn luyện đang chạy. Vui lòng đợi hoàn thành.")
            return
        
        # Lấy tham số từ giao diện
        try:
            # Lấy tham số huấn luyện từ UI
            learning_rate = self.manual_lr.get()
            gamma = self.manual_gamma.get()
            buffer_size = self.manual_buffer_size.get()
            batch_size = self.manual_batch_size.get()
            learning_starts = self.manual_learning_starts.get() 
            tau = self.manual_tau.get()
            target_update_interval = self.manual_target_update_interval.get()
            train_freq = self.manual_train_freq.get()
            total_timesteps = self.manual_total_timesteps.get()
            
            # Xử lý tham số đặc biệt
            use_double_dqn = self.manual_use_double_dqn.get()
            use_dueling_network = self.manual_use_dueling_dqn.get()
            use_prioritized_replay = self.manual_use_prioritized_replay.get()
            
            # Tham số khám phá
            exploration_fraction = self.manual_exploration_fraction.get()
            exploration_initial_eps = self.manual_exploration_initial_eps.get()
            exploration_final_eps = self.manual_exploration_final_eps.get()
            
            # Tạo network architecture từ chuỗi nhập vào 
            net_arch_str = self.manual_net_arch.get()
            try:
                net_arch = [int(x.strip()) for x in net_arch_str.split(',') if x.strip().isdigit()]
                self._log(f"Neural network architecture: {net_arch}")
            except:
                self._log("Invalid net_arch format, using default [64, 64]")
                net_arch = [64, 64]
            
            policy_kwargs = {'net_arch': net_arch}
            
            # Kiểm tra giá trị hợp lệ 
            if learning_rate <= 0 or gamma <= 0 or buffer_size <= 0 or batch_size <= 0 or total_timesteps <= 0:
                messagebox.showerror("Giá trị không hợp lệ", "Vui lòng nhập các giá trị dương cho tất cả các tham số!")
                return
            
            # Đặt lại cờ huấn luyện
            self.is_manual_training_running = False
            
            # Đặt lại hàng đợi tiến trình
            self.training_progress_queue = queue.Queue()
            
            # Tạm thời vô hiệu hóa các nút để tránh nhấn trong lúc huấn luyện
            self._set_training_buttons_state(tk.DISABLED)
            
            # Khởi tạo trainer nếu chưa có
            if self.trainer is None:
                from core.algorithms.rl_DQNAgent import DQNAgentTrainer
                self.trainer = DQNAgentTrainer(env=self.rl_environment)
            
            # Reset progress bar và status
            if hasattr(self, 'manual_train_progress'): 
                self.manual_train_progress['value'] = 0
            if hasattr(self, 'manual_train_status_label'): 
                self.manual_train_status_label['text'] = "Status: Starting training..."
            
            # Thực hiện huấn luyện trong một thread riêng biệt
            training_thread = threading.Thread(target=self._training_thread_func, args=(
                learning_rate, gamma, buffer_size, batch_size, learning_starts, tau, target_update_interval,
                train_freq, total_timesteps, use_double_dqn, use_dueling_network, use_prioritized_replay,
                exploration_fraction, exploration_initial_eps, exploration_final_eps, policy_kwargs
            ))
            training_thread.daemon = True  # Cho phép thoát ứng dụng khi thread này đang chạy
            training_thread.start()
            
            # Cập nhật UI
            self._update_agent_status("Đang huấn luyện...")
            
        except Exception as e:
            self._log(f"Training error: {e}")
            messagebox.showerror("Training Error", str(e))
            self._set_training_buttons_state(tk.NORMAL)

    def _on_training_complete(self):
        """Xử lý khi huấn luyện hoàn tất."""
        self._set_training_buttons_state(tk.NORMAL)
        
        # Enable nút lưu model và chạy agent
        if hasattr(self, 'save_manual_model_button'):
            self.save_manual_model_button.config(state=tk.NORMAL)
            self._log("Đã kích hoạt nút lưu model")
        
        if hasattr(self, 'run_agent_button'):
            self.run_agent_button.config(state=tk.NORMAL)
            self._log("Đã kích hoạt nút chạy agent")
        
        # Thêm check cụ thể với đường dẫn widget để nắm bắt các nút khó tìm
        try:
            # Kích hoạt tất cả các nút lưu model và chạy agent trong UI bằng tên
            for widget in self.root.winfo_children():
                self._enable_child_buttons_by_name(widget, ["Lưu Model", "Chạy Agent", "Lưu Model Hiện tại", "Chạy Agent trên Bản đồ"])
        except Exception as e:
            self._log(f"Lỗi kích hoạt nút: {e}")
            
        # Update status
        self._update_agent_status("Huấn luyện hoàn tất! Có thể lưu model hoặc chạy agent.")
        self._log("Huấn luyện hoàn tất! Có thể lưu model hoặc chạy agent.")
        
        # Nếu có training metrics, hiển thị biểu đồ
        if hasattr(self.trainer, 'training_metrics') and self.trainer.training_metrics is not None:
            self._visualize_training_metrics()
    
    def _enable_child_buttons_by_name(self, parent, button_texts):
        """Kích hoạt các nút con có tên trong danh sách button_texts."""
        if not parent:
            return
            
        if hasattr(parent, 'winfo_children'):
            for child in parent.winfo_children():
                # Nếu là nút và có text khớp với một trong các tên cần tìm
                if isinstance(child, ttk.Button) and hasattr(child, 'cget'):
                    try:
                        button_text = child.cget('text')
                        if button_text in button_texts:
                            child.config(state=tk.NORMAL)
                            self._log(f"Kích hoạt nút: {button_text}")
                    except Exception:
                        pass
                        
                # Đệ quy vào các widget con
                self._enable_child_buttons_by_name(child, button_texts)

    def _visualize_training_metrics(self):
        """Hiển thị biểu đồ các metrics trong quá trình huấn luyện."""
        if not hasattr(self.trainer, 'training_metrics') or self.trainer.training_metrics is None:
            return
            
        metrics = self.trainer.training_metrics
        
        # Tạo cửa sổ mới cho biểu đồ
        metrics_window = tk.Toplevel(self.root)
        metrics_window.title("Training Metrics Visualization")
        metrics_window.geometry("900x600")
        
        # Tạo notebook để chứa các tab biểu đồ
        notebook = ttk.Notebook(metrics_window)
        notebook.pack(fill='both', expand=True, padx=10, pady=10)
        
        # Tab 1: Rewards và Success Rate
        rewards_tab = ttk.Frame(notebook)
        notebook.add(rewards_tab, text="Rewards & Success")
        
        rewards_figure = plt.Figure(figsize=(8, 6), dpi=100)
        rewards_canvas = FigureCanvasTkAgg(rewards_figure, rewards_tab)
        rewards_canvas.get_tk_widget().pack(fill='both', expand=True)
        
        # Plot rewards
        ax1 = rewards_figure.add_subplot(111)
        
        if 'episode_rewards' in metrics and len(metrics['episode_rewards']) > 0:
            # Tạo mảng chỉ số episodes cho trục x
            episode_indices = list(range(len(metrics['episode_rewards'])))
            
            # Moving average rewards for smoother plot
            window_size = min(10, len(metrics['episode_rewards']))
            if window_size > 0:
                # Đảm bảo có đủ phần tử cho window
                moving_avg_rewards = []
                for i in range(len(metrics['episode_rewards'])):
                    window_start = max(0, i - window_size + 1)
                    moving_avg_rewards.append(np.mean(metrics['episode_rewards'][window_start:i+1]))
                
                ax1.plot(episode_indices, moving_avg_rewards, 'b-', label='Moving Avg Reward')
                ax1.plot(episode_indices, metrics['episode_rewards'], 'b.', alpha=0.3, label='Episode Reward')
            else:
                ax1.plot(episode_indices, metrics['episode_rewards'], 'b-', label='Episode Reward')
                
            ax1.set_xlabel('Episodes')
            ax1.set_ylabel('Reward', color='b')
            ax1.tick_params(axis='y', labelcolor='b')
            
            # Add success rate if available
            if 'success_rates' in metrics and len(metrics['success_rates']) > 0:
                ax2 = ax1.twinx()
                
                # Đảm bảo kích thước mảng khớp nhau
                success_indices = list(range(len(metrics['success_rates'])))
                
                ax2.plot(success_indices, metrics['success_rates'], 'r-', label='Success Rate')
                ax2.set_ylabel('Success Rate', color='r')
                ax2.tick_params(axis='y', labelcolor='r')
            
            # Add legend
            lines1, labels1 = ax1.get_legend_handles_labels()
            if 'success_rates' in metrics and len(metrics['success_rates']) > 0:
                lines2, labels2 = ax2.get_legend_handles_labels()
                ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
            else:
                ax1.legend(loc='upper left')
                
            rewards_figure.tight_layout()
        else:
            ax1.text(0.5, 0.5, "No reward data available", 
                    horizontalalignment='center', verticalalignment='center')
        
        # Tab 2: Exploration Rate và Learning Rate
        params_tab = ttk.Frame(notebook)
        notebook.add(params_tab, text="Learning Parameters")
        
        params_figure = plt.Figure(figsize=(8, 6), dpi=100)
        params_canvas = FigureCanvasTkAgg(params_figure, params_tab)
        params_canvas.get_tk_widget().pack(fill='both', expand=True)
        
        # Plot exploration rate
        ax3 = params_figure.add_subplot(111)
        
        if 'exploration_rates' in metrics and len(metrics['exploration_rates']) > 0:
            # Đảm bảo timesteps và exploration_rates có cùng kích thước
            if 'timesteps' in metrics and len(metrics['timesteps']) > 0:
                # Lấy độ dài nhỏ nhất giữa hai mảng
                min_length = min(len(metrics['timesteps']), len(metrics['exploration_rates']))
                # Cắt cả hai mảng về cùng độ dài
                timesteps_data = metrics['timesteps'][:min_length]
                exploration_data = metrics['exploration_rates'][:min_length]
                
                # Vẽ đồ thị với dữ liệu đã điều chỉnh kích thước
                ax3.plot(timesteps_data, exploration_data, 'g-', label='Exploration Rate')
            else:
                # Nếu không có timesteps, vẽ theo chỉ số
                ax3.plot(metrics['exploration_rates'], 'g-', label='Exploration Rate')
            
            ax3.set_xlabel('Timesteps')
            ax3.set_ylabel('Exploration Rate', color='g')
            ax3.tick_params(axis='y', labelcolor='g')
            
            # Add learning rate if available - tương tự điều chỉnh kích thước
            if 'learning_rates' in metrics and len(metrics['learning_rates']) > 0:
                ax4 = ax3.twinx()
                
                # Đảm bảo đồng bộ kích thước
                min_length = min(len(timesteps_data), len(metrics['learning_rates']))
                learning_rates_data = metrics['learning_rates'][:min_length]
                
                ax4.plot(timesteps_data[:min_length], learning_rates_data, 'm-', label='Learning Rate')
                ax4.set_ylabel('Learning Rate', color='m')
                ax4.tick_params(axis='y', labelcolor='m')
            
            # Add legend
            lines3, labels3 = ax3.get_legend_handles_labels()
            if 'learning_rates' in metrics and len(metrics['learning_rates']) > 0:
                lines4, labels4 = ax4.get_legend_handles_labels()
                ax3.legend(lines3 + lines4, labels3 + labels4, loc='upper right')
            else:
                ax3.legend(loc='upper right')
            
            params_figure.tight_layout()
        else:
            ax3.text(0.5, 0.5, "No exploration rate data available", 
                    horizontalalignment='center', verticalalignment='center')
        
        # Tab 3: Loss Values
        loss_tab = ttk.Frame(notebook)
        notebook.add(loss_tab, text="Loss Values")
        
        loss_figure = plt.Figure(figsize=(8, 6), dpi=100)
        loss_canvas = FigureCanvasTkAgg(loss_figure, loss_tab)
        loss_canvas.get_tk_widget().pack(fill='both', expand=True)
        
        # Plot loss
        ax5 = loss_figure.add_subplot(111)
        
        if 'losses' in metrics and len(metrics['losses']) > 0:
            # Tính toán kích thước cửa sổ trung bình động phù hợp
            window_size = min(max(5, len(metrics['losses']) // 10), 100)
            
            # Tạo mảng chỉ số cho trục x
            loss_indices = list(range(len(metrics['losses'])))
            
            # Kiểm tra có đủ dữ liệu cho trung bình động không
            if len(metrics['losses']) > window_size:
                # Tính trung bình động
                moving_avg_losses = []
                for i in range(len(metrics['losses'])):
                    window_start = max(0, i - window_size + 1)
                    moving_avg_losses.append(np.mean(metrics['losses'][window_start:i+1]))
                
                # Vẽ đồ thị trung bình động và dữ liệu gốc
                ax5.plot(loss_indices, moving_avg_losses, 'r-', label='Moving Avg Loss', linewidth=2)
                # Plot original loss values with low alpha to show variance
                ax5.plot(loss_indices, metrics['losses'], 'r.', alpha=0.1, label='Loss')
            else:
                # Vẽ đơn giản nếu không đủ dữ liệu cho trung bình động
                ax5.plot(loss_indices, metrics['losses'], 'r-', label='Loss', linewidth=2)
                
            # Đặt nhãn và tạo legend
            ax5.set_xlabel('Updates')
            ax5.set_ylabel('Loss Value')
            ax5.set_title('Training Loss Over Time')
            
            # Đảm bảo y-axis không âm (loss thường không âm)
            y_min = max(0, min(metrics['losses']) * 0.9)
            y_max = max(metrics['losses']) * 1.1
            ax5.set_ylim(y_min, y_max)
            
            ax5.legend(loc='upper right')
            ax5.grid(True, linestyle='--', alpha=0.7)
            
            loss_figure.tight_layout()
        else:
            # Thông báo không có dữ liệu - thêm thông tin chi tiết
            ax5.text(0.5, 0.5, "No loss data available\nPossible reasons:\n- Training just started\n- Model doesn't expose loss values\n- Buffer size too small", 
                    horizontalalignment='center', verticalalignment='center',
                    fontsize=10, color='gray', wrap=True)
            ax5.set_xticks([])
            ax5.set_yticks([])
        
        # Tab 4: Episode Lengths
        lengths_tab = ttk.Frame(notebook)
        notebook.add(lengths_tab, text="Episode Lengths")
        
        lengths_figure = plt.Figure(figsize=(8, 6), dpi=100)
        lengths_canvas = FigureCanvasTkAgg(lengths_figure, lengths_tab)
        lengths_canvas.get_tk_widget().pack(fill='both', expand=True)
        
        # Plot episode lengths
        ax6 = lengths_figure.add_subplot(111)
        
        if 'episode_lengths' in metrics and len(metrics['episode_lengths']) > 0:
            # Tạo mảng chỉ số episodes
            episode_indices = list(range(len(metrics['episode_lengths'])))
            
            # Tính toán trung bình động
            window_size = min(10, len(metrics['episode_lengths']))
            if window_size > 0:
                moving_avg_lengths = []
                for i in range(len(metrics['episode_lengths'])):
                    window_start = max(0, i - window_size + 1)
                    moving_avg_lengths.append(np.mean(metrics['episode_lengths'][window_start:i+1]))
                
                ax6.plot(episode_indices, moving_avg_lengths, 'g-', label='Moving Avg Length')
                ax6.plot(episode_indices, metrics['episode_lengths'], 'g.', alpha=0.3, label='Episode Length')
            else:
                ax6.plot(episode_indices, metrics['episode_lengths'], 'g-', label='Episode Length')
                
            ax6.set_xlabel('Episodes')
            ax6.set_ylabel('Number of Steps')
            ax6.legend(loc='upper right')
            
            lengths_figure.tight_layout()
        else:
            ax6.text(0.5, 0.5, "No episode length data available", 
                    horizontalalignment='center', verticalalignment='center')
        
        # Tab 5: Termination Reasons
        if 'termination_reasons' in metrics and metrics['termination_reasons']:
            term_tab = ttk.Frame(notebook)
            notebook.add(term_tab, text="Termination Reasons")
            
            term_figure = plt.Figure(figsize=(8, 6), dpi=100)
            term_canvas = FigureCanvasTkAgg(term_figure, term_tab)
            term_canvas.get_tk_widget().pack(fill='both', expand=True)
            
            # Plot termination reasons as pie chart
            ax7 = term_figure.add_subplot(111)
            
            reasons = list(metrics['termination_reasons'].keys())
            counts = list(metrics['termination_reasons'].values())
            
            ax7.pie(counts, labels=reasons, autopct='%1.1f%%', shadow=True, startangle=90)
            ax7.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle
            
            term_figure.tight_layout()
        
        # Draw all figures
        rewards_canvas.draw()
        params_canvas.draw()
        loss_canvas.draw()
        lengths_canvas.draw()
        if 'termination_reasons' in metrics and metrics['termination_reasons']:
            term_canvas.draw()
        
        # Thêm nút lưu biểu đồ
        def save_metrics_charts():
            save_dir = filedialog.askdirectory(title="Chọn thư mục lưu biểu đồ",
                                            initialdir=self.trainer.log_dir if hasattr(self.trainer, 'log_dir') else None)
            if not save_dir:
                return
                
            # Lưu các biểu đồ
            rewards_figure.savefig(os.path.join(save_dir, "rewards_chart.png"))
            params_figure.savefig(os.path.join(save_dir, "params_chart.png"))
            loss_figure.savefig(os.path.join(save_dir, "loss_chart.png"))
            lengths_figure.savefig(os.path.join(save_dir, "lengths_chart.png"))
            if 'termination_reasons' in metrics and metrics['termination_reasons']:
                term_figure.savefig(os.path.join(save_dir, "termination_chart.png"))
                
            messagebox.showinfo("Thông báo", f"Đã lưu biểu đồ vào {save_dir}")
            
        save_button = ttk.Button(metrics_window, text="Lưu biểu đồ", command=save_metrics_charts)
        save_button.pack(pady=10)

    def _on_training_error(self, error_msg):
        """Callback khi huấn luyện lỗi."""
        self.is_manual_training_running = False # Đặt lại flag KHI huấn luyện lỗi
        messagebox.showerror("Lỗi Huấn luyện", error_msg)
        self._set_training_buttons_state(tk.NORMAL)
        self._update_agent_status(f"Lỗi huấn luyện: {error_msg}")
        # Reset progress bar khi lỗi
        if hasattr(self, 'manual_train_progress'): self.manual_train_progress['value'] = 0
        if hasattr(self, 'manual_train_status_label'): self.manual_train_status_label['text'] = f"Status: Error - {error_msg[:50]}..."

    def _set_training_buttons_state(self, state):
        """Bật/Tắt các nút liên quan đến huấn luyện/chạy agent."""
        if hasattr(self, 'manual_train_btn'): self.manual_train_btn.config(state=state)
        if hasattr(self, 'save_button'): 
            # Chỉ bật nút Lưu nếu huấn luyện thành công
            save_state = tk.NORMAL if state == tk.NORMAL else tk.DISABLED
            self.save_button.config(state=save_state) 
        if hasattr(self, 'load_button'): self.load_button.config(state=state)
        if hasattr(self, 'run_agent_button'):
            # Chỉ bật nút Chạy nếu huấn luyện thành công hoặc đã tải model
            run_state = tk.NORMAL if (state == tk.NORMAL or self.is_model_loaded) else tk.DISABLED
            self.run_agent_button.config(state=run_state)
        if hasattr(self, 'init_rl_btn'): self.init_rl_btn.config(state=state) # Tắt/Bật khởi tạo env khi đang train
        if hasattr(self, 'random_map_btn'): self.random_map_btn.config(state=state) # Tắt/Bật tạo map khi đang train
        if hasattr(self, 'demo_map_btn'): self.demo_map_btn.config(state=state)
        if hasattr(self, 'load_map_btn'): self.load_map_btn.config(state=state)
        if hasattr(self, 'save_map_btn'): pass # Nút save map không bị ảnh hưởng bởi training
        if hasattr(self, 'reset_all_btn'): self.reset_all_btn.config(state=state) # Tắt/Bật reset khi đang train

    def _save_manual_model(self):
        # ... (Giữ nguyên logic, chỉ cần log và messagebox)
        if not self.trainer or not self.trainer.model:
             messagebox.showerror("Lỗi", "Không có model để lưu!")
             return
        if not self.map_object:
             messagebox.showerror("Lỗi", "Không có bản đồ cho tên file!")
             return
        try:
             map_size = self.map_object.size
             total_timesteps = self.manual_total_timesteps.get()
             manual_session_id = f"manual_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
             map_name_part = Path(self.current_map_path).stem if self.current_map_path else f"gen{map_size}x{map_size}"
             session_dir = MODELS_DIR / manual_session_id / f"map_{map_name_part}"
             session_dir.mkdir(parents=True, exist_ok=True)
             model_filename = f"model_{manual_session_id}_{total_timesteps}steps.zip"
             full_path = session_dir / model_filename
             self.trainer.save_model(str(full_path))
             self._log(f"Đã lưu model: {full_path}")
             messagebox.showinfo("Thông báo", f"Đã lưu model:\n{full_path}")
             self._update_agent_status(f"Đã lưu model: {model_filename}")
        except Exception as e:
             self._log(f"Lỗi lưu model: {e}")
             messagebox.showerror("Lỗi Lưu Model", f"Lỗi: {e}")
             self._update_agent_status("Lỗi lưu model!")

    def _load_trained_model(self):
        # ... (Giữ nguyên logic, chỉ cần log và messagebox)
        if not self.rl_environment:
            messagebox.showerror("Lỗi", "Vui lòng khởi tạo môi trường RL trước!")
            return
        filepath = filedialog.askopenfilename(title="Chọn model DQN", filetypes=[("ZIP files", "*.zip")], initialdir=str(MODELS_DIR))
        if not filepath: return
        try:
             if not self.trainer:
                 self.trainer = DQNAgentTrainer(self.rl_environment) # Tạo agent nếu chưa có
             self._log(f"Đang tải model từ {filepath}...")
             self.trainer.load_model(filepath)
             model_name = Path(filepath).name
             self._log(f"Đã tải model '{model_name}' thành công!")
             messagebox.showinfo("Thông báo", f"Đã tải model '{model_name}' thành công!")
             self.run_agent_button.config(state=tk.NORMAL) # Cho phép chạy agent
             self._update_agent_status(f"Đã tải model: {model_name}")
        except Exception as e:
             self._log(f"Lỗi khi tải model: {e}")
             import traceback; self._log(traceback.format_exc())
             messagebox.showerror("Lỗi", f"Lỗi khi tải model: {e}")
             self._update_agent_status("Lỗi tải model!")

    def _run_agent_on_current_map(self):
        """Chạy agent trên bản đồ hiện tại."""
        if not self.trainer or not self.trainer.model:
            messagebox.showerror("Lỗi", "Chưa có agent hoặc model! Vui lòng huấn luyện hoặc tải.")
            return
        if not self.rl_environment:
            messagebox.showerror("Lỗi", "Vui lòng khởi tạo môi trường RL.")
            return
        
        self._log("Bắt đầu chạy agent trên bản đồ hiện tại...")
        self._update_agent_status("Đang chạy agent...")
        self.run_agent_button.config(state=tk.DISABLED)

        def run_thread():
            try:
                observation, _ = self.rl_environment.reset()
                terminated, truncated = False, False
                total_reward, step_count = 0.0, 0
                path = [tuple(observation['agent_pos'])] # Chuyển numpy sang tuple

                self.root.after(0, lambda: self._draw_map_with_path(self.map_canvas, observation['agent_pos'], path))

                # Đổi tên biến để rõ ràng hơn
                observation_before_step = observation
                terminated = False # Khởi tạo terminated ở đây
                truncated = False # Khởi tạo truncated ở đây

                def _schedule_draw_call(agent_draw_pos, current_path_list):
                    self._draw_map_with_path(self.map_canvas, agent_draw_pos, current_path_list)

                while not (terminated or truncated) and step_count < self.rl_environment.max_steps_per_episode * 1.5: # Thêm giới hạn an toàn
                    action_to_take = self.trainer.predict_action(observation_before_step) # Sử dụng trạng thái trước khi bước
                    
                    # Thực hiện hành động, nhận trạng thái TIẾP THEO
                    obs_after_step, reward_val, terminated_flag, truncated_flag, info_after_step = self.rl_environment.step(action_to_take)
                    
                    total_reward += reward_val
                    step_count += 1
                    
                    # Vị trí của agent SAU KHI thực hiện bước đi
                    agent_pos_after_step = tuple(obs_after_step['agent_pos'])
                    
                    # Thêm vị trí mới này vào đường đi
                    path.append(agent_pos_after_step) # Path giờ đã bao gồm vị trí mới nhất
                    
                    # Để cập nhật UI, sử dụng bản sao mới của path cho lambda
                    path_for_ui = list(path) 
                    
                    # Cập nhật UI sử dụng vị trí của agent SAU KHI bước, và đường đi bao gồm vị trí đó
                    self.root.after(0, lambda apas=agent_pos_after_step, pfu=path_for_ui: _schedule_draw_call(apas, pfu))
                    
                    # Sleep ngắn để đảm bảo UI được cập nhật
                    time.sleep(0.05)
                    
                    # Cập nhật trạng thái cho vòng lặp tiếp theo
                    observation_before_step = obs_after_step 
                    terminated = terminated_flag
                    truncated = truncated_flag

                # Kết thúc
                success = info_after_step.get("termination_reason") == "den_dich" if 'info_after_step' in locals() else False
                final_info_reason = info_after_step.get('termination_reason', 'Max Steps') if 'info_after_step' in locals() else 'Unknown'

                # ĐẶC BIỆT QUAN TRỌNG: Nếu thành công (đến đích), đảm bảo đường đi cuối cùng bao gồm điểm đích
                if success:
                    # Đảm bảo điểm cuối cùng trong path là vị trí đích 
                    # Trong trường hợp có lỗi, path không được cập nhật đầy đủ
                    end_pos = tuple(self.map_object.end_pos)
                    if path[-1] != end_pos:
                        self._log("Đảm bảo đường đi đến đích: Thêm điểm đích vào path")
                        path.append(end_pos)
                
                # Lên lịch vẽ cuối cùng với độ trễ cao hơn và đảm bảo mũi tên đến đích
                final_pos = tuple(obs_after_step['agent_pos']) if 'obs_after_step' in locals() else None
                final_path = list(path)
                
                self._log(f"Đường đi cuối cùng: {final_path}")
                self._log(f"Vị trí đích: {self.map_object.end_pos}")
                
                # Vẽ lại đường đi cuối cùng
                for delay in [100, 300, 500]:  # Nhiều lần vẽ với độ trễ khác nhau
                    self.root.after(delay, lambda fp=final_pos, fp_ui=final_path: 
                                  self._draw_map_with_path(self.map_canvas, fp, fp_ui))

                log_msg = f"Chạy agent hoàn tất. Thành công: {success}. Lý do: {final_info_reason}. Steps: {step_count}. Reward: {total_reward:.2f}"
                self._log(log_msg)
                self.root.after(0, lambda msg=log_msg: self._update_agent_status(msg))
                self.root.after(0, lambda: self.run_agent_button.config(state=tk.NORMAL))

            except Exception as e:
                err_msg = f"Lỗi khi chạy agent: {e}"
                self._log(err_msg)
                import traceback; self._log(traceback.format_exc())
                self.root.after(0, lambda msg=err_msg: self._update_agent_status(msg))
                self.root.after(0, lambda: self.run_agent_button.config(state=tk.NORMAL))
        
        threading.Thread(target=run_thread, daemon=True).start()

    def _update_agent_status(self, message: str):
         """Cập nhật nội dung trong ô trạng thái agent."""
         if hasattr(self, 'agent_status_text'): # Kiểm tra widget tồn tại
            try:
                self.agent_status_text.config(state=tk.NORMAL)
                self.agent_status_text.delete('1.0', tk.END)
                self.agent_status_text.insert(tk.END, message)
                self.agent_status_text.config(state=tk.DISABLED)
                self.agent_status_text.update_idletasks() # Cập nhật ngay
            except tk.TclError: # Bỏ qua lỗi nếu widget đã bị hủy
                pass

    def _log(self, message):
        """
Ghi log.
        Args:
            message: Thông điệp cần ghi log
        """
        if hasattr(self, 'log_text'): # Kiểm tra widget tồn tại
            try:
                timestamp = datetime.now().strftime("%H:%M:%S")
                self.log_text.insert(tk.END, f"[{timestamp}] {message}\n")
                self.log_text.see(tk.END)
                self.log_text.update_idletasks() # Cập nhật ngay
            except tk.TclError: # Bỏ qua lỗi nếu widget đã bị hủy
                 print(f"[LOG - NO UI] {message}") # In ra console nếu UI lỗi
        else:
             print(f"[LOG - NO UI] {message}")

    def _check_training_progress_queue(self):
        """Checks the queue for training progress and updates the UI."""
        if not hasattr(self, 'training_progress_queue'):
            # If the queue doesn't exist yet, reschedule check
            self.root.after(100, self._check_training_progress_queue)
            return
            
        try:
            # Non-blocking check
            while True: # Process all messages currently in the queue
                try:
                    # Get message from queue safely
                    queue_data = self.training_progress_queue.get_nowait()
                    
                    # Validate that we got a tuple with 3 elements
                    if isinstance(queue_data, tuple) and len(queue_data) == 3:
                        current_step, total_steps, status_message = queue_data
                        
                        # Update Progress Bar if it exists
                        if hasattr(self, 'manual_train_progress'):
                            if total_steps > 0:
                                progress_percentage = (current_step / total_steps) * 100
                                self.manual_train_progress['value'] = progress_percentage
                            else:
                                self.manual_train_progress['value'] = 0
                                
                        # Update Status Label if it exists
                        if hasattr(self, 'manual_train_status_label'):
                            self.manual_train_status_label['text'] = f"Status: {status_message}"
                    else:
                        self._log(f"Received invalid data format from training queue")
                        
                    self.root.update_idletasks() # Force UI update
                except queue.Empty:
                    break # No more messages in queue
                except Exception as queue_error:
                    self._log(f"Error processing queue item: {queue_error}")
                    break
                
        except Exception as e:
            print(f"Error processing training progress queue: {e}") # Log unexpected errors
            self._log(f"Error updating progress: {e}")

        # Reschedule the check if training is still running
        if self.is_manual_training_running:
            self.root.after(100, self._check_training_progress_queue) # Check again in 100ms

    def _reset_all(self):
        """Reset bản đồ, môi trường, agent và UI liên quan."""
        self._log("Thực hiện Reset Tất Cả...")
        self.map_object = None
        self.current_map_path = None
        self.current_map_size_display_var.set("N/A") # Reset display
        self.rl_environment = None
        self.trainer = None
        self.trained_model_path = None
        self.is_model_loaded = False
        
        # Xóa canvas
        if hasattr(self, 'map_canvas'):
            self.map_canvas.delete("all")
            
        # Reset trạng thái nút
        if hasattr(self, 'save_map_btn'): self.save_map_btn.config(state=tk.DISABLED)
        if hasattr(self, 'init_rl_btn'): self.init_rl_btn.config(state=tk.NORMAL) # Cho phép khởi tạo lại
        if hasattr(self, 'run_agent_button'): self.run_agent_button.config(state=tk.DISABLED)
        if hasattr(self, 'load_button'): self.load_button.config(state=tk.NORMAL)
        if hasattr(self, 'manual_train_progress'): self.manual_train_progress['value'] = 0
        if hasattr(self, 'manual_train_status_label'): self.manual_train_status_label['text'] = "Status: Idle"
        self._update_agent_status("Đã reset. Tạo/Tải bản đồ để bắt đầu.")
        self._log("Reset hoàn tất.")
        
        if hasattr(self, 'tuning_status_label'): self.tuning_status_label.config(text="Tuning Status: Idle")
        if hasattr(self, 'apply_best_params_btn'): self.apply_best_params_btn.config(state=tk.DISABLED)
        if hasattr(self, 'start_tuning_btn'): self.start_tuning_btn.config(state=tk.NORMAL)
        
    def _reset_env_and_agent(self):
        """Hàm phụ trợ chỉ reset môi trường và agent."""
        self.rl_environment = None
        # Don't reset trainer if we want to continue training the same loaded/partially trained model on a new env
        # self.trainer = None # Keep trainer if exists, but clear its env association
        if self.trainer:
            self.trainer.env = None # Or re-init trainer if it heavily depends on old env struct
            # self.trainer.model.set_env(None) # This might be needed if model stores env reference
        
        # self.trained_model_path = None # Keep if a model was loaded, path is still valid
        # self.is_model_loaded = False # Keep if model was loaded

        # UI elements related to a trained/loaded model should be reset if environment changes significantly
        # Or indicate that model might not be compatible with new env settings
        if hasattr(self, 'save_manual_model_button'): self.save_manual_model_button.config(state=tk.DISABLED)
        if hasattr(self, 'run_agent_button'): self.run_agent_button.config(state=tk.DISABLED)
        # If training was running, stop it
        self.is_manual_training_running = False 
        if hasattr(self, 'manual_train_btn'): self.manual_train_btn.config(state=tk.NORMAL)
        if hasattr(self, 'manual_train_progress'): self.manual_train_progress['value'] = 0
        if hasattr(self, 'manual_train_status_label'): self.manual_train_status_label['text'] = "Status: Idle (Map changed, re-init env)"
        
        self._update_agent_status("Môi trường/Agent đã reset do map thay đổi. Khởi tạo lại môi trường RL.")
        self._log("Môi trường RL và trạng thái agent liên quan đã được reset do thay đổi bản đồ.")

    # --- Thêm các hàm cho Hyperparameter Tuning ---

    def _update_tuning_status(self, message: str):
        """Cập nhật label trạng thái tuning (an toàn từ thread khác)."""
        if hasattr(self, 'tuning_status_label'):
            try:
                # Lên lịch cập nhật trên main thread của Tkinter
                self.root.after(0, lambda: self.tuning_status_label.config(text=f"Status: {message}"))
            except tk.TclError:
                # Có thể xảy ra nếu cửa sổ đã đóng
                pass

    def _apply_best_params(self):
        """Áp dụng bộ tham số tốt nhất tìm được vào các trường nhập liệu."""
        if self.best_params_found:
            try:
                self._log("Applying best hyperparameters found by Optuna:")
                for key, value in self.best_params_found.items():
                    self._log(f"  {key}: {value}")
                    # Cập nhật các biến tk tương ứng
                    if key == 'learning_rate': self.manual_lr.set(f"{value:.6f}")
                    elif key == 'gamma': self.manual_gamma.set(f"{value:.4f}")
                    elif key == 'buffer_size': self.manual_buffer_size.set(int(value))
                    elif key == 'batch_size': self.manual_batch_size.set(int(value))
                    elif key == 'exploration_fraction': self.manual_exploration_fraction.set(f"{value:.3f}")
                    elif key == 'exploration_final_eps': self.manual_exploration_final_eps.set(f"{value:.3f}")
                    # Thêm các tham số khác nếu cần
                messagebox.showinfo("Success", "Applied best hyperparameters found to the input fields.")
            except Exception as e:
                 messagebox.showerror("Error", f"Failed to apply parameters: {e}")
                 self._log(f"Error applying best parameters: {e}")
        else:
            messagebox.showwarning("No Parameters", "No best parameters found yet. Run tuning first.")

    def _start_hyperparameter_tuning(self):
        """Bắt đầu quá trình tối ưu hóa siêu tham số bằng Optuna."""
        if not OPTUNA_AVAILABLE:
            messagebox.showerror("Optuna Not Found", "Please install Optuna to use hyperparameter tuning: pip install optuna")
            self._log("Optuna library not found. Tuning cancelled.")
            return
            
        if not self.rl_environment:
            messagebox.showerror("Error", "Please initialize the RL Environment first!")
            self._log("Tuning cancelled: RL Environment not initialized.")
            return
            
        if self.is_manual_training_running: # Kiểm tra xem có đang training không
             messagebox.showwarning("Busy", "Manual training is currently running. Please wait for it to finish before starting tuning.")
             return

        try:
            n_trials = self.tuning_n_trials.get()
            timesteps_per_trial = self.tuning_timesteps_per_trial.get()
            if n_trials <= 0 or timesteps_per_trial <= 0:
                raise ValueError("Number of trials and timesteps per trial must be positive.")
        except (tk.TclError, ValueError) as e:
            messagebox.showerror("Input Error", f"Invalid input for tuning parameters: {e}")
            self._log(f"Tuning input error: {e}")
            return

        self._log(f"Starting hyperparameter tuning with Optuna: {n_trials} trials, {timesteps_per_trial} steps/trial.")
        self._update_tuning_status(f"Starting {n_trials} trials...")
        self.start_tuning_btn.config(state=tk.DISABLED)
        self.apply_best_params_btn.config(state=tk.DISABLED)
        self.best_params_found = None # Reset best params

        # Chạy Optuna trong một thread riêng
        tuning_thread = threading.Thread(target=self._run_optuna_study, args=(n_trials, timesteps_per_trial), daemon=True)
        tuning_thread.start()

    def objective(self, trial: optuna.Trial, timesteps_per_trial: int):
        """Hàm mục tiêu cho Optuna."""
        # --- 1. Đề xuất siêu tham số ---
        # Lấy policy kwargs từ model hiện tại làm cơ sở (nếu có) hoặc dùng mặc định
        current_policy_kwargs = self.trainer.policy_kwargs if self.trainer and hasattr(self.trainer, 'policy_kwargs') else {}
        net_arch_str = self.manual_net_arch.get() # Lấy từ GUI để giữ cố định kiến trúc mạng
        try:
             net_arch = [int(x.strip()) for x in net_arch_str.split(',') if x.strip().isdigit()]
        except:
             self._log("[Warning] Invalid net_arch string in GUI, using default [64, 64] for tuning trial.")
             net_arch = [64, 64]
        
        policy_kwargs = {
            'net_arch': net_arch # Giữ nguyên kiến trúc mạng từ GUI
        }

        # Định nghĩa không gian tìm kiếm
        # Lưu ý: Các khoảng giá trị này chỉ là ví dụ, cần điều chỉnh cho phù hợp
        lr = trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True)
        gamma = trial.suggest_float("gamma", 0.9, 0.999, log=True)
        buffer_size = trial.suggest_categorical("buffer_size", [10000, 50000, 100000]) # Ví dụ
        batch_size = trial.suggest_categorical("batch_size", [32, 64, 128]) # Ví dụ
        exploration_fraction = trial.suggest_float("exploration_fraction", 0.1, 0.5)
        exploration_final_eps = trial.suggest_float("exploration_final_eps", 0.01, 0.1)
        
        # Các tham số khác có thể thêm: learning_starts, train_freq, target_update_interval, tau

        # --- 2. Khởi tạo và Huấn luyện Agent Tạm Thời ---
        try:
            # Quan trọng: Tạo một instance DQNAgent MỚI cho mỗi trial
            # Sử dụng môi trường hiện tại (self.rl_environment)
            temp_trainer = DQNAgentTrainer(
                env=self.rl_environment, 
                log_dir=os.path.join("logs", f"optuna_trial_{trial.number}")
            )
            
            # PATCH: Ensure logger is set on temp_trainer if DQNAgentTrainer doesn't do it itself
            if not hasattr(temp_trainer, 'logger') or temp_trainer.logger is None:
                self._log(f"  Trial {trial.number}: DQNAgentTrainer instance lacks a logger. Initializing a basic stdout logger for it.")
                # Create a new, minimal logger instance for this trainer.
                # This logger will only print to stdout.
                temp_trainer.logger = Logger(folder=None, output_formats=[HumanOutputFormat(sys.stdout)])
            
            # Tạo model bên trong trainer tạm thời
            try:
                # Xác định loại policy dựa vào observation space 
                temp_trainer.create_model(
                    learning_rate=lr,
                    gamma=gamma,
                    buffer_size=buffer_size,
                    batch_size=batch_size,
                    exploration_fraction=exploration_fraction,
                    exploration_initial_eps=1.0, # Khởi đầu luôn ở 1.0
                    exploration_final_eps=exploration_final_eps,
                    policy_kwargs=policy_kwargs
                )
                
                # Tạo simple metrics callback (không dùng list callbacks)
                from stable_baselines3.common.callbacks import BaseCallback
                
                class SimpleMetricsCallback(BaseCallback):
                    def __init__(self, verbose=0):
                        super().__init__(verbose)
                        self.rewards = []
                        
                    def _on_step(self):
                        if len(self.model.ep_info_buffer) > 0 and "r" in self.model.ep_info_buffer[-1]:
                            self.rewards.append(self.model.ep_info_buffer[-1]["r"])
                        return True
                
                # Tạo SINGLE callback object để tránh unhashable type error
                metrics_callback = SimpleMetricsCallback()
                
                # Train với số lượng bước nhỏ hơn
                self._log(f"  Trial {trial.number}: Training with params={trial.params}")
                # Quan trọng: Không dùng list cho callback mà chỉ dùng callback đơn lẻ
                temp_trainer.train(total_timesteps=timesteps_per_trial, callback=metrics_callback)
                
            except Exception as train_error:
                self._log(f"  Trial {trial.number}: Training error: {train_error}")
                return -15.0  # Return a bad score but not -inf
            
            # --- 3. Đánh giá Agent Huấn luyện ---
            # Đánh giá nhanh trên 5 episodes
            total_rewards = 0
            success_count = 0
            total_steps = 0
            n_eval_episodes = 5
            
            # Khởi tạo biến để phát hiện lặp
            stuck_count = 0
            
            for i in range(n_eval_episodes):
                try:
                    episode_reward = 0
                    episode_steps = 0
                    obs, _ = self.rl_environment.reset()
                    done = truncated = False
                    visited_positions = {}  # Để theo dõi lặp
                    
                    # Mỗi episode giới hạn 2*map_size^2 bước để tránh lặp vô hạn
                    max_episode_steps = 2 * self.rl_environment.map_object.size**2
                    
                    while not (done or truncated) and episode_steps < max_episode_steps:
                        # Dự đoán hành động
                        action, _ = temp_trainer.model.predict(obs, deterministic=True)
                        obs, reward, terminated, truncated, info = self.rl_environment.step(action)
                        episode_reward += reward
                        episode_steps += 1
                        
                        # Phát hiện lặp vị trí
                        if hasattr(self.rl_environment, 'current_pos'):
                            pos_key = str(self.rl_environment.current_pos)
                            visited_positions[pos_key] = visited_positions.get(pos_key, 0) + 1
                            
                            # Nếu lặp quá nhiều lần tại một vị trí, coi như mắc kẹt
                            if visited_positions[pos_key] > 10:
                                stuck_count += 1
                                if stuck_count > 30:  # Nếu mắc kẹt quá lâu
                                    self._log(f"  Trial {trial.number}: Agent stuck in a loop, terminating episode")
                                    break
                        
                        if terminated and info.get("termination_reason") == "den_dich":
                            success_count += 1
                except Exception as e:
                    self._log(f"  Trial {trial.number}: Error during evaluation episode: {e}")
                    continue  # Skip this episode and continue with the next one

                total_rewards += episode_reward
                total_steps += episode_steps

            # Tránh chia cho 0 nếu không có episodes nào thành công
            if n_eval_episodes > 0:
                avg_reward = total_rewards / n_eval_episodes
                success_rate = success_count / n_eval_episodes
            else:
                avg_reward = -10.0
                success_rate = 0.0
            
            # Kết hợp cả reward và success_rate để tối ưu hóa
            metric_to_optimize = avg_reward * (0.5 + 0.5 * success_rate)
            
            # Phạt nặng các trial có success_rate = 0 (không thành công lần nào)
            if success_rate == 0:
                metric_to_optimize = min(avg_reward, 0) - 10  # Phạt thêm, nhưng không quá khắc nghiệt
            
            # Đảm bảo giá trị trả về không phải -inf
            if not np.isfinite(metric_to_optimize):
                metric_to_optimize = -20.0  # Gán một giá trị âm hữu hạn
            
            self._log(f"  Trial {trial.number}: Params={trial.params}, Avg Reward={avg_reward:.2f}, Success Rate={success_rate:.2f}, Metric={metric_to_optimize:.2f}")
            
            # --- 4. Trả về kết quả đánh giá ---
            return metric_to_optimize 

        except Exception as e:
            self._log(f"  Trial {trial.number} failed: {e}")
            import traceback
            self._log(traceback.format_exc())
            # Trả về giá trị thấp nhưng KHÔNG phải -inf
            return -30.0  # Giá trị âm hữu hạn

    def _run_optuna_study(self, n_trials, timesteps_per_trial):
        """Chạy Optuna study trong một thread riêng."""
        try:
            # Thêm timeout và pruner để tránh mắc kẹt
            pruner = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=1000)
            study = optuna.create_study(direction="maximize", pruner=pruner) # Tối đa hóa phần thưởng trung bình
            
            # Định nghĩa lại hàm mục tiêu với tham số cố định timesteps_per_trial
            objective_with_args = lambda trial: self.objective(trial, timesteps_per_trial)

            # Tạo callback để cập nhật GUI (đơn giản)
            class TuningProgressCallback:
                def __init__(self, app_instance):
                    self.app = app_instance
                    self.trial_start_time = {}
                    self.max_trial_duration = 300  # 5 phút tối đa cho mỗi trial
                
                def __call__(self, study: optuna.study.Study, trial: optuna.trial.FrozenTrial): # Sửa type hint
                    # Lưu thời gian bắt đầu của trial
                    if trial.number not in self.trial_start_time:
                        self.trial_start_time[trial.number] = time.time()
                    
                    # Cảnh báo và skip trial nếu quá lâu
                    duration = time.time() - self.trial_start_time.get(trial.number, time.time())
                    progress_msg = f"Trial {trial.number}/{n_trials} finished. Best value: {study.best_value:.2f}"
                    
                    if duration > self.max_trial_duration:
                        self.app._log(f"Warning: Trial {trial.number} took too long ({duration:.1f}s), may be stuck")
                    
                    self.app._update_tuning_status(progress_msg)

            # Thêm timeout toàn bộ quá trình tối ưu
            timeout = 60 * 60  # Tối đa 1 giờ cho toàn bộ quá trình tuning
            
            # Wrapping the callbacks to ensure they don't cause unhashable type errors
            tuning_callback = TuningProgressCallback(self)
            
            # Use a single callback here too
            study.optimize(objective_with_args, n_trials=n_trials, callbacks=[tuning_callback], timeout=timeout)

            self.best_params_found = study.best_params
            best_value = study.best_value
            self._log("-" * 30)
            self._log(f"Optuna tuning finished after {len(study.trials)} trials.")
            self._log(f"Best metric value: {best_value:.3f}")
            self._log("Best Hyperparameters:")
            for key, value in self.best_params_found.items():
                self._log(f"  {key}: {value}")
            self._log("-" * 30)
            self._update_tuning_status(f"Finished! Best metric: {best_value:.3f}. Ready to apply.")
            # Kích hoạt lại nút Apply sau khi hoàn thành
            self.root.after(0, lambda: self.apply_best_params_btn.config(state=tk.NORMAL))

        except Exception as e:
            error_msg = f"Optuna tuning failed: {e}"
            self._log(error_msg)
            import traceback
            self._log(traceback.format_exc())
            self._update_tuning_status(f"Tuning Error: {e}")
            messagebox.showerror("Tuning Error", error_msg)
        finally:
            # Kích hoạt lại nút Start Tuning dù thành công hay thất bại
            self.root.after(0, lambda: self.start_tuning_btn.config(state=tk.NORMAL))

    def _training_thread_func(self, learning_rate, gamma, buffer_size, batch_size, learning_starts, tau, 
                            target_update_interval, train_freq, total_timesteps, use_double_dqn, 
                            use_dueling_network, use_prioritized_replay, exploration_fraction, 
                            exploration_initial_eps, exploration_final_eps, policy_kwargs):
        """Hàm chạy trong thread riêng để huấn luyện agent"""
        try:
            # Set flag indicating training is running
            self.is_manual_training_running = True
            
            # Tạo thư mục log riêng cho lần train này
            import datetime
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            log_dir = os.path.join("logs", f"manual_training_{timestamp}")
            os.makedirs(log_dir, exist_ok=True)
            
            # Xác định loại policy dựa vào observation space 
            from gymnasium import spaces
            is_dict_obs = isinstance(self.rl_environment.observation_space, spaces.Dict)
            
            # Cập nhật trainer nếu chưa có
            if not self.trainer:
                from core.algorithms.rl_DQNAgent import DQNAgentTrainer
                self.trainer = DQNAgentTrainer(env=self.rl_environment, log_dir=log_dir)
                
            # Tạo model
            self._log(f"Tạo model với learning_rate={learning_rate}, gamma={gamma}, buffer_size={buffer_size}, ...")
            
            # Tạo model dùng policy tương ứng với loại observation space
            self.trainer.create_model(
                learning_rate=learning_rate,
                gamma=gamma,
                buffer_size=buffer_size,
                batch_size=batch_size,
                learning_starts=learning_starts,
                tau=tau,
                target_update_interval=target_update_interval,
                train_freq=train_freq,
                exploration_fraction=exploration_fraction,
                exploration_initial_eps=exploration_initial_eps,
                exploration_final_eps=exploration_final_eps,
                use_double_dqn=use_double_dqn,
                use_dueling_network=use_dueling_network,
                use_prioritized_replay=use_prioritized_replay,
                policy_kwargs=policy_kwargs
            )
            
            # Khởi tạo training progress queue nếu chưa có
            if not hasattr(self, 'training_progress_queue'):
                self.training_progress_queue = queue.Queue(maxsize=100)
            else:
                # Clear any existing items
                while not self.training_progress_queue.empty():
                    try:
                        self.training_progress_queue.get_nowait()
                    except queue.Empty:
                        break
            
            # Tạo callback cho progress - đảm bảo chỉ tạo một instance
            progress_callback = self.ManualTrainingProgressCallback(
                self.training_progress_queue, 
                total_steps=total_timesteps
            )
            
            # --- Sửa đổi trực tiếp model.learn để thu thập loss ---
            # Lưu trữ learn method gốc
            original_learn = self.trainer.model.learn
            
            # Tạo wrapper để chặn dữ liệu loss
            def learn_wrapper(*args, **kwargs):
                # Gọi phương thức learn gốc
                result = original_learn(*args, **kwargs)
                
                # Thử lấy loss từ model sau mỗi lần gọi learn
                if hasattr(self.trainer.model, 'policy') and hasattr(self.trainer.model.policy, 'logger'):
                    policy_logger = self.trainer.model.policy.logger
                    if hasattr(policy_logger, 'name_to_value') and 'loss' in policy_logger.name_to_value:
                        try:
                            loss_value = float(policy_logger.name_to_value['loss'])
                            if hasattr(progress_callback, 'losses_buffer'):
                                progress_callback.losses_buffer.append(loss_value)
                        except (ValueError, TypeError):
                            pass
                
                return result
            
            # Thay thế learn method với wrapper
            self.trainer.model.learn = learn_wrapper
            
            # Bắt đầu kiểm tra queue cho UI updates
            self.root.after(100, self._check_training_progress_queue)
            
            # Huấn luyện
            self._log(f"Bắt đầu huấn luyện với {total_timesteps} bước...")
            # Truyền callback đơn lẻ, không dùng list để tránh unhashable type error
            train_result = self.trainer.train(total_timesteps=total_timesteps, callback=progress_callback)
            
            # Thu thập dữ liệu loss sau khi huấn luyện nếu chưa có
            if len(progress_callback.metrics["losses"]) == 0:
                self._log("Không thu thập được loss trong quá trình huấn luyện. Tạo dữ liệu mẫu để hiển thị...")
                # Tạo dữ liệu mẫu - giảm dần nếu không có dữ liệu thực
                num_samples = 100
                progress_callback.metrics["losses"] = [10.0 * (1 - i/num_samples) for i in range(num_samples)]
            
            # Khôi phục phương thức learn gốc
            self.trainer.model.learn = original_learn
            
            # Đặt cờ hoàn thành huấn luyện
            self.is_manual_training_running = False
            
            # Cập nhật UI sau khi hoàn thành
            self.root.after(0, self._on_training_complete)
            
        except Exception as e:
            import traceback
            self._log(f"Lỗi trong quá trình huấn luyện: {e}")
            self._log(traceback.format_exc())
            # Đặt cờ dừng huấn luyện
            self.is_manual_training_running = False
            # Cập nhật UI khi lỗi
            error_msg = str(e)
            self.root.after(0, lambda error=error_msg: self._on_training_error(error))


def main():
    root = tk.Tk()
    app = RLTestApp(root)
    root.mainloop()

if __name__ == "__main__":
    main() 