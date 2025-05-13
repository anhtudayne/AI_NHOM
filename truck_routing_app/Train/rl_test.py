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
        self.root.geometry("1000x850") # Increased height for logs/progress
        
        # Style configuration
        self.style = ttk.Style()
        self.style.theme_use('clam') # Use a modern theme

        # Map and Environment related variables
        self.map_object = None
        self.current_map_path = None
        self.rl_environment = None
        self.map_size = tk.IntVar(value=8)
        # Variables for customizing map generation counts
        self.num_tolls_var = tk.IntVar(value=2) # Default for 8x8 based on ~0.05 ratio
        self.num_gas_var = tk.IntVar(value=3)   # Default for 8x8 based on ~0.05 ratio
        self.num_obstacles_var = tk.IntVar(value=10) # Default for 8x8 based on ~0.2 ratio

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

            def _on_step(self) -> bool:
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
                
        self.ManualTrainingProgressCallback = ManualTrainingProgressCallback # Store the class

        self._create_ui()
    
    def _setup_manual_training_vars(self):
        """Khởi tạo các biến tk cho cấu hình huấn luyện thủ công."""
        # Lấy giá trị mặc định từ OPTIMAL_HYPERPARAMS[8]
        defaults = OPTIMAL_HYPERPARAMS.get(8, OPTIMAL_HYPERPARAMS[list(OPTIMAL_HYPERPARAMS.keys())[0]]) # Lấy 8 hoặc key đầu tiên

        self.manual_lr = tk.DoubleVar(value=defaults.get("learning_rate", 0.0001))
        self.manual_gamma = tk.DoubleVar(value=defaults.get("gamma", 0.99))
        self.manual_buffer_size = tk.IntVar(value=defaults.get("buffer_size", 50000))
        self.manual_learning_starts = tk.IntVar(value=10000)  # Tăng từ mặc định để agent khám phá hơn
        self.manual_batch_size = tk.IntVar(value=defaults.get("batch_size", 64))
        self.manual_tau = tk.DoubleVar(value=defaults.get("tau", 0.005))
        self.manual_train_freq = tk.IntVar(value=defaults.get("train_freq", 4))
        self.manual_target_update_interval = tk.IntVar(value=defaults.get("target_update_interval", 1000))
        self.manual_exploration_fraction = tk.DoubleVar(value=0.5)  # Giảm từ 0.9 xuống 0.5
        self.manual_exploration_initial_eps = tk.DoubleVar(value=1.0)  # Giữ nguyên giá trị mặc định
        self.manual_exploration_final_eps = tk.DoubleVar(value=0.02)  # Giảm từ 0.05 xuống 0.02 để agent khai thác tốt hơn cuối cùng
        self.manual_total_timesteps = tk.IntVar(value=1000000)  # Tăng từ 200000 lên 1,000,000 bước

        # Lấy giá trị cho các tính năng nâng cao từ cấu hình tối ưu
        self.manual_use_double_dqn = tk.BooleanVar(value=True) # Mặc định bật Double DQN
        self.manual_use_dueling_dqn = tk.BooleanVar(value=True) # Mặc định bật Dueling
        self.manual_use_prioritized_replay = tk.BooleanVar(value=True) # Mặc định bật PER
        self.manual_render_training = tk.BooleanVar(value=False)

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
        Tạo tab huấn luyện thủ công.
        
        Args:
            parent: Frame cha
        """
        # Tạo giao diện dạng nhiều cột
        main_frame = ttk.Frame(parent, padding=10)
        main_frame.pack(fill='both', expand=True)
        
        left_frame = ttk.Frame(main_frame)
        left_frame.pack(side='left', fill='both', expand=True, padx=(0, 5))
        
        right_frame = ttk.Frame(main_frame)
        right_frame.pack(side='right', fill='both', expand=True, padx=(5, 0))
        
        # Thông tin kích thước bản đồ và tham số môi trường
        env_info_frame = ttk.LabelFrame(left_frame, text="Thông tin môi trường", padding=5)
        env_info_frame.pack(fill='x', padx=5, pady=5)
        
        ttk.Label(env_info_frame, text="Map Size: ").grid(row=0, column=0, sticky='w')
        ttk.Label(env_info_frame, textvariable=self.map_size).grid(row=0, column=1, sticky='w')
        
        # Frame cho tham số huấn luyện
        training_params_frame = ttk.LabelFrame(left_frame, text="Tham số huấn luyện DQN", padding=5)
        training_params_frame.pack(fill='x', padx=5, pady=5)
        
        # Cấu hình tham số trong grid
        row = 0
        ttk.Label(training_params_frame, text="Timesteps:").grid(row=row, column=0, sticky='w')
        self.manual_timesteps_var = tk.IntVar(value=50000)
        ttk.Spinbox(training_params_frame, from_=10000, to=1000000, increment=10000, 
                    textvariable=self.manual_timesteps_var, width=10).grid(row=row, column=1, sticky='w')
        
        row += 1
        ttk.Label(training_params_frame, text="Learning rate:").grid(row=row, column=0, sticky='w')
        ttk.Spinbox(training_params_frame, from_=0.0001, to=0.01, increment=0.0001, 
                    textvariable=self.manual_lr, width=10, format="%.4f").grid(row=row, column=1, sticky='w')
        
        row += 1
        ttk.Label(training_params_frame, text="Gamma:").grid(row=row, column=0, sticky='w')
        ttk.Spinbox(training_params_frame, from_=0.8, to=0.999, increment=0.01, 
                    textvariable=self.manual_gamma, width=10, format="%.3f").grid(row=row, column=1, sticky='w')
        
        row += 1
        ttk.Label(training_params_frame, text="Buffer size:").grid(row=row, column=0, sticky='w')
        ttk.Spinbox(training_params_frame, from_=5000, to=1000000, increment=5000, 
                    textvariable=self.manual_buffer_size, width=10).grid(row=row, column=1, sticky='w')
        
        row += 1
        ttk.Label(training_params_frame, text="Learning starts:").grid(row=row, column=0, sticky='w')
        ttk.Spinbox(training_params_frame, from_=1000, to=100000, increment=1000, 
                    textvariable=self.manual_learning_starts, width=10).grid(row=row, column=1, sticky='w')
        
        row += 1
        ttk.Label(training_params_frame, text="Batch size:").grid(row=row, column=0, sticky='w')
        ttk.Spinbox(training_params_frame, from_=32, to=512, increment=32, 
                    textvariable=self.manual_batch_size, width=10).grid(row=row, column=1, sticky='w')
        
        # Frame cho cài đặt nâng cao
        row += 1
        advanced_frame = ttk.LabelFrame(left_frame, text="Cài đặt nâng cao", padding=5)
        advanced_frame.pack(fill='x', padx=5, pady=5)
        
        # Checkboxes for advanced settings
        self.use_double_dqn_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(advanced_frame, text="Sử dụng Double DQN", variable=self.use_double_dqn_var).grid(row=0, column=0, sticky='w')
        
        self.use_dueling_network_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(advanced_frame, text="Dueling DQN", variable=self.use_dueling_network_var).grid(row=0, column=1, sticky='w')
        ttk.Checkbutton(advanced_frame, text="PER", variable=self.manual_use_prioritized_replay).grid(row=0, column=2, sticky='w')

        # Lấy giá trị mặc định từ OPTIMAL_HYPERPARAMS
        defaults = OPTIMAL_HYPERPARAMS.get(8, OPTIMAL_HYPERPARAMS[list(OPTIMAL_HYPERPARAMS.keys())[0]])
        
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

    def _populate_map_controls(self, parent):
        """Hàm phụ trợ: Tạo các control cho bản đồ."""
        # Frame kích thước
        size_frame = ttk.Frame(parent)
        size_frame.pack(side="left", padx=5, pady=5)
        ttk.Label(size_frame, text="Kích thước (8-15):").grid(row=0, column=0, padx=5, pady=5)
        self.map_size_var = tk.IntVar(value=8)
        size_spinner = ttk.Spinbox(size_frame, from_=8, to=15, textvariable=self.map_size_var, width=5)
        size_spinner.grid(row=0, column=1, padx=5, pady=5)

        # Frame for special point counts
        counts_frame = ttk.Frame(parent)
        counts_frame.pack(side="left", padx=5, pady=5, anchor="nw")

        ttk.Label(counts_frame, text="# Tolls:").grid(row=0, column=0, padx=2, pady=2, sticky="w")
        ttk.Spinbox(counts_frame, from_=0, to=20, textvariable=self.num_tolls_var, width=4).grid(row=0, column=1, padx=2, pady=2, sticky="w")

        ttk.Label(counts_frame, text="# Gas Stations:").grid(row=1, column=0, padx=2, pady=2, sticky="w")
        ttk.Spinbox(counts_frame, from_=0, to=20, textvariable=self.num_gas_var, width=4).grid(row=1, column=1, padx=2, pady=2, sticky="w")

        ttk.Label(counts_frame, text="# Obstacles:").grid(row=2, column=0, padx=2, pady=2, sticky="w")
        ttk.Spinbox(counts_frame, from_=0, to=50, textvariable=self.num_obstacles_var, width=4).grid(row=2, column=1, padx=2, pady=2, sticky="w")

        # Frame nút
        buttons_frame = ttk.Frame(parent)
        buttons_frame.pack(side="left", padx=10, pady=5)
        self.random_map_btn = ttk.Button(buttons_frame, text="Tạo Ngẫu nhiên", command=self._create_random_map)
        self.random_map_btn.grid(row=0, column=0, padx=2, pady=2)
        self.demo_map_btn = ttk.Button(buttons_frame, text="Tạo Mẫu", command=self._create_demo_map)
        self.demo_map_btn.grid(row=0, column=1, padx=2, pady=2)
        self.load_map_btn = ttk.Button(buttons_frame, text="Tải", command=self._load_map)
        self.load_map_btn.grid(row=0, column=2, padx=2, pady=2)
        self.save_map_btn = ttk.Button(buttons_frame, text="Lưu", command=self._save_map, state=tk.DISABLED) # Disable ban đầu
        self.save_map_btn.grid(row=0, column=3, padx=2, pady=2)
        # Thêm nút Reset
        self.reset_all_btn = ttk.Button(buttons_frame, text="Reset Tất Cả", command=self._reset_all)
        self.reset_all_btn.grid(row=0, column=4, padx=5, pady=2)

    def _populate_env_init_controls(self, parent):
        """Hàm phụ trợ: Tạo các control khởi tạo môi trường."""
        # Biến lưu trữ các tham số (Lấy từ _create_rl_init_tab)
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
        ttk.Label(params_grid, text="Nhiên liệu:").grid(row=row_idx, column=0, padx=5, pady=2, sticky="w")
        ttk.Entry(params_grid, textvariable=self.initial_fuel_var, width=8).grid(row=row_idx, column=1, padx=5, pady=2, sticky="w")
        ttk.Label(params_grid, text="Tiền:").grid(row=row_idx, column=2, padx=5, pady=2, sticky="w")
        ttk.Entry(params_grid, textvariable=self.initial_money_var, width=8).grid(row=row_idx, column=3, padx=5, pady=2, sticky="w")
        row_idx += 1
        ttk.Label(params_grid, text="N.liệu/bước:").grid(row=row_idx, column=0, padx=5, pady=2, sticky="w")
        ttk.Entry(params_grid, textvariable=self.fuel_per_move_var, width=8).grid(row=row_idx, column=1, padx=5, pady=2, sticky="w")
        ttk.Label(params_grid, text="Phí xăng:").grid(row=row_idx, column=2, padx=5, pady=2, sticky="w")
        ttk.Entry(params_grid, textvariable=self.gas_station_cost_var, width=8).grid(row=row_idx, column=3, padx=5, pady=2, sticky="w")
        row_idx += 1
        ttk.Label(params_grid, text="Phí cầu đường:").grid(row=row_idx, column=0, padx=5, pady=2, sticky="w")
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
                self._log(f"Generated random map ({map_size}x{map_size}) with requested counts: T={num_tolls}, G={num_gas}, O={num_obstacles}.")
                # Log thêm số lượng thực tế từ map_object.get_statistics() để so sánh
                stats = self.map_object.get_statistics()
                self._log(f"  Actual counts: Tolls={stats['toll_stations']}, Gas={stats['gas_stations']}, Obstacles={stats['brick_cells']}")
                self._draw_map(self.map_canvas)
                self.save_map_btn.config(state=tk.NORMAL)
            else:
                messagebox.showwarning("Map Generation Failed", f"Could not generate a valid random map with the specified counts after several attempts. Check console warnings.")
                self._log(f"Failed to generate a valid map for size {map_size} with T={num_tolls}, G={num_gas}, O={num_obstacles}.")

        except tk.TclError as e:
            messagebox.showerror("Input Error", f"Invalid input for map generation parameters: {e}")
            self._log(f"Map generation input error: {e}")
        except Exception as e:
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
            self.map_object = Map.load(filepath)
            if self.map_object:
                self.current_map_path = filepath # Lưu đường dẫn file đã tải
                self._log(f"Đã tải bản đồ: {filepath} (Size: {self.map_object.size}x{self.map_object.size})")
                # self._display_map_info()
                self._draw_map(self.map_canvas) # Vẽ lên canvas của tab mới
                self.map_size_var.set(self.map_object.size) # Cập nhật size spinner
                self.save_map_btn.config(state=tk.DISABLED) # Không cho lưu lại map vừa tải
            else:
                messagebox.showerror("Lỗi", "Không thể tải bản đồ từ file!")
        except Exception as e:
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
        # ... (Giữ nguyên logic, nhưng cập nhật log và thông báo)
        if not self.map_object:
            messagebox.showerror("Lỗi", "Vui lòng tạo hoặc tải bản đồ trước khi khởi tạo môi trường!")
            return
        
        try:
            # ... (Lấy tham số từ self.xxx_var)
            initial_fuel = self.initial_fuel_var.get()
            initial_money = self.initial_money_var.get()
            fuel_per_move = self.fuel_per_move_var.get()
            gas_station_cost = self.gas_station_cost_var.get()
            toll_base_cost = self.toll_base_cost_var.get()
            max_steps = self.max_steps_var.get()
            
            # Khởi tạo môi trường RL
            self.rl_environment = TruckRoutingEnv(
                map_object=self.map_object,
                initial_fuel_config=initial_fuel, 
                initial_money_config=initial_money, 
                fuel_per_move_config=fuel_per_move, 
                gas_station_cost_config=gas_station_cost, 
                toll_base_cost_config=toll_base_cost, 
                max_steps_per_episode=max_steps
            )
            
            self._log("Đã khởi tạo môi trường RL thành công!")
            # Cập nhật trạng thái agent (nếu cần)
            self._update_agent_status("Môi trường đã sẵn sàng. Có thể huấn luyện.")
            # Không cần hiển thị thông tin RL chi tiết ở đây nữa

        except Exception as e:
            self._log(f"Lỗi khi khởi tạo môi trường RL: {e}")
            import traceback; self._log(traceback.format_exc())
            messagebox.showerror("Lỗi", f"Lỗi khi khởi tạo môi trường RL: {e}")
            self._update_agent_status("Lỗi khởi tạo môi trường!")

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
        if agent_pos is not None and agent_pos.size > 0:
            agent_x, agent_y = agent_pos[0], agent_pos[1] # Lấy giá trị từ numpy array
            pixel_x = offset_x + agent_x * cell_size
            pixel_y = offset_y + agent_y * cell_size
            canvas.create_oval(pixel_x + cell_size // 4, pixel_y + cell_size // 4, 
                                pixel_x + cell_size * 3 // 4, pixel_y + cell_size * 3 // 4, fill="blue", outline="black")
    
    def _draw_map_with_path(self, canvas, agent_pos, path):
        """Vẽ bản đồ với đường đi của agent."""
        # Sử dụng lại _draw_map để vẽ nền
        self._draw_map(canvas, agent_pos)
        canvas.after(100, lambda c=canvas, p=path: self._draw_path_internal(c, p))

    def _draw_path_internal(self, canvas, path):
        """Vẽ đường đi sau khi bản đồ nền đã được vẽ."""
        if not path or len(path) <= 1 or not self.map_object:
             return

        map_size = self.map_object.size
        canvas_width = canvas.winfo_width()
        canvas_height = canvas.winfo_height()
        if canvas_width <= 1 or canvas_height <= 1: return # Canvas chưa sẵn sàng
        cell_size = min(canvas_width // map_size, canvas_height // map_size)
        if cell_size <= 0: cell_size = 1
        offset_x = (canvas_width - cell_size * map_size) // 2
        offset_y = (canvas_height - cell_size * map_size) // 2

        for i in range(len(path) - 1):
            p1 = path[i]
            p2 = path[i+1]
            # Đảm bảo p1, p2 là tuple
            x1, y1 = tuple(p1) if hasattr(p1, '__len__') else (0,0)
            x2, y2 = tuple(p2) if hasattr(p2, '__len__') else (0,0)

            center_x1 = offset_x + x1 * cell_size + cell_size // 2
            center_y1 = offset_y + y1 * cell_size + cell_size // 2
            center_x2 = offset_x + x2 * cell_size + cell_size // 2
            center_y2 = offset_y + y2 * cell_size + cell_size // 2
            canvas.create_line(center_x1, center_y1, center_x2, center_y2, fill="orange", width=max(1, cell_size // 10), arrow=tk.LAST)

    def _start_manual_training(self):
        """Bắt đầu quá trình huấn luyện thủ công DQN agent."""
        if not self.rl_environment:
            messagebox.showerror("Lỗi", "Vui lòng khởi tạo môi trường RL trước!")
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
            total_timesteps = self.manual_total_timesteps.get()  # Thêm lại dòng này
            
            # Lấy các tham số exploration từ UI
            exploration_fraction = self.manual_exploration_fraction.get()
            exploration_initial_eps = self.manual_exploration_initial_eps.get()
            exploration_final_eps = self.manual_exploration_final_eps.get()
            
            # Lấy giá trị các checkbox
            double_q = self.manual_use_double_dqn.get()
            dueling_net = self.manual_use_dueling_dqn.get()
            prioritized_replay = self.manual_use_prioritized_replay.get()
            
            # Tạo thư mục log riêng cho lần train này
            import datetime
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            log_dir = os.path.join("logs", f"manual_training_{timestamp}")
            os.makedirs(log_dir, exist_ok=True)

            # Các tham số nâng cao cho khả năng thích ứng thoát khỏi lặp
            advanced_params_for_training_logic = {
                "loop_detection_active": True,  # Kích hoạt phát hiện lặp
                "adapt_exploration": True,      # Cho phép điều chỉnh exploration
                "min_reward_improvement": 0.1,  # Ngưỡng cải thiện tối thiểu để không tăng exploration
            }
            
            # Các tham số chỉ dành cho việc tạo model
            model_creation_params = {
                "learning_rate": learning_rate,
                "gamma": gamma,
                "buffer_size": buffer_size,
                "batch_size": batch_size,
                "learning_starts": learning_starts,
                "tau": tau,
                "train_freq": train_freq,
                "target_update_interval": target_update_interval,
                "exploration_fraction": exploration_fraction,
                "exploration_initial_eps": exploration_initial_eps,
                "exploration_final_eps": exploration_final_eps,
                "policy_kwargs": {"net_arch": [256, 256]},  # Mạng neural lớn hơn
                "use_double_dqn": double_q, 
                "use_dueling_network": dueling_net, 
                "use_prioritized_replay": prioritized_replay, 
                "prioritized_replay_alpha": 0.6,
                "prioritized_replay_beta0": 0.4
                # Không bao gồm **advanced_params ở đây
            }
            
        except tk.TclError as e:
            # Lỗi khi parse giá trị từ widget
            messagebox.showerror("Lỗi Tham số", f"Lỗi đọc tham số huấn luyện: {e}")
            self.is_manual_training_running = False # Reset flag on input error
            return
        
        # Thiết lập giao diện để thể hiện đang huấn luyện
        self._set_training_buttons_state(tk.DISABLED)
        self.is_manual_training_running = True # Đặt flag TRƯỚC khi bắt đầu thread và check_queue
        
        # Kiểm tra xem model đã được tải lên chưa
        is_continuing_training = hasattr(self, 'trainer') and self.trainer and hasattr(self.trainer, 'model') and self.trainer.model
        
        # Hiển thị thông báo phù hợp dựa trên việc tiếp tục train hay tạo mới
        if is_continuing_training:
            self._update_agent_status("Đang tiếp tục huấn luyện model đã tải...")
            self._log("Tiếp tục huấn luyện model đã tải lên...")
        else:
            self._update_agent_status("Đang khởi tạo model DQN mới...")
            self._log("Đang tạo model DQN mới...")
        
        # Hiển thị progress bar và status
        if hasattr(self, 'manual_train_progress'): self.manual_train_progress['value'] = 0
        if hasattr(self, 'manual_train_status_label'): self.manual_train_status_label['text'] = "Status: Initializing..."
        self.root.update_idletasks() # Cập nhật UI ngay

        # Tạo log directory riêng cho lần train này
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        session_id = f"manual_train_{timestamp}"
        log_dir = os.path.join("./training_logs", session_id)
        os.makedirs(log_dir, exist_ok=True)
        self._log(f"Log directory created: {log_dir}")

        # Chỉ tạo trainer mới nếu chưa có hoặc không có model đã tải
        if not is_continuing_training:
            self.trainer = DQNAgentTrainer(env=self.rl_environment, log_dir=log_dir)

        def training_thread():
            try:
                # Chỉ tạo model mới nếu không tiếp tục huấn luyện model đã tải
                if not is_continuing_training:
                    self._log("Đang tạo model DQN...")
                # Lấy kiến trúc mạng từ input, phân tách và chuyển đổi thành list các số nguyên
                try:
                    net_arch_str = self.manual_net_arch.get().split(',')
                    net_arch = [int(n.strip()) for n in net_arch_str if n.strip().isdigit()]
                    if not net_arch: # Nếu trống hoặc không hợp lệ, dùng mặc định
                        net_arch = [256, 256]
                        self._log("Cảnh báo: Network Arch không hợp lệ, sử dụng [256, 256]")
                    # Cập nhật net_arch trực tiếp vào model_creation_params
                    current_model_creation_params = model_creation_params.copy() # Tạo bản sao để thay đổi
                    current_model_creation_params["policy_kwargs"]["net_arch"] = net_arch
                except Exception as parse_e:
                    self._log(f"Lỗi phân tích Network Arch: {parse_e}. Sử dụng [256, 256]")
                    current_model_creation_params = model_creation_params.copy()
                    current_model_creation_params["policy_kwargs"]["net_arch"] = [256, 256]

                # Tạo model với các tham số được định nghĩa cho việc tạo model
                    self.trainer.create_model(**current_model_creation_params)
                    self._log(f"Model mới đã được tạo. Bắt đầu huấn luyện {total_timesteps} bước...")
                else:
                    self._log(f"Tiếp tục huấn luyện model đã tải thêm {total_timesteps} bước...")
                
                # Tạo callback với total_timesteps
                progress_callback = self.ManualTrainingProgressCallback(self.training_progress_queue, total_steps=total_timesteps)
                
                # Bắt đầu huấn luyện
                self.trainer.train(total_timesteps=total_timesteps, callback=progress_callback)
                
                # Khi huấn luyện xong (thành công)
                self.root.after(0, self._on_training_complete) 
            except ImportError as e_import:
                 self._log(f"LỖI IMPORT: {e_import}. Cài đặt sb3-contrib?")
                 self.root.after(0, self._on_training_error, f"Lỗi Import: {e_import}")
            except Exception as e_train:
                self._log(f"Lỗi huấn luyện: {e_train}")
                import traceback; self._log(traceback.format_exc())
                # Khi huấn luyện lỗi
                self.root.after(0, lambda err=e_train: self._on_training_error(f"Lỗi: {err}"))

        # Bắt đầu thread huấn luyện
        self.training_thread_handle = threading.Thread(target=training_thread, daemon=True)
        self.training_thread_handle.start()
        
        # Bắt đầu kiểm tra queue sau khi thread đã khởi chạy
        self.root.after(100, self._check_training_progress_queue) 
        self._update_agent_status("Đang huấn luyện...") # Cập nhật trạng thái lần nữa

    def _on_training_complete(self):
        """Xử lý khi huấn luyện hoàn tất."""
        self._set_training_buttons_state('normal')
        
        # Enable nút lưu model
        for child in self.root.winfo_children():
            if isinstance(child, ttk.Frame):
                for subchild in child.winfo_children():
                    if isinstance(subchild, ttk.Button) and subchild['text'] == "Lưu model":
                        subchild.configure(state='normal')
                        
        # Enable nút chạy agent
        for child in self.root.winfo_children():
            if isinstance(child, ttk.Frame):
                for subchild in child.winfo_children():
                    if isinstance(subchild, ttk.Button) and subchild['text'] == "Chạy agent trên bản đồ":
                        subchild.configure(state='normal')
        
        # Nếu có training metrics, hiển thị biểu đồ
        if hasattr(self.trainer, 'training_metrics') and self.trainer.training_metrics is not None:
            self._visualize_training_metrics()

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
            # Moving average rewards for smoother plot
            window_size = min(10, len(metrics['episode_rewards']))
            if window_size > 0:
                # Đảm bảo có đủ phần tử cho window
                moving_avg_rewards = []
                for i in range(len(metrics['episode_rewards'])):
                    window_start = max(0, i - window_size + 1)
                    moving_avg_rewards.append(np.mean(metrics['episode_rewards'][window_start:i+1]))
                
                ax1.plot(moving_avg_rewards, 'b-', label='Moving Avg Reward')
                ax1.plot(metrics['episode_rewards'], 'b.', alpha=0.3, label='Episode Reward')
            else:
                ax1.plot(metrics['episode_rewards'], 'b-', label='Episode Reward')
                
            ax1.set_xlabel('Episodes')
            ax1.set_ylabel('Reward', color='b')
            ax1.tick_params(axis='y', labelcolor='b')
            
            # Add success rate if available
            if 'success_rates' in metrics and len(metrics['success_rates']) > 0:
                ax2 = ax1.twinx()
                ax2.plot(metrics['success_rates'], 'r-', label='Success Rate')
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
            ax3.plot(metrics['timesteps'], metrics['exploration_rates'], 'g-', label='Exploration Rate')
            ax3.set_xlabel('Timesteps')
            ax3.set_ylabel('Exploration Rate', color='g')
            ax3.tick_params(axis='y', labelcolor='g')
            
            # Add learning rate if available
            if 'learning_rates' in metrics and len(metrics['learning_rates']) > 0:
                ax4 = ax3.twinx()
                ax4.plot(metrics['timesteps'], metrics['learning_rates'], 'm-', label='Learning Rate')
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
            # Moving average for losses to smooth out the plot
            window_size = min(100, len(metrics['losses']))
            if window_size > 0:
                moving_avg_losses = []
                for i in range(len(metrics['losses'])):
                    window_start = max(0, i - window_size + 1)
                    moving_avg_losses.append(np.mean(metrics['losses'][window_start:i+1]))
                
                ax5.plot(moving_avg_losses, 'r-', label='Moving Avg Loss')
                # Plot original loss values with low alpha to show variance
                ax5.plot(metrics['losses'], 'r.', alpha=0.1, label='Loss')
            else:
                ax5.plot(metrics['losses'], 'r-', label='Loss')
                
            ax5.set_xlabel('Updates')
            ax5.set_ylabel('Loss Value')
            ax5.legend(loc='upper right')
            
            loss_figure.tight_layout()
        else:
            ax5.text(0.5, 0.5, "No loss data available", 
                    horizontalalignment='center', verticalalignment='center')
        
        # Tab 4: Episode Lengths
        lengths_tab = ttk.Frame(notebook)
        notebook.add(lengths_tab, text="Episode Lengths")
        
        lengths_figure = plt.Figure(figsize=(8, 6), dpi=100)
        lengths_canvas = FigureCanvasTkAgg(lengths_figure, lengths_tab)
        lengths_canvas.get_tk_widget().pack(fill='both', expand=True)
        
        # Plot episode lengths
        ax6 = lengths_figure.add_subplot(111)
        
        if 'episode_lengths' in metrics and len(metrics['episode_lengths']) > 0:
            # Moving average for smoother visualization
            window_size = min(10, len(metrics['episode_lengths']))
            if window_size > 0:
                moving_avg_lengths = []
                for i in range(len(metrics['episode_lengths'])):
                    window_start = max(0, i - window_size + 1)
                    moving_avg_lengths.append(np.mean(metrics['episode_lengths'][window_start:i+1]))
                
                ax6.plot(moving_avg_lengths, 'g-', label='Moving Avg Length')
                ax6.plot(metrics['episode_lengths'], 'g.', alpha=0.3, label='Episode Length')
            else:
                ax6.plot(metrics['episode_lengths'], 'g-', label='Episode Length')
                
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

                while not (terminated or truncated) and step_count < self.rl_environment.max_steps_per_episode * 1.5: # Thêm giới hạn an toàn
                    action = self.trainer.predict_action(observation)
                    next_obs, reward, terminated, truncated, info = self.rl_environment.step(action)
                    total_reward += reward
                    step_count += 1
                    observation = next_obs
                    path.append(tuple(observation['agent_pos'])) # Chuyển numpy sang tuple
                    
                    # Cập nhật UI sau mỗi bước
                    self.root.after(0, lambda obs=observation, p=list(path): self._draw_map_with_path(self.map_canvas, obs['agent_pos'], p))
                    self.root.after(10) # Delay nhỏ

                # Kết thúc
                success = info.get("termination_reason") == "den_dich"
                log_msg = f"Chạy agent hoàn tất. Thành công: {success}. Lý do: {info.get('termination_reason', 'Max Steps')}. Steps: {step_count}. Reward: {total_reward:.2f}"
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
        try:
            # Non-blocking check
            while True: # Process all messages currently in the queue
                current_step, total_steps, status_message = self.training_progress_queue.get_nowait()
                
                # Update Progress Bar
                if total_steps > 0:
                    progress_percentage = (current_step / total_steps) * 100
                    self.manual_train_progress['value'] = progress_percentage
                else:
                    self.manual_train_progress['value'] = 0
                    
                # Update Status Label
                self.manual_train_status_label['text'] = f"Status: {status_message}"
                self.root.update_idletasks() # Force UI update
                
        except queue.Empty:
            pass # No new messages
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
        
    def _reset_env_and_agent(self):
        """Hàm phụ trợ chỉ reset môi trường và agent."""
        self.rl_environment = None
        self.trainer = None
        self.trained_model_path = None
        self.is_model_loaded = False
        if hasattr(self, 'save_button'): self.save_button.config(state=tk.DISABLED)
        if hasattr(self, 'run_agent_button'): self.run_agent_button.config(state=tk.DISABLED)
        self._update_agent_status("Môi trường/Agent đã reset do map thay đổi. Khởi tạo lại môi trường.")

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
                verbose=0, # Ít log hơn trong quá trình tuning
                policy_kwargs=policy_kwargs,
                learning_rate=lr,
                gamma=gamma,
                buffer_size=buffer_size,
                batch_size=batch_size,
                exploration_fraction=exploration_fraction,
                exploration_final_eps=exploration_final_eps,
                # Thêm các tham số khác đã suggest ở trên nếu có
            )
            
            # Tạo model bên trong trainer tạm thời
            temp_trainer.create_model(
                policy_kwargs=policy_kwargs,
                learning_rate=lr,
                gamma=gamma,
                buffer_size=buffer_size,
                batch_size=batch_size,
                exploration_fraction=exploration_fraction,
                exploration_final_eps=exploration_final_eps
                # Add other suggested hyperparameters if needed
                # Keeping Dueling/Double/PER options out for simplicity in tuning trials
            )
            
            # Huấn luyện agent tạm thời
            # Sử dụng callback rỗng hoặc callback tối thiểu để tránh ảnh hưởng GUI chính
            temp_trainer.train(total_timesteps=timesteps_per_trial, callback=None) 
            
            # --- 3. Đánh giá Agent ---
            n_eval_episodes = 10 # Số episode để đánh giá
            total_rewards = 0
            total_steps = 0
            success_count = 0

            # Thiết lập giới hạn bước đánh giá để tránh mắc kẹt
            max_eval_steps = 1000  # Giới hạn số bước cho mỗi episode đánh giá

            for _ in range(n_eval_episodes):
                obs, _ = self.rl_environment.reset() # Reset môi trường hiện tại
                episode_reward = 0
                episode_steps = 0
                terminated = truncated = False
                
                # Thêm giới hạn bước và timeout để tránh vòng lặp vô hạn
                start_time = time.time()
                timeout = 5.0  # Timeout 5 giây cho mỗi episode

                # Lưu các vị trí đã đi qua để phát hiện lặp
                visited_positions = {}
                stuck_count = 0
                
                while not terminated and not truncated and episode_steps < max_eval_steps:
                    # Kiểm tra timeout
                    if time.time() - start_time > timeout:
                        self._log(f"  Trial {trial.number}: Episode timed out after {episode_steps} steps")
                        break
                        
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

                total_rewards += episode_reward
                total_steps += episode_steps

            avg_reward = total_rewards / n_eval_episodes
            success_rate = success_count / n_eval_episodes
            
            # Kết hợp cả reward và success_rate để tối ưu hóa
            metric_to_optimize = avg_reward * (0.5 + 0.5 * success_rate)
            
            # Phạt nặng các trial có success_rate = 0 (không thành công lần nào)
            if success_rate == 0:
                metric_to_optimize = min(avg_reward, 0)  # Chắc chắn không dương nếu không thành công

            self._log(f"  Trial {trial.number}: Params={trial.params}, Avg Reward={avg_reward:.2f}, Success Rate={success_rate:.2f}, Metric={metric_to_optimize:.2f}")
            
            # --- 4. Trả về kết quả đánh giá ---
            return metric_to_optimize 

        except Exception as e:
            self._log(f"  Trial {trial.number} failed: {e}")
            import traceback
            self._log(traceback.format_exc())
            # Trả về giá trị rất thấp để Optuna tránh xa vùng tham số này
            return -float('inf') 

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
            study.optimize(objective_with_args, n_trials=n_trials, callbacks=[TuningProgressCallback(self)], timeout=timeout)

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


def main():
    root = tk.Tk()
    app = RLTestApp(root)
    root.mainloop()

if __name__ == "__main__":
    main() 