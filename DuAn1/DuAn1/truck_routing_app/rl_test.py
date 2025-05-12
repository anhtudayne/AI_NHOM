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

# Đường dẫn chính xác cho vị trí hiện tại
# File hiện tại: C:\Users\Win 11\Desktop\DoAnTTNT-Nhom\AI_NHOM\DuAn1\DuAn1\truck_routing_app\rl_test.py

# Kiểm tra đường dẫn và tệp tin
current_dir = os.path.dirname(os.path.abspath(__file__))
core_dir = os.path.join(current_dir, 'core')
rl_env_file = os.path.join(core_dir, 'rl_environment.py')

if not os.path.exists(rl_env_file):
    print(f"CẢNH BÁO: File rl_environment.py không tồn tại tại {rl_env_file}")
    print("Các file trong thư mục core:", os.listdir(core_dir) if os.path.exists(core_dir) else "thư mục core không tồn tại")
    
    # Tạo liên kết đến file rl_environment.py trong thư mục cha nếu có
    parent_dir = os.path.dirname(current_dir)
    parent_core_dir = os.path.join(parent_dir, 'core')
    parent_rl_env_file = os.path.join(parent_core_dir, 'rl_environment.py')
    
    if os.path.exists(parent_rl_env_file):
        print(f"Tìm thấy file rl_environment.py tại {parent_rl_env_file}")
        # Thêm thư mục cha vào sys.path
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
                        avg_reward = f"{np.mean([ep_info['r'] for ep_info in self.model.ep_info_buffer[-10:]]):.2f}"
                    
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
        self.manual_learning_starts = tk.IntVar(value=defaults.get("learning_starts", 1000))
        self.manual_batch_size = tk.IntVar(value=defaults.get("batch_size", 64))
        self.manual_tau = tk.DoubleVar(value=defaults.get("tau", 0.005))
        self.manual_train_freq = tk.IntVar(value=defaults.get("train_freq", 4))
        self.manual_target_update_interval = tk.IntVar(value=defaults.get("target_update_interval", 1000))
        self.manual_exploration_fraction = tk.DoubleVar(value=defaults.get("exploration_fraction", 0.2))
        self.manual_exploration_initial_eps = tk.DoubleVar(value=defaults.get("exploration_initial_eps", 1.0))
        self.manual_exploration_final_eps = tk.DoubleVar(value=defaults.get("exploration_final_eps", 0.05))
        self.manual_total_timesteps = tk.IntVar(value=10000) # Mặc định huấn luyện 10k bước

        self.manual_use_double_dqn = tk.BooleanVar(value=False)
        self.manual_use_dueling_dqn = tk.BooleanVar(value=False)
        self.manual_use_prioritized_replay = tk.BooleanVar(value=False)
        self.manual_render_training = tk.BooleanVar(value=False)
        self.manual_net_arch = tk.StringVar(value="64,64")

    
    def _create_ui(self):
        """Tạo giao diện người dùng (Đã TỐI ƯU)."""
        # Frame chính chứa Notebook và Log
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill="both", expand=True)

        # Tạo notebook với các tab chính
        self.notebook = ttk.Notebook(main_frame)
        self.notebook.pack(fill="both", expand=True, padx=10, pady=10)
        
        # Tab 0: Hướng dẫn sử dụng (Giữ lại)
        guide_frame = ttk.Frame(self.notebook)
        self.notebook.add(guide_frame, text="Hướng dẫn sử dụng")
        self._create_guide_tab(guide_frame)
        
        # Tab for RL Agent (DQN) - This will be the main training tab
        rl_agent_frame = ttk.Frame(self.notebook)
        self.notebook.add(rl_agent_frame, text="RL Agent (DQN Training)") # Renamed for clarity
        self._create_manual_train_tab(rl_agent_frame) # This creates all manual training controls
        
        # Khu vực log (Giữ lại)
        log_frame = ttk.LabelFrame(main_frame, text="Log")
        log_frame.pack(fill="x", expand=False, padx=10, pady=(0, 10), side="bottom") # Đặt log ở dưới
        
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
        """Tạo tab Huấn luyện Thủ công (Tab MỚI tổng hợp)."""
        # Chia tab thành 2 phần chính: Trái (Bản đồ & Môi trường), Phải (Agent & Điều khiển)
        left_frame = ttk.Frame(parent)
        left_frame.pack(side="left", fill="both", expand=True, padx=5, pady=5)
        right_frame = ttk.Frame(parent)
        right_frame.pack(side="right", fill="both", expand=True, padx=5, pady=5)

        # --- Phần Trái: Bản đồ & Môi trường ---
        map_control_frame = ttk.LabelFrame(left_frame, text="1. Bản đồ")
        map_control_frame.pack(fill="x", padx=5, pady=5, anchor='n')
        self._populate_map_controls(map_control_frame)

        self.map_canvas = tk.Canvas(left_frame, bg="white", height=300) # Canvas bản đồ ở đây
        self.map_canvas.pack(fill="both", expand=True, padx=5, pady=5)

        env_init_frame = ttk.LabelFrame(left_frame, text="2. Môi trường RL")
        env_init_frame.pack(fill="x", padx=5, pady=5, anchor='s')
        self._populate_env_init_controls(env_init_frame)

        # --- Phần Phải: Agent & Điều khiển ---
        agent_config_frame = ttk.LabelFrame(right_frame, text="3. Cấu hình Agent (DQN)")
        agent_config_frame.pack(fill="x", padx=5, pady=5, anchor='n')
        self._populate_agent_config_controls(agent_config_frame)

        # --- Phần Hành động & Trạng thái ---
        # Khung chứa các nút hành động chính
        action_buttons_frame = ttk.Frame(right_frame)
        action_buttons_frame.pack(pady=5, fill="x", padx=5)

        # Gom nhóm các nút huấn luyện và điều khiển
        self.train_button = ttk.Button(action_buttons_frame, text="Bắt đầu Huấn luyện", command=self._start_manual_training)
        self.train_button.pack(side=tk.LEFT, padx=2, pady=2)
        self.save_button = ttk.Button(action_buttons_frame, text="Lưu Model", command=self._save_manual_model, state=tk.DISABLED)
        self.save_button.pack(side=tk.LEFT, padx=2, pady=2)
        self.load_button = ttk.Button(action_buttons_frame, text="Tải Model", command=self._load_trained_model)
        self.load_button.pack(side=tk.LEFT, padx=2, pady=2)
        self.run_agent_button = ttk.Button(action_buttons_frame, text="Chạy Agent", command=self._run_agent_on_current_map, state=tk.DISABLED)
        self.run_agent_button.pack(side=tk.LEFT, padx=2, pady=2)

        # Trạng thái Agent
        agent_status_frame = ttk.LabelFrame(right_frame, text="Trạng thái Agent")
        agent_status_frame.pack(pady=5, fill="both", expand=True, padx=5)
        ttk.Label(agent_status_frame, text="Trạng thái:").pack(anchor='w')
        self.agent_status_text = scrolledtext.ScrolledText(agent_status_frame, height=5, state=tk.DISABLED)
        self.agent_status_text.pack(fill="both", expand=True, padx=5, pady=2)

        # Training Button
        manual_train_btn = ttk.Button(action_buttons_frame, text="Start Manual Training", command=self._start_manual_training, style='Accent.TButton')
        manual_train_btn.pack(fill=tk.X, padx=5, pady=(10, 5))

        # --- Progress Display ---
        progress_frame = ttk.Frame(action_buttons_frame)
        progress_frame.pack(fill=tk.X, padx=5, pady=5)

        progress_label = ttk.Label(progress_frame, text="Training Progress:")
        progress_label.pack(side=tk.LEFT, padx=(0, 5))

        self.manual_train_progress = ttk.Progressbar(progress_frame, orient=tk.HORIZONTAL, length=300, mode='determinate')
        self.manual_train_progress.pack(side=tk.LEFT, fill=tk.X, expand=True)

        # Status Label below progress
        self.manual_train_status_label = ttk.Label(action_buttons_frame, text="Status: Idle")
        self.manual_train_status_label.pack(fill=tk.X, padx=5, pady=(0, 5), anchor='w')

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

    def _populate_env_init_controls(self, parent):
        """Hàm phụ trợ: Tạo các control khởi tạo môi trường."""
        # Biến lưu trữ các tham số (Lấy từ _create_rl_init_tab)
        self.initial_fuel_var = tk.DoubleVar(value=MovementCosts.MAX_FUEL)
        self.initial_money_var = tk.DoubleVar(value=1500.0)
        self.fuel_per_move_var = tk.DoubleVar(value=MovementCosts.FUEL_PER_MOVE)
        self.gas_station_cost_var = tk.DoubleVar(value=StationCosts.BASE_GAS_COST)
        self.toll_base_cost_var = tk.DoubleVar(value=StationCosts.BASE_TOLL_COST)
        self.max_steps_var = tk.IntVar(value=200)

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

    def _create_random_map(self):
        """Tạo bản đồ ngẫu nhiên với số lượng điểm đặc biệt tùy chỉnh."""
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
                self._log(f"  Actual counts: Tolls={stats['toll_stations']}, Gas={stats['gas_stations']}, Obstacles={stats['obstacles']}")
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
        """Bắt đầu huấn luyện thủ công."""
        if self.rl_environment is None:
            messagebox.showerror("Error", "Please initialize the RL environment first.")
            return
        # ... (Logic lấy hyperparams từ self.manual_xxx vars) ...
        try:
             hyperparams = {
                 # ... (lấy giá trị từ self.manual_... vars)
                 "learning_rate": self.manual_lr.get(),
                 "gamma": self.manual_gamma.get(),
                 "buffer_size": self.manual_buffer_size.get(),
                 "learning_starts": self.manual_learning_starts.get(),
                 "batch_size": self.manual_batch_size.get(),
                 "tau": self.manual_tau.get(),
                 "train_freq": self.manual_train_freq.get(),
                 "target_update_interval": self.manual_target_update_interval.get(),
                 "exploration_fraction": self.manual_exploration_fraction.get(),
                 "exploration_initial_eps": self.manual_exploration_initial_eps.get(),
                 "exploration_final_eps": self.manual_exploration_final_eps.get()
             }
             total_timesteps = self.manual_total_timesteps.get()
             use_double_dqn = self.manual_use_double_dqn.get()
             use_dueling_network = self.manual_use_dueling_dqn.get()
             use_prioritized_replay = self.manual_use_prioritized_replay.get()
             # Kiểm tra giá trị hợp lệ
             if total_timesteps <= 0: raise ValueError("Total Timesteps > 0")
             # ... (thêm kiểm tra khác)
        except (ValueError, tk.TclError) as e:
             messagebox.showerror("Input Error", f"Invalid input for training parameters: {e}")
             self._log(f"Input Error: {e}")
             self.is_manual_training_running = False # Reset flag on input error
             return

        self._log("Bắt đầu huấn luyện thủ công...")
        self.train_button.config(state=tk.DISABLED)
        self.save_button.config(state=tk.DISABLED)
        self.run_agent_button.config(state=tk.DISABLED)
        self.load_button.config(state=tk.DISABLED) # Disable load khi đang train

        # Tạo agent mới
        self.trainer = DQNAgentTrainer(env=self.rl_environment, log_dir="./rl_models_logs/manual_train") # Log vào thư mục chung

        def training_thread():
            try:
                self._log("Đang tạo model DQN...")
                policy_kwargs = dict(net_arch=[64, 64]) # Kiến trúc mạng đơn giản
                self.trainer.create_model(
                    policy_kwargs=policy_kwargs,
                    use_double_dqn=use_double_dqn,
                    use_dueling_network=use_dueling_network,
                    use_prioritized_replay=use_prioritized_replay,
                    **hyperparams # Truyền các siêu tham số khác
                )
                self._log(f"Model đã tạo. Bắt đầu huấn luyện {total_timesteps} bước...")
                self.trainer.train(total_timesteps=total_timesteps)
                self._log("Huấn luyện hoàn thành!")
                self.root.after(0, self._on_training_complete)
            except ImportError as e_import:
                 self._log(f"LỖI IMPORT: {e_import}. Cài đặt sb3-contrib?")
                 self.root.after(0, self._on_training_error, f"Lỗi Import: {e_import}")
            except Exception as e_train:
                self._log(f"Lỗi huấn luyện: {e_train}")
                import traceback; self._log(traceback.format_exc())
                self.root.after(0, lambda: self._on_training_error(f"Lỗi: {e_train}"))

        threading.Thread(target=training_thread, daemon=True).start()
        self._update_agent_status("Đang huấn luyện...")

    def _on_training_complete(self):
        """Callback khi huấn luyện thành công."""
        self.train_button.config(state=tk.NORMAL)
        self.save_button.config(state=tk.NORMAL)
        self.run_agent_button.config(state=tk.NORMAL)
        self.load_button.config(state=tk.NORMAL)
        self._update_agent_status("Huấn luyện hoàn thành. Sẵn sàng lưu/chạy.")

    def _on_training_error(self, error_msg):
        """Callback khi huấn luyện lỗi."""
        messagebox.showerror("Lỗi Huấn luyện", error_msg)
        self.train_button.config(state=tk.NORMAL)
        self.load_button.config(state=tk.NORMAL)
        self._update_agent_status(f"Lỗi huấn luyện: {error_msg}")

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


def main():
    root = tk.Tk()
    app = RLTestApp(root)
    root.mainloop()

if __name__ == "__main__":
    main() 