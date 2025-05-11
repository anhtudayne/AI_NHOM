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


class RLTestApp:
    """Ứng dụng kiểm tra môi trường RL."""
    
    def __init__(self, root):
        """
        Khởi tạo ứng dụng kiểm tra môi trường RL.
        
        Args:
            root: Cửa sổ gốc Tkinter
        """
        self.root = root
        self.root.title("Kiểm tra Môi trường RL")
        self.root.geometry("1000x700")
        
        self.map_object = None
        self.rl_env = None
        self.current_observation = None
        
        self._create_ui()
    
    def _create_ui(self):
        """Tạo giao diện người dùng."""
        # Tạo notebook với các tab
        notebook = ttk.Notebook(self.root)
        notebook.pack(fill="both", expand=True, padx=10, pady=10)
        
        # Tab 0: Hướng dẫn sử dụng
        guide_frame = ttk.Frame(notebook)
        notebook.add(guide_frame, text="Hướng dẫn sử dụng")
        self._create_guide_tab(guide_frame)
        
        # Tab 1: Tải/Tạo Bản đồ
        map_frame = ttk.Frame(notebook)
        notebook.add(map_frame, text="Tải/Tạo Bản đồ")
        self._create_map_tab(map_frame)
        
        # Tab 2: Khởi tạo Môi trường RL
        rl_init_frame = ttk.Frame(notebook)
        notebook.add(rl_init_frame, text="Khởi tạo Môi trường RL")
        self._create_rl_init_tab(rl_init_frame)
        
        # Tab 3: Điều khiển Môi trường
        rl_control_frame = ttk.Frame(notebook)
        notebook.add(rl_control_frame, text="Điều khiển Môi trường")
        self._create_rl_control_tab(rl_control_frame)
        
        # Tab 4: Agent RL (DQN)
        rl_agent_frame = ttk.Frame(notebook)
        notebook.add(rl_agent_frame, text="Agent RL (DQN)")
        self._create_rl_agent_tab(rl_agent_frame)
        
        # Tab 5: RL Nâng cao
        rl_advanced_frame = ttk.Frame(notebook)
        notebook.add(rl_advanced_frame, text="RL Nâng cao")
        self._create_rl_advanced_tab(rl_advanced_frame)
        
        # Khu vực log
        log_frame = ttk.LabelFrame(self.root, text="Log")
        log_frame.pack(fill="both", expand=True, padx=10, pady=10)
        
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
TAB 3: ĐIỀU KHIỂN MÔI TRƯỜNG
----------------------------------------------

Mục đích: Kiểm tra môi trường RL bằng cách thực hiện các hành động thủ công.

Cách sử dụng:
- Nhấn "Reset Môi trường" để bắt đầu một episode mới
- Chọn hành động từ radio buttons (Lên, Xuống, Trái, Phải, Đổ xăng, Bỏ qua)
- Nhấn "Thực hiện Hành động" hoặc "Hành động Ngẫu nhiên"
- Quan sát kết quả ở khu vực "Kết quả hành động" và "Trạng thái hiện tại"

Kiểm tra đúng/sai:
✓ ĐÚNG: 
  - Trạng thái hiện tại cập nhật sau mỗi hành động (vị trí, nhiên liệu, tiền thay đổi)
  - Reward phù hợp với hành động (âm khi di chuyển, dương khi tiến gần đích, lớn khi đến đích)
  - Khi đến đích, done = True và termination_reason = "den_dich"
  - Khi hết nhiên liệu ở đường thường, done = True và termination_reason = "het_nhien_lieu"
  - Đổ xăng chỉ hoạt động khi ở trạm xăng (ô màu xanh)

✗ SAI:
  - Không thể di chuyển hoặc vị trí không thay đổi sau hành động
  - Reward không nhất quán (ví dụ: phần thưởng dương khi va chạm vật cản)
  - Không thể đổ xăng ở trạm xăng hoặc đổ xăng ở đường thường vẫn được
  - Nhiên liệu không giảm khi di chuyển
  - Tiền không giảm khi qua trạm thu phí

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

----------------------------------------------
TAB 5: RL NÂNG CAO
----------------------------------------------

Mục đích: Cung cấp các tính năng nâng cao cho việc cải thiện agent RL, bao gồm tinh chỉnh siêu tham số, đánh giá chi tiết và so sánh thuật toán.

Tab này gồm ba tab con:

1. Tinh chỉnh Siêu tham số:
   - Cho phép tối ưu hóa tự động các siêu tham số của model DQN
   - Sử dụng Optuna để tìm kiếm các tham số tốt nhất
   
   Cách sử dụng:
   - Nhấn "Tạo Thư mục Bản đồ" để tạo các thư mục huấn luyện/đánh giá (nếu chưa có)
   - Điều chỉnh các tham số: số lần thử, số bước huấn luyện, số episodes đánh giá
   - Nhấn "Bắt đầu Tinh chỉnh Siêu tham số" và chờ quá trình hoàn tất
   - Sau khi tìm được tham số tốt nhất, nhấn "Huấn luyện Model với Tham số Tốt nhất"
   
   Kiểm tra đúng/sai:
   ✓ ĐÚNG: Quá trình tinh chỉnh hiển thị tiến độ và kết quả tìm được các tham số tốt nhất
   ✗ SAI: Thông báo lỗi hoặc không tìm được tham số tốt

2. Đánh giá Chi tiết:
   - Đánh giá chi tiết hiệu suất của agent RL trên nhiều bản đồ khác nhau
   - Cung cấp các chỉ số chi tiết về thành công, phần thưởng, độ dài đường đi, nhiên liệu, chi phí
   
   Cách sử dụng:
   - Chọn model đã huấn luyện (file .zip)
   - Chọn thư mục bản đồ để đánh giá (thường là ./maps/test)
   - Điều chỉnh số episodes mỗi bản đồ
   - Nhấn "Bắt đầu Đánh giá Chi tiết" và chờ kết quả
   
   Kiểm tra đúng/sai:
   ✓ ĐÚNG: Hiển thị kết quả đánh giá chi tiết với các biểu đồ
   ✗ SAI: Thông báo lỗi hoặc không hiển thị kết quả đánh giá

3. So sánh Thuật toán:
   - So sánh hiệu suất của agent RL với các thuật toán tìm đường truyền thống
   - Hỗ trợ so sánh với A*, Greedy, Genetic Algorithm, Simulated Annealing, Local Beam Search
   
   Cách sử dụng:
   - Chọn model RL đã huấn luyện
   - Chọn thuật toán để so sánh (từ danh sách)
   - Chọn sử dụng bản đồ hiện tại hoặc tải bản đồ khác
   - Nhấn "Bắt đầu So sánh" và xem kết quả
   
   Kiểm tra đúng/sai:
   ✓ ĐÚNG: Hiển thị bảng so sánh và đường đi của cả hai thuật toán trên cùng bản đồ
   ✗ SAI: Thông báo lỗi hoặc không hiển thị kết quả so sánh

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
    
    def _create_map_tab(self, parent):
        """
        Tạo tab Tải/Tạo Bản đồ.
        
        Args:
            parent: Widget cha
        """
        # Frame chứa các tùy chọn
        options_frame = ttk.Frame(parent)
        options_frame.pack(fill="x", padx=10, pady=10)
        
        # Frame kích thước bản đồ
        size_frame = ttk.LabelFrame(options_frame, text="Kích thước bản đồ")
        size_frame.pack(side="left", padx=5, pady=5)
        
        ttk.Label(size_frame, text="Kích thước:").grid(row=0, column=0, padx=5, pady=5)
        self.map_size_var = tk.IntVar(value=8)
        size_spinner = ttk.Spinbox(
            size_frame, 
            from_=8, 
            to=15, 
            textvariable=self.map_size_var, 
            width=5
        )
        size_spinner.grid(row=0, column=1, padx=5, pady=5)
        
        # Frame các nút
        buttons_frame = ttk.Frame(options_frame)
        buttons_frame.pack(side="left", padx=20, pady=5)
        
        # Nút tạo bản đồ ngẫu nhiên
        self.random_map_btn = ttk.Button(
            buttons_frame, 
            text="Tạo Bản đồ Ngẫu nhiên", 
            command=self._create_random_map
        )
        self.random_map_btn.grid(row=0, column=0, padx=5, pady=5)
        
        # Nút tạo bản đồ mẫu
        self.demo_map_btn = ttk.Button(
            buttons_frame, 
            text="Tạo Bản đồ Mẫu", 
            command=self._create_demo_map
        )
        self.demo_map_btn.grid(row=0, column=1, padx=5, pady=5)
        
        # Nút tải bản đồ
        self.load_map_btn = ttk.Button(
            buttons_frame, 
            text="Tải Bản đồ", 
            command=self._load_map
        )
        self.load_map_btn.grid(row=0, column=2, padx=5, pady=5)
        
        # Nút lưu bản đồ
        self.save_map_btn = ttk.Button(
            buttons_frame, 
            text="Lưu Bản đồ", 
            command=self._save_map
        )
        self.save_map_btn.grid(row=0, column=3, padx=5, pady=5)
        
        # Frame thông tin bản đồ
        info_frame = ttk.LabelFrame(parent, text="Thông tin bản đồ")
        info_frame.pack(fill="x", padx=10, pady=10)
        
        self.map_info_text = scrolledtext.ScrolledText(info_frame, height=5)
        self.map_info_text.pack(fill="both", expand=True, padx=5, pady=5)
        
        # Canvas hiển thị bản đồ
        canvas_frame = ttk.LabelFrame(parent, text="Bản đồ")
        canvas_frame.pack(fill="both", expand=True, padx=10, pady=10)
        
        self.map_canvas = tk.Canvas(canvas_frame, bg="white")
        self.map_canvas.pack(fill="both", expand=True, padx=5, pady=5)
    
    def _create_rl_init_tab(self, parent):
        """
        Tạo tab Khởi tạo Môi trường RL.
        
        Args:
            parent: Widget cha
        """
        # Frame chứa các tham số
        params_frame = ttk.LabelFrame(parent, text="Tham số môi trường RL")
        params_frame.pack(fill="x", padx=10, pady=10)
        
        # Biến lưu trữ các tham số
        self.initial_fuel_var = tk.DoubleVar(value=MovementCosts.MAX_FUEL)
        self.initial_money_var = tk.DoubleVar(value=2000.0)
        self.fuel_per_move_var = tk.DoubleVar(value=MovementCosts.FUEL_PER_MOVE)
        self.gas_station_cost_var = tk.DoubleVar(value=StationCosts.BASE_GAS_COST)
        self.toll_base_cost_var = tk.DoubleVar(value=StationCosts.BASE_TOLL_COST)
        self.max_steps_var = tk.IntVar(value=200)
        
        # Tạo các trường nhập liệu cho các tham số
        params = [
            ("Nhiên liệu ban đầu:", self.initial_fuel_var),
            ("Tiền ban đầu:", self.initial_money_var),
            ("Nhiên liệu tiêu thụ mỗi bước:", self.fuel_per_move_var),
            ("Chi phí trạm xăng:", self.gas_station_cost_var),
            ("Chi phí trạm thu phí:", self.toll_base_cost_var),
            ("Số bước tối đa mỗi episode:", self.max_steps_var)
        ]
        
        for i, (label_text, var) in enumerate(params):
            ttk.Label(params_frame, text=label_text).grid(row=i, column=0, padx=5, pady=5, sticky="w")
            ttk.Entry(params_frame, textvariable=var, width=10).grid(row=i, column=1, padx=5, pady=5, sticky="w")
        
        # Nút khởi tạo môi trường RL
        self.init_rl_btn = ttk.Button(
            params_frame, 
            text="Khởi tạo Môi trường RL", 
            command=self._initialize_rl_env
        )
        self.init_rl_btn.grid(row=len(params), column=0, columnspan=2, padx=5, pady=10)
        
        # Frame thông tin môi trường RL
        info_frame = ttk.LabelFrame(parent, text="Thông tin môi trường RL")
        info_frame.pack(fill="both", expand=True, padx=10, pady=10)
        
        self.rl_info_text = scrolledtext.ScrolledText(info_frame, height=15)
        self.rl_info_text.pack(fill="both", expand=True, padx=5, pady=5)
    
    def _create_rl_control_tab(self, parent):
        """
        Tạo tab Điều khiển Môi trường.
        
        Args:
            parent: Widget cha
        """
        # Frame chứa các nút điều khiển
        control_frame = ttk.Frame(parent)
        control_frame.pack(fill="x", padx=10, pady=10)
        
        # Nút Reset môi trường
        self.reset_env_btn = ttk.Button(
            control_frame, 
            text="Reset Môi trường", 
            command=self._reset_environment
        )
        self.reset_env_btn.pack(side="left", padx=5, pady=5)
        
        # Frame chọn hành động
        action_frame = ttk.LabelFrame(control_frame, text="Chọn hành động")
        action_frame.pack(side="left", padx=20, pady=5)
        
        self.action_var = tk.IntVar(value=0)
        actions = [
            ("Lên (0)", 0),
            ("Xuống (1)", 1),
            ("Trái (2)", 2),
            ("Phải (3)", 3),
            ("Đổ xăng (4)", 4),
            ("Bỏ qua (5)", 5)
        ]
        
        # Tạo layout lưới 2x3 cho các radio button
        for i, (text, value) in enumerate(actions):
            row = i // 3
            col = i % 3
            ttk.Radiobutton(
                action_frame, 
                text=text, 
                variable=self.action_var, 
                value=value
            ).grid(row=row, column=col, padx=5, pady=5)
        
        # Nút thực hiện hành động
        self.step_btn = ttk.Button(
            control_frame, 
            text="Thực hiện Hành động", 
            command=self._step_environment
        )
        self.step_btn.pack(side="left", padx=5, pady=5)
        
        # Nút hành động ngẫu nhiên
        self.random_action_btn = ttk.Button(
            control_frame, 
            text="Hành động Ngẫu nhiên", 
            command=self._random_action
        )
        self.random_action_btn.pack(side="left", padx=5, pady=5)
        
        # Frame thông tin trạng thái
        state_frame = ttk.LabelFrame(parent, text="Trạng thái hiện tại")
        state_frame.pack(fill="x", padx=10, pady=10)
        
        self.state_text = scrolledtext.ScrolledText(state_frame, height=8)
        self.state_text.pack(fill="both", expand=True, padx=5, pady=5)
        
        # Frame kết quả hành động
        result_frame = ttk.LabelFrame(parent, text="Kết quả hành động")
        result_frame.pack(fill="x", padx=10, pady=10)
        
        self.result_text = scrolledtext.ScrolledText(result_frame, height=8)
        self.result_text.pack(fill="both", expand=True, padx=5, pady=5)
        
        # Canvas hiển thị bản đồ và vị trí agent
        canvas_frame = ttk.LabelFrame(parent, text="Bản đồ và Vị trí Agent")
        canvas_frame.pack(fill="both", expand=True, padx=10, pady=10)
        
        self.rl_canvas = tk.Canvas(canvas_frame, bg="white")
        self.rl_canvas.pack(fill="both", expand=True, padx=5, pady=5)
    
    def _create_rl_agent_tab(self, parent):
        """
        Tạo tab Agent RL (DQN).
        
        Args:
            parent: Widget cha
        """
        # Khởi tạo biến trạng thái cho agent
        self.dqn_agent = None
        
        # Frame cấu hình agent
        config_frame = ttk.LabelFrame(parent, text="Cấu hình Agent RL (DQN)")
        config_frame.pack(fill="x", padx=10, pady=10)
        
        # Biến lưu trữ các siêu tham số
        self.learning_rate_var = tk.DoubleVar(value=1e-4)
        self.total_timesteps_var = tk.IntVar(value=10000)
        self.buffer_size_var = tk.IntVar(value=50000)
        self.batch_size_var = tk.IntVar(value=32)
        self.training_freq_var = tk.IntVar(value=4)
        
        # Tạo các trường nhập liệu cho các siêu tham số
        params = [
            ("Learning rate:", self.learning_rate_var),
            ("Số bước huấn luyện:", self.total_timesteps_var),
            ("Buffer size:", self.buffer_size_var),
            ("Batch size:", self.batch_size_var),
            ("Training frequency:", self.training_freq_var)
        ]
        
        for i, (label_text, var) in enumerate(params):
            ttk.Label(config_frame, text=label_text).grid(row=i, column=0, padx=5, pady=5, sticky="w")
            ttk.Entry(config_frame, textvariable=var, width=10).grid(row=i, column=1, padx=5, pady=5, sticky="w")
        
        # Frame huấn luyện & model
        training_frame = ttk.LabelFrame(parent, text="Huấn luyện & Model")
        training_frame.pack(fill="x", padx=10, pady=10)
        
        # Nút bắt đầu huấn luyện ngắn
        self.train_btn = ttk.Button(
            training_frame, 
            text="Bắt đầu Huấn luyện Ngắn", 
            command=self._start_short_training
        )
        self.train_btn.grid(row=0, column=0, padx=5, pady=5)
        
        # Nút lưu model
        self.save_model_btn = ttk.Button(
            training_frame, 
            text="Lưu Model Hiện tại", 
            command=self._save_current_model
        )
        self.save_model_btn.grid(row=0, column=1, padx=5, pady=5)
        
        # Nút tải model
        self.load_model_btn = ttk.Button(
            training_frame, 
            text="Tải Model Đã Huấn luyện", 
            command=self._load_trained_model
        )
        self.load_model_btn.grid(row=0, column=2, padx=5, pady=5)
        
        # Frame tên model
        model_frame = ttk.Frame(training_frame)
        model_frame.grid(row=1, column=0, columnspan=3, padx=5, pady=5, sticky="w")
        
        ttk.Label(model_frame, text="Tên model:").grid(row=0, column=0, padx=5, pady=5, sticky="w")
        self.model_name_var = tk.StringVar(value="dqn_truck_router")
        ttk.Entry(model_frame, textvariable=self.model_name_var, width=30).grid(row=0, column=1, padx=5, pady=5, sticky="w")
        
        # Frame chạy agent đã huấn luyện
        run_frame = ttk.LabelFrame(parent, text="Chạy Agent Đã Huấn luyện")
        run_frame.pack(fill="x", padx=10, pady=10)
        
        # Nút chạy agent trên bản đồ hiện tại
        self.run_agent_btn = ttk.Button(
            run_frame, 
            text="Chạy Agent trên Bản đồ Hiện tại", 
            command=self._run_agent_on_current_map
        )
        self.run_agent_btn.grid(row=0, column=0, padx=5, pady=5)
        
        # Tùy chọn hiển thị
        self.visualize_steps_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(
            run_frame, 
            text="Hiển thị từng bước", 
            variable=self.visualize_steps_var
        ).grid(row=0, column=1, padx=5, pady=5)
        
        # Tùy chọn tốc độ hiển thị
        ttk.Label(run_frame, text="Tốc độ hiển thị (ms):").grid(row=0, column=2, padx=5, pady=5)
        self.visualization_speed_var = tk.IntVar(value=200)
        ttk.Spinbox(
            run_frame, 
            from_=50, 
            to=1000, 
            increment=50,
            textvariable=self.visualization_speed_var, 
            width=5
        ).grid(row=0, column=3, padx=5, pady=5)
        
        # Frame kết quả đánh giá
        results_frame = ttk.LabelFrame(parent, text="Kết quả Đánh giá")
        results_frame.pack(fill="both", expand=True, padx=10, pady=10)
        
        self.results_text = scrolledtext.ScrolledText(results_frame, height=10)
        self.results_text.pack(fill="both", expand=True, padx=5, pady=5)
    
    def _create_rl_advanced_tab(self, parent):
        """
        Tạo tab RL nâng cao với các tính năng tinh chỉnh siêu tham số và đánh giá chi tiết.
        
        Args:
            parent: Widget cha
        """
        # Chia tab thành hai phần: tinh chỉnh và đánh giá
        notebook = ttk.Notebook(parent)
        notebook.pack(fill="both", expand=True, padx=5, pady=5)
        
        # Tab con 1: Tinh chỉnh siêu tham số
        tuning_frame = ttk.Frame(notebook)
        notebook.add(tuning_frame, text="Tinh chỉnh Siêu tham số")
        self._create_hyperparameter_tuning_tab(tuning_frame)
        
        # Tab con 2: Đánh giá chi tiết
        evaluation_frame = ttk.Frame(notebook)
        notebook.add(evaluation_frame, text="Đánh giá Chi tiết")
        self._create_evaluation_tab(evaluation_frame)
        
        # Tab con 3: So sánh thuật toán
        comparison_frame = ttk.Frame(notebook)
        notebook.add(comparison_frame, text="So sánh Thuật toán")
        self._create_algorithm_comparison_tab(comparison_frame)
    
    def _create_hyperparameter_tuning_tab(self, parent):
        """
        Tạo tab tinh chỉnh siêu tham số.
        
        Args:
            parent: Widget cha
        """
        # Frame cấu hình tinh chỉnh
        config_frame = ttk.LabelFrame(parent, text="Cấu hình Tinh chỉnh Siêu tham số")
        config_frame.pack(fill="x", padx=10, pady=10)
        
        # Frame cho các thư mục bản đồ
        maps_frame = ttk.Frame(config_frame)
        maps_frame.pack(fill="x", padx=5, pady=5)
        
        # Thư mục bản đồ huấn luyện
        ttk.Label(maps_frame, text="Thư mục bản đồ huấn luyện:").grid(row=0, column=0, padx=5, pady=5, sticky="w")
        self.train_maps_dir_var = tk.StringVar(value="./maps/train")
        ttk.Entry(maps_frame, textvariable=self.train_maps_dir_var, width=30).grid(row=0, column=1, padx=5, pady=5, sticky="w")
        ttk.Button(maps_frame, text="Chọn", command=lambda: self._select_directory(self.train_maps_dir_var)).grid(row=0, column=2, padx=5, pady=5)
        
        # Thư mục bản đồ đánh giá
        ttk.Label(maps_frame, text="Thư mục bản đồ đánh giá:").grid(row=1, column=0, padx=5, pady=5, sticky="w")
        self.eval_maps_dir_var = tk.StringVar(value="./maps/eval")
        ttk.Entry(maps_frame, textvariable=self.eval_maps_dir_var, width=30).grid(row=1, column=1, padx=5, pady=5, sticky="w")
        ttk.Button(maps_frame, text="Chọn", command=lambda: self._select_directory(self.eval_maps_dir_var)).grid(row=1, column=2, padx=5, pady=5)
        
        # Frame cho các tham số tinh chỉnh
        params_frame = ttk.Frame(config_frame)
        params_frame.pack(fill="x", padx=5, pady=5)
        
        # Số lần thử
        ttk.Label(params_frame, text="Số lần thử (trials):").grid(row=0, column=0, padx=5, pady=5, sticky="w")
        self.n_trials_var = tk.IntVar(value=20)
        ttk.Spinbox(params_frame, from_=10, to=100, textvariable=self.n_trials_var, width=8).grid(row=0, column=1, padx=5, pady=5, sticky="w")
        
        # Số bước huấn luyện mỗi thử nghiệm
        ttk.Label(params_frame, text="Số bước huấn luyện mỗi thử nghiệm:").grid(row=1, column=0, padx=5, pady=5, sticky="w")
        self.tuning_timesteps_var = tk.IntVar(value=10000)
        ttk.Spinbox(params_frame, from_=1000, to=100000, increment=1000, textvariable=self.tuning_timesteps_var, width=8).grid(row=1, column=1, padx=5, pady=5, sticky="w")
        
        # Số episodes đánh giá
        ttk.Label(params_frame, text="Số episodes đánh giá mỗi thử nghiệm:").grid(row=2, column=0, padx=5, pady=5, sticky="w")
        self.eval_episodes_var = tk.IntVar(value=3)
        ttk.Spinbox(params_frame, from_=1, to=10, textvariable=self.eval_episodes_var, width=8).grid(row=2, column=1, padx=5, pady=5, sticky="w")
        
        # Số công việc song song
        ttk.Label(params_frame, text="Số công việc song song (n_jobs):").grid(row=3, column=0, padx=5, pady=5, sticky="w")
        self.n_jobs_var = tk.IntVar(value=1)
        ttk.Spinbox(params_frame, from_=1, to=8, textvariable=self.n_jobs_var, width=8).grid(row=3, column=1, padx=5, pady=5, sticky="w")
        
        # Frame các nút
        buttons_frame = ttk.Frame(config_frame)
        buttons_frame.pack(fill="x", padx=5, pady=10)
        
        # Tạo thư mục nếu chưa tồn tại
        ttk.Button(buttons_frame, text="Tạo Thư mục Bản đồ", command=self._create_map_dirs).grid(row=0, column=0, padx=5, pady=5)
        
        # Nút bắt đầu tinh chỉnh
        self.start_tuning_btn = ttk.Button(
            buttons_frame, 
            text="Bắt đầu Tinh chỉnh Siêu tham số", 
            command=self._start_hyperparameter_tuning
        )
        self.start_tuning_btn.grid(row=0, column=1, padx=5, pady=5)
        
        # Nút huấn luyện với tham số tốt nhất
        self.train_best_btn = ttk.Button(
            buttons_frame, 
            text="Huấn luyện Model với Tham số Tốt nhất", 
            command=self._train_with_best_params
        )
        self.train_best_btn.grid(row=0, column=2, padx=5, pady=5)
        
        # Frame kết quả tinh chỉnh
        results_frame = ttk.LabelFrame(parent, text="Kết quả Tinh chỉnh")
        results_frame.pack(fill="both", expand=True, padx=10, pady=10)
        
        self.tuning_results_text = scrolledtext.ScrolledText(results_frame)
        self.tuning_results_text.pack(fill="both", expand=True, padx=5, pady=5)
    
    def _create_evaluation_tab(self, parent):
        """
        Tạo tab đánh giá chi tiết.
        
        Args:
            parent: Widget cha
        """
        # Frame chọn model và bản đồ
        selection_frame = ttk.LabelFrame(parent, text="Chọn Model và Bản đồ")
        selection_frame.pack(fill="x", padx=10, pady=10)
        
        # Chọn model
        ttk.Label(selection_frame, text="Chọn Model RL:").grid(row=0, column=0, padx=5, pady=5, sticky="w")
        self.eval_model_path_var = tk.StringVar()
        ttk.Entry(selection_frame, textvariable=self.eval_model_path_var, width=40).grid(row=0, column=1, padx=5, pady=5, sticky="w")
        ttk.Button(selection_frame, text="Chọn", command=self._select_model_for_evaluation).grid(row=0, column=2, padx=5, pady=5)
        
        # Chọn thư mục bản đồ đánh giá
        ttk.Label(selection_frame, text="Thư mục bản đồ đánh giá:").grid(row=1, column=0, padx=5, pady=5, sticky="w")
        self.detailed_eval_maps_dir_var = tk.StringVar(value="./maps/test")
        ttk.Entry(selection_frame, textvariable=self.detailed_eval_maps_dir_var, width=40).grid(row=1, column=1, padx=5, pady=5, sticky="w")
        ttk.Button(selection_frame, text="Chọn", command=lambda: self._select_directory(self.detailed_eval_maps_dir_var)).grid(row=1, column=2, padx=5, pady=5)
        
        # Tham số đánh giá
        params_frame = ttk.Frame(selection_frame)
        params_frame.grid(row=2, column=0, columnspan=3, padx=5, pady=5, sticky="w")
        
        ttk.Label(params_frame, text="Số episodes mỗi bản đồ:").grid(row=0, column=0, padx=5, pady=5, sticky="w")
        self.detailed_eval_episodes_var = tk.IntVar(value=5)
        ttk.Spinbox(params_frame, from_=1, to=20, textvariable=self.detailed_eval_episodes_var, width=5).grid(row=0, column=1, padx=5, pady=5, sticky="w")
        
        # Nút bắt đầu đánh giá
        self.start_evaluation_btn = ttk.Button(
            selection_frame, 
            text="Bắt đầu Đánh giá Chi tiết", 
            command=self._start_detailed_evaluation
        )
        self.start_evaluation_btn.grid(row=3, column=0, columnspan=3, padx=5, pady=10)
        
        # Frame hiển thị kết quả
        results_frame = ttk.LabelFrame(parent, text="Kết quả Đánh giá")
        results_frame.pack(fill="both", expand=True, padx=10, pady=10)
        
        # Phần kết quả số liệu
        metrics_frame = ttk.Frame(results_frame)
        metrics_frame.pack(fill="x", padx=5, pady=5)
        
        self.evaluation_results_text = scrolledtext.ScrolledText(metrics_frame, height=8)
        self.evaluation_results_text.pack(fill="both", expand=True, padx=5, pady=5)
        
        # Phần đồ thị
        self.eval_plot_frame = ttk.Frame(results_frame)
        self.eval_plot_frame.pack(fill="both", expand=True, padx=5, pady=5)
    
    def _create_algorithm_comparison_tab(self, parent):
        """
        Tạo tab so sánh thuật toán.
        
        Args:
            parent: Widget cha
        """
        # Frame chọn model và thuật toán
        selection_frame = ttk.LabelFrame(parent, text="Chọn Model và Thuật toán So sánh")
        selection_frame.pack(fill="x", padx=10, pady=10)
        
        # Chọn model
        ttk.Label(selection_frame, text="Chọn Model RL:").grid(row=0, column=0, padx=5, pady=5, sticky="w")
        self.comp_model_path_var = tk.StringVar()
        ttk.Entry(selection_frame, textvariable=self.comp_model_path_var, width=40).grid(row=0, column=1, padx=5, pady=5, sticky="w")
        ttk.Button(selection_frame, text="Chọn", command=self._select_model_for_comparison).grid(row=0, column=2, padx=5, pady=5)
        
        # Chọn thuật toán so sánh
        ttk.Label(selection_frame, text="Thuật toán so sánh:").grid(row=1, column=0, padx=5, pady=5, sticky="w")
        self.compare_algorithm_var = tk.StringVar(value="A*")
        algorithms = ["A*", "Greedy", "Genetic Algorithm", "Simulated Annealing", "Local Beam Search"]
        ttk.Combobox(selection_frame, textvariable=self.compare_algorithm_var, values=algorithms).grid(row=1, column=1, padx=5, pady=5, sticky="w")
        
        # Có thể chọn bản đồ cụ thể hoặc dùng bản đồ hiện tại
        ttk.Label(selection_frame, text="Sử dụng bản đồ:").grid(row=2, column=0, padx=5, pady=5, sticky="w")
        self.use_current_map_var = tk.BooleanVar(value=True)
        ttk.Radiobutton(selection_frame, text="Bản đồ hiện tại", variable=self.use_current_map_var, value=True).grid(row=2, column=1, padx=5, pady=5, sticky="w")
        ttk.Radiobutton(selection_frame, text="Chọn bản đồ khác", variable=self.use_current_map_var, value=False).grid(row=2, column=2, padx=5, pady=5, sticky="w")
        
        # Nút chọn bản đồ (chỉ hiển thị khi chọn "Chọn bản đồ khác")
        self.select_comp_map_btn = ttk.Button(
            selection_frame, 
            text="Chọn Bản đồ", 
            command=self._select_map_for_comparison
        )
        self.select_comp_map_btn.grid(row=3, column=1, padx=5, pady=5, sticky="w")
        
        # Nút bắt đầu so sánh
        self.start_comparison_btn = ttk.Button(
            selection_frame, 
            text="Bắt đầu So sánh", 
            command=self._start_algorithm_comparison
        )
        self.start_comparison_btn.grid(row=4, column=0, columnspan=3, padx=5, pady=10)
        
        # Frame hiển thị kết quả
        results_frame = ttk.LabelFrame(parent, text="Kết quả So sánh")
        results_frame.pack(fill="both", expand=True, padx=10, pady=10)
        
        # Chia kết quả thành hai phần
        results_paned = ttk.PanedWindow(results_frame, orient=tk.HORIZONTAL)
        results_paned.pack(fill="both", expand=True, padx=5, pady=5)
        
        # Frame hiển thị thông tin
        info_frame = ttk.Frame(results_paned)
        self.comparison_results_text = scrolledtext.ScrolledText(info_frame)
        self.comparison_results_text.pack(fill="both", expand=True, padx=5, pady=5)
        results_paned.add(info_frame, weight=1)
        
        # Frame hiển thị bản đồ với hai đường đi
        map_frame = ttk.Frame(results_paned)
        self.comparison_canvas = tk.Canvas(map_frame, bg="white")
        self.comparison_canvas.pack(fill="both", expand=True, padx=5, pady=5)
        results_paned.add(map_frame, weight=1)
    
    def _log(self, message):
        """
        Ghi log.
        
        Args:
            message: Thông điệp cần ghi log
        """
        self.log_text.insert(tk.END, f"{message}\n")
        self.log_text.see(tk.END)
    
    def _create_random_map(self):
        """Tạo bản đồ ngẫu nhiên."""
        map_size = self.map_size_var.get()
        self.map_object = Map.generate_random(
            size=map_size, 
            toll_ratio=0.05, 
            gas_ratio=0.05, 
            brick_ratio=0.2
        )
        self._log(f"Đã tạo bản đồ ngẫu nhiên kích thước {map_size}x{map_size}")
        self._display_map_info()
        self._draw_map(self.map_canvas)
    
    def _create_demo_map(self):
        """Tạo bản đồ mẫu."""
        map_size = self.map_size_var.get()
        self.map_object = Map.create_demo_map(size=map_size)
        self._log(f"Đã tạo bản đồ mẫu kích thước {map_size}x{map_size}")
        self._display_map_info()
        self._draw_map(self.map_canvas)
    
    def _load_map(self):
        """Tải bản đồ từ file."""
        self.map_object = Map.load()
        if self.map_object:
            self._log(f"Đã tải bản đồ từ file, kích thước {self.map_object.size}x{self.map_object.size}")
            self._display_map_info()
            self._draw_map(self.map_canvas)
        else:
            messagebox.showerror("Lỗi", "Không thể tải bản đồ từ file!")
    
    def _display_map_info(self):
        """Hiển thị thông tin bản đồ."""
        if not self.map_object:
            return
        
        stats = self.map_object.get_statistics()
        info = f"Kích thước bản đồ: {self.map_object.size}x{self.map_object.size}\n"
        info += f"Điểm bắt đầu: {self.map_object.start_pos}\n"
        info += f"Điểm kết thúc: {self.map_object.end_pos}\n\n"
        info += f"Số ô đường thường: {stats['normal_roads']}\n"
        info += f"Số trạm thu phí: {stats['toll_stations']}\n"
        info += f"Số trạm xăng: {stats['gas_stations']}\n"
        info += f"Số ô vật cản: {stats['brick_cells']}\n"
        
        self.map_info_text.delete(1.0, tk.END)
        self.map_info_text.insert(tk.END, info)
    
    def _draw_map(self, canvas, agent_pos=None):
        """
        Vẽ bản đồ lên canvas.
        
        Args:
            canvas: Canvas để vẽ
            agent_pos: Vị trí hiện tại của agent (tùy chọn)
        """
        if not self.map_object:
            return
        
        # Xóa canvas
        canvas.delete("all")
        
        # Tính kích thước ô
        map_size = self.map_object.size
        canvas_width = canvas.winfo_width()
        canvas_height = canvas.winfo_height()
        
        # Đảm bảo kích thước tối thiểu cho canvas
        if canvas_width < 50:
            canvas_width = 400
        if canvas_height < 50:
            canvas_height = 400
        
        cell_size = min(canvas_width // map_size, canvas_height // map_size)
        offset_x = (canvas_width - cell_size * map_size) // 2
        offset_y = (canvas_height - cell_size * map_size) // 2
        
        # Vẽ lưới
        for y in range(map_size):
            for x in range(map_size):
                # Tính tọa độ pixel
                pixel_x = offset_x + x * cell_size
                pixel_y = offset_y + y * cell_size
                
                # Xác định màu sắc dựa trên loại ô
                cell_type = self.map_object.grid[y, x]
                if cell_type == CellType.OBSTACLE:
                    color = "darkgray"  # Vật cản
                elif cell_type == CellType.TOLL:
                    color = "red"  # Trạm thu phí
                elif cell_type == CellType.GAS:
                    color = "green"  # Trạm xăng
                else:
                    color = "white"  # Đường thường
                
                # Vẽ ô
                canvas.create_rectangle(
                    pixel_x, pixel_y, 
                    pixel_x + cell_size, pixel_y + cell_size,
                    fill=color, outline="black"
                )
                
                # Hiển thị nhãn nếu là điểm bắt đầu hoặc kết thúc
                if (x, y) == self.map_object.start_pos:
                    canvas.create_text(
                        pixel_x + cell_size // 2, 
                        pixel_y + cell_size // 2,
                        text="S", font=("Arial", 12, "bold")
                    )
                elif (x, y) == self.map_object.end_pos:
                    canvas.create_text(
                        pixel_x + cell_size // 2, 
                        pixel_y + cell_size // 2,
                        text="E", font=("Arial", 12, "bold")
                    )
        
        # Vẽ agent nếu có
        if agent_pos is not None and agent_pos.size > 0:
            agent_x, agent_y = agent_pos
            pixel_x = offset_x + agent_x * cell_size
            pixel_y = offset_y + agent_y * cell_size
            
            # Vẽ agent (hình tròn màu xanh dương)
            canvas.create_oval(
                pixel_x + cell_size // 4, 
                pixel_y + cell_size // 4,
                pixel_x + cell_size * 3 // 4, 
                pixel_y + cell_size * 3 // 4,
                fill="blue", outline="black"
            )
    
    def _initialize_rl_env(self):
        """Khởi tạo môi trường RL."""
        # Kiểm tra xem đã tạo bản đồ chưa
        if not self.map_object:
            messagebox.showerror("Lỗi", "Vui lòng tạo hoặc tải bản đồ trước!")
            return
        
        try:
            # Lấy các tham số từ giao diện
            initial_fuel = self.initial_fuel_var.get()
            initial_money = self.initial_money_var.get()
            fuel_per_move = self.fuel_per_move_var.get()
            gas_station_cost = self.gas_station_cost_var.get()
            toll_base_cost = self.toll_base_cost_var.get()
            max_steps = self.max_steps_var.get()
            
            # Khởi tạo môi trường RL
            self.rl_env = TruckRoutingEnv(
                map_object=self.map_object,
                initial_fuel=initial_fuel,
                initial_money=initial_money,
                fuel_per_move=fuel_per_move,
                gas_station_cost=gas_station_cost,
                toll_base_cost=toll_base_cost,
                max_steps_per_episode=max_steps
            )
            
            # Hiển thị thông tin môi trường
            self._display_rl_info()
            
            # Vẽ bản đồ trên canvas ở tab Điều khiển
            self._draw_map(self.rl_canvas)
            
            self._log("Đã khởi tạo môi trường RL thành công!")
            messagebox.showinfo("Thông báo", "Đã khởi tạo môi trường RL thành công!")
        except Exception as e:
            self._log(f"Lỗi khi khởi tạo môi trường RL: {e}")
            import traceback
            self._log(traceback.format_exc())
            messagebox.showerror("Lỗi", f"Lỗi khi khởi tạo môi trường RL: {e}")
    
    def _display_rl_info(self):
        """Hiển thị thông tin môi trường RL."""
        if not self.rl_env:
            return
        
        info = "THÔNG TIN MÔI TRƯỜNG RL:\n\n"
        
        # Thông tin không gian hành động
        info += "Action Space (Không gian Hành động):\n"
        info += f"Type: {self.rl_env.action_space}\n"
        info += f"Num Actions: {self.rl_env.action_space.n}\n\n"
        
        # Thông tin không gian quan sát
        info += "Observation Space (Không gian Quan sát):\n"
        for key, space in self.rl_env.observation_space.spaces.items():
            info += f"- {key}: {space}\n"
            if hasattr(space, 'shape'):
                info += f"  Shape: {space.shape}, Type: {space.dtype}\n"
            if hasattr(space, 'low') and hasattr(space, 'high'):
                info += f"  Range: [{space.low[0]}, {space.high[0]}]\n"
        
        # Thông tin tham số
        info += "\nTHÔNG SỐ CẤU HÌNH:\n"
        info += f"Nhiên liệu ban đầu: {self.rl_env.initial_fuel}\n"
        info += f"Tiền ban đầu: {self.rl_env.initial_money}\n"
        info += f"Nhiên liệu tiêu thụ mỗi bước: {self.rl_env.fuel_per_move}\n"
        info += f"Chi phí trạm xăng: {self.rl_env.gas_station_cost}\n"
        info += f"Chi phí trạm thu phí: {self.rl_env.toll_base_cost}\n"
        info += f"Số bước tối đa mỗi episode: {self.rl_env.max_steps_per_episode}\n"
        
        self.rl_info_text.delete(1.0, tk.END)
        self.rl_info_text.insert(tk.END, info)
    
    def _reset_environment(self):
        """Reset môi trường RL."""
        if not self.rl_env:
            messagebox.showerror("Lỗi", "Vui lòng khởi tạo môi trường RL trước!")
            return
        
        try:
            # Gymnasium API trả về (observation, info)
            observation, info = self.rl_env.reset()
            self.current_observation = observation
            
            # Hiển thị trạng thái hiện tại
            self._display_current_state()
            
            # Kiểm tra xem current_observation có phải là None không
            if self.current_observation is not None:
                self._draw_map(self.rl_canvas, self.current_observation['agent_pos'])
            else:
                self._log("Cảnh báo: Reset môi trường trả về None")
                
            # Kích hoạt các nút thực hiện hành động
            self.step_btn.config(state="normal")
            self.random_action_btn.config(state="normal")
            
            # Xóa kết quả hành động
            self.result_text.delete(1.0, tk.END)
            
            self._log("Đã reset môi trường RL")
        except Exception as e:
            self._log(f"Lỗi khi reset môi trường: {e}")
            import traceback
            self._log(traceback.format_exc())
            messagebox.showerror("Lỗi", f"Lỗi khi reset môi trường: {e}")
    
    def _display_current_state(self):
        """Hiển thị trạng thái hiện tại."""
        if not self.current_observation:
            return
        
        state_text = "TRẠNG THÁI HIỆN TẠI:\n\n"
        state_text += f"Vị trí agent: {tuple(self.current_observation['agent_pos'])}\n"
        state_text += f"Vị trí đích: {tuple(self.current_observation['target_pos'])}\n"
        state_text += f"Nhiên liệu: {self.current_observation['fuel'][0]:.2f}\n"
        state_text += f"Tiền: {self.current_observation['money'][0]:.2f}\n"
        state_text += f"Số bước: {self.rl_env.current_step_in_episode}/{self.rl_env.max_steps_per_episode}\n"
        
        self.state_text.delete(1.0, tk.END)
        self.state_text.insert(tk.END, state_text)
    
    def _step_environment(self):
        """Thực hiện một bước trong môi trường RL."""
        if not self.rl_env or self.current_observation is None:
            messagebox.showerror("Lỗi", "Vui lòng reset môi trường RL trước!")
            return
        
        try:
            # Lấy hành động từ giao diện
            action = self.action_var.get()
            
            # Thực hiện hành động (Gymnasium API trả về 5 giá trị)
            next_observation, reward, terminated, truncated, info = self.rl_env.step(action)
            
            # Cập nhật trạng thái hiện tại
            self.current_observation = next_observation
            
            # Hiển thị kết quả
            self._display_result(action, reward, terminated or truncated, info)
            self._display_current_state()
            
            # Kiểm tra xem current_observation có hợp lệ không
            if self.current_observation is not None:
                self._draw_map(self.rl_canvas, self.current_observation['agent_pos'])
            
            # Kiểm tra xem episode đã kết thúc chưa
            if terminated or truncated:
                self._log(f"Episode kết thúc! Lý do: {info.get('termination_reason', 'unknown')}")
                self.step_btn.config(state="disabled")
                self.random_action_btn.config(state="disabled")
        except Exception as e:
            self._log(f"Lỗi khi thực hiện hành động: {e}")
            import traceback
            self._log(traceback.format_exc())
            messagebox.showerror("Lỗi", f"Lỗi khi thực hiện hành động: {e}")
    
    def _random_action(self):
        """Thực hiện hành động ngẫu nhiên."""
        if not self.rl_env:
            messagebox.showerror("Lỗi", "Vui lòng khởi tạo môi trường RL trước!")
            return
            
        if self.current_observation is None:
            # Nếu môi trường chưa được reset, thực hiện reset trước
            self._log("Môi trường chưa được reset. Thực hiện reset trước khi thực hiện hành động ngẫu nhiên.")
            self._reset_environment()
            if self.current_observation is None:
                return  # Nếu reset không thành công, thoát

        try:
            # Chọn hành động ngẫu nhiên
            action = np.random.randint(0, 6)  # 0-5 (Lên, Xuống, Trái, Phải, Đổ xăng, Bỏ qua)
            self.action_var.set(action)
            
            # Thực hiện hành động (Gymnasium API trả về 5 giá trị)
            next_observation, reward, terminated, truncated, info = self.rl_env.step(action)
            
            # Hiển thị kết quả
            self._display_result(action, reward, terminated or truncated, info)
            
            # Cập nhật trạng thái hiện tại
            self.current_observation = next_observation
            
            # Cập nhật hiển thị
            if self.rl_env.current_pos is not None:
                self._draw_map(self.rl_canvas, self.rl_env.current_pos)
            self._display_current_state()
            
            # Kiểm tra xem episode đã kết thúc chưa
            if terminated or truncated:
                self._log(f"Episode kết thúc sau {self.rl_env.current_step_in_episode} bước.")
                if "termination_reason" in info:
                    self._log(f"Lý do kết thúc: {info['termination_reason']}")
                    
                if info.get("termination_reason") == "den_dich":
                    messagebox.showinfo("Thông báo", "Đã đến đích!")
                else:
                    messagebox.showinfo("Thông báo", f"Episode kết thúc: {info.get('termination_reason', 'Không xác định')}")
        except Exception as e:
            self._log(f"Lỗi khi thực hiện hành động ngẫu nhiên: {e}")
            import traceback
            self._log(traceback.format_exc())
            messagebox.showerror("Lỗi", f"Lỗi khi thực hiện hành động ngẫu nhiên: {e}")
    
    def _display_result(self, action, reward, done, info):
        """
        Hiển thị kết quả của hành động.
        
        Args:
            action: Hành động đã thực hiện
            reward: Phần thưởng nhận được
            done: Cờ đánh dấu episode đã kết thúc
            info: Thông tin bổ sung
        """
        action_names = ["Lên", "Xuống", "Trái", "Phải", "Đổ xăng", "Bỏ qua"]
        result_text = "KẾT QUẢ HÀNH ĐỘNG:\n\n"
        result_text += f"Hành động: {action_names[action]}\n"
        result_text += f"Phần thưởng: {reward:.2f}\n"
        result_text += f"Episode kết thúc: {'Có' if done else 'Không'}\n\n"
        
        if info:
            result_text += "Thông tin bổ sung:\n"
            for key, value in info.items():
                result_text += f"- {key}: {value}\n"
        
        self.result_text.delete(1.0, tk.END)
        self.result_text.insert(tk.END, result_text)
        
        # Ghi log
        self._log(f"Thực hiện hành động {action_names[action]}, Phần thưởng: {reward:.2f}, Kết thúc: {done}")
    
    def _start_short_training(self):
        """Bắt đầu huấn luyện ngắn cho agent."""
        try:
            # Kiểm tra xem đã khởi tạo môi trường RL chưa
            if not self.rl_env:
                messagebox.showerror("Lỗi", "Vui lòng khởi tạo môi trường RL trước!")
                return
            
            # Đọc các siêu tham số
            learning_rate = self.learning_rate_var.get()
            total_timesteps = self.total_timesteps_var.get()
            buffer_size = self.buffer_size_var.get()
            batch_size = self.batch_size_var.get()
            train_freq = self.training_freq_var.get()
            
            # Khởi tạo thư mục logs nếu chưa có
            log_dir = "./rl_models_logs/"
            os.makedirs(log_dir, exist_ok=True)
            
            # Khởi tạo DQNAgentTrainer
            self._log("Đang khởi tạo DQNAgentTrainer...")
            self.dqn_agent = DQNAgentTrainer(
                env=self.rl_env,
                log_dir=log_dir
            )
            
            # Cấu hình lại model với các siêu tham số đã nhập
            self._log("Tạo model DQN với các siêu tham số đã nhập...")
            from stable_baselines3 import DQN
            self.dqn_agent.model = DQN(
                "MultiInputPolicy",
                self.rl_env,
                verbose=1,
                tensorboard_log=log_dir,
                learning_rate=learning_rate,
                buffer_size=buffer_size,
                batch_size=batch_size,
                train_freq=train_freq
            )
            
            # Bắt đầu huấn luyện
            self._log(f"Bắt đầu huấn luyện với {total_timesteps} timesteps...")
            self.dqn_agent.train(total_timesteps=total_timesteps)
            
            # Hoàn thành huấn luyện
            self._log("Huấn luyện đã hoàn tất!")
            messagebox.showinfo("Thông báo", "Huấn luyện đã hoàn tất!")
            
            # Đánh giá model vừa huấn luyện
            self._evaluate_agent()
            
        except Exception as e:
            self._log(f"Lỗi trong quá trình huấn luyện: {e}")
            import traceback
            self._log(traceback.format_exc())
            messagebox.showerror("Lỗi", f"Lỗi trong quá trình huấn luyện: {e}")
    
    def _save_current_model(self):
        """Lưu model hiện tại."""
        if not self.dqn_agent:
            messagebox.showerror("Lỗi", "Không có model để lưu! Vui lòng huấn luyện trước.")
            return
        
        model_name = self.model_name_var.get()
        if not model_name:
            model_name = "dqn_truck_router"
        
        try:
            # Đảm bảo thư mục saved_models tồn tại
            os.makedirs("./saved_models", exist_ok=True)
            
            # Lưu model
            save_path = f"./saved_models/{model_name}"
            self.dqn_agent.save_model(save_path)
            self._log(f"Đã lưu model vào {save_path}.zip")
            messagebox.showinfo("Thông báo", f"Đã lưu model vào {save_path}.zip")
        except Exception as e:
            self._log(f"Lỗi khi lưu model: {e}")
            messagebox.showerror("Lỗi", f"Lỗi khi lưu model: {e}")
    
    def _load_trained_model(self):
        """Tải model đã huấn luyện."""
        # Kiểm tra xem đã khởi tạo môi trường RL chưa
        if not self.rl_env:
            messagebox.showerror("Lỗi", "Vui lòng khởi tạo môi trường RL trước!")
            return
        
        from tkinter import filedialog
        
        try:
            # Mở hộp thoại chọn file
            file_path = filedialog.askopenfilename(
                title="Chọn model DQN",
                filetypes=[("ZIP files", "*.zip")],
                initialdir="./saved_models"
            )
            
            if not file_path:
                return  # Người dùng đã hủy
            
            # Import DQNAgentTrainer nếu chưa import
            if not self.dqn_agent:
                try:
                    from core.algorithms.rl_DQNAgent import DQNAgentTrainer
                    self.dqn_agent = DQNAgentTrainer(self.rl_env)
                except ImportError as e:
                    self._log(f"Lỗi import DQNAgentTrainer: {e}")
                    messagebox.showerror("Lỗi", f"Không thể import DQNAgentTrainer: {e}")
                    return
            
            # Tải model
            self._log(f"Đang tải model từ {file_path}...")
            self.dqn_agent.load_model(file_path.replace(".zip", ""))
            
            # Cập nhật tên model
            model_name = os.path.basename(file_path).replace(".zip", "")
            self.model_name_var.set(model_name)
            
            self._log(f"Đã tải model thành công!")
            messagebox.showinfo("Thông báo", "Đã tải model thành công!")
            
            # Đánh giá model vừa tải
            self._evaluate_agent()
            
        except Exception as e:
            self._log(f"Lỗi khi tải model: {e}")
            import traceback
            self._log(traceback.format_exc())
            messagebox.showerror("Lỗi", f"Lỗi khi tải model: {e}")
    
    def _run_agent_on_current_map(self):
        """Chạy agent trên bản đồ hiện tại."""
        if not self.dqn_agent:
            messagebox.showerror("Lỗi", "Không có agent! Vui lòng huấn luyện hoặc tải model trước.")
            return
        
        if not self.rl_env:
            messagebox.showerror("Lỗi", "Vui lòng khởi tạo môi trường RL trước!")
            return
        
        try:
            # Reset môi trường (Gymnasium API trả về observation, info)
            observation, _ = self.rl_env.reset()
            terminated = False
            truncated = False
            total_reward = 0
            step_count = 0
            path = [self.rl_env.current_pos]
            
            # Hiển thị trạng thái ban đầu
            self._draw_map(self.rl_canvas, self.rl_env.current_pos)
            self._display_current_state()
            
            self.results_text.delete(1.0, tk.END)
            self.results_text.insert(tk.END, "Đang chạy agent...\n")
            self.results_text.update()
            
            visualize = self.visualize_steps_var.get()
            delay_ms = self.visualization_speed_var.get()
            
            # Vòng lặp episode
            while not (terminated or truncated):
                # Dự đoán hành động từ agent
                action = self.dqn_agent.predict_action(observation)
                
                # Thực hiện hành động trong môi trường (Gymnasium API trả về 5 giá trị)
                next_observation, reward, terminated, truncated, info = self.rl_env.step(action)
                
                # Cập nhật tổng phần thưởng và đếm bước
                total_reward += reward
                step_count += 1
                
                # Lưu vị trí vào đường đi
                if self.rl_env.current_pos is not None:
                    path.append(self.rl_env.current_pos)
                
                # Hiển thị từng bước nếu được yêu cầu
                if visualize:
                    self._draw_map_with_path(self.rl_canvas, self.rl_env.current_pos, path)
                    self._display_current_state()
                    self._display_result(action, reward, terminated or truncated, info)
                    self.rl_canvas.update()
                    self.root.after(delay_ms)  # Delay để hiển thị
                
                # Cập nhật observation
                observation = next_observation
                
                # Kiểm tra nếu số bước vượt quá giới hạn an toàn
                if step_count > self.rl_env.max_steps_per_episode * 2:
                    self._log("Cảnh báo: Đã vượt quá số bước tối đa. Dừng chạy agent.")
                    break
            
            # Hiển thị đường đi cuối cùng
            self._draw_map_with_path(self.rl_canvas, self.rl_env.current_pos, path)
            
            # Hiển thị kết quả
            success = info.get("termination_reason") == "den_dich"
            result_text = "KẾT QUẢ CHẠY AGENT:\n"
            result_text += f"Hoàn thành nhiệm vụ: {'Thành công' if success else 'Thất bại'}\n"
            result_text += f"Lý do kết thúc: {info.get('termination_reason', 'Không xác định')}\n"
            result_text += f"Tổng phần thưởng: {total_reward:.2f}\n"
            result_text += f"Số bước di chuyển: {step_count}\n"
            
            if observation is not None and 'fuel' in observation and 'money' in observation:
                result_text += f"Nhiên liệu còn lại: {float(observation['fuel'][0]):.2f}\n"
                result_text += f"Tiền còn lại: {float(observation['money'][0]):.2f}\n"
            
            self.results_text.delete(1.0, tk.END)
            self.results_text.insert(tk.END, result_text)
            
            self._log(f"Agent chạy xong. Thành công: {success}")
            
        except Exception as e:
            self._log(f"Lỗi khi chạy agent: {e}")
            import traceback
            self._log(traceback.format_exc())
            messagebox.showerror("Lỗi", f"Lỗi khi chạy agent: {e}")
    
    def _evaluate_agent(self, n_episodes=10):
        """Đánh giá agent qua nhiều episodes."""
        if not self.dqn_agent:
            return
        
        try:
            self._log(f"Đang đánh giá agent qua {n_episodes} episodes...")
            metrics = self.dqn_agent.evaluate(n_episodes=n_episodes)
            
            # Hiển thị kết quả đánh giá
            results = "KẾT QUẢ ĐÁNH GIÁ AGENT:\n"
            results += f"Tỷ lệ thành công: {metrics['success_rate']:.2f}\n"
            results += f"Phần thưởng trung bình: {metrics['avg_reward']:.2f}\n"
            results += f"Độ dài đường đi trung bình: {metrics['avg_path_length']:.2f}\n"
            results += f"Nhiên liệu còn lại trung bình: {metrics['avg_remaining_fuel']:.2f}\n"
            results += f"Tiền còn lại trung bình: {metrics['avg_remaining_money']:.2f}\n"
            
            self.results_text.delete(1.0, tk.END)
            self.results_text.insert(tk.END, results)
            
            self._log("Đánh giá hoàn tất!")
        except Exception as e:
            self._log(f"Lỗi khi đánh giá agent: {e}")
    
    def _draw_map_with_path(self, canvas, agent_pos, path):
        """
        Vẽ bản đồ với đường đi của agent.
        
        Args:
            canvas: Canvas để vẽ
            agent_pos: Vị trí hiện tại của agent
            path: Danh sách các vị trí trên đường đi
        """
        try:
            # Vẽ bản đồ cơ bản
            self._draw_map(canvas, agent_pos)
            
            if not path or len(path) <= 1:
                return
            
            # Tính kích thước ô
            map_size = self.map_object.size
            canvas_width = canvas.winfo_width()
            canvas_height = canvas.winfo_height()
            
            # Đảm bảo kích thước tối thiểu cho canvas
            if canvas_width < 50:
                canvas_width = 400
            if canvas_height < 50:
                canvas_height = 400
            
            cell_size = min(canvas_width // map_size, canvas_height // map_size)
            offset_x = (canvas_width - cell_size * map_size) // 2
            offset_y = (canvas_height - cell_size * map_size) // 2
            
            # Vẽ đường đi
            for i in range(len(path) - 1):
                # Chuyển đổi từ numpy array sang tuple nếu cần
                if hasattr(path[i], 'shape'):
                    x1, y1 = path[i][0], path[i][1]
                else:
                    x1, y1 = path[i]
                    
                if hasattr(path[i+1], 'shape'):
                    x2, y2 = path[i+1][0], path[i+1][1]
                else:
                    x2, y2 = path[i+1]
                
                # Tính tọa độ tâm của các ô
                center_x1 = offset_x + x1 * cell_size + cell_size // 2
                center_y1 = offset_y + y1 * cell_size + cell_size // 2
                center_x2 = offset_x + x2 * cell_size + cell_size // 2
                center_y2 = offset_y + y2 * cell_size + cell_size // 2
                
                # Vẽ đường nối giữa hai ô
                canvas.create_line(
                    center_x1, center_y1, center_x2, center_y2,
                    fill="blue", width=2, arrow=tk.LAST
                )
        except Exception as e:
            self._log(f"Lỗi khi vẽ đường đi: {e}")
    
    def _select_directory(self, string_var):
        """
        Mở hộp thoại chọn thư mục và cập nhật biến StringVar.
        
        Args:
            string_var: Biến StringVar để cập nhật
        """
        directory = filedialog.askdirectory()
        if directory:
            string_var.set(directory)
    
    def _create_map_dirs(self):
        """Tạo các thư mục bản đồ nếu chưa tồn tại."""
        train_dir = self.train_maps_dir_var.get()
        eval_dir = self.eval_maps_dir_var.get()
        test_dir = self.detailed_eval_maps_dir_var.get()
        
        # Tạo thư mục
        os.makedirs(train_dir, exist_ok=True)
        os.makedirs(eval_dir, exist_ok=True)
        os.makedirs(test_dir, exist_ok=True)
        
        # Kiểm tra xem có cần tạo bản đồ mẫu không
        if len(os.listdir(train_dir)) == 0:
            self._log("Tạo bản đồ huấn luyện mẫu...")
            for i in range(5):
                map_obj = Map.generate_random(size=10, toll_ratio=0.05, gas_ratio=0.05, brick_ratio=0.2)
                map_obj.save(os.path.join(train_dir, f"train_map_{i}.json"))
        
        if len(os.listdir(eval_dir)) == 0:
            self._log("Tạo bản đồ đánh giá mẫu...")
            for i in range(3):
                map_obj = Map.generate_random(size=10, toll_ratio=0.05, gas_ratio=0.05, brick_ratio=0.2)
                map_obj.save(os.path.join(eval_dir, f"eval_map_{i}.json"))
        
        if len(os.listdir(test_dir)) == 0:
            self._log("Tạo bản đồ kiểm thử mẫu...")
            for i in range(3):
                map_obj = Map.generate_random(size=12, toll_ratio=0.05, gas_ratio=0.05, brick_ratio=0.2)
                map_obj.save(os.path.join(test_dir, f"test_map_{i}.json"))
        
        self._log(f"Các thư mục bản đồ đã được tạo và khởi tạo:")
        self._log(f"- Huấn luyện: {train_dir} ({len(os.listdir(train_dir))} bản đồ)")
        self._log(f"- Đánh giá: {eval_dir} ({len(os.listdir(eval_dir))} bản đồ)")
        self._log(f"- Kiểm thử: {test_dir} ({len(os.listdir(test_dir))} bản đồ)")
        
        messagebox.showinfo("Hoàn tất", "Các thư mục bản đồ đã được tạo và khởi tạo thành công!")
    
    def _start_hyperparameter_tuning(self):
        """Bắt đầu tinh chỉnh siêu tham số trong một luồng riêng biệt."""
        # Kiểm tra xem đã có Optuna chưa
        try:
            import optuna
        except ImportError:
            self._log("Không tìm thấy thư viện Optuna. Đang cài đặt...")
            import subprocess
            subprocess.check_call([sys.executable, "-m", "pip", "install", "optuna"])
            self._log("Đã cài đặt Optuna thành công.")
        
        # Kiểm tra xem các thư mục có tồn tại không
        train_dir = self.train_maps_dir_var.get()
        eval_dir = self.eval_maps_dir_var.get()
        
        if not os.path.exists(train_dir) or not os.path.exists(eval_dir):
            messagebox.showerror("Lỗi", "Thư mục bản đồ không tồn tại. Vui lòng tạo thư mục trước!")
            return
        
        if len(os.listdir(train_dir)) == 0 or len(os.listdir(eval_dir)) == 0:
            messagebox.showerror("Lỗi", "Thư mục bản đồ trống. Vui lòng tạo bản đồ trước!")
            return
        
        # Vô hiệu hóa nút khi đang tinh chỉnh
        self.start_tuning_btn.config(state="disabled")
        self.train_best_btn.config(state="disabled")
        
        # Xóa kết quả cũ
        self.tuning_results_text.delete(1.0, tk.END)
        self.tuning_results_text.insert(tk.END, "Đang bắt đầu tinh chỉnh siêu tham số...\n")
        self.tuning_results_text.update()
        
        # Lấy tham số
        n_trials = self.n_trials_var.get()
        n_timesteps = self.tuning_timesteps_var.get()
        n_eval_episodes = self.eval_episodes_var.get()
        n_jobs = self.n_jobs_var.get()
        
        # Bắt đầu tinh chỉnh trong một luồng riêng biệt
        self._log(f"Bắt đầu tinh chỉnh siêu tham số với {n_trials} lần thử...")
        
        def tuning_thread():
            try:
                # Thư mục kết quả
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                results_dir = f"./hyperparameter_tuning_results/{timestamp}"
                os.makedirs(results_dir, exist_ok=True)
                
                # Ghi log
                self._update_tuning_log(f"Tinh chỉnh với {n_trials} lần thử, {n_timesteps} bước mỗi lần\n")
                self._update_tuning_log(f"Thư mục huấn luyện: {train_dir}\n")
                self._update_tuning_log(f"Thư mục đánh giá: {eval_dir}\n")
                self._update_tuning_log(f"Kết quả sẽ được lưu tại: {results_dir}\n\n")
                
                # Bắt đầu tinh chỉnh
                start_time = time.time()
                
                best_params = optimize_hyperparameters(
                    train_maps_dir=train_dir,
                    eval_maps_dir=eval_dir,
                    n_trials=n_trials,
                    n_timesteps=n_timesteps,
                    n_eval_episodes=n_eval_episodes,
                    n_jobs=n_jobs,
                    study_name=f"dqn_optimization_{timestamp}"
                )
                
                end_time = time.time()
                elapsed_time = end_time - start_time
                
                # Lưu tham số tốt nhất
                self.best_params = best_params
                self.best_params_dir = results_dir
                
                # Hiển thị kết quả
                self._update_tuning_log(f"\nTinh chỉnh hoàn tất sau {elapsed_time:.2f} giây!\n")
                self._update_tuning_log(f"Tham số tốt nhất đã được lưu tại: {results_dir}/best_params.json\n\n")
                self._update_tuning_log("Tham số tốt nhất:\n")
                
                for param, value in best_params.items():
                    self._update_tuning_log(f"- {param}: {value}\n")
                
                self._log(f"Tinh chỉnh siêu tham số hoàn tất! Kết quả được lưu tại {results_dir}")
                
                # Kích hoạt nút
                self.root.after(0, lambda: self.start_tuning_btn.config(state="normal"))
                self.root.after(0, lambda: self.train_best_btn.config(state="normal"))
                
            except Exception as e:
                self._log(f"Lỗi khi tinh chỉnh siêu tham số: {e}")
                import traceback
                self._log(traceback.format_exc())
                
                self._update_tuning_log(f"\nLỗi khi tinh chỉnh siêu tham số: {e}\n")
                
                # Kích hoạt nút
                self.root.after(0, lambda: self.start_tuning_btn.config(state="normal"))
                self.root.after(0, lambda: self.train_best_btn.config(state="normal"))
        
        # Bắt đầu luồng
        threading.Thread(target=tuning_thread, daemon=True).start()
    
    def _update_tuning_log(self, message):
        """
        Cập nhật log tinh chỉnh siêu tham số.
        
        Args:
            message: Thông điệp cần ghi log
        """
        self.root.after(0, lambda: self.tuning_results_text.insert(tk.END, message))
        self.root.after(0, lambda: self.tuning_results_text.see(tk.END))
    
    def _train_with_best_params(self):
        """Huấn luyện model với tham số tốt nhất."""
        # Kiểm tra xem đã có tham số tốt nhất chưa
        if not hasattr(self, 'best_params'):
            # Thử đọc từ file
            try:
                # Tìm thư mục kết quả gần nhất
                results_dir = "./hyperparameter_tuning_results"
                if os.path.exists(results_dir):
                    subdirs = [os.path.join(results_dir, d) for d in os.listdir(results_dir) 
                              if os.path.isdir(os.path.join(results_dir, d))]
                    latest_dir = max(subdirs, key=os.path.getmtime) if subdirs else None
                    
                    if latest_dir and os.path.exists(os.path.join(latest_dir, "best_params.json")):
                        with open(os.path.join(latest_dir, "best_params.json"), "r") as f:
                            self.best_params = json.load(f)
                        self.best_params_dir = latest_dir
                        self._log(f"Đã tải tham số tốt nhất từ {latest_dir}")
                    else:
                        messagebox.showerror("Lỗi", "Không tìm thấy tham số tốt nhất. Vui lòng tinh chỉnh trước!")
                        return
                else:
                    messagebox.showerror("Lỗi", "Không tìm thấy thư mục kết quả tinh chỉnh!")
                    return
            except Exception as e:
                messagebox.showerror("Lỗi", f"Không thể tải tham số tốt nhất: {e}")
                return
        
        # Kiểm tra xem đã tạo bản đồ chưa
        if not self.map_object:
            messagebox.showerror("Lỗi", "Vui lòng tạo hoặc tải bản đồ trước!")
            return
        
        # Vô hiệu hóa nút khi đang huấn luyện
        self.train_best_btn.config(state="disabled")
        
        # Hiển thị thông báo
        self._update_tuning_log("\nBắt đầu huấn luyện với tham số tốt nhất...\n")
        
        # Lấy tham số
        n_timesteps = 50000  # Huấn luyện lâu hơn
        
        # Bắt đầu huấn luyện trong một luồng riêng biệt
        self._log(f"Bắt đầu huấn luyện model với tham số tốt nhất...")
        
        def training_thread():
            try:
                # Tạo thư mục lưu model
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                save_path = f"./saved_models/best_dqn_agent_{timestamp}"
                
                # Ghi log
                self._update_tuning_log(f"Huấn luyện với {n_timesteps} bước\n")
                self._update_tuning_log(f"Model sẽ được lưu tại: {save_path}\n\n")
                
                # Bắt đầu huấn luyện
                start_time = time.time()
                
                agent = train_agent_with_best_params(
                    best_params=self.best_params,
                    train_map=self.map_object,
                    n_timesteps=n_timesteps,
                    save_path=save_path
                )
                
                end_time = time.time()
                elapsed_time = end_time - start_time
                
                # Đánh giá model
                metrics = agent.evaluate(n_episodes=5)
                
                # Hiển thị kết quả
                self._update_tuning_log(f"\nHuấn luyện hoàn tất sau {elapsed_time:.2f} giây!\n")
                self._update_tuning_log(f"Model đã được lưu tại: {save_path}.zip\n\n")
                self._update_tuning_log("Kết quả đánh giá:\n")
                self._update_tuning_log(f"- Tỷ lệ thành công: {metrics['success_rate']:.2f}\n")
                self._update_tuning_log(f"- Phần thưởng trung bình: {metrics['avg_reward']:.2f}\n")
                self._update_tuning_log(f"- Độ dài đường đi trung bình: {metrics['avg_path_length']:.2f}\n")
                self._update_tuning_log(f"- Nhiên liệu còn lại trung bình: {metrics['avg_remaining_fuel']:.2f}\n")
                self._update_tuning_log(f"- Tiền còn lại trung bình: {metrics['avg_remaining_money']:.2f}\n")
                
                self._log(f"Huấn luyện hoàn tất! Model được lưu tại {save_path}.zip")
                
                # Kích hoạt nút
                self.root.after(0, lambda: self.train_best_btn.config(state="normal"))
                
            except Exception as e:
                self._log(f"Lỗi khi huấn luyện model: {e}")
                import traceback
                self._log(traceback.format_exc())
                
                self._update_tuning_log(f"\nLỗi khi huấn luyện model: {e}\n")
                
                # Kích hoạt nút
                self.root.after(0, lambda: self.train_best_btn.config(state="normal"))
        
        # Bắt đầu luồng
        threading.Thread(target=training_thread, daemon=True).start()
    
    def _select_model_for_evaluation(self):
        """Mở hộp thoại chọn model cho đánh giá."""
        file_path = filedialog.askopenfilename(
            title="Chọn model DQN",
            filetypes=[("ZIP files", "*.zip")],
            initialdir="./saved_models"
        )
        
        if file_path:
            self.eval_model_path_var.set(file_path.replace(".zip", ""))
    
    def _select_model_for_comparison(self):
        """Mở hộp thoại chọn model cho so sánh."""
        file_path = filedialog.askopenfilename(
            title="Chọn model DQN",
            filetypes=[("ZIP files", "*.zip")],
            initialdir="./saved_models"
        )
        
        if file_path:
            self.comp_model_path_var.set(file_path.replace(".zip", ""))
    
    def _select_map_for_comparison(self):
        """Mở hộp thoại chọn bản đồ cho so sánh."""
        file_path = filedialog.askopenfilename(
            title="Chọn bản đồ",
            filetypes=[("JSON files", "*.json")],
            initialdir="./maps"
        )
        
        if file_path:
            try:
                self.comparison_map_obj = Map.load(file_path)
                messagebox.showinfo("Thành công", f"Đã tải bản đồ {file_path}")
            except Exception as e:
                messagebox.showerror("Lỗi", f"Không thể tải bản đồ: {e}")
    
    def _start_detailed_evaluation(self):
        """Bắt đầu đánh giá chi tiết agent RL."""
        # Kiểm tra model path
        model_path = self.eval_model_path_var.get()
        if not model_path:
            messagebox.showerror("Lỗi", "Vui lòng chọn model RL!")
            return
        
        # Kiểm tra thư mục bản đồ
        maps_dir = self.detailed_eval_maps_dir_var.get()
        if not os.path.exists(maps_dir) or len(os.listdir(maps_dir)) == 0:
            messagebox.showerror("Lỗi", "Thư mục bản đồ không tồn tại hoặc trống!")
            return
        
        # Vô hiệu hóa nút khi đang đánh giá
        self.start_evaluation_btn.config(state="disabled")
        
        # Xóa kết quả cũ
        self.evaluation_results_text.delete(1.0, tk.END)
        self.evaluation_results_text.insert(tk.END, "Đang bắt đầu đánh giá chi tiết...\n")
        self.evaluation_results_text.update()
        
        # Xóa đồ thị cũ
        for widget in self.eval_plot_frame.winfo_children():
            widget.destroy()
        
        # Lấy tham số
        n_episodes = self.detailed_eval_episodes_var.get()
        
        # Bắt đầu đánh giá trong một luồng riêng biệt
        self._log(f"Bắt đầu đánh giá chi tiết model {model_path}...")
        
        def evaluation_thread():
            try:
                # Tạo evaluator
                results_dir = "./evaluation_results"
                os.makedirs(results_dir, exist_ok=True)
                
                evaluator = RLEvaluator(maps_dir=maps_dir, results_dir=results_dir)
                
                # Ghi log
                self._update_eval_log(f"Đánh giá model: {model_path}\n")
                self._update_eval_log(f"Thư mục bản đồ: {maps_dir}\n")
                self._update_eval_log(f"Số episodes mỗi bản đồ: {n_episodes}\n\n")
                
                # Bắt đầu đánh giá
                start_time = time.time()
                
                results_df = evaluator.evaluate_rl_agent(
                    model_path=model_path,
                    n_episodes=n_episodes
                )
                
                end_time = time.time()
                elapsed_time = end_time - start_time
                
                # Hiển thị kết quả
                self._update_eval_log(f"Đánh giá hoàn tất sau {elapsed_time:.2f} giây!\n\n")
                
                # Tính toán các chỉ số thống kê
                success_rate = results_df["success"].mean()
                avg_reward = results_df["total_reward"].mean()
                avg_path_length = results_df["path_length"].mean() if "path_length" in results_df.columns else 0
                avg_fuel_consumed = results_df["fuel_consumed"].mean() if "fuel_consumed" in results_df.columns else 0
                avg_money_spent = results_df["money_spent"].mean() if "money_spent" in results_df.columns else 0
                avg_execution_time = results_df["execution_time"].mean() if "execution_time" in results_df.columns else 0
                
                self._update_eval_log("Kết quả tổng quan:\n")
                self._update_eval_log(f"- Tỷ lệ thành công: {success_rate:.2f}\n")
                self._update_eval_log(f"- Phần thưởng trung bình: {avg_reward:.2f}\n")
                self._update_eval_log(f"- Độ dài đường đi trung bình: {avg_path_length:.2f}\n")
                self._update_eval_log(f"- Nhiên liệu tiêu thụ trung bình: {avg_fuel_consumed:.2f}\n")
                self._update_eval_log(f"- Chi phí trung bình: {avg_money_spent:.2f}\n")
                self._update_eval_log(f"- Thời gian thực thi trung bình: {avg_execution_time:.4f} giây\n\n")
                
                # Phân tích lỗi
                if success_rate < 1.0:
                    failure_reasons = results_df[results_df["success"] == False]["termination_reason"].value_counts()
                    self._update_eval_log("Nguyên nhân thất bại:\n")
                    for reason, count in failure_reasons.items():
                        percentage = (count / len(results_df)) * 100
                        self._update_eval_log(f"- {reason}: {count} lần ({percentage:.2f}%)\n")
                
                # Vẽ đồ thị
                self._create_evaluation_plots(results_df)
                
                self._log(f"Đánh giá chi tiết hoàn tất! Kết quả được lưu tại {results_dir}")
                
                # Kích hoạt nút
                self.root.after(0, lambda: self.start_evaluation_btn.config(state="normal"))
                
            except Exception as e:
                self._log(f"Lỗi khi đánh giá chi tiết: {e}")
                import traceback
                self._log(traceback.format_exc())
                
                self._update_eval_log(f"\nLỗi khi đánh giá chi tiết: {e}\n")
                
                # Kích hoạt nút
                self.root.after(0, lambda: self.start_evaluation_btn.config(state="normal"))
        
        # Bắt đầu luồng
        threading.Thread(target=evaluation_thread, daemon=True).start()
    
    def _update_eval_log(self, message):
        """
        Cập nhật log đánh giá chi tiết.
        
        Args:
            message: Thông điệp cần ghi log
        """
        self.root.after(0, lambda: self.evaluation_results_text.insert(tk.END, message))
        self.root.after(0, lambda: self.evaluation_results_text.see(tk.END))
    
    def _create_evaluation_plots(self, results_df):
        """
        Tạo đồ thị từ kết quả đánh giá.
        
        Args:
            results_df: DataFrame kết quả đánh giá
        """
        # Tạo figure với 2x2 subplots
        fig, axs = plt.subplots(2, 2, figsize=(10, 8))
        fig.tight_layout(pad=4.0)
        
        # 1. Success Rate by Map
        success_by_map = results_df.groupby("map_name")["success"].mean().sort_values(ascending=False)
        axs[0, 0].bar(success_by_map.index, success_by_map.values, color="green")
        axs[0, 0].set_title("Tỷ lệ thành công theo bản đồ")
        axs[0, 0].set_ylabel("Tỷ lệ thành công")
        axs[0, 0].tick_params(axis='x', rotation=45)
        
        # 2. Average Path Length
        axs[0, 1].hist(results_df["path_length"], bins=10, color="blue", alpha=0.7)
        axs[0, 1].set_title("Phân phối độ dài đường đi")
        axs[0, 1].set_xlabel("Độ dài đường đi")
        axs[0, 1].set_ylabel("Tần suất")
        
        # 3. Average Reward
        axs[1, 0].hist(results_df["total_reward"], bins=10, color="purple", alpha=0.7)
        axs[1, 0].set_title("Phân phối phần thưởng")
        axs[1, 0].set_xlabel("Tổng phần thưởng")
        axs[1, 0].set_ylabel("Tần suất")
        
        # 4. Resource usage (fuel consumed vs money spent)
        if "fuel_consumed" in results_df.columns and "money_spent" in results_df.columns:
            axs[1, 1].scatter(results_df["fuel_consumed"], results_df["money_spent"], 
                             alpha=0.7, c=results_df["success"], cmap="coolwarm")
            axs[1, 1].set_title("Mối quan hệ giữa nhiên liệu và chi phí")
            axs[1, 1].set_xlabel("Nhiên liệu tiêu thụ")
            axs[1, 1].set_ylabel("Chi phí")
            
            # Thêm colorbar
            cbar = fig.colorbar(axs[1, 1].collections[0], ax=axs[1, 1])
            cbar.set_label("Thành công")
        
        # Thêm canvas vào frame
        canvas = FigureCanvasTkAgg(fig, master=self.eval_plot_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
    
    def _start_algorithm_comparison(self):
        """Bắt đầu so sánh RL agent với thuật toán khác."""
        # Kiểm tra model path
        model_path = self.comp_model_path_var.get()
        if not model_path:
            messagebox.showerror("Lỗi", "Vui lòng chọn model RL!")
            return
        
        # Kiểm tra thuật toán
        algorithm_name = self.compare_algorithm_var.get()
        
        # Kiểm tra bản đồ
        use_current_map = self.use_current_map_var.get()
        
        if use_current_map:
            if not self.map_object:
                messagebox.showerror("Lỗi", "Không có bản đồ hiện tại! Vui lòng tạo hoặc tải bản đồ trước.")
                return
            map_obj = self.map_object
        else:
            if not hasattr(self, 'comparison_map_obj'):
                messagebox.showerror("Lỗi", "Vui lòng chọn bản đồ cho so sánh!")
                return
            map_obj = self.comparison_map_obj
        
        # Vô hiệu hóa nút khi đang so sánh
        self.start_comparison_btn.config(state="disabled")
        
        # Xóa kết quả cũ
        self.comparison_results_text.delete(1.0, tk.END)
        self.comparison_results_text.insert(tk.END, "Đang bắt đầu so sánh thuật toán...\n")
        self.comparison_results_text.update()
        
        # Bắt đầu so sánh trong một luồng riêng biệt
        self._log(f"Bắt đầu so sánh model {model_path} với thuật toán {algorithm_name}...")
        
        def comparison_thread():
            try:
                # Thư mục kết quả
                results_dir = "./evaluation_results"
                os.makedirs(results_dir, exist_ok=True)
                
                # Tạo evaluator với một bản đồ
                evaluator = RLEvaluator(maps_dir=results_dir, results_dir=results_dir)
                
                # Ghi log
                self._update_comparison_log(f"So sánh model: {model_path}\n")
                self._update_comparison_log(f"Thuật toán so sánh: {algorithm_name}\n")
                self._update_comparison_log(f"Bản đồ: {'Hiện tại' if use_current_map else 'Đã chọn'}\n\n")
                
                # Bắt đầu so sánh
                start_time = time.time()
                
                rl_path, rl_metrics, algorithm_path, algorithm_metrics = evaluator.compare_with_algorithm(
                    model_path=model_path,
                    algorithm_name=algorithm_name,
                    map_obj=map_obj
                )
                
                end_time = time.time()
                elapsed_time = end_time - start_time
                
                # Hiển thị kết quả
                self._update_comparison_log(f"So sánh hoàn tất sau {elapsed_time:.2f} giây!\n\n")
                
                # So sánh các chỉ số
                self._update_comparison_log("KẾT QUẢ SO SÁNH:\n\n")
                
                # Tạo bảng so sánh
                self._update_comparison_log("Chỉ số                    | RL Agent        | " + algorithm_name + "\n")
                self._update_comparison_log("-" * 60 + "\n")
                
                # Thành công
                rl_success = rl_metrics.get("success", False)
                algo_success = algorithm_metrics.get("success", False) if algorithm_metrics else False
                self._update_comparison_log(f"Thành công               | {'Có' if rl_success else 'Không':<15} | {'Có' if algo_success else 'Không'}\n")
                
                # Độ dài đường đi
                rl_path_length = rl_metrics.get("path_length", 0)
                algo_path_length = len(algorithm_path) - 1 if algorithm_path else 0
                self._update_comparison_log(f"Độ dài đường đi          | {rl_path_length:<15} | {algo_path_length}\n")
                
                # Nhiên liệu tiêu thụ
                rl_fuel = rl_metrics.get("fuel_consumed", 0)
                algo_fuel = algorithm_metrics.get("fuel_used", 0) if algorithm_metrics else 0
                self._update_comparison_log(f"Nhiên liệu tiêu thụ      | {rl_fuel:<15.2f} | {algo_fuel:.2f}\n")
                
                # Chi phí
                rl_cost = rl_metrics.get("money_spent", 0)
                algo_cost = algorithm_metrics.get("total_cost", 0) if algorithm_metrics else 0
                self._update_comparison_log(f"Chi phí                  | {rl_cost:<15.2f} | {algo_cost:.2f}\n")
                
                # Số bước thực hiện
                rl_steps = rl_metrics.get("total_steps", 0)
                algo_steps = algorithm_metrics.get("iterations", 0) if algorithm_metrics else 0
                self._update_comparison_log(f"Số bước thực hiện        | {rl_steps:<15} | {algo_steps}\n")
                
                # Thời gian thực thi
                rl_time = rl_metrics.get("execution_time", 0)
                algo_time = algorithm_metrics.get("execution_time", 0) if algorithm_metrics else 0
                self._update_comparison_log(f"Thời gian thực thi (giây)| {rl_time:<15.4f} | {algo_time:.4f}\n")
                
                # Vẽ hai đường đi trên cùng bản đồ
                self._draw_comparison_paths(map_obj, rl_path, algorithm_path)
                
                self._log(f"So sánh thuật toán hoàn tất!")
                
                # Kích hoạt nút
                self.root.after(0, lambda: self.start_comparison_btn.config(state="normal"))
                
            except Exception as e:
                self._log(f"Lỗi khi so sánh thuật toán: {e}")
                import traceback
                self._log(traceback.format_exc())
                
                self._update_comparison_log(f"\nLỗi khi so sánh thuật toán: {e}\n")
                
                # Kích hoạt nút
                self.root.after(0, lambda: self.start_comparison_btn.config(state="normal"))
        
        # Bắt đầu luồng
        threading.Thread(target=comparison_thread, daemon=True).start()
    
    def _update_comparison_log(self, message):
        """
        Cập nhật log so sánh thuật toán.
        
        Args:
            message: Thông điệp cần ghi log
        """
        self.root.after(0, lambda: self.comparison_results_text.insert(tk.END, message))
        self.root.after(0, lambda: self.comparison_results_text.see(tk.END))
    
    def _draw_comparison_paths(self, map_obj, rl_path, algorithm_path):
        """
        Vẽ hai đường đi trên cùng bản đồ để so sánh.
        
        Args:
            map_obj: Đối tượng bản đồ
            rl_path: Đường đi của RL agent
            algorithm_path: Đường đi của thuật toán so sánh
        """
        # Xóa canvas
        self.comparison_canvas.delete("all")
        
        # Tính kích thước ô
        map_size = map_obj.size
        canvas_width = self.comparison_canvas.winfo_width()
        canvas_height = self.comparison_canvas.winfo_height()
        
        # Đảm bảo kích thước tối thiểu cho canvas
        if canvas_width < 50:
            canvas_width = 400
        if canvas_height < 50:
            canvas_height = 400
        
        cell_size = min(canvas_width // map_size, canvas_height // map_size)
        offset_x = (canvas_width - cell_size * map_size) // 2
        offset_y = (canvas_height - cell_size * map_size) // 2
        
        # Vẽ lưới
        for y in range(map_size):
            for x in range(map_size):
                # Tính tọa độ pixel
                pixel_x = offset_x + x * cell_size
                pixel_y = offset_y + y * cell_size
                
                # Xác định màu sắc dựa trên loại ô
                cell_type = map_obj.grid[y, x]
                if cell_type == CellType.OBSTACLE:
                    color = "darkgray"  # Vật cản
                elif cell_type == CellType.TOLL:
                    color = "red"  # Trạm thu phí
                elif cell_type == CellType.GAS:
                    color = "green"  # Trạm xăng
                else:
                    color = "white"  # Đường thường
                
                # Vẽ ô
                self.comparison_canvas.create_rectangle(
                    pixel_x, pixel_y, 
                    pixel_x + cell_size, pixel_y + cell_size,
                    fill=color, outline="black"
                )
                
                # Hiển thị nhãn nếu là điểm bắt đầu hoặc kết thúc
                if (x, y) == map_obj.start_pos:
                    self.comparison_canvas.create_text(
                        pixel_x + cell_size // 2, 
                        pixel_y + cell_size // 2,
                        text="S", font=("Arial", 12, "bold")
                    )
                elif (x, y) == map_obj.end_pos:
                    self.comparison_canvas.create_text(
                        pixel_x + cell_size // 2, 
                        pixel_y + cell_size // 2,
                        text="E", font=("Arial", 12, "bold")
                    )
        
        # Vẽ đường đi RL agent (màu xanh dương)
        if rl_path and len(rl_path) > 1:
            for i in range(len(rl_path) - 1):
                # Chuyển đổi từ numpy array sang tuple nếu cần
                if hasattr(rl_path[i], 'shape'):
                    x1, y1 = rl_path[i][0], rl_path[i][1]
                else:
                    x1, y1 = rl_path[i]
                    
                if hasattr(rl_path[i+1], 'shape'):
                    x2, y2 = rl_path[i+1][0], rl_path[i+1][1]
                else:
                    x2, y2 = rl_path[i+1]
                
                # Tính tọa độ tâm của các ô
                center_x1 = offset_x + x1 * cell_size + cell_size // 2
                center_y1 = offset_y + y1 * cell_size + cell_size // 2
                center_x2 = offset_x + x2 * cell_size + cell_size // 2
                center_y2 = offset_y + y2 * cell_size + cell_size // 2
                
                # Vẽ đường nối giữa hai ô
                self.comparison_canvas.create_line(
                    center_x1, center_y1, center_x2, center_y2,
                    fill="blue", width=2, arrow=tk.LAST
                )
        
        # Vẽ đường đi thuật toán so sánh (màu cam)
        if algorithm_path and len(algorithm_path) > 1:
            for i in range(len(algorithm_path) - 1):
                # Tính tọa độ tâm của các ô
                x1, y1 = algorithm_path[i]
                x2, y2 = algorithm_path[i+1]
                
                center_x1 = offset_x + x1 * cell_size + cell_size // 2
                center_y1 = offset_y + y1 * cell_size + cell_size // 2
                center_x2 = offset_x + x2 * cell_size + cell_size // 2
                center_y2 = offset_y + y2 * cell_size + cell_size // 2
                
                # Vẽ đường nối giữa hai ô
                self.comparison_canvas.create_line(
                    center_x1, center_y1, center_x2, center_y2,
                    fill="orange", width=2, arrow=tk.LAST
                )
        
        # Thêm chú thích
        legend_x = offset_x
        legend_y = offset_y + map_size * cell_size + 10
        
        # RL agent (màu xanh dương)
        self.comparison_canvas.create_line(
            legend_x, legend_y, legend_x + 30, legend_y,
            fill="blue", width=2
        )
        self.comparison_canvas.create_text(
            legend_x + 100, legend_y,
            text="RL Agent", anchor="w"
        )
        
        # Thuật toán so sánh (màu cam)
        self.comparison_canvas.create_line(
            legend_x + 200, legend_y, legend_x + 230, legend_y,
            fill="orange", width=2
        )
        self.comparison_canvas.create_text(
            legend_x + 300, legend_y,
            text=self.compare_algorithm_var.get(), anchor="w"
        )
    
    def _save_map(self):
        """Lưu bản đồ hiện tại."""
        if not self.map_object:
            messagebox.showerror("Lỗi", "Không có bản đồ để lưu!")
            return
        
        try:
            # Mở hộp thoại lưu file
            file_path = filedialog.asksaveasfilename(
                title="Lưu bản đồ",
                filetypes=[("JSON files", "*.json")],
                initialdir="./maps",
                defaultextension=".json"
            )
            
            if not file_path:
                return  # Người dùng đã hủy
            
            # Lưu bản đồ
            # Kiểm tra xem file_path có thư mục maps không, nếu không thì lấy tên file đơn giản
            if os.path.dirname(file_path) == os.path.abspath('./maps'):
                # Nếu người dùng chọn trong thư mục maps, chỉ sử dụng tên file
                filename = os.path.basename(file_path)
                saved_path = self.map_object.save(filename)
            else:
                # Nếu người dùng chọn thư mục khác, sao chép file sau khi lưu
                # Lưu vào thư mục maps trước với tên file tạm thời
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                temp_filename = f"temp_map_{self.map_object.size}x{self.map_object.size}_{timestamp}.json"
                saved_path = self.map_object.save(temp_filename)
                
                # Sao chép file từ maps sang đường dẫn mà người dùng chọn
                shutil.copy2(saved_path, file_path)
                
                # Xóa file tạm (tùy chọn)
                try:
                    os.remove(saved_path)
                except:
                    pass
                
                saved_path = file_path
            
            self._log(f"Đã lưu bản đồ vào {saved_path}")
            messagebox.showinfo("Thông báo", f"Đã lưu bản đồ vào {saved_path}")
        except Exception as e:
            self._log(f"Lỗi khi lưu bản đồ: {e}")
            import traceback
            self._log(traceback.format_exc())
            messagebox.showerror("Lỗi", f"Lỗi khi lưu bản đồ: {e}")


def main():
    """Hàm chính để chạy ứng dụng kiểm tra môi trường RL."""
    root = tk.Tk()
    app = RLTestApp(root)
    root.mainloop()


if __name__ == "__main__":
    main() 