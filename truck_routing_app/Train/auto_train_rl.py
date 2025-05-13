#!/usr/bin/env python
"""
╔═══════════════════════════════════════════════════════════════════════╗
║                         AUTO TRAIN RL - MASTER                        ║
║                                                                       ║
║        Công cụ tự động huấn luyện RL agent cho định tuyến xe tải     ║
╚═══════════════════════════════════════════════════════════════════════╝

Công cụ này tự động hóa toàn bộ quá trình huấn luyện:
1. Tự động tạo bản đồ train và test theo kích thước yêu cầu
2. Tự động cấu hình môi trường và tham số phù hợp với kích thước bản đồ
3. Huấn luyện agent với siêu tham số tối ưu cho từng kích thước bản đồ
4. Đánh giá chi tiết và lưu kết quả

Cách sử dụng: 
python auto_train_rl.py

hoặc với tham số chỉ định:
python auto_train_rl.py --map-size 8 --num-maps 20 --training-steps 100000 --advanced

Các tùy chọn:
--map-size: Kích thước bản đồ (mặc định: 8, hỗ trợ: 8, 9, 10, 12, 15)
--num-maps: Số lượng bản đồ tạo cho mỗi loại
--training-steps: Số bước huấn luyện
--advanced: Sử dụng các kỹ thuật nâng cao (Double DQN, Dueling, PER)
--render: Hiển thị quá trình huấn luyện
"""

import os
import sys
import time
import json
import random
import argparse
import numpy as np
from datetime import datetime
from pathlib import Path
import uuid
import shutil
import rich
import re
import traceback # THÊM IMPORT NÀY

# Thêm thư viện UI cho terminal
try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table
    from rich.progress import Progress, TextColumn, BarColumn, TaskProgressColumn, TimeRemainingColumn, SpinnerColumn
    from rich.prompt import Prompt, Confirm
    from rich.markdown import Markdown
    from rich.style import Style
    from rich.text import Text
    from rich.layout import Layout
    from rich.live import Live
    from rich import box
    from rich.align import Align
    from tqdm import tqdm
    
    RICH_AVAILABLE = True
except ImportError:
    print("Thư viện Rich không khả dụng. Đang cài đặt...")
    import subprocess
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "rich", "tqdm"])
        from rich.console import Console
        from rich.panel import Panel
        from rich.table import Table
        from rich.progress import Progress, TextColumn, BarColumn, TaskProgressColumn, TimeRemainingColumn, SpinnerColumn
        from rich.prompt import Prompt, Confirm
        from rich.markdown import Markdown
        from rich.style import Style
        from rich.text import Text
        from rich.layout import Layout
        from rich.live import Live
        from rich import box
        from rich.align import Align
        from tqdm import tqdm
        
        RICH_AVAILABLE = True
    except:
        print("Không thể cài đặt Rich. Sử dụng giao diện cơ bản.")
        RICH_AVAILABLE = False

# Khởi tạo console
if RICH_AVAILABLE:
    console = Console()
else:
    class SimpleConsole:
        def print(self, *args, **kwargs):
            print(*args)
        def rule(self, title):
            print("\n" + "=" * 80)
            print(title.center(80))
            print("=" * 80)
    console = SimpleConsole()

# Đảm bảo thư mục hiện tại được thêm vào path
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

# Tạo ID phiên duy nhất
SESSION_ID = datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + str(uuid.uuid4())[:8]

# Import các module cần thiết
try:
    from core.map import Map
    from core.constants import CellType, MovementCosts, StationCosts
    from core.rl_environment import TruckRoutingEnv
    from core.rl_environment import (
        DEFAULT_MAX_FUEL_RANGE, DEFAULT_INITIAL_FUEL_RANGE, 
        DEFAULT_INITIAL_MONEY_RANGE, DEFAULT_FUEL_PER_MOVE_RANGE,
        DEFAULT_GAS_STATION_COST_RANGE, DEFAULT_TOLL_BASE_COST_RANGE
    )
    from core.algorithms.rl_DQNAgent import DQNAgentTrainer
    from core.algorithms.greedy import GreedySearch
    from truck_routing_app.statistics.rl_evaluation import RLEvaluator
    from stable_baselines3.common.callbacks import BaseCallback # <--- IMPORT THÊM
except ImportError as e:
    if RICH_AVAILABLE:
        console.print(f"[bold red]Lỗi khi import module:[/] {e}")
    else:
        print(f"Lỗi khi import module: {e}")
    sys.exit(1)

# Các siêu tham số tối ưu cho từng kích thước bản đồ
# Được xác định trước qua thử nghiệm
OPTIMAL_HYPERPARAMS = {
    8: {
        "learning_rate": 0.0003,
        "gamma": 0.99,
        "buffer_size": 50000,
        "learning_starts": 5000,
        "batch_size": 64,
        "tau": 0.005,
        "train_freq": 4,
        "target_update_interval": 5000,
        "exploration_fraction": 0.8,  # Tăng từ 0.6 lên 0.8 để agent khám phá nhiều hơn
        "exploration_initial_eps": 1.0,
        "exploration_final_eps": 0.02,  # Giảm từ 0.05 xuống 0.02 để có khai thác tốt hơn cuối cùng
        "policy_kwargs": {"net_arch": [256, 256]},  # Tăng từ 128,128 lên 256,256
        "double_q": True,
        "dueling_net": True,
        "prioritized_replay": True,
        "prioritized_replay_alpha": 0.6,
        "prioritized_replay_beta0": 0.4
    },
    9: {
        "learning_rate": 0.00005,
        "gamma": 0.99,
        "buffer_size": 100000,
        "learning_starts": 2000,
        "batch_size": 64,
        "tau": 0.005,
        "train_freq": 4,
        "target_update_interval": 1000,
        "exploration_fraction": 0.4,
        "exploration_initial_eps": 1.0,
        "exploration_final_eps": 0.05,
        "policy_kwargs": {"net_arch": [128, 128]},
        "double_q": True,
        "dueling_net": True,
        "prioritized_replay": True,
        "prioritized_replay_alpha": 0.6,
        "prioritized_replay_beta0": 0.4
    },
    10: {
        "learning_rate": 0.00005,
        "gamma": 0.995,
        "buffer_size": 100000,
        "learning_starts": 2000,
        "batch_size": 128,
        "tau": 0.005,
        "train_freq": 4,
        "target_update_interval": 1000,
        "exploration_fraction": 0.4,
        "exploration_initial_eps": 1.0,
        "exploration_final_eps": 0.05,
        "policy_kwargs": {"net_arch": [256, 256]},
        "double_q": True,
        "dueling_net": True,
        "prioritized_replay": True,
        "prioritized_replay_alpha": 0.6,
        "prioritized_replay_beta0": 0.4
    },
    12: {
        "learning_rate": 0.00003,
        "gamma": 0.995,
        "buffer_size": 150000,
        "learning_starts": 5000,
        "batch_size": 128,
        "tau": 0.005,
        "train_freq": 4,
        "target_update_interval": 1000,
        "exploration_fraction": 0.4,
        "exploration_initial_eps": 1.0,
        "exploration_final_eps": 0.05,
        "policy_kwargs": {"net_arch": [256, 256]},
        "double_q": True,
        "dueling_net": True,
        "prioritized_replay": True,
        "prioritized_replay_alpha": 0.6,
        "prioritized_replay_beta0": 0.4
    },
    15: {
        "learning_rate": 0.00002,
        "gamma": 0.995,
        "buffer_size": 200000,
        "learning_starts": 10000,
        "batch_size": 256,
        "tau": 0.001,
        "train_freq": 4,
        "target_update_interval": 1000,
        "exploration_fraction": 0.4,
        "exploration_initial_eps": 1.0,
        "exploration_final_eps": 0.05,
        "policy_kwargs": {"net_arch": [512, 256]},
        "double_q": True,
        "dueling_net": True,
        "prioritized_replay": True,
        "prioritized_replay_alpha": 0.6,
        "prioritized_replay_beta0": 0.4
    }
}

# Tham số môi trường tối ưu cho từng kích thước bản đồ
OPTIMAL_ENV_PARAMS = {
    8: {
        "initial_fuel": 50.0,
        "initial_money": 1000.0,
        "fuel_per_move": 0.5,
        "gas_station_cost": 10,
        "toll_base_cost": 5,
        "max_steps": 200
    },
    9: {
        "initial_fuel": 50.0,
        "initial_money": 1500.0,
        "fuel_per_move": 0.5,
        "gas_station_cost": 10,
        "toll_base_cost": 5,
        "max_steps": 250
    },
    10: {
        "initial_fuel": 50.0,
        "initial_money": 1500.0,
        "fuel_per_move": 0.5,
        "gas_station_cost": 10,
        "toll_base_cost": 5,
        "max_steps": 300
    },
    12: {
        "initial_fuel": 50.0,
        "initial_money": 1500.0,
        "fuel_per_move": 0.5,
        "gas_station_cost": 10,
        "toll_base_cost": 5,
        "max_steps": 350
    },
    15: {
        "initial_fuel": 50.0,
        "initial_money": 1500.0,
        "fuel_per_move": 0.5,
        "gas_station_cost": 10,
        "toll_base_cost": 5,
        "max_steps": 500
    }
}

# Tỷ lệ tối ưu cho các loại ô đặc biệt theo kích thước bản đồ
OPTIMAL_MAP_RATIOS = {
    8: {"toll_ratio": 0.03, "gas_ratio": 0.07, "brick_ratio": 0.12},
    9: {"toll_ratio": 0.03, "gas_ratio": 0.07, "brick_ratio": 0.12},
    10: {"toll_ratio": 0.03, "gas_ratio": 0.07, "brick_ratio": 0.12},
    12: {"toll_ratio": 0.025, "gas_ratio": 0.07, "brick_ratio": 0.12},
    15: {"toll_ratio": 0.02, "gas_ratio": 0.07, "brick_ratio": 0.12}
}

# Màu sắc và biểu tượng
COLORS = {
    "info": "cyan",
    "success": "green",
    "warning": "yellow",
    "error": "red",
    "highlight": "magenta",
    "title": "bright_blue",
    "step": "bright_green",
    "training": "blue",
    "header": "#00ff00",  # Bright green for main header (hacker style)
    "subheader": "#1fa8b7",  # Cyan-ish for subheaders
    "border": "#00ffff",  # Cyan border
    "menu_number": "#ff00ff",  # Magenta for menu numbers
    "menu_text": "#ffffff",  # White for menu text
    "accent": "#ff0000",  # Red for accent
    "progress": "#00ff00",  # Green for progress bars
    "prompt": "#ffff00",  # Yellow for prompts
    "tech": "#0000ff"     # Blue for technical elements
}

ICONS = {
    "info": "ℹ️",
    "success": "✅",
    "warning": "⚠️",
    "error": "❌",
    "map": "🗺️",
    "training": "🚀",
    "model": "💾",
    "evaluation": "📊",
    "time": "⏱️",
    "config": "⚙️",
    "analytics": "📈",
    "folder": "📁",
    "map_size": "📏",
    "steps": "👣"
}

# Regex để tìm SESSION_ID trong tên file, ví dụ: _20231027_153000_abcdef12_
# Nó tìm một chuỗi bắt đầu bằng gạch dưới, theo sau là 8 chữ số (YYYYMMDD),
# gạch dưới, 6 chữ số (HHMMSS), gạch dưới, và 8 ký tự hex.
SESSION_ID_PATTERN_IN_FILENAME_RE = re.compile(r"(\d{8})_(\d{6})_([a-f0-9]{8})")

# Cấu trúc thư mục
DIRECTORIES = {
    "maps": { # Note: MAPS_DIR will be _ROOT_DIR / "maps" due to this being a dict
        "train": "maps/train",
        "eval": "maps/eval",
        "test": "maps/test",
    },
    "models": "saved_models",
    "logs": "training_logs",
    "results": "evaluation_results",
    "sessions": "sessions" # Ensure this key exists
}

# Define global Path constants immediately after DIRECTORIES
_ROOT_DIR = Path(__file__).resolve().parent

# DIRECTORIES["maps"] is a dict, so MAPS_DIR needs to point to the general 'maps' folder directly.
MAPS_DIR = _ROOT_DIR / "maps" 
MODELS_DIR = _ROOT_DIR / DIRECTORIES["models"]
LOGS_DIR = _ROOT_DIR / DIRECTORIES["logs"]
RESULTS_DIR = _ROOT_DIR / DIRECTORIES["results"]
SESSIONS_DIR = _ROOT_DIR / DIRECTORIES["sessions"]

# Create these base directories if they don't exist to be safe, though specific session/type dirs are created later.
MAPS_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)
LOGS_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
SESSIONS_DIR.mkdir(parents=True, exist_ok=True)

# Mẫu tên file
FILE_TEMPLATES = {
    "map": "map_{size}x{size}_{index}_{timestamp}.json",
    "model": "rl_agent_size_{size}{variant}_{steps}_{session_id}.zip",
    "results": "results_{size}x{size}_success{success_rate:.0f}_{session_id}.json"
}

# Logo và banner
RL_TOOL_LOGO = """
████████╗██████╗ ██╗   ██╗ ██████╗██╗  ██╗    ██████╗ ██╗     
╚══██╔══╝██╔══██╗██║   ██║██╔════╝██║ ██╔╝    ██╔══██╗██║     
   ██║   ██████╔╝██║   ██║██║     █████╔╝     ██████╔╝██║     
   ██║   ██╔══██╗██║   ██║██║     ██╔═██╗     ██╔══██╗██║     
   ██║   ██║  ██║╚██████╔╝╚██████╗██║  ██╗    ██║  ██║███████╗
   ╚═╝   ╚═╝  ╚═╝ ╚═════╝  ╚═════╝╚═╝  ╚═╝    ╚═╝  ╚═╝╚══════╝
                                                               
  █████╗ ██╗    ███╗   ███╗ █████╗ ███████╗████████╗███████╗██████╗ 
 ██╔══██╗██║    ████╗ ████║██╔══██╗██╔════╝╚══██╔══╝██╔════╝██╔══██╗
 ███████║██║    ██╔████╔██║███████║███████╗   ██║   █████╗  ██████╔╝
 ██╔══██║██║    ██║╚██╔╝██║██╔══██║╚════██║   ██║   ██╔══╝  ██╔══██╗
 ██║  ██║██║    ██║ ╚═╝ ██║██║  ██║███████║   ██║   ███████╗██║  ██║
 ╚═╝  ╚═╝╚═╝    ╚═╝     ╚═╝╚═╝  ╚═╝╚══════╝   ╚═╝   ╚══════╝╚═╝  ╚═╝
                                                           v2.0
"""

MENU_SEPARATOR = "═" * 70  # Using solid unicode separator for a more technical look

def display_header():
    """Hiển thị header của ứng dụng"""
    if RICH_AVAILABLE:
        # Header với logo
        console.print()
        header_panel = Panel(
            Align.center(Text(RL_TOOL_LOGO, style=f"bold {COLORS['header']}")),
            border_style=COLORS["border"],
            box=box.DOUBLE,
            width=90,
            padding=(0, 2)
        )
        console.print(header_panel)
        
        # Subtitle
        subtitle = Text("[ ADVANCED REINFORCEMENT LEARNING TRAINING FRAMEWORK ]", style=f"bold {COLORS['subheader']}")
        console.print(Align.center(subtitle))
        
        # Separator
        console.print(Align.center(Text(MENU_SEPARATOR, style=COLORS["border"])))
    else:
        print("\n" + "=" * 80)
        print("TRUCK RL AGENT - Công cụ huấn luyện tự động".center(80))
        print("=" * 80)

def display_menu():
    """Hiển thị menu chức năng"""
    if RICH_AVAILABLE:
        menu_table = Table(show_header=False, box=None, padding=(0, 2))
        menu_table.add_column("Option", style=COLORS["menu_number"], justify="right", width=6)
        menu_table.add_column("Description", style=COLORS["menu_text"])
        
        menu_table.add_row("[1]", "QUICK TRAIN (Huấn luyện nhanh - 8x8, cấu hình mặc định)")
        menu_table.add_row("[2]", "MASTER MODE (Chỉ cần nhập kích thước bản đồ, tự động tối ưu hóa)")
        menu_table.add_row("[3]", "ADVANCED TRAIN (Double DQN, Dueling, PER)")
        menu_table.add_row("[4]", "EVALUATE MODEL (Đánh giá model đã huấn luyện)")
        menu_table.add_row("[5]", "GENERATE MAPS (Tạo bộ bản đồ mới)")
        menu_table.add_row("[6]", "HELP MANUAL (Thông tin trợ giúp)")
        menu_table.add_row("[0]", "EXIT (Thoát)")
        
        menu_panel = Panel(
            menu_table,
            title="[ MAIN MENU ]",
            title_align="center",
            border_style=COLORS["border"],
            box=box.HEAVY,
            padding=(1, 2)
        )
        console.print(menu_panel)
        
        # Technical info display
        tech_info = Table(show_header=False, box=None, padding=(0, 1))
        tech_info.add_column("Label", style=COLORS["accent"], width=15)
        tech_info.add_column("Value", style="white")
        
        tech_info.add_row("Session ID:", SESSION_ID)
        tech_info.add_row("System:", f"Python {sys.version.split()[0]}")
        tech_info.add_row("Time:", datetime.now().strftime("%H:%M:%S %d/%m/%Y"))
        
        tech_panel = Panel(
            tech_info,
            title="[ SYSTEM INFO ]",
            title_align="center",
            border_style=COLORS["border"],
            box=box.SIMPLE,
            width=60
        )
        console.print(Align.center(tech_panel))
        
        # Footer
        console.print(Align.center(Text(MENU_SEPARATOR, style=COLORS["border"])))
        footer = Text("Coded by NHOM AI © 2023 | https://github.com/AI-NHOM", style=COLORS["accent"])
        console.print(Align.center(footer))
        console.print()
    else:
        print("\nMenu chức năng:")
        print("  [1] QUICK TRAIN (Huấn luyện nhanh - 8x8, cấu hình mặc định)")
        print("  [2] MASTER MODE (Chỉ cần nhập kích thước bản đồ, tự động tối ưu hóa)")
        print("  [3] ADVANCED TRAIN (Double DQN, Dueling, PER)")
        print("  [4] EVALUATE MODEL (Đánh giá model đã huấn luyện)")
        print("  [5] GENERATE MAPS (Tạo bộ bản đồ mới)")
        print("  [6] HELP MANUAL (Thông tin trợ giúp)")
        print("  [0] EXIT (Thoát)")
        print(f"\nSession ID: {SESSION_ID}")
        print(f"Thời gian: {datetime.now().strftime('%H:%M:%S %d/%m/%Y')}")

def get_user_choice():
    """Lấy lựa chọn từ người dùng"""
    if RICH_AVAILABLE:
        choice = Prompt.ask(
            "\n[bold cyan]Nhập lựa chọn của bạn[/bold cyan]",
            choices=["0", "1", "2", "3", "4", "5", "6"],
            default="1"
        )
    else:
        choice = input("\nNhập lựa chọn của bạn [1-6, 0 để thoát]: ")
    return choice

def display_training_config(map_size, num_maps, training_steps, use_advanced):
    """Hiển thị cấu hình huấn luyện hiện tại"""
    if RICH_AVAILABLE:
        config_table = Table(show_header=True, box=box.SIMPLE)
        config_table.add_column("Tham số", style="cyan")
        config_table.add_column("Giá trị", style="green")
        
        config_table.add_row("Kích thước bản đồ", f"{map_size}x{map_size}")
        config_table.add_row("Số lượng bản đồ", str(num_maps))
        config_table.add_row("Số bước huấn luyện", f"{training_steps:,}")
        config_table.add_row("Sử dụng kỹ thuật nâng cao", "✅ Có" if use_advanced else "❌ Không")
        
        config_panel = Panel(
            config_table,
            title="[bold]Cấu hình huấn luyện[/bold]",
            title_align="center",
            border_style="blue",
            padding=(1, 2)
        )
        console.print(config_panel)
    else:
        print("\nCấu hình huấn luyện:")
        print(f"  Kích thước bản đồ: {map_size}x{map_size}")
        print(f"  Số lượng bản đồ: {num_maps}")
        print(f"  Số bước huấn luyện: {training_steps:,}")
        print(f"  Sử dụng kỹ thuật nâng cao: {'Có' if use_advanced else 'Không'}")

def clear_screen():
    """Xóa màn hình terminal"""
    if sys.platform == "win32":
        os.system("cls")
    else:
        os.system("clear")

def setup_directories():
    """Tạo các thư mục cần thiết nếu chưa tồn tại."""
    directories = [
        "maps/train",
        "maps/eval",
        "maps/test",
        "saved_models",
        "training_logs",
        "evaluation_results"
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
    
    if RICH_AVAILABLE:
        console.print("[bold green]✅ Đã tạo các thư mục cần thiết[/bold green]")
    else:
        print("✅ Đã tạo các thư mục cần thiết")
    return True

def generate_maps(base_log_dir, map_size, num_maps_per_set=10, map_ratios_override=None):
    """
    Generates a set of maps for a given size and saves them.
    Ensures each map has a path from start to end.
    Uses ratios from OPTIMAL_MAP_RATIOS unless overridden.
    """
    map_dir = base_log_dir / f"map_size_{map_size}" / "generated_maps"
    map_dir.mkdir(parents=True, exist_ok=True)
    print(f"Generating maps in: {map_dir}")

    generated_map_files = []

    if map_ratios_override:
        current_ratios = map_ratios_override
    else:
        # Lấy ratio từ OPTIMAL_MAP_RATIOS, nếu không có cho size cụ thể, dùng của size 10
        current_ratios = OPTIMAL_MAP_RATIOS.get(map_size, OPTIMAL_MAP_RATIOS.get(10))
        if not current_ratios:
            print(f"[Error] No OPTIMAL_MAP_RATIOS defined for map size {map_size} or fallback size 10.")
            return [] # Trả về list rỗng nếu không có ratio
            
    toll_ratio = current_ratios.get("toll_ratio", 0.05)
    gas_ratio = current_ratios.get("gas_ratio", 0.05)
    brick_ratio = current_ratios.get("brick_ratio", 0.2)

    print(f"Using ratios for map size {map_size}: Toll={toll_ratio:.3f}, Gas={gas_ratio:.3f}, Brick={brick_ratio:.3f}")

    maps_generated = 0
    attempts = 0
    max_total_attempts = num_maps_per_set * 20 # Giới hạn tổng số lần thử để tránh kẹt vô hạn

    while maps_generated < num_maps_per_set and attempts < max_total_attempts:
        attempts += 1

        # Tính toán số lượng từ ratio (logic này cần phải có vì OPTIMAL_MAP_RATIOS dùng ratio)
        total_cells = map_size * map_size
        effective_area = max(1, total_cells - 2) # Trừ start/end để tính toán số lượng
        
        # Thêm random.uniform để có sự biến thiên nhẹ về số lượng, tương tự như cách các ratio này có thể được hiểu
        # là một khoảng giá trị mong muốn.
        num_tolls_calc = max(0, int(toll_ratio * effective_area * random.uniform(0.8, 1.2)))
        num_gas_calc = max(0, int(gas_ratio * effective_area * random.uniform(0.8, 1.2)))
        num_obstacles_calc = max(0, int(brick_ratio * effective_area * random.uniform(0.8, 1.2)))

        # Tạo bản đồ bằng cách gọi phương thức đã được cập nhật của Map, truyền số lượng
        map_obj = Map.generate_random(
            size=map_size,
            num_tolls=num_tolls_calc,
            num_gas=num_gas_calc,
            num_obstacles=num_obstacles_calc
        )

        if map_obj:
            # Đã có kiểm tra has_path_from_start_to_end bên trong generate_random
            map_filename = f"map_{map_size}x{map_size}_{maps_generated + 1}.json"
            full_map_path = map_dir / map_filename
            if map_obj.save(str(full_map_path)):
                generated_map_files.append(str(full_map_path))
                maps_generated += 1
                print(f"  Generated and saved: {map_filename} (Attempt {attempts})")
                stats = map_obj.get_statistics()
                print(f"    Actual counts: Tolls={stats['toll_stations']}, Gas={stats['gas_stations']}, Obstacles={stats['obstacles']}")
            else:
                print(f"  Failed to save map generated on attempt {attempts}.")
        else:
            # generate_random đã in warning nếu không tạo được map hợp lệ
            print(f"  Map generation failed on attempt {attempts} for size {map_size}. Retrying...")
            time.sleep(0.1) # Chờ một chút trước khi thử lại

    if maps_generated < num_maps_per_set:
        print(f"[Warning] Could only generate {maps_generated}/{num_maps_per_set} maps for size {map_size} after {max_total_attempts} attempts.")
    
    return generated_map_files

def create_environment(map_obj, map_size, render_mode=None):
    """Tạo môi trường TruckRoutingEnv với cấu hình tối ưu cho map_size."""
    # Lấy tham số tối ưu cố định cho map_size này
    env_params_optimal = OPTIMAL_ENV_PARAMS.get(map_size, OPTIMAL_ENV_PARAMS[8]) 
    max_steps_for_env = env_params_optimal.get('max_steps', 2 * map_obj.size * map_obj.size)
    
    # Lấy các giá trị max_fuel từ optimal hoặc default (cần thiết cho TruckRoutingEnv init)
    # Sử dụng initial_fuel làm max_fuel nếu không có, hoặc giá trị trung bình từ default range
    max_fuel_val = env_params_optimal.get('max_fuel', 
                                          env_params_optimal.get('initial_fuel', 
                                                                 (DEFAULT_MAX_FUEL_RANGE[0] + DEFAULT_MAX_FUEL_RANGE[1]) / 2))

    console.print(f"[info]Tạo môi trường với Tham số Tối ưu cho map_size {map_size}:[/info]")
    console.print(f"  Max Fuel: {max_fuel_val}") # Sử dụng max_fuel_val đã xác định
    console.print(f"  Initial Fuel: {env_params_optimal['initial_fuel']}")
    console.print(f"  Initial Money: {env_params_optimal['initial_money']}")
    console.print(f"  Fuel Per Move: {env_params_optimal['fuel_per_move']}")
    console.print(f"  Gas Station Cost: {env_params_optimal['gas_station_cost']}")
    console.print(f"  Toll Base Cost: {env_params_optimal['toll_base_cost']}")
    console.print(f"  Max Steps Per Episode: {max_steps_for_env}")

    env = TruckRoutingEnv(
        map_object=map_obj,
        # Sử dụng giá trị tối ưu thay vì easy_params
        max_fuel_config=max_fuel_val, # Truyền max_fuel đã xác định
        initial_fuel_config=env_params_optimal['initial_fuel'],
        initial_money_config=env_params_optimal['initial_money'],
        fuel_per_move_config=env_params_optimal['fuel_per_move'],
        gas_station_cost_config=env_params_optimal['gas_station_cost'],
        toll_base_cost_config=env_params_optimal['toll_base_cost'],
        max_steps_per_episode=max_steps_for_env 
    )
    return env

def create_agent(env, map_size, use_advanced, log_dir):
    """
    Tạo agent RL với tham số tối ưu
    
    Args:
        env: Môi trường RL
        map_size: Kích thước bản đồ
        use_advanced: Sử dụng các kỹ thuật nâng cao
        log_dir: Đường dẫn đến thư mục log đã được tạo bởi hàm gọi
    
    Returns:
        agent: Đối tượng DQNAgentTrainer
    """
    # Thư mục log được cung cấp bởi hàm gọi và đã được tạo
    # Path(log_dir).mkdir(parents=True, exist_ok=True) # Dòng này không cần thiết nữa
    
    # Tạo agent
    agent = DQNAgentTrainer(env=env, log_dir=log_dir)
    
    # Lấy siêu tham số tối ưu
    hyperparams = OPTIMAL_HYPERPARAMS.get(map_size, OPTIMAL_HYPERPARAMS[10])
    
    # Cấu hình cho agent
    if use_advanced:
        # Sử dụng các kỹ thuật nâng cao: Double DQN, Dueling Network, PER
        agent.create_model(
            learning_rate=hyperparams["learning_rate"],
            buffer_size=hyperparams["buffer_size"], 
            learning_starts=hyperparams["learning_starts"],
            batch_size=hyperparams["batch_size"],
            tau=hyperparams["tau"],
            gamma=hyperparams["gamma"],
            train_freq=hyperparams["train_freq"],
            gradient_steps=1,
            target_update_interval=hyperparams["target_update_interval"],
            exploration_fraction=hyperparams["exploration_fraction"],
            exploration_initial_eps=hyperparams["exploration_initial_eps"],
            exploration_final_eps=hyperparams["exploration_final_eps"],
            use_double_dqn=True,
            use_dueling_network=True,
            use_prioritized_replay=True,
            prioritized_replay_alpha=0.6,
            prioritized_replay_beta0=0.4
        )
    else:
        # Sử dụng DQN cơ bản
        agent.create_model(
            learning_rate=hyperparams["learning_rate"],
            buffer_size=hyperparams["buffer_size"], 
            learning_starts=hyperparams["learning_starts"],
            batch_size=hyperparams["batch_size"],
            tau=hyperparams["tau"],
            gamma=hyperparams["gamma"],
            train_freq=hyperparams["train_freq"],
            target_update_interval=hyperparams["target_update_interval"],
            exploration_fraction=hyperparams["exploration_fraction"],
            exploration_initial_eps=hyperparams["exploration_initial_eps"],
            exploration_final_eps=hyperparams["exploration_final_eps"]
        )
    
    return agent

def train_agent(agent: DQNAgentTrainer, total_timesteps, map_size, callback=None):
    """Huấn luyện agent đã được tạo."""
    console.print(f"[info]Bắt đầu huấn luyện agent với {total_timesteps} timesteps...[/info]")
    agent.train(total_timesteps=total_timesteps, callback=callback)
    console.print("[green]✓[/green] Huấn luyện agent hoàn thành.")

def evaluate_agent(agent: DQNAgentTrainer, map_size, num_episodes=10, eval_map_obj=None):
    """Đánh giá nhanh agent bằng cách sử dụng evaluate_robust_performance với kịch bản tối ưu.
    Hàm này được giữ lại cho mục đích đánh giá nhanh hoặc gỡ lỗi.
    """
    console.print(f"[info]Đánh giá nhanh agent (evaluate_agent) trên map_size={map_size} với {num_episodes} episodes...[/info]")
    
    if eval_map_obj is None:
        console.print("[warning]Không có bản đồ đánh giá cụ thể cho evaluate_agent, tạo bản đồ ngẫu nhiên...[/warning]")
        eval_map_obj = Map.generate_random(map_size, 0.05, 0.05, 0.1)
        # Đảm bảo start/end được thiết lập đúng trên map_obj này
        if not eval_map_obj.ensure_start_end_connected(): # ensure_start_end_connected có thể trả về False
            console.print("[error]Không thể tạo vị trí bắt đầu/kết thúc hợp lệ trên bản đồ ngẫu nhiên cho đánh giá (evaluate_agent).[/error]")
            return { 
                "overall_score": 0,
                "avg_success_rate": 0,
                "avg_reward_overall": 0,
                "avg_path_length_overall": 0,
                "detailed_results_by_scenario": {}
            } 

    # Tạo môi trường đánh giá.
    eval_env = create_environment(eval_map_obj, map_size) 
    
    # Chỉ sử dụng kịch bản tối ưu cho việc đánh giá nhanh này
    # Lưu ý: get_optimal_env_scenario cần map_size, không phải eval_env.map_size trực tiếp ở đây
    # vì eval_env có thể chưa có map_size nếu map_obj không hợp lệ.
    quick_eval_scenario = get_optimal_env_scenario(eval_map_obj.size) # Sử dụng eval_map_obj.size
    
    console.print(f"[info]Sử dụng kịch bản đánh giá nhanh: {quick_eval_scenario.get('name')}[/info]")

    robust_metrics = evaluate_robust_performance(
        agent_model=agent,
        eval_env=eval_env, 
        num_episodes_per_scenario=num_episodes, 
        scenarios=[quick_eval_scenario] 
    )

    # Không cần in lại các metrics ở đây vì evaluate_robust_performance đã in rất chi tiết.
    # Chỉ cần trả về kết quả.
    return robust_metrics

def evaluate_robust_performance(agent_model: DQNAgentTrainer, eval_env: TruckRoutingEnv, 
                                num_episodes_per_scenario=5, scenarios=None,
                                outer_rich_progress_active: bool = False):
    """
    Đánh giá hiệu suất của agent trên nhiều kịch bản môi trường.
    Sử dụng lại eval_env và gọi reset với evaluation_params.
    Args:
        agent_model: Model agent đã huấn luyện.
        eval_env: Môi trường để đánh giá.
        num_episodes_per_scenario: Số episodes để chạy cho mỗi kịch bản.
        scenarios: Danh sách các kịch bản để đánh giá. Nếu None, dùng default.
        outer_rich_progress_active: True nếu có một Rich Progress bar bên ngoài đang chạy.
    """
    if scenarios is None:
        current_scenarios = list(DEFAULT_EVALUATION_SCENARIOS) 
        current_scenarios.append(get_optimal_env_scenario(eval_env.map_size))
    else:
        current_scenarios = scenarios

    total_episodes = 0
    total_successes = 0
    all_episode_rewards = []
    all_episode_path_lengths = []
    all_episode_money_spent = []
    all_episode_fuel_consumed = []
    
    detailed_results_by_scenario = {}

    console.print(f"[info]Bắt đầu đánh giá hiệu suất trên {len(current_scenarios)} kịch bản...[/info]")

    try: # Bọc vòng lặp kịch bản
        for scenario_idx, scenario in enumerate(current_scenarios):
            scenario_name = scenario.get("name", f"Scenario {scenario_idx + 1}")
            console.print(f"  [info]Đang đánh giá kịch bản ({scenario_idx+1}/{len(current_scenarios)}): {scenario_name}[/info]")
            
            progress_bar_description = f"Episodes cho '{shorten_text(scenario_name, 30)}'"
            episode_run_results = None

            if not outer_rich_progress_active and RICH_AVAILABLE:
                with Progress(
                    SpinnerColumn(),
                    TextColumn("[progress.description]{task.description}"),
                    BarColumn(),
                    TaskProgressColumn(),
                    TimeRemainingColumn(),
                    console=console,
                    transient=True 
                ) as progress:
                    task_id = progress.add_task(progress_bar_description, total=num_episodes_per_scenario)
                    episode_run_results = _run_scenario_episodes(
                        eval_env, agent_model, scenario, num_episodes_per_scenario,
                        scenario_name, # scenario_name_logging
                        rich_progress=progress, rich_task_id=task_id
                    )
            else:
                episode_run_results = _run_scenario_episodes(
                    eval_env, agent_model, scenario, num_episodes_per_scenario,
                    scenario_name # scenario_name_logging
                )
            
            scenario_successes_count = episode_run_results["success_count"]
            scenario_rewards_list = episode_run_results["rewards_list"]
            scenario_path_lengths_list = episode_run_results["path_lengths_list"]
            scenario_money_spent_list = episode_run_results["money_spent_list"]
            scenario_fuel_consumed_list = episode_run_results["fuel_consumed_list"]
            
            total_successes += scenario_successes_count
            all_episode_rewards.extend(scenario_rewards_list)
            all_episode_path_lengths.extend(scenario_path_lengths_list)
            all_episode_money_spent.extend(scenario_money_spent_list)
            all_episode_fuel_consumed.extend(scenario_fuel_consumed_list)

            detailed_results_by_scenario[scenario_name] = {
                "success_rate": scenario_successes_count / num_episodes_per_scenario if num_episodes_per_scenario > 0 else 0,
                "avg_reward": np.mean(scenario_rewards_list) if scenario_rewards_list else 0,
                "avg_path_length": np.mean(scenario_path_lengths_list) if scenario_path_lengths_list else 0,
                "avg_money_spent": np.mean(scenario_money_spent_list) if scenario_money_spent_list else 0,
                "avg_fuel_consumed": np.mean(scenario_fuel_consumed_list) if scenario_fuel_consumed_list else 0,
                "rewards_list": scenario_rewards_list,
                "path_lengths_list": scenario_path_lengths_list
            }
            console.print(f"  [green]✓[/green] Kịch bản '{scenario_name}': SR={detailed_results_by_scenario[scenario_name]['success_rate']:.2f}, AvgRew={detailed_results_by_scenario[scenario_name]['avg_reward']:.2f}")

    except Exception as e_eval_robust:
        console.print(f"[bold red]LỖI TRONG evaluate_robust_performance:[/bold red]")
        console.print(f"[red]{str(e_eval_robust)}[/red]")
        if RICH_AVAILABLE:
            console.print_exception(show_locals=False)
        else:
            traceback.print_exc()
        # Trả về kết quả lỗi để pipeline chính có thể xử lý
        return {
            "overall_score": 0,
            "avg_success_rate": 0,
            "avg_reward_overall": 0,
            "avg_path_length_overall": 0,
            "detailed_results_by_scenario": {},
            "error": str(e_eval_robust)
        }

    avg_success_rate = total_successes / total_episodes if total_episodes > 0 else 0
    avg_reward_overall = np.mean(all_episode_rewards) if all_episode_rewards else 0
    avg_path_length_overall = np.mean(all_episode_path_lengths) if all_episode_path_lengths else 0
    avg_money_spent_overall = np.mean(all_episode_money_spent) if all_episode_money_spent else 0
    avg_fuel_consumed_overall = np.mean(all_episode_fuel_consumed) if all_episode_fuel_consumed else 0
    
    score_reward_component = np.tanh(avg_reward_overall / 100) 
    score_path_component = 0
    if avg_path_length_overall > 0 and eval_env.max_steps_per_episode > 0:
        # Path length should be non-zero if successful, use max_steps as a rough normalizer
        normalized_path = avg_path_length_overall / eval_env.max_steps_per_episode
        score_path_component = 1 - normalized_path 
        score_path_component = max(-1, min(1, score_path_component)) # clamp between -1 and 1

    overall_score = (avg_success_rate * 0.6) + (score_reward_component * 0.2) + (score_path_component * 0.2)
    overall_score = max(0, min(1, overall_score)) 

    console.print(f"[highlight]Đánh giá hiệu suất hoàn thành:[/highlight]")
    console.print(f"  Tỷ lệ thành công TB (qua các kịch bản): {avg_success_rate:.2%}")
    console.print(f"  Phần thưởng TB tổng thể: {avg_reward_overall:.2f}")
    console.print(f"  Độ dài đường đi TB tổng thể: {avg_path_length_overall:.2f}")
    console.print(f"  Chi tiêu tiền TB tổng thể: {avg_money_spent_overall:.2f}")
    console.print(f"  Tiêu thụ nhiên liệu TB tổng thể: {avg_fuel_consumed_overall:.2f}")
    console.print(f"  [bold]Điểm hiệu năng tổng hợp (0-1): {overall_score:.4f}[/bold]")

    return {
        "overall_score": overall_score,
        "avg_success_rate": avg_success_rate,
        "avg_reward_overall": avg_reward_overall,
        "avg_path_length_overall": avg_path_length_overall,
        "avg_money_spent_overall": avg_money_spent_overall,
        "avg_fuel_consumed_overall": avg_fuel_consumed_overall,
        "detailed_results_by_scenario": detailed_results_by_scenario
    }

# Helper function to run evaluation episodes for a single scenario
def _run_scenario_episodes(eval_env: TruckRoutingEnv, agent_model: DQNAgentTrainer, scenario_params: dict, 
                           num_episodes: int, scenario_name_logging: str, 
                           rich_progress=None, rich_task_id=None):
    """
    Runs evaluation episodes for a single scenario and collects metrics.
    Logs progress to console if rich_progress is None and an outer Rich display is active.
    """
    scenario_successes_count = 0
    scenario_rewards_list = []
    scenario_path_lengths_list = []
    scenario_money_spent_list = []
    scenario_fuel_consumed_list = []

    try: # Bọc vòng lặp episodes
        for episode_idx in range(num_episodes):
            if rich_progress is None: 
                if RICH_AVAILABLE and hasattr(console, 'is_live') and console.is_live:
                    if num_episodes <=5 or (episode_idx + 1) % max(1, num_episodes // 3) == 0 or episode_idx == num_episodes - 1:
                        console.print(f"    Kịch bản '{shorten_text(scenario_name_logging, 25)}': Đang chạy episode {episode_idx + 1}/{num_episodes}...")
                elif not RICH_AVAILABLE: 
                    if (episode_idx + 1) % max(1, num_episodes // 5) == 0 or episode_idx == num_episodes - 1:
                        print(f"    Kịch bản '{scenario_name_logging}': Episode {episode_idx + 1}/{num_episodes}")
            
            obs, info = eval_env.reset(evaluation_params=scenario_params) 
            terminated = False
            truncated = False
            episode_reward = 0.0
            episode_length = 0
            initial_money_for_episode = eval_env.current_money 
            initial_fuel_for_episode = eval_env.current_fuel   

            while not (terminated or truncated):
                predicted_output = agent_model.predict_action(obs)
                if isinstance(predicted_output, tuple):
                    action = predicted_output[0]
                else:
                    action = predicted_output
                
                if hasattr(action, 'item'): 
                    action = action.item()
                
                obs, reward, term, trunc, info = eval_env.step(action) 
                terminated = term
                truncated = trunc
                episode_reward += reward
                episode_length += 1
            
            if info.get("termination_reason") == "den_dich":
                scenario_successes_count += 1
            
            scenario_rewards_list.append(episode_reward)
            actual_path_length = episode_length if info.get("termination_reason") == "den_dich" else eval_env.max_steps_per_episode
            scenario_path_lengths_list.append(actual_path_length)
            
            money_spent_episode = initial_money_for_episode - eval_env.current_money
            current_fuel_after_episode = eval_env.current_fuel 
            fuel_consumed_episode = initial_fuel_for_episode - current_fuel_after_episode
            scenario_money_spent_list.append(money_spent_episode)
            scenario_fuel_consumed_list.append(fuel_consumed_episode)
            
            if rich_progress and rich_task_id:
                rich_progress.update(rich_task_id, advance=1)

    except Exception as e_scenario_run:
        console.print(f"[bold red]LỖI TRONG _run_scenario_episodes (kịch bản: {scenario_name_logging}):[/bold red]")
        console.print(f"[red]{str(e_scenario_run)}[/red]")
        if RICH_AVAILABLE:
            console.print_exception(show_locals=False)
        else:
            traceback.print_exc()
        # Vẫn trả về những gì đã thu thập được, có thể không đầy đủ
        # Hoặc có thể raise lại lỗi nếu muốn dừng hẳn evaluate_robust_performance

    return {
        "success_count": scenario_successes_count,
        "rewards_list": scenario_rewards_list,
        "path_lengths_list": scenario_path_lengths_list,
        "money_spent_list": scenario_money_spent_list,
        "fuel_consumed_list": scenario_fuel_consumed_list
    }

def shorten_text(text, max_length):
    return text if len(text) <= max_length else text[:max_length-3] + "..."

# Thêm hằng số cho thư mục lưu model "tốt nhất"
BEST_ROBUST_MODELS_DIR = MODELS_DIR / "best_robust_by_type"
BEST_ROBUST_MODELS_DIR.mkdir(parents=True, exist_ok=True)

# Callback tùy chỉnh cho Rich Progress Bar trong quá trình huấn luyện SB3
class RichProgressSB3Callback(BaseCallback):
    """
    Callback tùy chỉnh cho Stable Baselines3 để cập nhật Rich Progress Bar.
    """
    def __init__(self, total_training_steps: int, 
                 pipeline_progress_callback, 
                 pipeline_progress_start_percent: int, 
                 pipeline_progress_span_percent: int, # Tổng % dành cho training
                 verbose: int = 0):
        super().__init__(verbose)
        self.total_training_steps = total_training_steps
        self.pipeline_progress_callback = pipeline_progress_callback
        self.pipeline_progress_start_percent = pipeline_progress_start_percent
        self.pipeline_progress_span_percent = pipeline_progress_span_percent
        self.training_status_message_template = "Huấn luyện: {current_steps}/{total_steps} ({percent_done:.1f}%)"

    def _on_step(self) -> bool:
        if self.pipeline_progress_callback:
            # Tính toán % hoàn thành của chỉ riêng phase huấn luyện
            training_completion_fraction = self.num_timesteps / self.total_training_steps
            
            # Tính toán % tổng thể trên pipeline progress bar
            current_pipeline_percent = self.pipeline_progress_start_percent + \
                                       (training_completion_fraction * self.pipeline_progress_span_percent)
            
            status_message = self.training_status_message_template.format(
                current_steps=self.num_timesteps,
                total_steps=self.total_training_steps,
                percent_done=training_completion_fraction * 100
            )
            
            self.pipeline_progress_callback(current_pipeline_percent, status_message)
        return True

def run_training_pipeline(map_size, num_maps=10, training_steps=50000, use_advanced=False, render=False, progress_callback=None, outer_rich_progress_active: bool = False):
    """Chạy toàn bộ pipeline huấn luyện và đánh giá."""
    # Define progress milestones
    P_START = 0
    P_MAP_GENERATION_TRAIN_END = 10  # Tạo map train xong
    P_MAP_GENERATION_EVAL_END = 15   # Tạo map eval xong
    P_ENV_CREATION_END = 20          # Tạo môi trường xong
    P_AGENT_CREATION_END = 25        # Tạo agent xong
    P_TRAINING_START = 25            # Bắt đầu huấn luyện
    P_TRAINING_SPAN = 50             # % dành cho training
    P_TRAINING_END = P_TRAINING_START + P_TRAINING_SPAN  # Huấn luyện xong
    P_SESSION_MODEL_SAVE_END = P_TRAINING_END + 5  # Lưu model session xong
    P_NEW_MODEL_EVAL_END = P_SESSION_MODEL_SAVE_END + 10  # Đánh giá model mới xong
    P_OLD_MODEL_EVAL_END = P_NEW_MODEL_EVAL_END + 5  # Đánh giá model cũ xong
    P_PIPELINE_END = 100             # Hoàn thành

    try:  # Bọc toàn bộ pipeline
        console.print(f"[title]╔══════════════════════════════════════════════════╗[/title]")
        console.print(f"[title]║    BẮT ĐẦU PIPELINE HUẤN LUYỆN TỰ ĐỘNG RL     ║[/title]")
        console.print(f"[title]╚══════════════════════════════════════════════════╝[/title]")
        console.print(f"[info]Phiên làm việc: {SESSION_ID}[/info]")
        console.print(f"[info]Chạy pipeline huấn luyện cho map_size={map_size}[/info]")
        console.print(f"  Số bản đồ train/eval/test: {num_maps}")
        console.print(f"  Số bước huấn luyện: {training_steps}")
        console.print(f"  Sử dụng kỹ thuật nâng cao: {'Có' if use_advanced else 'Không'}")

        # --- 1. Thiết lập thư mục ---    
        session_log_dir = LOGS_DIR / SESSION_ID / f"map_size_{map_size}"
        session_models_dir = MODELS_DIR / SESSION_ID / f"map_size_{map_size}"
        session_maps_dir = MAPS_DIR / SESSION_ID / f"map_size_{map_size}"
        
        session_log_dir.mkdir(parents=True, exist_ok=True)
        session_models_dir.mkdir(parents=True, exist_ok=True)
        session_maps_dir.mkdir(parents=True, exist_ok=True)

        # --- 2. Tạo bản đồ huấn luyện và đánh giá ---
        console.print("[step]Bước 1 & 2: Tạo bản đồ huấn luyện và đánh giá...[/step]")
        map_progress_callback_wrapper = progress_callback if progress_callback else lambda p, m: None
        
        if progress_callback:
            progress_callback(10, f"Đang tạo {num_maps} bản đồ huấn luyện size {map_size}x{map_size}...")

        map_train_paths = generate_maps(map_size, num_maps=num_maps, map_types=["train"], 
                                      map_save_dir=session_maps_dir, 
                                      progress_callback=map_progress_callback_wrapper)
        if not map_train_paths:
            console.print("[error]Không thể tạo bản đồ huấn luyện. Dừng pipeline.[/error]")
            return None
        console.print(f"[green]✓[/green] Đã tạo {len(map_train_paths)} bản đồ huấn luyện.")

        if progress_callback:
            progress_callback(15, f"Đang tạo {max(1, num_maps // 5)} bản đồ đánh giá size {map_size}x{map_size}...")

        map_eval_paths = generate_maps(map_size, num_maps=max(1, num_maps // 5), map_types=["eval"], 
                                     map_save_dir=session_maps_dir, 
                                     progress_callback=map_progress_callback_wrapper)
        console.print(f"[green]✓[/green] Đã tạo {len(map_eval_paths)} bản đồ đánh giá.")

        train_map_obj = Map.load(map_train_paths[0])  # Sử dụng bản đồ đầu tiên làm cơ sở
        if not train_map_obj:  # Kiểm tra nếu Map.load thất bại
            console.print(f"[error]Không thể tải bản đồ huấn luyện chính: {map_train_paths[0]}. Dừng pipeline.[/error]")
            return {
                "best_model_path": None,
                "best_model_performance": None,
                "session_model_path": None,
                "session_model_performance": None,
                "error": f"Failed to load main training map: {map_train_paths[0]}"
            }

        # Get map name for model identification
        map_specific_name_part = Path(map_train_paths[0]).stem
        map_type_folder_name = f"map_file_{map_specific_name_part}"
        
        # Setup best model directory
        specific_best_model_dir = BEST_ROBUST_MODELS_DIR / map_type_folder_name
        specific_best_model_dir.mkdir(parents=True, exist_ok=True)
        best_model_path_for_type = specific_best_model_dir / "best_robust_rl_model.zip"

        # --- 3. Tạo môi trường RL ---
        console.print("[step]Bước 3: Tạo môi trường RL...[/step]")
        if progress_callback:
            progress_callback(P_ENV_CREATION_END - 2, "Tạo môi trường RL...")  # Cập nhật gần cuối bước này
        
        env = create_environment(train_map_obj, map_size, "human" if render else None)
        console.print(f"[green]✓[/green] Môi trường RL đã được tạo.")
        if progress_callback:
            progress_callback(P_ENV_CREATION_END, "Môi trường RL đã tạo.")

        # --- 4. Tạo Agent --- 
        console.print("[step]Bước 4: Tạo agent RL...[/step]")
        if progress_callback:
            progress_callback(P_AGENT_CREATION_END - 2, "Tạo RL agent...")

        # Sử dụng các siêu tham số tối ưu
        agent_hyperparams = OPTIMAL_HYPERPARAMS.get(map_size, OPTIMAL_HYPERPARAMS[8])
        agent = create_agent(env, map_size, use_advanced, log_dir=str(session_log_dir))
        console.print(f"[green]✓[/green] Agent đã được tạo.")
        if progress_callback:
            progress_callback(P_AGENT_CREATION_END, "RL Agent đã tạo.")

        # --- 5. Huấn luyện Agent --- 
        console.print("[step]Bước 5: Huấn luyện agent...[/step]")
        if progress_callback:
            progress_callback(P_TRAINING_START, f"Chuẩn bị huấn luyện ({training_steps} bước)...")
        
        training_start_time = time.time()
        
        sb3_callback_list = []
        if progress_callback:
            rich_sb3_callback = RichProgressSB3Callback(
                total_training_steps=training_steps,
                pipeline_progress_callback=progress_callback,
                pipeline_progress_start_percent=P_TRAINING_START,
                pipeline_progress_span_percent=P_TRAINING_SPAN,
                verbose=0 
            )
            sb3_callback_list.append(rich_sb3_callback)

        agent.train(total_timesteps=training_steps, callback=sb3_callback_list[0] if sb3_callback_list else None) 

        training_duration = time.time() - training_start_time
        console.print(f"[green]✓[/green] Huấn luyện hoàn thành sau {training_duration:.2f} giây.")
        
        if progress_callback:
            progress_callback(P_TRAINING_END, f"Huấn luyện xong. Đang lưu model phiên...")
        
        current_session_model_name = f"model_session_{SESSION_ID}_{map_specific_name_part}_steps_{training_steps}.zip"
        current_session_model_path = session_models_dir / current_session_model_name
        agent.save_model(str(current_session_model_path))  # Đảm bảo truyền string
        console.print(f"[info]Mô hình huấn luyện trong phiên này được lưu tại: {current_session_model_path}[/info]")

        # --- 6. Đánh giá Agent đã huấn luyện và so sánh --- 
        console.print("[step]Bước 6: Đánh giá agent và so sánh với mô hình tốt nhất hiện có...[/step]")
        
        if progress_callback:
            progress_callback(P_SESSION_MODEL_SAVE_END + 2, "Đánh giá model mới...")
        
        # Tải bản đồ đánh giá hoặc dùng bản đồ huấn luyện nếu không có bản đồ đánh giá
        eval_map_obj_for_comparison = None
        if map_eval_paths and map_eval_paths[0]:
            eval_map_obj_for_comparison = Map.load(map_eval_paths[0])
        
        if not eval_map_obj_for_comparison:
            console.print(f"[warning]Không tải được bản đồ đánh giá từ {map_eval_paths[0] if map_eval_paths else 'N/A'}. Sử dụng bản đồ huấn luyện để đánh giá.[/warning]")
            eval_map_obj_for_comparison = train_map_obj
        
        eval_env_for_comparison = create_environment(eval_map_obj_for_comparison, map_size)

        console.print(f"[info]Đánh giá mô hình vừa huấn luyện (từ phiên {SESSION_ID})...[/info]")
        
        new_model_performance = evaluate_robust_performance(
            agent, eval_env_for_comparison, 
            num_episodes_per_scenario=10, 
            scenarios=None,
            outer_rich_progress_active=outer_rich_progress_active
        ) 
        
        if progress_callback:
            progress_callback(P_NEW_MODEL_EVAL_END, "Đánh giá model mới hoàn tất.")
        
        final_best_model_path = None
        final_best_model_performance = None

        if best_model_path_for_type.exists():
            console.print(f"[info]Đang tải mô hình tốt nhất hiện có từ: {best_model_path_for_type} để so sánh...[/info]")
            if progress_callback:
                progress_callback(P_NEW_MODEL_EVAL_END + 2, "Đánh giá model tốt nhất hiện có...")
            old_agent = DQNAgentTrainer(env=eval_env_for_comparison) 
            try:
                old_agent.load_model(str(best_model_path_for_type))
                console.print("[info]Đánh giá mô hình tốt nhất hiện có...[/info]")
                old_model_performance = evaluate_robust_performance(
                    old_agent, eval_env_for_comparison, 
                    num_episodes_per_scenario=10,
                    scenarios=None,
                    outer_rich_progress_active=outer_rich_progress_active
                )
                if progress_callback:
                    progress_callback(P_OLD_MODEL_EVAL_END, "So sánh model hoàn tất.")
                
                console.print(f"  Điểm mô hình mới: {new_model_performance['overall_score']:.4f}")
                console.print(f"  Điểm mô hình cũ: {old_model_performance['overall_score']:.4f}")

                if new_model_performance["overall_score"] > old_model_performance["overall_score"]:
                    console.print(f"[success]Mô hình mới TỐT HƠN. Đang lưu vào: {best_model_path_for_type}[/success]")
                    agent.save_model(str(best_model_path_for_type))  # Ghi đè mô hình tốt nhất
                    final_best_model_path = best_model_path_for_type
                    final_best_model_performance = new_model_performance
                else:
                    console.print(f"[warning]Mô hình mới KHÔNG cải thiện. Giữ lại mô hình cũ tại: {best_model_path_for_type}[/warning]")
                    final_best_model_path = best_model_path_for_type  # Vẫn là đường dẫn cũ
                    final_best_model_performance = old_model_performance  # Performance của model cũ tốt hơn
            except Exception as e:
                console.print(f"[error]Lỗi khi tải hoặc đánh giá mô hình cũ: {e}. Sẽ lưu mô hình mới.[/error]")
                agent.save_model(str(best_model_path_for_type))
                final_best_model_path = best_model_path_for_type
                final_best_model_performance = new_model_performance
        else:
            console.print(f"[info]Chưa có mô hình tốt nhất nào cho loại bản đồ '{map_type_folder_name}'. Lưu mô hình mới làm tốt nhất.[/info]")
            agent.save_model(str(best_model_path_for_type))
            final_best_model_path = best_model_path_for_type
            final_best_model_performance = new_model_performance
        
        console.print("[step]Bước 7: Hoàn thành pipeline và dọn dẹp (nếu có)...[/step]")
        if progress_callback:
            progress_callback(P_PIPELINE_END, "Pipeline hoàn thành!")
        
        return {
            "best_model_path": str(final_best_model_path) if final_best_model_path else None,
            "best_model_performance": final_best_model_performance,
            "session_model_path": str(current_session_model_path),
            "session_model_performance": new_model_performance,
            "map_type_folder_name": map_type_folder_name
        }
    except Exception as e_pipeline:
        console.print(f"[bold red]LỖI NGHIÊM TRỌNG TRONG PIPELINE HUẤN LUYỆN:[/bold red]")
        console.print(f"[red]{str(e_pipeline)}[/red]")
        if RICH_AVAILABLE:  # traceback có thể dài, chỉ in nếu có Rich
            console.print_exception(show_locals=False)  # show_locals=True có thể quá dài
        else:
            traceback.print_exc()  # In traceback tiêu chuẩn
        return {
            "best_model_path": None,
            "best_model_performance": None,
            "session_model_path": None,
            "session_model_performance": None,
            "error": str(e_pipeline),
            "map_type_folder_name": locals().get("map_type_folder_name", "unknown")
        }

def master_mode():
    """
    Chế độ Master: Chỉ cần nhập kích thước bản đồ, tự động tối ưu hóa tất cả các tham số khác
    """
    if RICH_AVAILABLE:
        console.print("\n[bold cyan]MASTER MODE - TỰ ĐỘNG TỐI ƯU HÓA[/bold cyan]")
        console.print("[bold yellow]Chỉ cần nhập kích thước bản đồ, mọi thứ sẽ được tự động cấu hình tối ưu![/bold yellow]")
        
        # Tạo bảng chọn kích thước bản đồ
        size_table = Table(box=box.SIMPLE, show_header=True)
        size_table.add_column("Option", header_style="bold cyan", style=COLORS["menu_number"], justify="center")
        size_table.add_column("Kích thước", header_style="bold cyan", style=COLORS["menu_text"], justify="center")
        size_table.add_column("Mức độ phức tạp", header_style="bold cyan", style=COLORS["menu_text"], justify="center")
        size_table.add_column("Thời gian huấn luyện", header_style="bold cyan", style=COLORS["menu_text"], justify="center")
        
        size_table.add_row("1", "8 x 8", "Cơ bản", "~5-10 phút")
        size_table.add_row("2", "9 x 9", "Trung bình", "~10-15 phút")
        size_table.add_row("3", "10 x 10", "Khó", "~15-25 phút")
        size_table.add_row("4", "12 x 12", "Rất khó", "~30-40 phút")
        size_table.add_row("5", "15 x 15", "Cực khó", "~45-60 phút")
        
        size_panel = Panel(
            size_table,
            title="[ CHỌN KÍCH THƯỚC BẢN ĐỒ ]",
            title_align="center",
            border_style=COLORS["border"],
            padding=(1, 2)
        )
        console.print(size_panel)
        
        choice = Prompt.ask(
            "\n[bold cyan]Chọn kích thước bản đồ[/bold cyan]",
            choices=["1", "2", "3", "4", "5", "0"],
            default="1"
        )
        
        if choice == "0":
            return
            
        # Map từ lựa chọn sang kích thước bản đồ
        map_sizes = {
            "1": 8,
            "2": 9,
            "3": 10,
            "4": 12,
            "5": 15
        }
        
        map_size = map_sizes[choice]
        
        # Cấu hình tối ưu tự động dựa trên kích thước
        # Số lượng bản đồ giảm khi kích thước tăng
        if map_size <= 9:
            num_maps = 12
        elif map_size <= 10:
            num_maps = 10
        elif map_size <= 12:
            num_maps = 8
        else:
            num_maps = 5
            
        # Số bước huấn luyện tăng theo kích thước
        if map_size <= 8:
            training_steps = 125000  # Original: 50000
            use_advanced = False
        elif map_size <= 9:
            training_steps = 187500  # Original: 75000
            use_advanced = False
        elif map_size <= 10:
            training_steps = 250000  # Original: 100000
            use_advanced = True
        elif map_size <= 12:
            training_steps = 375000  # Original: 150000
            use_advanced = True
        else:
            training_steps = 500000  # Original: 200000
            use_advanced = True
            
        # Hiển thị cấu hình tối ưu đã chọn
        config_table = Table(show_header=True, box=box.SIMPLE)
        config_table.add_column("Tham số", style="cyan")
        config_table.add_column("Giá trị", style="green")
        
        config_table.add_row("Kích thước bản đồ", f"{map_size}x{map_size}")
        config_table.add_row("Số lượng bản đồ", f"{num_maps}")
        config_table.add_row("Số bước huấn luyện", f"{training_steps:,}")
        config_table.add_row("Sử dụng kỹ thuật nâng cao", "✅ Có" if use_advanced else "❌ Không")
        
        hyperparams = OPTIMAL_HYPERPARAMS.get(map_size, OPTIMAL_HYPERPARAMS[10])
        env_params = OPTIMAL_ENV_PARAMS.get(map_size, OPTIMAL_ENV_PARAMS[10])
        
        config_table.add_row("Learning rate", f"{hyperparams['learning_rate']}")
        config_table.add_row("Replay buffer", f"{hyperparams['buffer_size']}")
        config_table.add_row("Nhiên liệu ban đầu", f"{env_params['initial_fuel']}")
        
        config_panel = Panel(
            config_table,
            title="[ CẤU HÌNH TỐI ƯU ]",
            title_align="center",
            border_style=COLORS["border"],
            padding=(1, 2)
        )
        console.print(config_panel)
        
        # Hiển thị thông báo tiến trình
        with console.status("[bold green]Đang chuẩn bị huấn luyện...", spinner="dots"):
            time.sleep(1.5)  # Tạo hiệu ứng loading
        
        confirm = Confirm.ask("[bold yellow]Xác nhận huấn luyện với cấu hình tối ưu?[/bold yellow]")
        
        if confirm:
            console.print("[bold green]Bắt đầu huấn luyện tối ưu cho bản đồ kích thước " + 
                        f"{map_size}x{map_size}...[/bold green]")
            
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(bar_width=40, style=COLORS["progress"]),
                TaskProgressColumn(),
                TimeRemainingColumn(),
                TextColumn("[cyan]{task.fields[status]}", justify="right"),
                console=console,
                expand=False
            ) as progress:
                task = progress.add_task(
                    f"[bold cyan]Huấn luyện Master cho bản đồ {map_size}x{map_size}[/bold cyan]", 
                    total=100,
                    status="Chuẩn bị..."
                )
                
                # Cập nhật trạng thái khi chạy pipeline
                def progress_update_callback(percent, status):
                    progress.update(task, completed=10 + percent * 0.9, status=status)
                
                progress.update(task, completed=0, status="Khởi tạo pipeline...") 
                progress_update_callback(0, "Tạo bản đồ...") # Directly call the locally defined callback
                
                # Chạy pipeline huấn luyện với một cờ hiệu đặc biệt để tránh xung đột progress bar
                results = run_training_pipeline(
                    map_size=map_size,
                    num_maps=num_maps,
                    training_steps=training_steps,
                    use_advanced=use_advanced,
                    render=False,
                    progress_callback=progress_update_callback,
                    outer_rich_progress_active=True # Master mode has an active Rich progress
                )
                
                progress.update(task, completed=100, status="Hoàn thành!")
                
            # Hiển thị kết quả
            if results:
                console.print("\n[bold underline bright_green]Tổng kết Pipeline Huấn luyện:[/bold underline bright_green]")
                
                # Retrieve map_type_folder_name from results
                map_type_folder_name = results.get("map_type_folder_name", "N/A")

                if results.get("session_model_path") and results.get("session_model_performance"):
                    console.print(f"  {ICONS['model']} Model của phiên này đã lưu tại: [cyan]{results['session_model_path']}[/cyan]")
                    session_perf = results['session_model_performance']
                    console.print(f"    Điểm hiệu năng (phiên): [yellow]{session_perf.get('overall_score', 'N/A'):.4f}[/yellow]")
                    console.print(f"    Tỷ lệ thành công TB (phiên): {session_perf.get('avg_success_rate', 'N/A'):.2%}")

                if results.get("best_model_path") and results.get("best_model_performance"):
                    console.print(f"  {ICONS['success']} Model tốt nhất cho loại bản đồ '{map_type_folder_name}' đã cập nhật/lưu tại: [bright_cyan]{results['best_model_path']}[/bright_cyan]")
                    best_perf = results['best_model_performance']
                    console.print(f"    Điểm hiệu năng (tốt nhất): [bold yellow]{best_perf.get('overall_score', 'N/A'):.4f}[/bold yellow]")
                    console.print(f"    Tỷ lệ thành công TB (tốt nhất): {best_perf.get('avg_success_rate', 'N/A'):.2%}")

                    # Hiển thị bảng chi tiết cho model tốt nhất
                    results_table = Table(show_header=True, box=box.ROUNDED, title_style="bold magenta", title=f"Chi tiết Model Tốt Nhất ({Path(results['best_model_path']).name})")
                    results_table.add_column("Chỉ số", style="cyan", overflow="fold")
                    results_table.add_column("Giá trị", style="green")
                
                    results_table.add_row("Điểm tổng hợp", f"{best_perf.get('overall_score', 'N/A'):.4f}")
                    results_table.add_row("Tỷ lệ thành công TB", f"{best_perf.get('avg_success_rate', 'N/A'):.2%}")
                    results_table.add_row("Phần thưởng TB tổng thể", f"{best_perf.get('avg_reward_overall', 'N/A'):.2f}")
                    results_table.add_row("Độ dài đường đi TB", f"{best_perf.get('avg_path_length_overall', 'N/A'):.2f}")
                    results_table.add_row("Tiền chi TB", f"{best_perf.get('avg_money_spent_overall', 'N/A'):.2f}")
                    results_table.add_row("Nhiên liệu tiêu thụ TB", f"{best_perf.get('avg_fuel_consumed_overall', 'N/A'):.2f}")
                    
                    console.print(results_table)

                    if best_perf.get('detailed_results_by_scenario'):
                        console.print("[bold magenta]  Chi tiết theo kịch bản đánh giá (model tốt nhất):[/bold magenta]")
                        for scenario_name, details in best_perf['detailed_results_by_scenario'].items():
                            short_name = shorten_text(scenario_name, 40)
                            console.print(f"    [italic cyan]{short_name}[/italic cyan]: SR={details.get('success_rate',0):.2f}, AvgRew={details.get('avg_reward',0):.2f}")
                else:
                    console.print("[yellow]Không có thông tin về model tốt nhất được trả về từ pipeline.[/yellow]")
            else:
                console.print("[bold red]Pipeline huấn luyện không trả về kết quả.[/bold red]")
            input("\nHuấn luyện hoàn tất! Nhấn Enter để tiếp tục...")
    else:
        print("\nMASTER MODE - TỰ ĐỘNG TỐI ƯU HÓA")
        print("Chỉ cần nhập kích thước bản đồ, mọi thứ sẽ được tự động cấu hình tối ưu!")
        
        print("\nCác kích thước bản đồ hỗ trợ:")
        print("  [1] 8 x 8  - Cơ bản      (~5-10 phút)")
        print("  [2] 9 x 9  - Trung bình  (~10-15 phút)")
        print("  [3] 10 x 10 - Khó        (~15-25 phút)")
        print("  [4] 12 x 12 - Rất khó    (~30-40 phút)")
        print("  [5] 15 x 15 - Cực khó    (~45-60 phút)")
        print("  [0] Quay lại")
        
        choice = input("\nChọn kích thước bản đồ [1-5, 0 để quay lại]: ")
        
        if choice == "0":
            return
            
        map_sizes = {
            "1": 8,
            "2": 9,
            "3": 10,
            "4": 12,
            "5": 15
        }
        
        if choice not in map_sizes:
            print("Lựa chọn không hợp lệ. Vui lòng chọn lại.")
            return
            
        map_size = map_sizes[choice]
        
        # Cấu hình tối ưu tự động
        if map_size <= 9:
            num_maps = 12
        elif map_size <= 10:
            num_maps = 10
        elif map_size <= 12:
            num_maps = 8
        else:
            num_maps = 5
            
        if map_size <= 8:
            training_steps = 125000  # Original: 50000
            use_advanced = False
        elif map_size <= 9:
            training_steps = 187500  # Original: 75000
            use_advanced = False
        elif map_size <= 10:
            training_steps = 250000  # Original: 100000
            use_advanced = True
        elif map_size <= 12:
            training_steps = 375000  # Original: 150000
            use_advanced = True
        else:
            training_steps = 500000  # Original: 200000
            use_advanced = True
            
        # Hiển thị cấu hình
        print(f"\nCấu hình tối ưu cho bản đồ {map_size}x{map_size}:")
        print(f"  Số lượng bản đồ: {num_maps}")
        print(f"  Số bước huấn luyện: {training_steps:,}")
        print(f"  Sử dụng kỹ thuật nâng cao: {'Có' if use_advanced else 'Không'}")
        
        confirm = input("\nXác nhận huấn luyện với cấu hình tối ưu? (y/n): ").lower() == 'y'
        
        if confirm:
            print(f"\nBắt đầu huấn luyện tối ưu cho bản đồ kích thước {map_size}x{map_size}...")
            
            # Chạy pipeline huấn luyện
            results = run_training_pipeline(
                map_size=map_size,
                num_maps=num_maps,
                training_steps=training_steps,
                use_advanced=use_advanced,
                render=False
            )
            
            # Hiển thị kết quả cơ bản
            if results:
                console.print("\n[bold underline bright_green]Tổng kết Pipeline Huấn luyện (CLI Mode):[/bold underline bright_green]")
                if results.get("session_model_path") and results.get("session_model_performance"):
                    console.print(f"  Model của phiên này: {results['session_model_path']}")
                    session_perf = results['session_model_performance']
                    console.print(f"    Điểm (phiên): {session_perf.get('overall_score', 'N/A'):.4f}")
                if results.get("best_model_path") and results.get("best_model_performance"):
                    console.print(f"  Model tốt nhất cho loại bản đồ: {results['best_model_path']}")
                    best_perf = results['best_model_performance']
                    console.print(f"    Điểm (tốt nhất): {best_perf.get('overall_score', 'N/A'):.4f}")
            else:
                console.print("Pipeline huấn luyện không trả về kết quả.")
            input("\nHuấn luyện hoàn tất! Nhấn Enter để tiếp tục...")

def evaluate_model_ui():
    """
    Giao diện đánh giá model đã huấn luyện
    """
    if RICH_AVAILABLE:
        console.print("\n[bold cyan]EVALUATE MODEL - Đánh giá model đã huấn luyện[/bold cyan]")
        
        # Lựa chọn nguồn model
        source_choice_table = Table(box=box.SIMPLE, show_header=True)
        source_choice_table.add_column("Option", header_style="bold cyan", style=COLORS["menu_number"], justify="center")
        source_choice_table.add_column("Nguồn Model", header_style="bold cyan", style=COLORS["menu_text"])
        source_choice_table.add_row("1", "Model tốt nhất theo loại bản đồ (Best Models by Map Type)")
        source_choice_table.add_row("2", "Tất cả model theo phiên (All Session Models)")
        source_choice_table.add_row("0", "Quay lại Menu Chính")

        source_panel = Panel(
            source_choice_table,
            title="[ CHỌN NGUỒN MODEL ĐỂ ĐÁNH GIÁ ]",
            title_align="center",
            border_style=COLORS["border"],
            padding=(1,2)
        )
        console.print(source_panel)
        
        source_choice = Prompt.ask(
            "\n[bold cyan]Chọn nguồn model[/bold cyan]",
            choices=["0", "1", "2"],
            default="1"
        )

        if source_choice == "0":
            return

        all_models = []
        search_path_description = ""

        if source_choice == "1":  # Model tốt nhất theo loại bản đồ
            search_path_description = f"trong {BEST_ROBUST_MODELS_DIR}"
            if BEST_ROBUST_MODELS_DIR.exists():
                # Quét các thư mục con (loại bản đồ), tìm file model .zip trong đó
                for map_type_dir in BEST_ROBUST_MODELS_DIR.iterdir():
                    if map_type_dir.is_dir():
                        all_models.extend(list(map_type_dir.glob("*.zip")))
        elif source_choice == "2":  # Tất cả model theo phiên
            search_path_description = f"trong {MODELS_DIR}"
            if MODELS_DIR.exists():
                # Quét MODELS_DIR và các thư mục con SESSION_ID
                all_models.extend(list(MODELS_DIR.rglob("*.zip")))
                # Loại trừ các model trong BEST_ROBUST_MODELS_DIR nếu MODELS_DIR là cha của nó
                if BEST_ROBUST_MODELS_DIR.is_relative_to(MODELS_DIR):
                    best_models_paths = {p for p in BEST_ROBUST_MODELS_DIR.rglob("*.zip")}
                    all_models = [m for m in all_models if m not in best_models_paths]

        if not all_models:
            console.print(f"[bold red]❌ Không tìm thấy model nào {search_path_description}. Vui lòng huấn luyện model trước![/bold red]")
            input("\nNhấn Enter để trở về menu chính...")
            return

        # Phân loại model theo kích thước bản đồ (nếu có thể từ tên file)
        models_by_display_group = {}  # Key sẽ là string mô tả (ví dụ "Size 8x8" hoặc tên loại bản đồ)
        
        for model_path in all_models:
            model_name = model_path.stem
            # Cố gắng trích xuất kích thước từ tên file hoặc thư mục cha
            size_match = None
            type_name_for_display = "Unknown Type / Session Model"

            # Trường hợp 1: Model từ BEST_ROBUST_MODELS_DIR
            if BEST_ROBUST_MODELS_DIR in model_path.parents:
                map_type_folder_name = model_path.parent.name
                type_name_for_display = f"Best: {map_type_folder_name}"
                # Cố gắng lấy size từ map_type_folder_name nếu có
                if "size_" in map_type_folder_name:
                    parts = map_type_folder_name.split('_')
                    for part in parts:
                        if 'x' in part and part.replace('x','').isdigit():
                            try:
                                size_val = int(part.split('x')[0])
                                if size_val in [8,9,10,12,15]:
                                    size_match = size_val
                                    break
                            except ValueError:
                                pass
                if size_match is None and "map_size_" in map_type_folder_name:
                    try:
                        size_val = int(map_type_folder_name.split("map_size_")[1].split("_")[0])
                        if size_val in [8,9,10,12,15]:
                            size_match = size_val
                    except ValueError:
                        pass

            # Trường hợp 2: Model từ thư mục session
            elif MODELS_DIR in model_path.parents and SESSION_ID_PATTERN_IN_FILENAME_RE.search(model_name):
                parent_dir_name = model_path.parent.name
                if parent_dir_name.startswith("map_size_"):
                    try:
                        size_val = int(parent_dir_name.split("_")[-1])
                        if size_val in [8,9,10,12,15]:
                            size_match = size_val
                    except ValueError:
                        pass
                if size_match is None:
                    for size_pattern in [f"size_{s}" for s in [8,9,10,12,15]]:
                        if size_pattern in model_name:
                            size_match = int(size_pattern.split('_')[1])
                            break

            group_key = f"Size {size_match}x{size_match}" if size_match else type_name_for_display
            
            if group_key not in models_by_display_group:
                models_by_display_group[group_key] = []
            models_by_display_group[group_key].append(model_path)

            # Lấy ngày tạo file
            date_str = "N/A"
            try:
                timestamp = model_path.stat().st_mtime
                date_obj = datetime.fromtimestamp(timestamp)
                date_str = date_obj.strftime("%d/%m/%Y %H:%M")
            except Exception:  # nosemgrep
                pass  # Bỏ qua nếu không lấy được ngày tạo

        # Display models in groups
        group_options = sorted(models_by_display_group.keys())
        for group_name in group_options:
            console.print(f"\n[bold cyan]Group: {group_name}[/bold cyan]")
            model_table = Table(show_header=True, box=box.SIMPLE)
            model_table.add_column("ID", style="cyan", justify="center")
            model_table.add_column("Model Path", style="green")
            model_table.add_column("Created", style="yellow", justify="right")
            
            for idx, model_path in enumerate(models_by_display_group[group_name], 1):
                try:
                    model_name_for_display = str(model_path.relative_to(_ROOT_DIR))
                except ValueError:
                    model_name_for_display = str(model_path)
                
                date_str = "N/A"
                try:
                    timestamp = model_path.stat().st_mtime
                    date_str = datetime.fromtimestamp(timestamp).strftime("%Y-%m-%d %H:%M")
                except Exception:  # nosemgrep
                    pass
                
                model_table.add_row(str(idx), model_name_for_display, date_str)
            
            console.print(model_table)

def generate_training_map(map_size):
    """
    Tạo một bản đồ huấn luyện với kích thước và tỷ lệ tối ưu.
    
    Args:
        map_size: Kích thước bản đồ
        
    Returns:
        Map: Đối tượng bản đồ đã tạo
    """
    # Lấy tỷ lệ tối ưu cho kích thước bản đồ
    map_ratios = OPTIMAL_MAP_RATIOS.get(map_size, OPTIMAL_MAP_RATIOS[10])
    toll_ratio = map_ratios["toll_ratio"]
    gas_ratio = map_ratios["gas_ratio"]
    brick_ratio = map_ratios["brick_ratio"]
    
    # Thêm một chút biến thể ngẫu nhiên để tạo sự đa dạng
    current_toll_ratio = toll_ratio * random.uniform(0.8, 1.2)
    current_gas_ratio = gas_ratio * random.uniform(0.8, 1.2)
    current_brick_ratio = brick_ratio * random.uniform(0.8, 1.2)
    
    # Tạo bản đồ mới
    return Map.generate_random(
        size=map_size,
        toll_ratio=current_toll_ratio,
        gas_ratio=current_gas_ratio,
        brick_ratio=current_brick_ratio
    )

DEFAULT_EVALUATION_SCENARIOS = [
    {
        "name": "Low Cost Focus",
        "max_fuel": 30.0, "initial_fuel": 25.0, "initial_money": 1500.0,
        "fuel_per_move": 0.3, "gas_station_cost": 15.0, "toll_base_cost": 200.0
    },
    {
        "name": "High Cost - Fuel Efficient",
        "max_fuel": 40.0, "initial_fuel": 35.0, "initial_money": 3000.0,
        "fuel_per_move": 0.2, "gas_station_cost": 50.0, "toll_base_cost": 150.0
    },
    {
        "name": "Low Fuel - Money Focus",
        "max_fuel": 20.0, "initial_fuel": 15.0, "initial_money": 4000.0,
        "fuel_per_move": 0.8, "gas_station_cost": 80.0, "toll_base_cost": 50.0
    },
    {
        "name": "Average Conditions", # Dựa trên giá trị giữa của UI sliders
        "max_fuel": (DEFAULT_MAX_FUEL_RANGE[0] + DEFAULT_MAX_FUEL_RANGE[1]) / 2, 
        "initial_fuel": (DEFAULT_INITIAL_FUEL_RANGE[0] + DEFAULT_INITIAL_FUEL_RANGE[1]) / 2, # Sẽ được clamp bởi max_fuel
        "initial_money": (DEFAULT_INITIAL_MONEY_RANGE[0] + DEFAULT_INITIAL_MONEY_RANGE[1]) / 2,
        "fuel_per_move": (DEFAULT_FUEL_PER_MOVE_RANGE[0] + DEFAULT_FUEL_PER_MOVE_RANGE[1]) / 2,
        "gas_station_cost": (DEFAULT_GAS_STATION_COST_RANGE[0] + DEFAULT_GAS_STATION_COST_RANGE[1]) / 2,
        "toll_base_cost": (DEFAULT_TOLL_BASE_COST_RANGE[0] + DEFAULT_TOLL_BASE_COST_RANGE[1]) / 2
    }
]

def get_optimal_env_scenario(map_size):
    params = OPTIMAL_ENV_PARAMS.get(map_size, OPTIMAL_ENV_PARAMS[8])
    # OPTIMAL_ENV_PARAMS không có max_fuel, nên ta sẽ lấy từ default range hoặc một giá trị hợp lý.
    # Trong trường hợp này, hãy sử dụng giá trị initial_fuel làm max_fuel cho kịch bản này nếu max_fuel không được định nghĩa cụ thể.
    max_fuel_val = params.get('max_fuel', params.get('initial_fuel',(DEFAULT_MAX_FUEL_RANGE[0] + DEFAULT_MAX_FUEL_RANGE[1]) / 2))

    return {
        "name": f"Optimal Env Params (Size {map_size})",
        "max_fuel": max_fuel_val, 
        "initial_fuel": params["initial_fuel"], # Sẽ được clamp bởi max_fuel ở trên trong env.reset
        "initial_money": params["initial_money"],
        "fuel_per_move": params["fuel_per_move"],
        "gas_station_cost": params["gas_station_cost"],
        "toll_base_cost": params["toll_base_cost"]
    }

def main():
    """Hàm chính để chạy công cụ từ command line"""
    parser = argparse.ArgumentParser(description="Công cụ tự động huấn luyện RL cho định tuyến xe tải")
    
    # Tham số
    parser.add_argument("--map-size", type=int, default=8, choices=[8, 9, 10, 12, 15],
                        help="Kích thước bản đồ (8, 9, 10, 12, 15)")
    parser.add_argument("--num-maps", type=int, default=10,
                        help="Số lượng bản đồ tạo cho mỗi loại (train, eval, test)")
    parser.add_argument("--training-steps", type=int, default=50000,
                        help="Số bước huấn luyện")
    parser.add_argument("--advanced", action="store_true",
                        help="Sử dụng các kỹ thuật nâng cao (Double DQN, Dueling, PER)")
    parser.add_argument("--render", action="store_true",
                        help="Hiển thị quá trình huấn luyện")
    parser.add_argument("--cli", action="store_true",
                        help="Chạy ở chế độ command line, không hiển thị UI")
    parser.add_argument("--master", action="store_true",
                        help="Chạy ở chế độ master, chỉ cần nhập kích thước bản đồ")
    
    # Parse tham số
    args = parser.parse_args()
    
    # Nếu là chế độ master qua command line
    if args.cli and args.master:
        map_size = args.map_size
        
        # Tự động cấu hình các tham số tối ưu
        if map_size <= 9:
            num_maps = 12
        elif map_size <= 10:
            num_maps = 10
        elif map_size <= 12:
            num_maps = 8
        else:
            num_maps = 5
            
        if map_size <= 8:
            training_steps = 125000  # Original: 50000
            use_advanced = False
        elif map_size <= 9:
            training_steps = 187500  # Original: 75000
            use_advanced = False
        elif map_size <= 10:
            training_steps = 250000  # Original: 100000
            use_advanced = True
        elif map_size <= 12:
            training_steps = 375000  # Original: 150000
            use_advanced = True
        else:
            training_steps = 500000  # Original: 200000
            use_advanced = True
        
        # Hiển thị thông tin và chạy
        print(f"\nChế độ Master - Kích thước bản đồ: {map_size}x{map_size}")
        print(f"Số lượng bản đồ: {num_maps}")
        print(f"Số bước huấn luyện: {training_steps:,}")
        print(f"Sử dụng kỹ thuật nâng cao: {'Có' if use_advanced else 'Không'}")
        
        run_training_pipeline(
            map_size=map_size,
            num_maps=num_maps,
            training_steps=training_steps,
            use_advanced=use_advanced,
            render=args.render
        )
        return
    
    # Nếu là chế độ command line thông thường
    elif args.cli:
        run_training_pipeline(
            map_size=args.map_size,
            num_maps=args.num_maps,
            training_steps=args.training_steps,
            use_advanced=args.advanced,
            render=args.render
        )
        return
    
    # Khởi tạo thư mục
    setup_directories()
    
    # Khởi tạo cấu hình mặc định
    map_size = args.map_size
    num_maps = args.num_maps
    training_steps = args.training_steps
    use_advanced = args.advanced
    
    # Hiển thị giao diện chính
    while True:
        clear_screen()
        display_header()
        display_menu()
        choice = get_user_choice()
        
        if choice == "0":
            # Thoát
            if RICH_AVAILABLE:
                console.print("\n[bold green]Cảm ơn bạn đã sử dụng công cụ![/bold green]")
            else:
                print("\nCảm ơn bạn đã sử dụng công cụ!")
            break
            
        elif choice == "1":
            # Huấn luyện nhanh
            if RICH_AVAILABLE:
                console.print("\n[bold cyan]QUICK TRAIN - Huấn luyện nhanh với cấu hình mặc định (8x8)[/bold cyan]")
            else:
                print("\nQUICK TRAIN - Huấn luyện nhanh với cấu hình mặc định (8x8)")
                
            display_training_config(8, 5, 30000, False)
            
            if RICH_AVAILABLE:
                confirm = Confirm.ask("Bạn có muốn tiếp tục không?")
            else:
                confirm = input("Bạn có muốn tiếp tục không? (y/n): ").lower() == 'y'
                
            if confirm:
                results = run_training_pipeline(
                    map_size=8,
                    num_maps=5,
                    training_steps=30000,
                    use_advanced=False,
                    render=False
                )
                
                if RICH_AVAILABLE:
                    if results:
                        console.print("\n[bold underline bright_green]Tổng kết Pipeline Huấn luyện (Quick Train):[/bold underline bright_green]")
                        if results.get("best_model_path") and results.get("best_model_performance"):
                            console.print(f"  {ICONS['success']} Model tốt nhất đã lưu/cập nhật tại: [bright_cyan]{results['best_model_path']}[/bright_cyan]")
                            best_perf = results['best_model_performance']
                            console.print(f"    Điểm hiệu năng: [bold yellow]{best_perf.get('overall_score', 'N/A'):.4f}[/bold yellow]")
                        else:
                            console.print("[yellow]Không có thông tin về model tốt nhất.[/yellow]")
                    else:
                        console.print("[red]Pipeline huấn luyện không trả về kết quả.[/red]")
                    console.print("\n[bold green]Huấn luyện hoàn tất! Nhấn Enter để tiếp tục...[/bold green]")
                else:
                    if results and results.get("best_model_path"):
                        print(f"Model tốt nhất đã lưu/cập nhật tại: {results['best_model_path']}")
                    else:
                        print("Không có thông tin model tốt nhất hoặc pipeline lỗi.")
                    input("\nHuấn luyện hoàn tất! Nhấn Enter để tiếp tục...")
            
        elif choice == "2":
            # Master Mode - Chỉ cần nhập kích thước bản đồ
            master_mode()
            
        elif choice == "3":
            # Huấn luyện nâng cao
            if RICH_AVAILABLE:
                console.print("\n[bold cyan]ADVANCED TRAIN - Huấn luyện nâng cao (Double DQN, Dueling, PER)[/bold cyan]")
                
                map_size = int(Prompt.ask(
                    "Chọn kích thước bản đồ", 
                    choices=["8", "9", "10", "12", "15"],
                    default="10"
                ))
                
                num_maps = int(Prompt.ask(
                    "Số lượng bản đồ mỗi loại",
                    default="10"
                ))
                
                training_steps = int(Prompt.ask(
                    "Số bước huấn luyện",
                    default="100000"
                ))
            else:
                print("\nADVANCED TRAIN - Huấn luyện nâng cao (Double DQN, Dueling, PER)")
                map_size = int(input("Chọn kích thước bản đồ (8, 9, 10, 12, 15): ") or "10")
                num_maps = int(input("Số lượng bản đồ mỗi loại: ") or "10")
                training_steps = int(input("Số bước huấn luyện: ") or "100000")
            
            display_training_config(map_size, num_maps, training_steps, True)
            
            if RICH_AVAILABLE:
                confirm = Confirm.ask("Bạn có muốn tiếp tục không?")
            else:
                confirm = input("Bạn có muốn tiếp tục không? (y/n): ").lower() == 'y'
                
            if confirm:
                results = run_training_pipeline(
                    map_size=map_size,
                    num_maps=num_maps,
                    training_steps=training_steps,
                    use_advanced=True,
                    render=False
                )
                
                if RICH_AVAILABLE:
                    if results:
                        console.print("\n[bold underline bright_green]Tổng kết Pipeline Huấn luyện (Advanced Train):[/bold underline bright_green]")
                        if results.get("best_model_path") and results.get("best_model_performance"):
                            console.print(f"  {ICONS['success']} Model tốt nhất đã lưu/cập nhật tại: [bright_cyan]{results['best_model_path']}[/bright_cyan]")
                            best_perf = results['best_model_performance']
                            console.print(f"    Điểm hiệu năng: [bold yellow]{best_perf.get('overall_score', 'N/A'):.4f}[/bold yellow]")
                        else:
                            console.print("[yellow]Không có thông tin về model tốt nhất.[/yellow]")
                    else:
                        console.print("[red]Pipeline huấn luyện không trả về kết quả.[/red]")
                    console.print("\n[bold green]Huấn luyện hoàn tất! Nhấn Enter để tiếp tục...[/bold green]")
                else:
                    if results and results.get("best_model_path"):
                        print(f"Model tốt nhất đã lưu/cập nhật tại: {results['best_model_path']}")
                    else:
                        print("Không có thông tin model tốt nhất hoặc pipeline lỗi.")
                    input("\nHuấn luyện hoàn tất! Nhấn Enter để tiếp tục...")
            
        elif choice == "4":
            # Đánh giá model - Use the new function
            evaluate_model_ui()
            
        elif choice == "5":
            # Tạo bộ bản đồ mới
            if RICH_AVAILABLE:
                console.print("\n[bold cyan]GENERATE MAPS - Tạo bộ bản đồ mới[/bold cyan]")
                
                map_size = int(Prompt.ask(
                    "Chọn kích thước bản đồ", 
                    choices=["8", "9", "10", "12", "15"],
                    default="8"
                ))
                
                num_maps = int(Prompt.ask(
                    "Số lượng bản đồ mỗi loại",
                    default="10"
                ))
            else:
                print("\nGENERATE MAPS - Tạo bộ bản đồ mới")
                map_size = int(input("Chọn kích thước bản đồ (8, 9, 10, 12, 15): ") or "8")
                num_maps = int(input("Số lượng bản đồ mỗi loại: ") or "10")
            
            # Tạo bản đồ
            generate_maps(map_size, num_maps, map_types=["train", "eval", "test"])
            
            if RICH_AVAILABLE:
                console.print("\n[bold green]Tạo bản đồ hoàn tất! Nhấn Enter để tiếp tục...[/bold green]")
            else:
                input("\nTạo bản đồ hoàn tất! Nhấn Enter để tiếp tục...")
            
        elif choice == "6":
            # Hiển thị thông tin giúp đỡ
            if RICH_AVAILABLE:
                help_text = """
                # HƯỚNG DẪN SỬ DỤNG TRUCK RL AGENT MASTER

                ## QUICK TRAIN
                Sử dụng cấu hình mặc định, kích thước bản đồ 8x8 và 30,000 bước huấn luyện.
                Phù hợp để thử nghiệm nhanh trong vòng 5-10 phút.
                
                ## MASTER MODE (Khuyên dùng)
                Chế độ thông minh - Bạn chỉ cần nhập kích thước bản đồ,
                công cụ sẽ tự động cấu hình tất cả các thông số còn lại
                để đạt hiệu quả tối ưu nhất.
                
                ## ADVANCED TRAIN
                Sử dụng các kỹ thuật nâng cao (Double DQN, Dueling Network, Prioritized Experience Replay)
                cho hiệu suất tốt hơn nhưng thời gian huấn luyện lâu hơn.
                
                ## EVALUATE MODEL
                Đánh giá hiệu suất của model đã huấn luyện trên các bản đồ test.
                
                ## GENERATE MAPS
                Tạo bộ bản đồ mới cho huấn luyện và đánh giá.
                
                ## Giải thích các thông số
                - **Kích thước bản đồ**: Kích thước của bản đồ, càng lớn càng phức tạp
                - **Số bước huấn luyện**: Số bước mà agent sẽ thực hiện để học
                - **Kỹ thuật nâng cao**: Double DQN, Dueling Networks và Prioritized Experience Replay
                  giúp cải thiện quá trình học nhưng tốn thời gian hơn
                """
                md = Markdown(help_text)
                help_panel = Panel(
                    md,
                    title="[ HELP MANUAL ]",
                    title_align="center",
                    border_style=COLORS["border"],
                    padding=(1, 2)
                )
                console.print(help_panel)
                input("\nNhấn Enter để tiếp tục...")
            else:
                print("\nHƯỚNG DẪN SỬ DỤNG TRUCK RL AGENT MASTER:")
                print("\n1. QUICK TRAIN:")
                print("   Sử dụng cấu hình mặc định, kích thước bản đồ 8x8 và 30,000 bước huấn luyện.")
                print("\n2. MASTER MODE (Khuyên dùng):")
                print("   Chế độ thông minh - Bạn chỉ cần nhập kích thước bản đồ,")
                print("   công cụ sẽ tự động cấu hình tất cả các thông số còn lại")
                print("   để đạt hiệu quả tối ưu nhất.")
                print("\n3. ADVANCED TRAIN:")
                print("   Sử dụng các kỹ thuật nâng cao (Double DQN, Dueling Network, Prioritized Experience Replay)")
                print("   cho hiệu suất tốt hơn nhưng thời gian huấn luyện lâu hơn.")
                print("\n4. EVALUATE MODEL:")
                print("   Đánh giá hiệu suất của model đã huấn luyện trên các bản đồ test.")
                print("\n5. GENERATE MAPS:")
                print("   Tạo bộ bản đồ mới cho huấn luyện và đánh giá.")
                input("\nNhấn Enter để tiếp tục...")

if __name__ == "__main__":
    if RICH_AVAILABLE:
        try:
            main()
        except KeyboardInterrupt:
            console.print("\n[bold yellow]Đã hủy thao tác. Thoát chương trình.[/bold yellow]")
    else:
        try:
            main()
        except KeyboardInterrupt:
            print("\nĐã hủy thao tác. Thoát chương trình.") 