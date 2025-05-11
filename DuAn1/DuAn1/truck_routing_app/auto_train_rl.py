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
    from core.algorithms.rl_DQNAgent import DQNAgentTrainer
    from core.algorithms.greedy import GreedySearch
    from truck_routing_app.statistics.rl_evaluation import RLEvaluator
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
        "learning_rate": 0.0001,
        "gamma": 0.99,
        "buffer_size": 50000,
        "learning_starts": 1000,
        "batch_size": 64,
        "tau": 0.005,
        "train_freq": 4,
        "target_update_interval": 1000,
        "exploration_fraction": 0.15,
        "exploration_initial_eps": 1.0,
        "exploration_final_eps": 0.05
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
        "exploration_fraction": 0.2,
        "exploration_initial_eps": 1.0,
        "exploration_final_eps": 0.05
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
        "exploration_fraction": 0.2,
        "exploration_initial_eps": 1.0,
        "exploration_final_eps": 0.05
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
        "exploration_fraction": 0.25,
        "exploration_initial_eps": 1.0,
        "exploration_final_eps": 0.05
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
        "exploration_fraction": 0.3,
        "exploration_initial_eps": 1.0,
        "exploration_final_eps": 0.05
    }
}

# Tham số môi trường tối ưu cho từng kích thước bản đồ
OPTIMAL_ENV_PARAMS = {
    8: {
        "initial_fuel": 50,
        "initial_money": 1000,
        "fuel_per_move": 1.0,
        "gas_station_cost": 10,
        "toll_base_cost": 5,
        "max_steps": 200
    },
    9: {
        "initial_fuel": 60,
        "initial_money": 1200,
        "fuel_per_move": 1.0,
        "gas_station_cost": 10,
        "toll_base_cost": 5,
        "max_steps": 250
    },
    10: {
        "initial_fuel": 70,
        "initial_money": 1500,
        "fuel_per_move": 1.0,
        "gas_station_cost": 10,
        "toll_base_cost": 5,
        "max_steps": 300
    },
    12: {
        "initial_fuel": 80,
        "initial_money": 2000,
        "fuel_per_move": 1.0,
        "gas_station_cost": 10,
        "toll_base_cost": 5,
        "max_steps": 350
    },
    15: {
        "initial_fuel": 100,
        "initial_money": 2500,
        "fuel_per_move": 1.0,
        "gas_station_cost": 10,
        "toll_base_cost": 5,
        "max_steps": 500
    }
}

# Tỷ lệ tối ưu cho các loại ô đặc biệt theo kích thước bản đồ
OPTIMAL_MAP_RATIOS = {
    8: {"toll_ratio": 0.03, "gas_ratio": 0.04, "brick_ratio": 0.15},
    9: {"toll_ratio": 0.03, "gas_ratio": 0.04, "brick_ratio": 0.15},
    10: {"toll_ratio": 0.03, "gas_ratio": 0.04, "brick_ratio": 0.18},
    12: {"toll_ratio": 0.025, "gas_ratio": 0.03, "brick_ratio": 0.2},
    15: {"toll_ratio": 0.02, "gas_ratio": 0.025, "brick_ratio": 0.2}
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

# Cấu trúc thư mục
DIRECTORIES = {
    "maps": {
        "train": "maps/train",
        "eval": "maps/eval",
        "test": "maps/test",
    },
    "models": "saved_models",
    "logs": "training_logs",
    "results": "evaluation_results",
    "sessions": "sessions"
}

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

def generate_maps(map_size, num_maps=10, map_types=["train", "eval", "test"], show_progress=True, progress_callback=None):
    """
    Tạo bản đồ với kích thước và số lượng chỉ định
    
    Args:
        map_size: Kích thước bản đồ (8x8, 9x9, v.v.)
        num_maps: Số lượng bản đồ mỗi loại
        map_types: Loại bản đồ cần tạo ("train", "eval", "test")
        show_progress: Có hiển thị thanh tiến trình không
        progress_callback: Hàm callback để báo cáo tiến trình (percent, message)
    
    Returns:
        success: True nếu tạo thành công
    """
    if RICH_AVAILABLE:
        console.print(f"\n[bold {COLORS['info']}]🗺️  Đang tạo bản đồ kích thước {map_size}x{map_size}...[/bold {COLORS['info']}]")
    else:
        print(f"\n🗺️  Đang tạo bản đồ kích thước {map_size}x{map_size}...")
    
    # Lấy tỷ lệ tối ưu cho kích thước bản đồ
    map_ratios = OPTIMAL_MAP_RATIOS.get(map_size, OPTIMAL_MAP_RATIOS[10])
    toll_ratio = map_ratios["toll_ratio"]
    gas_ratio = map_ratios["gas_ratio"]
    brick_ratio = map_ratios["brick_ratio"]
    
    total_maps = len(map_types) * num_maps
    maps_created = 0
    
    for map_type in map_types:
        map_dir = f"{map_type}"  # Map.save() already prepends 'maps/'
        Path(os.path.join("maps", map_dir)).mkdir(parents=True, exist_ok=True)
        
        # Xóa các bản đồ cũ với kích thước này
        for old_map in Path(os.path.join("maps", map_dir)).glob(f"map_{map_size}x{map_size}_*.json"):
            old_map.unlink()
        
        if RICH_AVAILABLE:
            console.print(f"  [italic]Đang tạo {num_maps} bản đồ {map_type}...[/italic]")
        else:
            print(f"  Đang tạo {num_maps} bản đồ {map_type}...")
        
        # Create maps with or without progress display
        # Kiểm tra xem có progress_callback đang hoạt động không để tránh nhiều thanh tiến trình
        if show_progress and RICH_AVAILABLE and not progress_callback:
            # Kiểm tra xem đã có thanh tiến trình nào đang hoạt động không
            try:
                # Create a new progress bar
                with Progress(
                    SpinnerColumn(),
                    TextColumn("[progress.description]{task.description}"),
                    BarColumn(),
                    TaskProgressColumn(),
                    TimeRemainingColumn(),
                    console=console,
                    transient=True
                ) as progress:
                    map_task = progress.add_task(f"[cyan]Tạo bản đồ {map_type}", total=num_maps)
                    
                    for i in range(num_maps):
                        # Thay đổi nhẹ tỷ lệ để tạo sự đa dạng
                        current_toll_ratio = toll_ratio * random.uniform(0.8, 1.2)
                        current_gas_ratio = gas_ratio * random.uniform(0.8, 1.2)
                        current_brick_ratio = brick_ratio * random.uniform(0.8, 1.2)
                        
                        # Tạo bản đồ mới
                        map_obj = Map.generate_random(
                            size=map_size,
                            toll_ratio=current_toll_ratio,
                            gas_ratio=current_gas_ratio,
                            brick_ratio=current_brick_ratio
                        )
                        
                        # Lưu bản đồ
                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                        filename = f"map_{map_size}x{map_size}_{i+1}_{timestamp}.json"
                        map_obj.save(os.path.join(map_dir, filename))
                        
                        # Cập nhật tiến độ
                        progress.update(map_task, advance=1)
                        
                        # Update overall progress
                        maps_created += 1
            except rich.errors.LiveError:
                # Nếu đã có thanh tiến trình đang chạy, thì không dùng thanh tiến trình mới
                console.print("  [yellow]Không thể hiển thị thanh tiến trình (đã có thanh khác đang chạy)[/yellow]")
                # Fall back to simple version without progress bar
                for i in range(num_maps):
                    current_toll_ratio = toll_ratio * random.uniform(0.8, 1.2)
                    current_gas_ratio = gas_ratio * random.uniform(0.8, 1.2)
                    current_brick_ratio = brick_ratio * random.uniform(0.8, 1.2)
                    
                    # Tạo bản đồ mới
                    map_obj = Map.generate_random(
                        size=map_size,
                        toll_ratio=current_toll_ratio,
                        gas_ratio=current_gas_ratio,
                        brick_ratio=current_brick_ratio
                    )
                    
                    # Lưu bản đồ
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    filename = f"map_{map_size}x{map_size}_{i+1}_{timestamp}.json"
                    map_obj.save(os.path.join(map_dir, filename))
                    
                    # Update progress
                    maps_created += 1
                    if i % max(1, num_maps // 5) == 0 or i == num_maps - 1:
                        console.print(f"    [dim]{i+1}/{num_maps} ({(i+1)/num_maps*100:.0f}%)[/dim]")
        else:
            # Create maps without progress bar or using callback
            for i in range(num_maps):
                current_toll_ratio = toll_ratio * random.uniform(0.8, 1.2)
                current_gas_ratio = gas_ratio * random.uniform(0.8, 1.2)
                current_brick_ratio = brick_ratio * random.uniform(0.8, 1.2)
                
                # Tạo bản đồ mới
                map_obj = Map.generate_random(
                    size=map_size,
                    toll_ratio=current_toll_ratio,
                    gas_ratio=current_gas_ratio,
                    brick_ratio=current_brick_ratio
                )
                
                # Lưu bản đồ
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"map_{map_size}x{map_size}_{i+1}_{timestamp}.json"
                map_obj.save(os.path.join(map_dir, filename))
                
                # Update progress
                maps_created += 1
                if progress_callback:
                    # Call external progress callback with percent complete and message
                    percent_complete = (maps_created / total_maps) * 100
                    progress_callback(percent_complete, f"Tạo bản đồ {map_type} ({i+1}/{num_maps})")
                elif i % max(1, num_maps // 5) == 0 or i == num_maps - 1:
                    # Print progress periodically
                    if RICH_AVAILABLE:
                        console.print(f"    [dim]{i+1}/{num_maps} ({(i+1)/num_maps*100:.0f}%)[/dim]")
                    else:
                        print(f"    {i+1}/{num_maps} ({(i+1)/num_maps*100:.0f}%)")
    
    if RICH_AVAILABLE:
        console.print(f"[bold green]✅ Đã tạo tổng cộng {num_maps * len(map_types)} bản đồ[/bold green]")
    else:
        print(f"✅ Đã tạo tổng cộng {num_maps * len(map_types)} bản đồ")
    return True

def create_environment(map_obj, map_size, render_mode=None):
    """
    Tạo môi trường RL với tham số phù hợp cho kích thước bản đồ
    
    Args:
        map_obj: Đối tượng Map
        map_size: Kích thước bản đồ
        render_mode: Chế độ hiển thị (human hoặc None) - currently not supported
    
    Returns:
        env: Đối tượng TruckRoutingEnv
    """
    # Lấy tham số môi trường tối ưu
    env_params = OPTIMAL_ENV_PARAMS.get(map_size, OPTIMAL_ENV_PARAMS[10])
    
    # Tạo môi trường
    env = TruckRoutingEnv(
        map_object=map_obj,
        initial_fuel=env_params["initial_fuel"],
        initial_money=env_params["initial_money"],
        fuel_per_move=env_params["fuel_per_move"],
        gas_station_cost=env_params["gas_station_cost"],
        toll_base_cost=env_params["toll_base_cost"],
        max_steps_per_episode=env_params["max_steps"]
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

def train_agent(agent, total_timesteps, map_size, callback=None):
    """
    Huấn luyện agent
    
    Args:
        agent: Đối tượng DQNAgentTrainer
        total_timesteps: Tổng số bước huấn luyện
        map_size: Kích thước bản đồ
        callback: Hàm callback trong quá trình huấn luyện
    
    Returns:
        agent: Agent đã huấn luyện
    """
    print(f"\n🚀 Bắt đầu huấn luyện agent RL cho bản đồ {map_size}x{map_size}...")
    print(f"  Số bước huấn luyện: {total_timesteps}")
    
    start_time = time.time()
    
    # Định nghĩa callback hiển thị tiến độ
    if callback is None:
        def default_callback(locals, globals):
            if locals['step'] % (total_timesteps // 10) == 0:
                step = locals['step']
                elapsed_time = time.time() - start_time
                remaining_time = (elapsed_time / step) * (total_timesteps - step) if step > 0 else 0
                
                print(f"  ⏳ Tiến độ: {step}/{total_timesteps} bước ({step/total_timesteps*100:.1f}%) - "
                      f"Thời gian: {elapsed_time/60:.1f} phút - "
                      f"Còn lại: {remaining_time/60:.1f} phút")
            return True  # Tiếp tục huấn luyện
        
        callback = default_callback
    
    # Huấn luyện
    agent.train(total_timesteps=total_timesteps, callback=callback)
    
    # Thời gian huấn luyện
    training_time = time.time() - start_time
    print(f"✅ Huấn luyện hoàn tất sau {training_time/60:.1f} phút")
    
    return agent

def evaluate_agent(agent, map_size, num_episodes=5):
    """
    Đánh giá agent trên tập bản đồ đánh giá
    
    Args:
        agent: Đối tượng DQNAgentTrainer đã huấn luyện
        map_size: Kích thước bản đồ
        num_episodes: Số episodes đánh giá cho mỗi bản đồ
    
    Returns:
        results: Kết quả đánh giá
    """
    print(f"\n📊 Đánh giá agent trên bản đồ {map_size}x{map_size}...")
    
    # Tìm các bản đồ đánh giá
    eval_maps_dir = "eval"  # Map.load will prepend 'maps/'
    os.makedirs(os.path.join("maps", eval_maps_dir), exist_ok=True)
    eval_map_files = list(Path(os.path.join("maps", eval_maps_dir)).glob(f"map_{map_size}x{map_size}_*.json"))
    
    if not eval_map_files:
        print("  ❌ Không tìm thấy bản đồ đánh giá phù hợp!")
        # Tạo một số bản đồ đánh giá nếu không có
        print("  🗺️ Tạo một số bản đồ đánh giá mới...")
        generate_maps(map_size, num_maps=3, map_types=["eval"], show_progress=True)
        eval_map_files = list(Path(os.path.join("maps", eval_maps_dir)).glob(f"map_{map_size}x{map_size}_*.json"))
        
        if not eval_map_files:
            print("  ❌ Không thể tạo bản đồ đánh giá.")
            return None
    
    # Giới hạn số lượng bản đồ đánh giá
    max_eval_maps = 5
    if len(eval_map_files) > max_eval_maps:
        eval_map_files = random.sample(eval_map_files, max_eval_maps)
    
    # Kết quả đánh giá
    all_results = {
        "success_rate": 0,
        "avg_reward": 0,
        "avg_steps": 0,
        "avg_remaining_fuel": 0,
        "avg_remaining_money": 0,
        "map_size": map_size,
        "num_maps": len(eval_map_files),
        "num_episodes": num_episodes,
        "map_results": []
    }
    
    # Đánh giá trên từng bản đồ
    total_episodes = 0
    total_success = 0
    total_reward = 0
    total_steps = 0
    total_remaining_fuel = 0
    total_remaining_money = 0
    
    for i, map_file in enumerate(eval_map_files):
        print(f"  Đánh giá trên bản đồ {map_file.name} ({i+1}/{len(eval_map_files)})...")
        
        # Tải bản đồ
        try:
            map_obj = Map.load(str(Path(map_file).relative_to(Path("maps"))))  # Just pass the filename without path
            if map_obj is None:
                raise FileNotFoundError(f"Map {map_file.name} not found")
        except Exception as e:
            print(f"    ❌ Không thể tải bản đồ {map_file}: {e}")
            continue
        
        # Tạo môi trường
        env = create_environment(map_obj, map_size)
        
        # Đánh giá
        map_success = 0
        map_reward = 0
        map_steps = 0
        map_remaining_fuel = 0
        map_remaining_money = 0
        
        for ep in range(num_episodes):
            observation, _ = env.reset()
            done = False
            truncated = False
            episode_reward = 0
            episode_steps = 0
            
            while not (done or truncated):
                action = agent.predict_action(observation)
                next_observation, reward, done, truncated, info = env.step(action)
                
                episode_reward += reward
                episode_steps += 1
                observation = next_observation
            
            # Kết quả episode
            success = info.get("termination_reason") == "den_dich"
            map_success += 1 if success else 0
            map_reward += episode_reward
            map_steps += episode_steps
            
            if success and "fuel" in observation and "money" in observation:
                map_remaining_fuel += float(observation["fuel"][0])
                map_remaining_money += float(observation["money"][0])
        
        # Kết quả trung bình trên bản đồ
        map_success_rate = map_success / num_episodes
        map_avg_reward = map_reward / num_episodes
        map_avg_steps = map_steps / num_episodes
        map_avg_remaining_fuel = map_remaining_fuel / max(1, map_success)
        map_avg_remaining_money = map_remaining_money / max(1, map_success)
        
        # Cập nhật tổng
        total_episodes += num_episodes
        total_success += map_success
        total_reward += map_reward
        total_steps += map_steps
        total_remaining_fuel += map_remaining_fuel
        total_remaining_money += map_remaining_money
        
        # Lưu kết quả cho bản đồ
        all_results["map_results"].append({
            "map_name": map_file.name,
            "success_rate": map_success_rate,
            "avg_reward": map_avg_reward,
            "avg_steps": map_avg_steps,
            "avg_remaining_fuel": map_avg_remaining_fuel,
            "avg_remaining_money": map_avg_remaining_money
        })
        
        print(f"    ✅ Tỷ lệ thành công: {map_success_rate:.2f} - Phần thưởng TB: {map_avg_reward:.2f}")
    
    # Tính kết quả tổng thể
    if total_episodes > 0:
        all_results["success_rate"] = total_success / total_episodes
        all_results["avg_reward"] = total_reward / total_episodes
        all_results["avg_steps"] = total_steps / total_episodes
        all_results["avg_remaining_fuel"] = total_remaining_fuel / max(1, total_success)
        all_results["avg_remaining_money"] = total_remaining_money / max(1, total_success)
    
    # Hiển thị kết quả tổng thể
    print(f"\n📈 Kết quả đánh giá tổng thể:")
    print(f"  Tỷ lệ thành công: {all_results['success_rate']:.2f}")
    print(f"  Phần thưởng trung bình: {all_results['avg_reward']:.2f}")
    print(f"  Số bước trung bình: {all_results['avg_steps']:.2f}")
    print(f"  Nhiên liệu còn lại trung bình: {all_results['avg_remaining_fuel']:.2f}")
    print(f"  Tiền còn lại trung bình: {all_results['avg_remaining_money']:.2f}")
    
    return all_results

def detailed_evaluation(model_path, map_size):
    """
    Đánh giá chi tiết model đã huấn luyện
    
    Args:
        model_path: Đường dẫn đến model đã lưu
        map_size: Kích thước bản đồ
    
    Returns:
        results: Kết quả đánh giá chi tiết
    """
    print(f"\n🔍 Đánh giá chi tiết model trên bản đồ {map_size}x{map_size}...")
    
    try:
        # Thư mục bản đồ test
        test_maps_dir = "maps/test"  # Use a correct path
        
        # Make sure test directory exists
        os.makedirs(test_maps_dir, exist_ok=True)
        
        # Kiểm tra xem có bản đồ test nào với kích thước phù hợp không
        map_filter = f"map_{map_size}x{map_size}"
        test_maps = [f for f in os.listdir(test_maps_dir) if map_filter in f and f.endswith('.json')]
        
        if not test_maps:
            print(f"  ⚠️ Không tìm thấy bản đồ test cho kích thước {map_size}x{map_size}")
            print(f"  🗺️ Tạo một số bản đồ test mới...")
            generate_maps(map_size, num_maps=3, map_types=["test"], show_progress=True)
            test_maps = [f for f in os.listdir(test_maps_dir) if map_filter in f and f.endswith('.json')]
            
            if not test_maps:
                print(f"  ❌ Không thể tạo bản đồ test. Bỏ qua đánh giá chi tiết.")
                return None
            
        # Tạo evaluator - chỉ truyền maps_dir, vì RLEvaluator đã tự đặt results_dir="evaluation_results"
        evaluator = RLEvaluator(maps_dir=test_maps_dir)
        
        # Đường dẫn đến model
        if not model_path.endswith('.zip'):
            model_path = f"{model_path}.zip"
        
        # Kiểm tra tồn tại của model
        if not os.path.exists(model_path):
            print(f"  ❌ Không tìm thấy model tại: {model_path}")
            print(f"  Đường dẫn hiện tại: {os.getcwd()}")
            return None
        
        # Đánh giá chi tiết
        print(f"  💻 Đánh giá model: {os.path.basename(model_path)}")
        print(f"  🗺️ Trên {len(test_maps)} bản đồ test với kích thước {map_size}x{map_size}")
        
        try:
            results_df = evaluator.evaluate_rl_agent(
                model_path=model_path.replace('.zip', ''),
                n_episodes=3,
                map_filter=map_filter
            )
            
            # Kiểm tra kết quả
            if results_df is None or len(results_df) == 0:
                print(f"  ❌ Không có kết quả đánh giá. Bỏ qua đánh giá chi tiết.")
                return None
            
            # Tính toán các chỉ số thống kê
            success_rate = results_df["success"].mean()
            
            # Xử lý trường hợp có thể tên cột reward thay đổi
            reward_col = None
            for col_name in ["total_reward", "reward", "episode_reward"]:
                if col_name in results_df.columns:
                    reward_col = col_name
                    break
            
            if reward_col:
                avg_reward = results_df[reward_col].mean()
            else:
                print("  ⚠️ Không tìm thấy cột reward trong kết quả")
                avg_reward = 0
            
            # Các chỉ số thống kê khác
            avg_path_length = results_df["path_length"].mean() if "path_length" in results_df.columns else 0
            
            # Tính fuel_consumed từ initial_fuel - remaining_fuel nếu có
            if "remaining_fuel" in results_df.columns and "initial_fuel" in results_df.columns:
                results_df["fuel_consumed"] = results_df["initial_fuel"] - results_df["remaining_fuel"]
            
            # Tính money_spent từ initial_money - remaining_money nếu có
            if "remaining_money" in results_df.columns and "initial_money" in results_df.columns:
                results_df["money_spent"] = results_df["initial_money"] - results_df["remaining_money"]
            
            avg_fuel_consumed = results_df["fuel_consumed"].mean() if "fuel_consumed" in results_df.columns else 0
            avg_money_spent = results_df["money_spent"].mean() if "money_spent" in results_df.columns else 0
            
            # Hiển thị kết quả
            print(f"\n📊 Kết quả đánh giá chi tiết:")
            print(f"  Tỷ lệ thành công: {success_rate:.2f}")
            print(f"  Phần thưởng trung bình: {avg_reward:.2f}")
            print(f"  Độ dài đường đi trung bình: {avg_path_length:.2f}")
            print(f"  Nhiên liệu tiêu thụ trung bình: {avg_fuel_consumed:.2f}")
            print(f"  Chi phí trung bình: {avg_money_spent:.2f}")
            
            # Phân tích lỗi
            if success_rate < 1.0:
                try:
                    failure_reasons = results_df[results_df["success"] == False]["termination_reason"].value_counts()
                    print("\n❌ Nguyên nhân thất bại:")
                    for reason, count in failure_reasons.items():
                        percentage = (count / len(results_df)) * 100
                        print(f"  - {reason}: {count} lần ({percentage:.2f}%)")
                except Exception as e:
                    print(f"  ⚠️ Không thể phân tích lỗi: {e}")
            
            return {
                "success_rate": success_rate,
                "avg_reward": avg_reward,
                "avg_path_length": avg_path_length,
                "avg_fuel_consumed": avg_fuel_consumed,
                "avg_money_spent": avg_money_spent,
                "dataframe": results_df
            }
            
        except Exception as e:
            print(f"  ❌ Lỗi khi đánh giá chi tiết: {e}")
            import traceback
            print(traceback.format_exc())
            return None
            
    except Exception as e:
        print(f"  ❌ Lỗi tổng thể khi đánh giá: {e}")
        import traceback
        print(traceback.format_exc())
        return None

def run_training_pipeline(map_size, num_maps=10, training_steps=50000, use_advanced=False, render=False, progress_callback=None):
    """
    Chạy pipeline huấn luyện RL agent.
    
    Args:
        map_size: Kích thước bản đồ
        num_maps: Số lượng bản đồ huấn luyện
        training_steps: Số bước huấn luyện
        use_advanced: Sử dụng cấu hình nâng cao
        render: Hiển thị quá trình huấn luyện
        progress_callback: Callback cập nhật tiến độ
        
    Returns:
        dict: Kết quả huấn luyện
    """
    start_time = time.time()
    
    # Đảm bảo các thư mục tồn tại
    os.makedirs("saved_models", exist_ok=True)
    os.makedirs("evaluation_results", exist_ok=True)
    os.makedirs("training_logs", exist_ok=True)
    os.makedirs("maps/train", exist_ok=True)
    
    # Tạo ID phiên huấn luyện duy nhất
    training_id = f"{map_size}x{map_size}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    log_dir = os.path.join("training_logs", training_id)
    os.makedirs(log_dir, exist_ok=True)
    
    # Tạo bản đồ huấn luyện
    if progress_callback:
        progress_callback(10, "Tạo bản đồ huấn luyện...")
    
    # Tạo bản đồ train thông qua function generate_maps
    # Nếu đang trong context của progress bar khác, tắt hiển thị progress bar mới
    print(f"Tạo bản đồ huấn luyện {map_size}x{map_size}...")
    
    # Map the percent from 0-100 in generate_maps to 10-25 in our overall progress
    def map_progress(percent, message):
        if progress_callback:
            overall_percent = 10 + (percent * 0.15 / 100)
            progress_callback(overall_percent, message)
    
    generate_maps(
        map_size=map_size, 
        num_maps=num_maps, 
        map_types=["train"], 
        show_progress=progress_callback is None,
        progress_callback=map_progress if progress_callback else None
    )
    
    # Lấy các file map train đã tạo 
    train_maps_path = os.path.join("maps", "train")
    map_files = list(Path(train_maps_path).glob(f"map_{map_size}x{map_size}_*.json"))
    
    if not map_files:
        raise ValueError(f"Không tìm thấy bản đồ train {map_size}x{map_size}")
    
    # Load map đầu tiên để tạo môi trường
    map_obj = Map.load(str(Path(map_files[0]).relative_to(Path("maps"))))
    
    # Khởi tạo agent
    if progress_callback:
        progress_callback(25, "Khởi tạo agent...")
    
    # Tạo môi trường
    env = create_environment(map_obj, map_size)
    
    # Khởi tạo agent
    agent = create_agent(env, map_size, use_advanced, log_dir)
    
    # Huấn luyện agent
    if progress_callback:
        progress_callback(30, "Bắt đầu huấn luyện...")
    
    # Định nghĩa callback cập nhật tiến độ
    if progress_callback:
        def training_callback(locals, globals):
            step = locals.get('step', 0)
            if step % (training_steps // 20) == 0 or step == training_steps:
                progress_percent = step / training_steps
                elapsed_time = time.time() - start_time
                remaining_time = (elapsed_time / max(1, step)) * (training_steps - step) if step > 0 else 0
                
                # Cập nhật tiến độ từ 30% đến 80%
                overall_percent = 30 + (progress_percent * 50)
                status = f"Huấn luyện: {step}/{training_steps} bước - Còn {remaining_time/60:.1f} phút"
                progress_callback(overall_percent, status)
                
                # Lưu checkpoint định kỳ
                if step % (training_steps // 5) == 0:
                    checkpoint_path = os.path.join("saved_models", f"checkpoint_{map_size}_{step}_{training_id}")
                    agent.save_model(checkpoint_path)
            return True
        
        agent = train_agent(agent, training_steps, map_size, callback=training_callback)
    else:
        agent = train_agent(agent, training_steps, map_size)
    
    # Lưu model cuối cùng
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    advanced_suffix = "_advanced" if use_advanced else ""
    model_filename = f"rl_agent_size_{map_size}{advanced_suffix}_{training_id}"
    model_path = os.path.join("saved_models", model_filename)
    
    # Đảm bảo lưu model thành công
    try:
        agent.save_model(model_path)
        if not os.path.exists(f"{model_path}.zip"):
            raise Exception("Model file not found after saving")
            
        if RICH_AVAILABLE and progress_callback is None:
            console.print(f"\n[bold {COLORS['success']}]💾 Đã lưu model tại: {model_path}.zip[/bold {COLORS['success']}]")
        elif progress_callback:
            progress_callback(80, "Đánh giá model...")
        else:
            print(f"\n💾 Đã lưu model tại: {model_path}.zip")
    except Exception as e:
        error_msg = f"Lỗi khi lưu model: {str(e)}"
        if RICH_AVAILABLE and progress_callback is None:
            console.print(f"\n[bold red]❌ {error_msg}[/bold red]")
        elif progress_callback:
            progress_callback(80, error_msg)
        else:
            print(f"\n❌ {error_msg}")
        raise
    
    # Đánh giá agent
    evaluation_results = evaluate_agent(agent, map_size)
    if progress_callback:
        progress_callback(90, "Đánh giá chi tiết...")
    
    # Đánh giá chi tiết
    detailed_results = detailed_evaluation(model_path, map_size)
    if progress_callback:
        progress_callback(95, "Lưu kết quả...")
    
    # Thời gian tổng
    total_time = time.time() - start_time
    if RICH_AVAILABLE and progress_callback is None:
        console.print(f"\n[bold {COLORS['info']}]⏱️  Tổng thời gian: {total_time/60:.1f} phút[/bold {COLORS['info']}]")
    elif progress_callback is None:
        print(f"\n⏱️  Tổng thời gian: {total_time/60:.1f} phút")
    
    # Lưu kết quả
    results = {
        "map_size": map_size,
        "num_maps": num_maps,
        "training_steps": training_steps,
        "use_advanced": use_advanced,
        "model_path": f"{model_path}.zip",
        "training_id": training_id,
        "training_time": total_time,
        "evaluation_results": evaluation_results,
        "detailed_results": {
            "success_rate": detailed_results["success_rate"] if detailed_results else None,
            "avg_reward": detailed_results["avg_reward"] if detailed_results else None,
            "avg_path_length": detailed_results["avg_path_length"] if detailed_results else None,
            "avg_fuel_consumed": detailed_results["avg_fuel_consumed"] if detailed_results else None,
            "avg_money_spent": detailed_results["avg_money_spent"] if detailed_results else None
        }
    }
    
    # Lưu kết quả vào file JSON
    results_path = os.path.join("evaluation_results", f"training_results_{map_size}x{map_size}_{training_id}.json")
    
    with open(results_path, 'w') as f:
        # Loại bỏ dataframe trước khi lưu
        if detailed_results and "dataframe" in detailed_results:
            del detailed_results["dataframe"]
        
        json.dump(results, f, indent=2)
    
    if RICH_AVAILABLE and progress_callback is None:
        console.print(f"\n[italic {COLORS['info']}]📝 Đã lưu kết quả tại: {results_path}[/italic {COLORS['info']}]")
        console.print(f"\n{'='*80}")
        console.print(f"[bold {COLORS['success']}]✅ HUẤN LUYỆN HOÀN TẤT[/bold {COLORS['success']}]")
        console.print(f"{'='*80}")
    elif progress_callback:
        progress_callback(100, "Huấn luyện hoàn tất!")
    else:
        print(f"\n📝 Đã lưu kết quả tại: {results_path}")
        print(f"\n{'='*80}")
        print(f"✅ HUẤN LUYỆN HOÀN TẤT")
        print(f"{'='*80}")
    
    return results

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
                
                progress.update(task, completed=10, status="Tạo bản đồ...")
                time.sleep(0.5)
                
                # Chạy pipeline huấn luyện với một cờ hiệu đặc biệt để tránh xung đột progress bar
                results = run_training_pipeline(
                    map_size=map_size,
                    num_maps=num_maps,
                    training_steps=training_steps,
                    use_advanced=use_advanced,
                    render=False,
                    progress_callback=progress_update_callback
                )
                
                progress.update(task, completed=100, status="Hoàn thành!")
                
            # Hiển thị kết quả
            if results and "detailed_results" in results:
                results_table = Table(show_header=True, box=box.SIMPLE)
                results_table.add_column("Chỉ số", style="cyan")
                results_table.add_column("Giá trị", style="green")
                
                dr = results["detailed_results"]
                if dr["success_rate"] is not None:
                    results_table.add_row("Tỷ lệ thành công", f"{dr['success_rate']*100:.1f}%")
                    results_table.add_row("Phần thưởng trung bình", f"{dr['avg_reward']:.2f}")
                    results_table.add_row("Độ dài đường đi TB", f"{dr['avg_path_length']:.2f}")
                    results_table.add_row("Nhiên liệu tiêu thụ TB", f"{dr['avg_fuel_consumed']:.2f}")
                    
                    results_panel = Panel(
                        results_table,
                        title="[ KẾT QUẢ HUẤN LUYỆN ]",
                        title_align="center",
                        border_style=COLORS["border"],
                        padding=(1, 2)
                    )
                    console.print(results_panel)
            
            console.print("\n[bold green]Huấn luyện hoàn tất! Nhấn Enter để trở về menu chính...[/bold green]")
            input()
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
            if results and "detailed_results" in results:
                dr = results["detailed_results"]
                if dr["success_rate"] is not None:
                    print("\nKết quả huấn luyện:")
                    print(f"  Tỷ lệ thành công: {dr['success_rate']*100:.1f}%")
                    print(f"  Phần thưởng trung bình: {dr['avg_reward']:.2f}")
                    print(f"  Độ dài đường đi TB: {dr['avg_path_length']:.2f}")
                    print(f"  Nhiên liệu tiêu thụ TB: {dr['avg_fuel_consumed']:.2f}")
            
            input("\nHuấn luyện hoàn tất! Nhấn Enter để trở về menu chính...")

def evaluate_model_ui():
    """
    Giao diện đánh giá model đã huấn luyện
    """
    if RICH_AVAILABLE:
        console.print("\n[bold cyan]EVALUATE MODEL - Đánh giá model đã huấn luyện[/bold cyan]")
        
        # Tìm tất cả các model đã huấn luyện
        model_dir = "saved_models"
        all_models = list(Path(model_dir).glob("*.zip"))
        
        if not all_models:
            console.print("[bold red]❌ Không tìm thấy model nào. Vui lòng huấn luyện model trước![/bold red]")
            input("\nNhấn Enter để trở về menu chính...")
            return
        
        # Phân loại model theo kích thước bản đồ
        models_by_size = {}
        for model_path in all_models:
            model_name = model_path.stem
            # Trích xuất kích thước từ tên file
            size_match = None
            for size in [8, 9, 10, 12, 15]:
                if f"size_{size}" in model_name:
                    size_match = size
                    break
            
            if size_match:
                if size_match not in models_by_size:
                    models_by_size[size_match] = []
                models_by_size[size_match].append(model_path)
        
        # Hiển thị danh sách kích thước bản đồ để chọn
        size_options = sorted(models_by_size.keys())
        if not size_options:
            console.print("[bold red]❌ Không thể xác định kích thước bản đồ từ tên các model![/bold red]")
            input("\nNhấn Enter để trở về menu chính...")
            return
        
        size_table = Table(box=box.SIMPLE, show_header=True)
        size_table.add_column("Option", header_style="bold cyan", style=COLORS["menu_number"], justify="center")
        size_table.add_column("Kích thước", header_style="bold cyan", style=COLORS["menu_text"], justify="center")
        size_table.add_column("Số model", header_style="bold cyan", style=COLORS["menu_text"], justify="center")
        
        for idx, size in enumerate(size_options, 1):
            size_table.add_row(str(idx), f"{size}x{size}", str(len(models_by_size[size])))
        
        size_table.add_row("0", "Quay lại", "")
        
        size_panel = Panel(
            size_table,
            title="[ CHỌN KÍCH THƯỚC BẢN ĐỒ ]",
            title_align="center",
            border_style=COLORS["border"],
            padding=(1, 2)
        )
        console.print(size_panel)
        
        size_choice = Prompt.ask(
            "\n[bold cyan]Chọn kích thước bản đồ[/bold cyan]",
            choices=[str(i) for i in range(len(size_options) + 1)],
            default="1"
        )
        
        if size_choice == "0":
            return evaluate_model_ui()  # Quay lại chọn kích thước bản đồ
        
        selected_size = size_options[int(size_choice) - 1]
        selected_models = models_by_size[selected_size]
        
        # Hiển thị danh sách model để chọn
        model_table = Table(box=box.SIMPLE, show_header=True)
        model_table.add_column("Option", header_style="bold cyan", style=COLORS["menu_number"], justify="center")
        model_table.add_column("Tên model", header_style="bold cyan", style=COLORS["menu_text"])
        model_table.add_column("Ngày tạo", header_style="bold cyan", style=COLORS["menu_text"], justify="center")
        model_table.add_column("Loại", header_style="bold cyan", style=COLORS["menu_text"], justify="center")
        
        for idx, model_path in enumerate(selected_models, 1):
            model_name = model_path.stem
            # Trích xuất timestamp từ tên file
            date_str = "N/A"
            if "_20" in model_name:
                date_parts = [part for part in model_name.split("_") if part.startswith("20")]
                if date_parts and len(date_parts) >= 2:
                    try:
                        date_obj = datetime.strptime(f"{date_parts[0]}_{date_parts[1]}", "%Y%m%d_%H%M%S")
                        date_str = date_obj.strftime("%d/%m/%Y %H:%M")
                    except:
                        pass
            
            model_type = "Advanced" if "advanced" in model_name.lower() else "Basic"
            model_table.add_row(str(idx), model_name, date_str, model_type)
        
        model_table.add_row("0", "Quay lại", "", "")
        
        model_panel = Panel(
            model_table,
            title=f"[ MODELS CHO BẢN ĐỒ {selected_size}x{selected_size} ]",
            title_align="center",
            border_style=COLORS["border"],
            padding=(1, 2)
        )
        console.print(model_panel)
        
        model_choice = Prompt.ask(
            "\n[bold cyan]Chọn model để đánh giá[/bold cyan]",
            choices=[str(i) for i in range(len(selected_models) + 1)],
            default="1"
        )
        
        if model_choice == "0":
            return evaluate_model_ui()  # Quay lại chọn kích thước bản đồ
        
        selected_model = selected_models[int(model_choice) - 1]
        
        # Hiển thị các tùy chọn đánh giá
        eval_table = Table(box=box.SIMPLE, show_header=True)
        eval_table.add_column("Option", header_style="bold cyan", style=COLORS["menu_number"], justify="center")
        eval_table.add_column("Loại đánh giá", header_style="bold cyan", style=COLORS["menu_text"])
        eval_table.add_column("Mô tả", header_style="bold cyan", style=COLORS["menu_text"])
        
        eval_table.add_row("1", "Đánh giá nhanh", "Kiểm tra hiệu suất cơ bản trên 3-5 bản đồ")
        eval_table.add_row("2", "Đánh giá chi tiết", "Phân tích sâu trên toàn bộ bản đồ test")
        eval_table.add_row("3", "Đánh giá và Trực quan hóa", "Hiển thị chi tiết và trực quan hóa quyết định của model")
        eval_table.add_row("0", "Quay lại", "")
        
        eval_panel = Panel(
            eval_table,
            title="[ CHỌN LOẠI ĐÁNH GIÁ ]",
            title_align="center",
            border_style=COLORS["border"],
            padding=(1, 2)
        )
        console.print(eval_panel)
        
        eval_choice = Prompt.ask(
            "\n[bold cyan]Chọn loại đánh giá[/bold cyan]",
            choices=["0", "1", "2", "3"],
            default="2"
        )
        
        if eval_choice == "0":
            return evaluate_model_ui()  # Quay lại chọn model
        
        num_episodes = 3  # Mặc định
        
        if eval_choice == "2" or eval_choice == "3":
            num_episodes = int(Prompt.ask(
                "\n[bold cyan]Số episodes đánh giá cho mỗi bản đồ[/bold cyan]",
                default="5"
            ))
        
        # Hiển thị thông tin đánh giá
        evaluation_info = f"""
        Model: [bold green]{selected_model.stem}[/bold green]
        Kích thước bản đồ: [bold green]{selected_size}x{selected_size}[/bold green]
        Số episodes mỗi bản đồ: [bold green]{num_episodes}[/bold green]
        """
        
        info_panel = Panel(
            evaluation_info,
            title="[ THÔNG TIN ĐÁNH GIÁ ]",
            title_align="center",
            border_style=COLORS["border"],
            padding=(1, 2)
        )
        console.print(info_panel)
        
        confirm = Confirm.ask("[bold yellow]Xác nhận đánh giá model này?[/bold yellow]")
        
        if not confirm:
            return evaluate_model_ui()  # Quay lại từ đầu
        
        # Thực hiện đánh giá
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
                f"[bold cyan]Đánh giá model cho bản đồ {selected_size}x{selected_size}[/bold cyan]", 
                total=100,
                status="Đang chuẩn bị..."
            )
            
            progress.update(task, completed=10, status="Tải bản đồ test...")
            time.sleep(0.5)
            
            progress.update(task, completed=20, status="Chuẩn bị môi trường...")
            time.sleep(0.5)
            
            progress.update(task, completed=30, status="Khởi tạo agent từ model...")
            time.sleep(0.5)
            
            progress.update(task, completed=40, status="Đánh giá trên bản đồ test...")
            
            # Chạy đánh giá chi tiết
            detailed_results = detailed_evaluation(str(selected_model).replace(".zip", ""), selected_size)
            
            progress.update(task, completed=90, status="Tổng hợp kết quả...")
            time.sleep(0.5)
            
            progress.update(task, completed=100, status="Hoàn thành!")
        
        # Hiển thị kết quả
        if detailed_results:
            results_table = Table(show_header=True, box=box.SIMPLE)
            results_table.add_column("Chỉ số", style="cyan")
            results_table.add_column("Giá trị", style="green")
            
            results_table.add_row("Tỷ lệ thành công", f"{detailed_results['success_rate']*100:.1f}%")
            results_table.add_row("Phần thưởng trung bình", f"{detailed_results['avg_reward']:.2f}")
            results_table.add_row("Độ dài đường đi TB", f"{detailed_results['avg_path_length']:.2f}")
            results_table.add_row("Nhiên liệu tiêu thụ TB", f"{detailed_results['avg_fuel_consumed']:.2f}")
            results_table.add_row("Chi phí tiêu thụ TB", f"{detailed_results['avg_money_spent']:.2f}")
            
            results_panel = Panel(
                results_table,
                title="[ KẾT QUẢ ĐÁNH GIÁ CHI TIẾT ]",
                title_align="center",
                border_style=COLORS["border"],
                padding=(1, 2)
            )
            console.print(results_panel)
            
            # Hiển thị phân tích lỗi nếu có thất bại
            if detailed_results['success_rate'] < 1.0 and 'dataframe' in detailed_results:
                df = detailed_results['dataframe']
                failure_df = df[df['success'] == False]
                
                if not failure_df.empty and 'termination_reason' in failure_df.columns:
                    failure_reasons = failure_df['termination_reason'].value_counts()
                    
                    failure_table = Table(show_header=True, box=box.SIMPLE)
                    failure_table.add_column("Nguyên nhân", style="red")
                    failure_table.add_column("Số lượng", style="yellow", justify="center")
                    failure_table.add_column("Tỷ lệ", style="yellow", justify="center")
                    
                    total_failures = len(failure_df)
                    for reason, count in failure_reasons.items():
                        percentage = (count / total_failures) * 100
                        failure_table.add_row(reason, str(count), f"{percentage:.1f}%")
                    
                    failure_panel = Panel(
                        failure_table,
                        title="[ PHÂN TÍCH THẤT BẠI ]",
                        title_align="center",
                        border_style="red",
                        padding=(1, 2)
                    )
                    console.print(failure_panel)
            
            # Lưu kết quả đánh giá
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            results_path = os.path.join("evaluation_results", f"eval_{selected_model.stem}_{timestamp}.json")
            
            eval_data = {
                "model_name": selected_model.stem,
                "map_size": selected_size,
                "num_episodes": num_episodes,
                "timestamp": timestamp,
                "success_rate": detailed_results['success_rate'],
                "avg_reward": detailed_results['avg_reward'],
                "avg_path_length": detailed_results['avg_path_length'],
                "avg_fuel_consumed": detailed_results['avg_fuel_consumed'],
                "avg_money_spent": detailed_results['avg_money_spent']
            }
            
            with open(results_path, 'w') as f:
                json.dump(eval_data, f, indent=2)
            
            console.print(f"\n[italic {COLORS['info']}]📝 Đã lưu kết quả đánh giá tại: {results_path}[/italic {COLORS['info']}]")
        else:
            console.print("\n[bold red]❌ Không thể hoàn thành đánh giá. Vui lòng kiểm tra lại model và bản đồ test.[/bold red]")
        
        input("\nNhấn Enter để trở về menu chính...")
    else:
        # Phiên bản đơn giản cho terminal không hỗ trợ rich
        print("\nEVALUATE MODEL - Đánh giá model đã huấn luyện")
        
        # Tìm tất cả các model đã huấn luyện
        model_dir = "saved_models"
        all_models = list(Path(model_dir).glob("*.zip"))
        
        if not all_models:
            print("❌ Không tìm thấy model nào. Vui lòng huấn luyện model trước!")
            input("\nNhấn Enter để trở về menu chính...")
            return
        
        # Phân loại model theo kích thước bản đồ
        models_by_size = {}
        for model_path in all_models:
            model_name = model_path.stem
            # Trích xuất kích thước từ tên file
            size_match = None
            for size in [8, 9, 10, 12, 15]:
                if f"size_{size}" in model_name:
                    size_match = size
                    break
            
            if size_match:
                if size_match not in models_by_size:
                    models_by_size[size_match] = []
                models_by_size[size_match].append(model_path)
        
        # Hiển thị danh sách kích thước bản đồ để chọn
        size_options = sorted(models_by_size.keys())
        if not size_options:
            print("❌ Không thể xác định kích thước bản đồ từ tên các model!")
            input("\nNhấn Enter để trở về menu chính...")
            return
        
        print("\nDanh sách kích thước bản đồ có model:")
        for idx, size in enumerate(size_options, 1):
            print(f"  [{idx}] {size}x{size} - {len(models_by_size[size])} model")
        print("  [0] Quay lại")
        
        size_choice = input("\nChọn kích thước bản đồ [0-{}]: ".format(len(size_options)))
        
        if size_choice == "0" or not size_choice.isdigit() or int(size_choice) < 1 or int(size_choice) > len(size_options):
            return
        
        selected_size = size_options[int(size_choice) - 1]
        selected_models = models_by_size[selected_size]
        
        # Hiển thị danh sách model để chọn
        print(f"\nDanh sách model cho bản đồ {selected_size}x{selected_size}:")
        for idx, model_path in enumerate(selected_models, 1):
            model_name = model_path.stem
            model_type = "Advanced" if "advanced" in model_name.lower() else "Basic"
            print(f"  [{idx}] {model_name} - {model_type}")
        print("  [0] Quay lại")
        
        model_choice = input("\nChọn model để đánh giá [0-{}]: ".format(len(selected_models)))
        
        if model_choice == "0" or not model_choice.isdigit() or int(model_choice) < 1 or int(model_choice) > len(selected_models):
            return evaluate_model_ui()  # Quay lại chọn kích thước bản đồ
        
        selected_model = selected_models[int(model_choice) - 1]
        
        # Hiển thị các tùy chọn đánh giá
        print("\nChọn loại đánh giá:")
        print("  [1] Đánh giá nhanh - Kiểm tra hiệu suất cơ bản trên 3-5 bản đồ")
        print("  [2] Đánh giá chi tiết - Phân tích sâu trên toàn bộ bản đồ test")
        print("  [3] Đánh giá và Trực quan hóa - Hiển thị chi tiết và trực quan hóa quyết định của model")
        print("  [0] Quay lại")
        
        eval_choice = input("\nChọn loại đánh giá [0-3]: ")
        
        if eval_choice == "0" or not eval_choice.isdigit() or int(eval_choice) < 1 or int(eval_choice) > 3:
            return evaluate_model_ui()  # Quay lại chọn model
        
        num_episodes = 3  # Mặc định
        
        if eval_choice == "2" or eval_choice == "3":
            num_episodes_input = input("\nSố episodes đánh giá cho mỗi bản đồ [mặc định: 5]: ")
            if num_episodes_input.isdigit() and int(num_episodes_input) > 0:
                num_episodes = int(num_episodes_input)
            else:
                num_episodes = 5
        
        # Hiển thị thông tin đánh giá
        print("\nTHÔNG TIN ĐÁNH GIÁ:")
        print(f"  Model: {selected_model.stem}")
        print(f"  Kích thước bản đồ: {selected_size}x{selected_size}")
        print(f"  Số episodes mỗi bản đồ: {num_episodes}")
        
        confirm = input("\nXác nhận đánh giá model này? (y/n): ").lower() == 'y'
        
        if not confirm:
            return evaluate_model_ui()  # Quay lại từ đầu
        
        # Thực hiện đánh giá
        print("\n🔍 Bắt đầu đánh giá model...")
        print("  ⏳ Đang chuẩn bị...")
        
        # Chạy đánh giá chi tiết
        detailed_results = detailed_evaluation(str(selected_model).replace(".zip", ""), selected_size)
        
        # Hiển thị kết quả
        if detailed_results:
            print("\nKẾT QUẢ ĐÁNH GIÁ CHI TIẾT:")
            print(f"  Tỷ lệ thành công: {detailed_results['success_rate']*100:.1f}%")
            print(f"  Phần thưởng trung bình: {detailed_results['avg_reward']:.2f}")
            print(f"  Độ dài đường đi TB: {detailed_results['avg_path_length']:.2f}")
            print(f"  Nhiên liệu tiêu thụ TB: {detailed_results['avg_fuel_consumed']:.2f}")
            print(f"  Chi phí tiêu thụ TB: {detailed_results['avg_money_spent']:.2f}")
            
            # Hiển thị phân tích lỗi nếu có thất bại
            if detailed_results['success_rate'] < 1.0 and 'dataframe' in detailed_results:
                df = detailed_results['dataframe']
                failure_df = df[df['success'] == False]
                
                if not failure_df.empty and 'termination_reason' in failure_df.columns:
                    failure_reasons = failure_df['termination_reason'].value_counts()
                    
                    print("\nPHÂN TÍCH THẤT BẠI:")
                    total_failures = len(failure_df)
                    for reason, count in failure_reasons.items():
                        percentage = (count / total_failures) * 100
                        print(f"  {reason}: {count} lần ({percentage:.1f}%)")
            
            # Lưu kết quả đánh giá
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            results_path = os.path.join("evaluation_results", f"eval_{selected_model.stem}_{timestamp}.json")
            
            eval_data = {
                "model_name": selected_model.stem,
                "map_size": selected_size,
                "num_episodes": num_episodes,
                "timestamp": timestamp,
                "success_rate": detailed_results['success_rate'],
                "avg_reward": detailed_results['avg_reward'],
                "avg_path_length": detailed_results['avg_path_length'],
                "avg_fuel_consumed": detailed_results['avg_fuel_consumed'],
                "avg_money_spent": detailed_results['avg_money_spent']
            }
            
            with open(results_path, 'w') as f:
                json.dump(eval_data, f, indent=2)
            
            print(f"\n📝 Đã lưu kết quả đánh giá tại: {results_path}")
        else:
            print("\n❌ Không thể hoàn thành đánh giá. Vui lòng kiểm tra lại model và bản đồ test.")
        
        input("\nNhấn Enter để trở về menu chính...")

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
                run_training_pipeline(
                    map_size=8,
                    num_maps=5,
                    training_steps=30000,
                    use_advanced=False,
                    render=False
                )
                
                if RICH_AVAILABLE:
                    console.print("\n[bold green]Huấn luyện hoàn tất! Nhấn Enter để tiếp tục...[/bold green]")
                else:
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
                run_training_pipeline(
                    map_size=map_size,
                    num_maps=num_maps,
                    training_steps=training_steps,
                    use_advanced=True,
                    render=False
                )
                
                if RICH_AVAILABLE:
                    console.print("\n[bold green]Huấn luyện hoàn tất! Nhấn Enter để tiếp tục...[/bold green]")
                else:
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