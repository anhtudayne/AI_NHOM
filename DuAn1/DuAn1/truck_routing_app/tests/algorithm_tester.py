"""
Hệ thống kiểm thử thuật toán tìm đường với giao diện đồ họa.
Ứng dụng này cho phép bạn kiểm thử các thuật toán trên các bản đồ ngẫu nhiên
và hiển thị kết quả trực quan.
"""

import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox, filedialog
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np
import time
import os
import sys
import threading
import traceback
from typing import List, Tuple, Dict, Any

# Thêm thư mục gốc vào đường dẫn để tìm module core
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import các thuật toán
from core.map import Map
from core.algorithms.bfs import BFS
from core.algorithms.dfs import DFS
from core.algorithms.astar import AStar
from core.algorithms.greedy import GreedySearch
from core.algorithms.simulated_annealing import SimulatedAnnealing
from core.algorithms.genetic_algorithm import GeneticAlgorithm
from core.algorithms.local_beam import LocalBeamSearch

# Hàm hỗ trợ tạo thư mục đầu ra
def create_output_dir():
    """Tạo thư mục output nếu chưa có."""
    output_dir = os.path.join(os.path.dirname(__file__), 'output')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    return output_dir

# Lớp trình diễn và hiển thị kết quả
class ResultVisualizer:
    """Các phương thức để trực quan hóa bản đồ và kết quả thuật toán."""
    
    @staticmethod
    def visualize_map(map_obj, algorithm=None, path=None, visited=None, ax=None, save_filename=None):
        """
        Trực quan hóa bản đồ với thuật toán và kết quả tìm đường nếu có.
        
        Args:
            map_obj: Đối tượng bản đồ cần hiển thị
            algorithm: (Tùy chọn) thuật toán đã chạy trên bản đồ
            path: (Tùy chọn) đường đi đã tìm được
            visited: (Tùy chọn) danh sách các ô đã thăm
            ax: (Tùy chọn) trục matplotlib để vẽ
            save_filename: (Tùy chọn) tên file để lưu hình
        """
        # Tạo axes nếu chưa có
        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 8))
            standalone = True
        else:
            standalone = False
        
        # Tạo bảng màu cho bản đồ
        grid = map_obj.grid
        size = map_obj.size
        color_map = np.zeros((size, size, 3))
        
        # Định nghĩa màu cho các loại ô
        road_color = [0.95, 0.95, 0.95]    # Xám nhạt cho đường
        toll_color = [0.9, 0.4, 0.4]       # Đỏ cho trạm thu phí
        gas_color = [0.4, 0.8, 0.4]        # Xanh lá cho trạm xăng
        brick_color = [0.5, 0.5, 0.5]      # Xám đậm cho vật cản
        
        # Tô màu các ô dựa vào loại
        for i in range(size):
            for j in range(size):
                if grid[j, i] == 0:      # Đường thường
                    color_map[j, i] = road_color
                elif grid[j, i] == 1:    # Trạm thu phí
                    color_map[j, i] = toll_color
                elif grid[j, i] == 2:    # Trạm xăng
                    color_map[j, i] = gas_color
                elif grid[j, i] == -1:    # Vật cản
                    color_map[j, i] = brick_color
        
        # Vẽ bản đồ
        ax.imshow(color_map, origin='upper')
        
        # Vẽ đường kẻ lưới
        for i in range(size + 1):
            ax.axhline(i - 0.5, color='black', linewidth=0.5)
            ax.axvline(i - 0.5, color='black', linewidth=0.5)
        
        # Vẽ các ô đã thăm
        if visited:
            x_visited = [pos[0] for pos in visited]
            y_visited = [pos[1] for pos in visited]
            ax.scatter(x_visited, y_visited, c='lightskyblue', s=15, alpha=0.5)
        
        # Vẽ đường đi
        if path:
            x_path = [pos[0] for pos in path]
            y_path = [pos[1] for pos in path]
            ax.plot(x_path, y_path, c='blue', linewidth=2, alpha=0.7)
        
        # Vẽ điểm bắt đầu và kết thúc
        if map_obj.start_pos:
            ax.plot(map_obj.start_pos[0], map_obj.start_pos[1], 'go', markersize=10)
        if map_obj.end_pos:
            ax.plot(map_obj.end_pos[0], map_obj.end_pos[1], 'ro', markersize=10)
        
        # Thêm chú thích cho các loại ô
        ax.text(0.02, 0.02, 'Đường', transform=ax.transAxes, color='black', fontsize=8)
        ax.text(0.02, 0.05, 'Trạm thu phí', transform=ax.transAxes, color='red', fontsize=8)
        ax.text(0.02, 0.08, 'Trạm xăng', transform=ax.transAxes, color='green', fontsize=8)
        ax.text(0.02, 0.11, 'Vật cản', transform=ax.transAxes, color='dimgray', fontsize=8)
        
        # Thêm thông tin về thuật toán nếu có
        if algorithm:
            title = f"Bản đồ {size}x{size}"
            if hasattr(algorithm, '__class__'):
                title += f" - {algorithm.__class__.__name__}"
            if path:
                toll_cost = getattr(algorithm, "current_toll_cost", 0)
                fuel_cost = getattr(algorithm, "current_fuel_cost", 0)
                title += f"\nĐộ dài: {len(path) - 1}, Chi phí: {algorithm.cost:.2f} (Xăng: {fuel_cost:.2f}, Phí: {toll_cost:.2f}), Bước: {algorithm.steps}"
            ax.set_title(title, fontsize=10)
        else:
            ax.set_title(f"Bản đồ {size}x{size}", fontsize=10)
        
        # Lưu hình nếu cần
        if save_filename and standalone:
            output_dir = create_output_dir()
            plt.savefig(os.path.join(output_dir, save_filename), dpi=150, bbox_inches='tight')
            plt.close()
        elif standalone:
            plt.tight_layout()
            plt.show()
        
        return ax
    
    @staticmethod
    def plot_performance_comparison(results, metric='execution_time', title='So sánh hiệu suất', ax=None, save_filename=None):
        """
        Tạo biểu đồ cột so sánh hiệu suất các thuật toán.
        
        Args:
            results: Danh sách kết quả của các thuật toán
            metric: Chỉ số để so sánh ('execution_time', 'path_length', 'cost', 'fuel_cost', 'toll_cost', 'fuel_consumed', 'money_remaining', hoặc 'steps')
            title: Tiêu đề biểu đồ
            ax: (Tùy chọn) trục matplotlib để vẽ
            save_filename: (Tùy chọn) tên file để lưu hình
        """
        if not results:
            return None
            
        # Tạo axes nếu chưa có
        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 5))
            standalone = True
        else:
            standalone = False
        
        # Bảo đảm các kết quả có metric cần thiết
        valid_results = [r for r in results if metric in r or (metric in ['fuel_cost', 'toll_cost', 'money_remaining'] and 'cost' in r)]
        
        if not valid_results:
            ax.clear()
            ax.text(0.5, 0.5, f"Không có dữ liệu cho metric: {metric}", ha='center', va='center', fontsize=12)
            if standalone:
                plt.tight_layout()
                plt.show()
            return ax
        
        algorithms = [r['algorithm'] for r in valid_results]
        values = []
        
        # Lấy giá trị metric
        for r in valid_results:
            if metric in r:
                values.append(r[metric])
            elif metric == 'fuel_cost' and 'cost' in r:
                values.append(r.get('fuel_cost', r['cost'] * 0.7))  # Ước tính nếu không có
            elif metric == 'toll_cost' and 'cost' in r:
                values.append(r.get('toll_cost', r['cost'] * 0.3))  # Ước tính nếu không có
            elif metric == 'money_remaining' and 'money_remaining' in r:
                values.append(r['money_remaining'])
            else:
                values.append(0)
        
        # Tạo biểu đồ cột
        bars = ax.bar(algorithms, values)
        
        # Thêm giá trị trên các cột
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.01 * max(values) if values else 0,
                   f'{height:.2f}', ha='center', va='bottom', fontsize=8)
        
        ax.set_title(title, fontsize=10)
        ax.set_xlabel('Thuật toán', fontsize=9)
        
        # Đặt nhãn trục y phù hợp
        if metric == 'execution_time':
            ax.set_ylabel('Thời gian (giây)', fontsize=9)
        elif metric == 'path_length':
            ax.set_ylabel('Độ dài đường đi', fontsize=9)
        elif metric == 'cost':
            ax.set_ylabel('Chi phí đường đi', fontsize=9)
        elif metric == 'fuel_cost':
            ax.set_ylabel('Chi phí nhiên liệu', fontsize=9)
        elif metric == 'toll_cost':
            ax.set_ylabel('Chi phí trạm thu phí', fontsize=9)
        elif metric == 'fuel_consumed':
            ax.set_ylabel('Nhiên liệu tiêu thụ (lít)', fontsize=9)
        elif metric == 'money_remaining':
            ax.set_ylabel('Số tiền còn lại (đ)', fontsize=9)
            # Đảo ngược màu để số tiền càng cao càng tốt (xanh)
            for i, bar in enumerate(bars):
                bar.set_color('green')
        elif metric == 'steps':
            ax.set_ylabel('Số bước thuật toán', fontsize=9)
        
        ax.tick_params(axis='x', rotation=45, labelsize=8)
        ax.tick_params(axis='y', labelsize=8)
        
        # Lưu hình nếu cần
        if save_filename and standalone:
            output_dir = create_output_dir()
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, save_filename), dpi=150, bbox_inches='tight')
            plt.close()
        elif standalone:
            plt.tight_layout()
            plt.show()
        
        return ax
    
    @staticmethod
    def plot_trend_analysis(results, x_key, y_key, title, ax=None, save_filename=None):
        """
        Tạo biểu đồ đường phân tích xu hướng trong kết quả.
        
        Args:
            results: Danh sách kết quả
            x_key: Khóa dùng cho trục x
            y_key: Khóa dùng cho trục y
            title: Tiêu đề biểu đồ
            ax: (Tùy chọn) trục matplotlib để vẽ
            save_filename: (Tùy chọn) tên file để lưu hình
        """
        if not results:
            return None
            
        # Tạo axes nếu chưa có
        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 5))
            standalone = True
        else:
            standalone = False
        
        # Tổng hợp dữ liệu theo x_key
        x_values = sorted(set(r[x_key] for r in results))
        y_values = []
        
        for x in x_values:
            matching_results = [r[y_key] for r in results if r[x_key] == x]
            y_values.append(sum(matching_results) / len(matching_results) if matching_results else 0)
        
        # Vẽ biểu đồ đường
        ax.plot(x_values, y_values, 'o-', linewidth=2)
        
        # Thêm điểm dữ liệu
        for i, (x, y) in enumerate(zip(x_values, y_values)):
            ax.text(x, y + 0.01 * max(y_values) if y_values else 0, f'{y:.2f}', ha='center', fontsize=8)
        
        ax.set_title(title, fontsize=10)
        ax.set_xlabel(x_key.replace('_', ' ').title(), fontsize=9)
        ax.set_ylabel(y_key.replace('_', ' ').title(), fontsize=9)
        ax.grid(True, linestyle='--', alpha=0.7)
        ax.tick_params(axis='both', labelsize=8)
        
        # Lưu hình nếu cần
        if save_filename and standalone:
            output_dir = create_output_dir()
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, save_filename), dpi=150, bbox_inches='tight')
            plt.close()
        elif standalone:
            plt.tight_layout()
            plt.show()
        
        return ax

# Lớp AlgorithmTester chứa logic kiểm thử
class AlgorithmTester:
    """Quản lý việc kiểm thử thuật toán trên bản đồ ngẫu nhiên."""
    
    def __init__(self, log_callback=None):
        """Khởi tạo trình kiểm thử."""
        self.log_callback = log_callback or (lambda msg: print(msg))
        self.maps = []
        self.results = []
        self.initial_money = None
        self.max_fuel = None
        self.initial_fuel = None
        self.fuel_per_move = None
        self.gas_station_cost = None
        self.toll_base_cost = None
    
    def log(self, message):
        """Ghi log thông báo."""
        if self.log_callback:
            self.log_callback(message)
    
    def generate_test_maps(self, configs):
        """
        Tạo bản đồ kiểm thử dựa trên cấu hình.
        
        Args:
            configs: Danh sách cấu hình bản đồ
        
        Returns:
            List[Map]: Danh sách các bản đồ đã tạo
        """
        self.maps = []
        
        for config in configs:
            self.log(f"Đang tạo bản đồ kích thước {config['size']}x{config['size']}...")
            map_obj = Map.generate_random(
                config["size"], 
                config["toll_ratio"], 
                config["gas_ratio"], 
                config["brick_ratio"]
            )
            
            # Chọn điểm bắt đầu và kết thúc
            positions = [(i, j) for i in range(map_obj.size) for j in range(map_obj.size) 
                        if map_obj.grid[j, i] == 0]  # Chỉ xét các ô đường thường
            if len(positions) >= 2:
                # Chọn các vị trí cách xa nhau
                positions.sort(key=lambda pos: pos[0] + pos[1])
                map_obj.start_pos = positions[0]  # Vị trí gần góc trên trái
                map_obj.end_pos = positions[-1]  # Vị trí gần góc dưới phải
                self.maps.append(map_obj)
                self.log(f"Đã tạo bản đồ: Bắt đầu={map_obj.start_pos}, Kết thúc={map_obj.end_pos}")
            else:
                self.log("Không thể tạo bản đồ, không đủ ô đường.")
        
        return self.maps
    
    def create_algorithm(self, algorithm_name, grid):
        """
        Tạo đối tượng thuật toán dựa trên tên.
        
        Args:
            algorithm_name: Tên thuật toán
            grid: Lưới bản đồ
            
        Returns:
            Đối tượng thuật toán
        """
        # Get custom parameters if set
        initial_money = getattr(self, 'initial_money', None)
        max_fuel = getattr(self, 'max_fuel', None)
        initial_fuel = getattr(self, 'initial_fuel', None)
        fuel_per_move = getattr(self, 'fuel_per_move', None)
        gas_station_cost = getattr(self, 'gas_station_cost', None)
        toll_base_cost = getattr(self, 'toll_base_cost', None)
        
        if algorithm_name == 'BFS':
            return BFS(grid, initial_money, max_fuel, fuel_per_move, gas_station_cost, toll_base_cost, initial_fuel)
        elif algorithm_name == 'DFS':
            return DFS(grid, initial_money, max_fuel, fuel_per_move, gas_station_cost, toll_base_cost, initial_fuel)
        elif algorithm_name == 'AStar':
            return AStar(grid, initial_money, max_fuel, fuel_per_move, gas_station_cost, toll_base_cost, initial_fuel)
        elif algorithm_name == 'Greedy':
            return GreedySearch(grid, initial_money, max_fuel, fuel_per_move, gas_station_cost, toll_base_cost, initial_fuel)
        elif algorithm_name == 'LocalBeam':
            return LocalBeamSearch(grid, beam_width=5, initial_money=initial_money, max_fuel=max_fuel, 
                                  fuel_per_move=fuel_per_move, gas_station_cost=gas_station_cost, 
                                  toll_base_cost=toll_base_cost, initial_fuel=initial_fuel)
        elif algorithm_name == 'SimulatedAnnealing':
            return SimulatedAnnealing(grid, initial_temperature=100, cooling_rate=0.95, steps_per_temp=20, 
                                     initial_money=initial_money, max_fuel=max_fuel, fuel_per_move=fuel_per_move, 
                                     gas_station_cost=gas_station_cost, toll_base_cost=toll_base_cost, 
                                     initial_fuel=initial_fuel)
        elif algorithm_name == 'GeneticAlgorithm':
            return GeneticAlgorithm(grid, population_size=20, crossover_rate=0.7, mutation_rate=0.3, generations=50, 
                                   initial_money=initial_money, max_fuel=max_fuel, fuel_per_move=fuel_per_move, 
                                   gas_station_cost=gas_station_cost, toll_base_cost=toll_base_cost, 
                                   initial_fuel=initial_fuel)
        else:
            raise ValueError(f"Thuật toán không hỗ trợ: {algorithm_name}")
    
    def test_algorithm(self, algorithm_name, map_obj):
        """
        Kiểm thử một thuật toán trên một bản đồ.
        
        Args:
            algorithm_name: Tên thuật toán
            map_obj: Đối tượng bản đồ
            
        Returns:
            Dict: Kết quả kiểm thử
        """
        self.log(f"Đang chạy {algorithm_name}...")
        
        # Khởi tạo biến start_time ở đây để đảm bảo nó luôn được định nghĩa
        start_time = time.time()
        
        try:
            # Tạo đối tượng thuật toán
            algorithm = self.create_algorithm(algorithm_name, map_obj.grid)
            
            # Đo thời gian thực thi
            path = algorithm.search(map_obj.start_pos, map_obj.end_pos)
            execution_time = time.time() - start_time
            
            # Thu thập kết quả
            result = {
                "algorithm": algorithm_name,
                "path_length": len(path) - 1 if path else 0,
                "path_found": len(path) > 0,
                "cost": algorithm.cost,
                "steps": algorithm.steps,
                "execution_time": execution_time,
                "path": path,
                "visited": algorithm.visited
            }
            
            # Phân tích chi phí và nhiên liệu chi tiết nếu có thể
            toll_cost = getattr(algorithm, "current_toll_cost", 0)
            fuel_cost = getattr(algorithm, "current_fuel_cost", 0)
            fuel_consumed = getattr(algorithm, "fuel_consumed", 0)
            fuel_refilled = getattr(algorithm, "fuel_refilled", 0)
            fuel_remaining = getattr(algorithm, "current_fuel", 0)
            money_remaining = getattr(algorithm, "current_money", 0)
            
            # Thêm thông tin chi tiết vào kết quả
            result["toll_cost"] = toll_cost
            result["fuel_cost"] = fuel_cost
            result["fuel_consumed"] = fuel_consumed
            result["fuel_refilled"] = fuel_refilled
            result["fuel_remaining"] = fuel_remaining
            result["money_remaining"] = money_remaining
            
            # In kết quả
            if path:
                self.log(f"  ✓ Tìm thấy đường đi với độ dài {result['path_length']}")
                self.log(f"  ✓ Chi phí tổng: {result['cost']:.2f}")
                self.log(f"    - Chi phí nhiên liệu: {fuel_cost:.2f}")
                self.log(f"    - Chi phí trạm thu phí: {toll_cost:.2f}")
                self.log(f"  ✓ Thông tin nhiên liệu:")
                self.log(f"    - Tiêu thụ: {fuel_consumed:.2f} lít")
                self.log(f"    - Đổ thêm tại trạm xăng: {fuel_refilled:.2f} lít")
                self.log(f"    - Còn lại: {fuel_remaining:.2f} lít")
                self.log(f"  ✓ Số tiền:")
                initial_money = self.initial_money if self.initial_money is not None else algorithm.MAX_MONEY
                self.log(f"    - Ban đầu: {initial_money:.2f} đ")
                self.log(f"    - Còn lại: {money_remaining:.2f} đ")
                self.log(f"    - Chi tiêu: {initial_money - money_remaining:.2f} đ") 
                self.log(f"  ✓ Số bước thuật toán: {result['steps']}")
                self.log(f"  ✓ Thời gian: {result['execution_time']:.4f} giây")
            else:
                self.log(f"  ✗ Không tìm thấy đường đi sau {result['steps']} bước")
                self.log(f"  ✗ Thời gian: {result['execution_time']:.4f} giây")
            
            return result, algorithm
        except Exception as e:
            # Ghi log lỗi chi tiết
            self.log(f"  ✗ Lỗi khi chạy {algorithm_name}: {str(e)}")
            self.log(f"  ✗ Chi tiết lỗi: {traceback.format_exc()}")
            
            # Trả về kết quả lỗi
            result = {
                "algorithm": algorithm_name,
                "error": str(e),
                "path_found": False,
                "execution_time": time.time() - start_time,
                "path": [],
                "visited": []
            }
            return result, None
    
    def compare_algorithms(self, algorithm_names, map_obj):
        """
        So sánh nhiều thuật toán trên cùng một bản đồ.
        
        Args:
            algorithm_names: Danh sách tên thuật toán
            map_obj: Đối tượng bản đồ
            
        Returns:
            List[Dict]: Danh sách kết quả
        """
        results = []
        
        for algorithm_name in algorithm_names:
            result, _ = self.test_algorithm(algorithm_name, map_obj)
            if "error" not in result:  # Chỉ thêm kết quả không có lỗi
                results.append(result)
        
        # So sánh kết quả
        self.log("\nTổng kết:")
        
        # Tốt nhất về độ dài
        results_with_path = [r for r in results if r["path_found"]]
        if results_with_path:
            best_path = min(results_with_path, key=lambda x: x["path_length"])
            self.log(f"  Tốt nhất về độ dài: {best_path['algorithm']} ({best_path['path_length']} bước)")
        
        # Tốt nhất về chi phí
        if results_with_path:
            best_cost = min(results_with_path, key=lambda x: x["cost"])
            self.log(f"  Tốt nhất về chi phí: {best_cost['algorithm']} ({best_cost['cost']:.2f})")
            self.log(f"    - Chi phí nhiên liệu: {best_cost.get('fuel_cost', 0):.2f}")
            self.log(f"    - Chi phí trạm thu phí: {best_cost.get('toll_cost', 0):.2f}")
            
        # Tốt nhất về nhiên liệu tiêu thụ
        if results_with_path:
            best_fuel = min(results_with_path, key=lambda x: x.get("fuel_consumed", float('inf')))
            if "fuel_consumed" in best_fuel:
                self.log(f"  Tiết kiệm nhiên liệu nhất: {best_fuel['algorithm']} ({best_fuel['fuel_consumed']:.2f} lít)")
                self.log(f"    - Đổ thêm: {best_fuel.get('fuel_refilled', 0):.2f} lít")
                self.log(f"    - Còn lại: {best_fuel.get('fuel_remaining', 0):.2f} lít")
        
        # Tốt nhất về tiết kiệm tiền
        if results_with_path:
            # Lọc kết quả có thông tin về tiền
            results_with_money = [r for r in results_with_path if "money_remaining" in r]
            if results_with_money:
                best_money = max(results_with_money, key=lambda x: x["money_remaining"])
                initial_money = self.initial_money if self.initial_money is not None else 2000.0
                money_spent = initial_money - best_money["money_remaining"]
                self.log(f"  Tiết kiệm tiền nhất: {best_money['algorithm']} (Còn lại: {best_money['money_remaining']:.2f} đ)")
                self.log(f"    - Chi tiêu: {money_spent:.2f} đ")
                self.log(f"    - Tỷ lệ tiết kiệm: {(best_money['money_remaining']/initial_money)*100:.1f}%")
        
        # Nhanh nhất về thời gian
        fastest = min(results, key=lambda x: x["execution_time"])
        self.log(f"  Nhanh nhất: {fastest['algorithm']} ({fastest['execution_time']:.4f} giây)")
        
        # Hiệu quả nhất (tỷ lệ bước:độ dài)
        if results_with_path:
            most_efficient = min(results_with_path, 
                              key=lambda x: x["steps"]/max(1, x["path_length"]))
            self.log(f"  Hiệu quả nhất: {most_efficient['algorithm']} " 
                   f"({most_efficient['steps']}/{most_efficient['path_length']} = "
                   f"{most_efficient['steps']/max(1, most_efficient['path_length']):.2f} bước/đơn vị đường)")
        
        return results

# Lớp giao diện đồ họa chính
class AlgorithmTesterApp:
    """Giao diện đồ họa để kiểm thử thuật toán."""
    
    def __init__(self, root):
        self.root = root
        self.root.title("Hệ thống kiểm thử thuật toán tìm đường")
        self.root.geometry("1200x800")
        
        # Tạo đối tượng kiểm thử
        self.tester = AlgorithmTester(self.log)
        
        # Tạo các thành phần GUI
        self.create_widgets()
        
        # Cấu hình bản đồ mặc định
        self.map_configs = [
            {"size": 10, "toll_ratio": 0.05, "gas_ratio": 0.05, "brick_ratio": 0.15}
        ]
        
        # Chế độ chạy mặc định
        self.current_map_index = 0
        self.current_algorithm = "AStar"
        
    def create_widgets(self):
        """Tạo các thành phần giao diện."""
        # Tạo khung chính
        main_frame = ttk.Frame(self.root, padding=10)
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Chia làm 2 phần: control panel và visualization panel
        control_frame = ttk.Frame(main_frame, padding=5, borderwidth=1, relief=tk.GROOVE)
        control_frame.pack(side=tk.LEFT, fill=tk.Y, padx=5, pady=5)
        
        viz_frame = ttk.Frame(main_frame, padding=5)
        viz_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # === PANEL ĐIỀU KHIỂN ===
        # Phần cấu hình bản đồ
        map_frame = ttk.LabelFrame(control_frame, text="Cấu hình bản đồ", padding=5)
        map_frame.pack(fill=tk.X, padx=5, pady=5)
        
        # Kích thước bản đồ
        ttk.Label(map_frame, text="Kích thước:").grid(row=0, column=0, sticky=tk.W)
        self.size_var = tk.StringVar(value="10")
        ttk.Combobox(map_frame, textvariable=self.size_var, values=["8", "10", "12", "15"], width=5).grid(row=0, column=1, sticky=tk.W, padx=5, pady=2)
        
        # Tỷ lệ trạm thu phí
        ttk.Label(map_frame, text="Tỷ lệ trạm thu phí:").grid(row=1, column=0, sticky=tk.W)
        self.toll_var = tk.StringVar(value="0.05")
        ttk.Combobox(map_frame, textvariable=self.toll_var, values=["0.03", "0.05", "0.08", "0.1"], width=5).grid(row=1, column=1, sticky=tk.W, padx=5, pady=2)
        
        # Tỷ lệ trạm xăng
        ttk.Label(map_frame, text="Tỷ lệ trạm xăng:").grid(row=2, column=0, sticky=tk.W)
        self.gas_var = tk.StringVar(value="0.05")
        ttk.Combobox(map_frame, textvariable=self.gas_var, values=["0.03", "0.05", "0.08", "0.1"], width=5).grid(row=2, column=1, sticky=tk.W, padx=5, pady=2)
        
        # Tỷ lệ vật cản
        ttk.Label(map_frame, text="Tỷ lệ vật cản:").grid(row=3, column=0, sticky=tk.W)
        self.brick_var = tk.StringVar(value="0.15")
        ttk.Combobox(map_frame, textvariable=self.brick_var, values=["0.1", "0.15", "0.2", "0.25"], width=5).grid(row=3, column=1, sticky=tk.W, padx=5, pady=2)
        
        # Thêm tuỳ chọn chọn kiểu bản đồ
        ttk.Label(map_frame, text="Kiểu bản đồ:").grid(row=4, column=0, sticky=tk.W)
        self.map_type_var = tk.StringVar(value="random")
        ttk.Radiobutton(map_frame, text="Ngẫu nhiên", value="random", variable=self.map_type_var).grid(row=4, column=1, sticky=tk.W, padx=0, pady=1)
        ttk.Radiobutton(map_frame, text="Mặc định", value="default", variable=self.map_type_var).grid(row=5, column=1, sticky=tk.W, padx=0, pady=1)
        
        # Khung cấu hình xe và chi phí
        vehicle_frame = ttk.LabelFrame(control_frame, text="Cấu hình xe và chi phí", padding=5)
        vehicle_frame.pack(fill=tk.X, padx=5, pady=5)
        
        # Thêm cấu hình dung tích bình xăng
        ttk.Label(vehicle_frame, text="Dung tích bình xăng (L):").grid(row=0, column=0, sticky=tk.W)
        self.max_fuel_var = tk.StringVar(value="20.0")
        ttk.Entry(vehicle_frame, textvariable=self.max_fuel_var, width=8).grid(row=0, column=1, sticky=tk.W, padx=5, pady=2)
        
        # Thêm cấu hình nhiên liệu ban đầu
        ttk.Label(vehicle_frame, text="Nhiên liệu ban đầu (L):").grid(row=1, column=0, sticky=tk.W)
        self.initial_fuel_var = tk.StringVar(value="20.0")
        ttk.Entry(vehicle_frame, textvariable=self.initial_fuel_var, width=8).grid(row=1, column=1, sticky=tk.W, padx=5, pady=2)
        
        # Thêm cấu hình nhiên liệu tiêu thụ 1 ô
        ttk.Label(vehicle_frame, text="Nhiên liệu/ô (L):").grid(row=2, column=0, sticky=tk.W)
        self.fuel_per_move_var = tk.StringVar(value="0.4")
        ttk.Entry(vehicle_frame, textvariable=self.fuel_per_move_var, width=8).grid(row=2, column=1, sticky=tk.W, padx=5, pady=2)
        
        # Thêm cấu hình giá xăng 1L
        ttk.Label(vehicle_frame, text="Giá xăng (đ/L):").grid(row=3, column=0, sticky=tk.W)
        self.gas_cost_var = tk.StringVar(value="30.0")
        ttk.Entry(vehicle_frame, textvariable=self.gas_cost_var, width=8).grid(row=3, column=1, sticky=tk.W, padx=5, pady=2)
        
        # Thêm cấu hình phí trạm thu phí
        ttk.Label(vehicle_frame, text="Phí trạm thu phí (đ):").grid(row=4, column=0, sticky=tk.W)
        self.toll_cost_var = tk.StringVar(value="50.0")
        ttk.Entry(vehicle_frame, textvariable=self.toll_cost_var, width=8).grid(row=4, column=1, sticky=tk.W, padx=5, pady=2)
        
        # Thêm cấu hình số tiền ban đầu
        ttk.Label(vehicle_frame, text="Tiền ban đầu (đ):").grid(row=5, column=0, sticky=tk.W)
        self.money_var = tk.StringVar(value="2000.0")
        ttk.Entry(vehicle_frame, textvariable=self.money_var, width=8).grid(row=5, column=1, sticky=tk.W, padx=5, pady=2)
        
        # Nút tạo bản đồ
        ttk.Button(map_frame, text="Tạo bản đồ", command=self.generate_map).grid(row=7, column=0, columnspan=2, pady=5)
        
        # Phần chọn thuật toán
        alg_frame = ttk.LabelFrame(control_frame, text="Thuật toán", padding=5)
        alg_frame.pack(fill=tk.X, padx=5, pady=5)
        
        self.algorithm_var = tk.StringVar(value="AStar")
        algorithms = ["BFS", "DFS", "AStar", "Greedy", "LocalBeam", "SimulatedAnnealing", "GeneticAlgorithm"]
        
        for i, alg in enumerate(algorithms):
            ttk.Radiobutton(alg_frame, text=alg, value=alg, variable=self.algorithm_var).grid(row=i, column=0, sticky=tk.W, padx=5, pady=2)
        
        # Nút chạy thuật toán
        ttk.Button(alg_frame, text="Chạy thuật toán", command=self.run_algorithm).grid(row=len(algorithms), column=0, pady=5)
        
        # Nút so sánh tất cả các thuật toán
        ttk.Button(alg_frame, text="So sánh tất cả", command=self.compare_all).grid(row=len(algorithms)+1, column=0, pady=5)
        
        # Nút reset
        ttk.Button(alg_frame, text="Reset", command=self.reset).grid(row=len(algorithms)+2, column=0, pady=5)
        
        # Phần log
        log_frame = ttk.LabelFrame(control_frame, text="Log", padding=5)
        log_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        self.log_text = scrolledtext.ScrolledText(log_frame, width=40, height=15)
        self.log_text.pack(fill=tk.BOTH, expand=True)
        
        # === PANEL HIỂN THỊ ===
        # Tab control cho phần hiển thị
        self.tab_control = ttk.Notebook(viz_frame)
        self.tab_control.pack(fill=tk.BOTH, expand=True)
        
        # Tab hiển thị bản đồ
        map_tab = ttk.Frame(self.tab_control)
        self.tab_control.add(map_tab, text="Bản đồ")
        
        # Tạo hình cho matplotlib
        self.map_fig, self.map_ax = plt.subplots(figsize=(8, 8))
        self.map_canvas = FigureCanvasTkAgg(self.map_fig, master=map_tab)
        self.map_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Tab hiển thị so sánh hiệu suất
        perf_tab = ttk.Frame(self.tab_control)
        self.tab_control.add(perf_tab, text="Hiệu suất")
        self.perf_tab = perf_tab
        
        # Tab control cho các biểu đồ hiệu suất
        self.perf_tab_control = ttk.Notebook(perf_tab)
        self.perf_tab_control.pack(fill=tk.BOTH, expand=True)
        
        # Tab thời gian thực thi
        time_tab = ttk.Frame(self.perf_tab_control)
        self.perf_tab_control.add(time_tab, text="Thời gian")
        
        self.time_fig, self.time_ax = plt.subplots(figsize=(8, 6))
        self.time_canvas = FigureCanvasTkAgg(self.time_fig, master=time_tab)
        self.time_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Tab chi phí
        cost_tab = ttk.Frame(self.perf_tab_control)
        self.perf_tab_control.add(cost_tab, text="Chi phí")
        
        self.cost_fig, self.cost_ax = plt.subplots(figsize=(8, 6))
        self.cost_canvas = FigureCanvasTkAgg(self.cost_fig, master=cost_tab)
        self.cost_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Tab độ dài đường đi
        path_tab = ttk.Frame(self.perf_tab_control)
        self.perf_tab_control.add(path_tab, text="Độ dài đường đi")
        
        self.path_fig, self.path_ax = plt.subplots(figsize=(8, 6))
        self.path_canvas = FigureCanvasTkAgg(self.path_fig, master=path_tab)
        self.path_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Tab nhiên liệu
        fuel_tab = ttk.Frame(self.perf_tab_control)
        self.perf_tab_control.add(fuel_tab, text="Nhiên liệu")
        
        self.fuel_fig, self.fuel_ax = plt.subplots(figsize=(8, 6))
        self.fuel_canvas = FigureCanvasTkAgg(self.fuel_fig, master=fuel_tab)
        self.fuel_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Tab tiền còn lại
        money_tab = ttk.Frame(self.perf_tab_control)
        self.perf_tab_control.add(money_tab, text="Tiền còn lại")
        
        self.money_fig, self.money_ax = plt.subplots(figsize=(8, 6))
        self.money_canvas = FigureCanvasTkAgg(self.money_fig, master=money_tab)
        self.money_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Tab số bước thực hiện
        steps_tab = ttk.Frame(self.perf_tab_control)
        self.perf_tab_control.add(steps_tab, text="Số bước")
        
        self.steps_fig, self.steps_ax = plt.subplots(figsize=(8, 6))
        self.steps_canvas = FigureCanvasTkAgg(self.steps_fig, master=steps_tab)
        self.steps_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def log(self, message):
        """Ghi log vào khung văn bản và đảm bảo hiển thị luôn cập nhật."""
        # Thêm thông báo mới vào cuối
        self.log_text.insert(tk.END, message + "\n")
        
        # Giới hạn số dòng log hiển thị để tránh quá nhiều dòng
        max_lines = 100
        content = self.log_text.get("1.0", tk.END).split("\n")
        if len(content) > max_lines:
            # Chỉ giữ lại max_lines dòng gần nhất
            self.log_text.delete("1.0", tk.END)
            self.log_text.insert(tk.END, "\n".join(content[-max_lines:]))
        
        # Cuộn xuống để hiển thị log mới nhất
        self.log_text.see(tk.END)
        
        # Cập nhật giao diện ngay lập tức
        self.root.update_idletasks()
    
    def generate_map(self):
        """Tạo và hiển thị bản đồ mới."""
        try:
            size = int(self.size_var.get())
            toll_ratio = float(self.toll_var.get())
            gas_ratio = float(self.gas_var.get())
            brick_ratio = float(self.brick_var.get())
            map_type = self.map_type_var.get()
            
            # Đọc các tham số cấu hình
            try:
                initial_money = float(self.money_var.get())
                max_fuel = float(self.max_fuel_var.get())
                initial_fuel = float(self.initial_fuel_var.get())
                fuel_per_move = float(self.fuel_per_move_var.get())
                gas_station_cost = float(self.gas_cost_var.get())
                toll_base_cost = float(self.toll_cost_var.get())
                
                # Cài đặt tham số cho tester
                self.tester.initial_money = initial_money
                self.tester.max_fuel = max_fuel
                self.tester.initial_fuel = initial_fuel
                self.tester.fuel_per_move = fuel_per_move
                self.tester.gas_station_cost = gas_station_cost
                self.tester.toll_base_cost = toll_base_cost
                
                self.log(f"Đã cài đặt cấu hình xe:")
                self.log(f"  - Dung tích bình xăng: {max_fuel:.1f}L")
                self.log(f"  - Nhiên liệu ban đầu: {initial_fuel:.1f}L")
                self.log(f"  - Tiêu thụ/ô: {fuel_per_move:.2f}L")
                self.log(f"  - Giá xăng: {gas_station_cost:.1f}đ/L")
                self.log(f"  - Phí trạm thu phí: {toll_base_cost:.1f}đ")
                self.log(f"  - Tiền ban đầu: {initial_money:.1f}đ")
            except ValueError:
                self.log(f"Không thể chuyển đổi các thông số cấu hình. Sử dụng giá trị mặc định.")
                self.tester.initial_money = None
                self.tester.max_fuel = None
                self.tester.initial_fuel = None
                self.tester.fuel_per_move = None
                self.tester.gas_station_cost = None
                self.tester.toll_base_cost = None
            
            self.log(f"\n=== Đang tạo bản đồ {size}x{size} (Kiểu: {map_type}) ===")
            
            # Tạo bản đồ dựa trên loại đã chọn
            if map_type == "default":
                # Tạo bản đồ mặc định
                map_obj = Map.create_demo_map(size)
                self.tester.maps = [map_obj]
                self.log(f"Đã tạo bản đồ mặc định: Bắt đầu={map_obj.start_pos}, Kết thúc={map_obj.end_pos}")
            else:
                # Tạo bản đồ ngẫu nhiên
                self.map_configs = [
                    {"size": size, "toll_ratio": toll_ratio, "gas_ratio": gas_ratio, "brick_ratio": brick_ratio}
                ]
                self.tester.generate_test_maps(self.map_configs)
            
            if self.tester.maps:
                # Hiển thị bản đồ mới
                self.map_ax.clear()
                ResultVisualizer.visualize_map(self.tester.maps[0], ax=self.map_ax)
                self.map_canvas.draw()
                self.tab_control.select(0)  # Chuyển đến tab bản đồ
                
                # Hiển thị thống kê
                map_obj = self.tester.maps[0]
                stats = map_obj.get_statistics()
                self.log(f"Thống kê bản đồ:")
                self.log(f"  - Ô đường thường: {stats['normal_roads']}")
                self.log(f"  - Trạm thu phí: {stats['toll_stations']}")
                self.log(f"  - Trạm xăng: {stats['gas_stations']}")
                self.log(f"  - Vật cản: {stats['brick_cells']}")
            else:
                messagebox.showwarning("Lỗi", "Không thể tạo bản đồ. Vui lòng thử với cấu hình khác.")
        except Exception as e:
            messagebox.showerror("Lỗi", f"Lỗi khi tạo bản đồ: {str(e)}")
            self.log(f"Lỗi: {str(e)}")
            self.log(traceback.format_exc())
    
    def run_algorithm(self):
        """Chạy thuật toán đã chọn trên bản đồ hiện tại."""
        if not self.tester.maps:
            messagebox.showinfo("Thông báo", "Vui lòng tạo bản đồ trước.")
            return
            
        algorithm_name = self.algorithm_var.get()
        map_obj = self.tester.maps[self.current_map_index]
        
        try:
            # Chạy thuật toán trong một luồng riêng
            def run_in_thread():
                self.log(f"\n=== Chạy thuật toán {algorithm_name} ===")
                result, algorithm = self.tester.test_algorithm(algorithm_name, map_obj)
                
                # Cập nhật giao diện trong luồng chính
                self.root.after(0, lambda: self.update_results(result, algorithm, map_obj))
                
            threading.Thread(target=run_in_thread).start()
            
        except Exception as e:
            messagebox.showerror("Lỗi", f"Lỗi khi chạy thuật toán: {str(e)}")
    
    def update_results(self, result, algorithm, map_obj):
        """Cập nhật kết quả sau khi chạy thuật toán."""
        # Hiển thị bản đồ với đường đi
        self.map_ax.clear()
        ResultVisualizer.visualize_map(
            map_obj, 
            algorithm=algorithm, 
            path=result["path"], 
            visited=result["visited"], 
            ax=self.map_ax
        )
        self.map_canvas.draw()
        self.tab_control.select(0)  # Chuyển đến tab bản đồ
    
    def compare_all(self):
        """So sánh tất cả các thuật toán trên bản đồ hiện tại."""
        if not self.tester.maps:
            messagebox.showinfo("Thông báo", "Vui lòng tạo bản đồ trước.")
            return
            
        map_obj = self.tester.maps[self.current_map_index]
        algorithms = ["BFS", "DFS", "AStar", "Greedy", "LocalBeam", "SimulatedAnnealing", "GeneticAlgorithm"]
        
        try:
            # Chạy so sánh trong một luồng riêng
            def run_in_thread():
                self.log(f"\n=== So sánh tất cả các thuật toán ===")
                results = self.tester.compare_algorithms(algorithms, map_obj)
                
                # Cập nhật giao diện trong luồng chính
                self.root.after(0, lambda: self.update_comparison(results))
                
            threading.Thread(target=run_in_thread).start()
            
        except Exception as e:
            messagebox.showerror("Lỗi", f"Lỗi khi so sánh thuật toán: {str(e)}")
    
    def update_comparison(self, results):
        """
        Hiển thị kết quả so sánh hiệu suất.
        
        Args:
            results: Danh sách kết quả của các thuật toán
        """
        if not results:
            return
        
        # Chọn tab hiệu suất
        self.tab_control.select(self.perf_tab)
        
        # Cập nhật các biểu đồ
        # Xóa dữ liệu cũ
        self.time_ax.clear()
        self.cost_ax.clear()
        self.path_ax.clear()
        self.fuel_ax.clear()
        self.money_ax.clear()
        self.steps_ax.clear()
        
        # Vẽ các biểu đồ mới
        ResultVisualizer.plot_performance_comparison(results, 'execution_time', 'So sánh thời gian thực thi', self.time_ax)
        ResultVisualizer.plot_performance_comparison(results, 'cost', 'So sánh chi phí đường đi', self.cost_ax)
        ResultVisualizer.plot_performance_comparison(results, 'path_length', 'So sánh độ dài đường đi', self.path_ax)
        ResultVisualizer.plot_performance_comparison(results, 'fuel_consumed', 'So sánh lượng nhiên liệu tiêu thụ', self.fuel_ax)
        ResultVisualizer.plot_performance_comparison(results, 'money_remaining', 'So sánh số tiền còn lại sau hành trình', self.money_ax)
        ResultVisualizer.plot_performance_comparison(results, 'steps', 'So sánh số bước thuật toán', self.steps_ax)
        
        # Cập nhật canvas
        self.time_canvas.draw()
        self.cost_canvas.draw()
        self.path_canvas.draw() 
        self.fuel_canvas.draw()
        self.money_canvas.draw()
        self.steps_canvas.draw()

    def reset(self):
        """Reset bản đồ và xóa kết quả."""
        # Xóa bản đồ hiện tại
        self.tester.maps = []
        self.tester.results = []
        
        # Xóa bản đồ hiện tại
        self.map_ax.clear()
        self.map_ax.set_title("Chưa có bản đồ", fontsize=10)
        self.map_canvas.draw()
        
        # Xóa các biểu đồ hiệu suất
        self.time_ax.clear()
        self.time_ax.set_title("Chưa có dữ liệu", fontsize=10)
        self.time_canvas.draw()
        
        self.cost_ax.clear()
        self.cost_ax.set_title("Chưa có dữ liệu", fontsize=10)
        self.cost_canvas.draw()
        
        self.path_ax.clear()
        self.path_ax.set_title("Chưa có dữ liệu", fontsize=10)
        self.path_canvas.draw()
        
        self.fuel_ax.clear()
        self.fuel_ax.set_title("Chưa có dữ liệu", fontsize=10)
        self.fuel_canvas.draw()
        
        self.money_ax.clear()
        self.money_ax.set_title("Chưa có dữ liệu", fontsize=10)
        self.money_canvas.draw()
        
        self.steps_ax.clear()
        self.steps_ax.set_title("Chưa có dữ liệu", fontsize=10)
        self.steps_canvas.draw()
        
        # Xóa log
        self.log_text.delete(1.0, tk.END)
        self.log("Đã reset hệ thống. Vui lòng tạo bản đồ mới.")

# Chạy ứng dụng nếu file được thực thi trực tiếp
if __name__ == "__main__":
    root = tk.Tk()
    app = AlgorithmTesterApp(root)
    root.mainloop() 