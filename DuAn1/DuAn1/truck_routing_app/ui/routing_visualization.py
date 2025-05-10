"""
Module for routing visualization and algorithm comparison.
This module provides the UI for visualizing different routing algorithms.
"""

import streamlit as st
import numpy as np
import time
import json
import os
from datetime import datetime
from typing import List, Tuple, Dict
from core.algorithms.base_search import BaseSearch, OBSTACLE_CELL, ROAD_CELL, TOLL_CELL, GAS_STATION_CELL
from core.algorithms.bfs import BFS
from core.algorithms.dfs import DFS
from core.algorithms.astar import AStar
from core.algorithms.greedy import GreedySearch
from core.algorithms.local_beam import LocalBeamSearch
from core.algorithms.simulated_annealing import SimulatedAnnealing
from core.algorithms.genetic_algorithm import GeneticAlgorithm
from ui import map_display

# Hằng số xác định loại ô (đồng bộ với base_search.py)
OBSTACLE_CELL = -1    # Ô chướng ngại vật
ROAD_CELL = 0         # Ô đường thường
TOLL_CELL = 1         # Ô trạm thu phí
GAS_STATION_CELL = 2  # Ô trạm xăng

def get_grid_from_map_data(map_data):
    """Trích xuất grid từ map_data một cách nhất quán."""
    if hasattr(map_data, 'grid'):
        return map_data.grid
    return map_data

def is_obstacle_cell(grid, pos):
    """Kiểm tra xem một ô có phải là chướng ngại vật không."""
    try:
        if pos[0] < 0 or pos[1] < 0 or pos[0] >= grid.shape[1] or pos[1] >= grid.shape[0]:
            return True  # Coi như ô ngoài biên là chướng ngại vật
        return grid[pos[1], pos[0]] == OBSTACLE_CELL
    except Exception as e:
        print(f"Error checking cell at {pos}: {str(e)}")
        return True  # Coi như ô lỗi là chướng ngại vật để an toàn

def filter_obstacle_cells(map_data, path):
    """Lọc bỏ các ô chướng ngại vật khỏi đường đi.
    
    Args:
        map_data: Đối tượng Map hoặc numpy array chứa thông tin bản đồ
        path: Danh sách các vị trí trên đường đi
        
    Returns:
        List[Tuple[int, int]]: Đường đi đã lọc bỏ các ô chướng ngại vật
    """
    if not path:
        return []
        
    grid = get_grid_from_map_data(map_data)
    filtered_path = []
    obstacles_found = False
    obstacles_count = 0
    
    # Kiểm tra trước các biên của grid để tránh lỗi
    rows, cols = grid.shape[0], grid.shape[1]
    
    for pos in path:
        try:
            # Kiểm tra tính hợp lệ của vị trí
            if not (0 <= pos[0] < cols and 0 <= pos[1] < rows):
                obstacles_count += 1
                obstacles_found = True
                print(f"WARNING: Vị trí {pos} nằm ngoài lưới {cols}x{rows}")
                continue
                
            # Kiểm tra xem ô có phải là chướng ngại vật hay không
            cell_value = grid[pos[1], pos[0]]
            if cell_value != OBSTACLE_CELL:
                filtered_path.append(pos)
            else:
                obstacles_found = True
                obstacles_count += 1
                print(f"WARNING: Bỏ qua ô chướng ngại vật tại vị trí {pos}")
        except Exception as e:
            print(f"LỖI: Không thể kiểm tra vị trí {pos}: {str(e)}")
            obstacles_count += 1
            obstacles_found = True
            
    if obstacles_found:
        print(f"CẢNH BÁO: Đã lọc bỏ {obstacles_count} ô chướng ngại vật từ đường đi (còn lại {len(filtered_path)})")
        
    return filtered_path

def draw_visualization_step(map_data, visited, current_pos, path=None, current_neighbors=None):
    """Vẽ một bước của quá trình minh họa thuật toán.
    
    Args:
        map_data: Đối tượng Map hoặc numpy array chứa thông tin bản đồ
        visited: Danh sách các vị trí đã thăm
        current_pos: Vị trí hiện tại đang xét
        path: Đường đi cuối cùng (nếu có)
        current_neighbors: Danh sách các vị trí lân cận của vị trí hiện tại
    """
    try:
        # Kiểm tra nếu không có dữ liệu đầu vào
        if map_data is None:
            st.error("Lỗi: Không có dữ liệu bản đồ")
            return
            
        # Thực hiện kiểm tra an toàn để tránh các ô chướng ngại vật trong đường đi
        if path:
            original_path_len = len(path)
            path = filter_obstacle_cells(map_data, path)
            if len(path) < original_path_len:
                st.warning(f"⚠️ Đã lọc bỏ {original_path_len - len(path)} ô chướng ngại vật khỏi đường đi")
        
        # Lọc các ô visited để đảm bảo không có ô chướng ngại vật
        if visited:
            visited = filter_obstacle_cells(map_data, visited)
        
        # Lọc các ô lân cận để đảm bảo không có ô chướng ngại vật
        if current_neighbors:
            current_neighbors = filter_obstacle_cells(map_data, current_neighbors)
        
        # Kiểm tra vị trí hiện tại
        if current_pos and is_obstacle_cell(get_grid_from_map_data(map_data), current_pos):
            print(f"CẢNH BÁO: Vị trí hiện tại {current_pos} là ô chướng ngại vật!")
            current_pos = None

        # Vẽ bản đồ với các thành phần đã được lọc
        map_display.draw_map(map_data, visited=visited, current_pos=current_pos, 
                           path=path, current_neighbors=current_neighbors)
    except Exception as e:
        st.error(f"Lỗi khi vẽ bước minh họa: {str(e)}")
        print(f"Exception in draw_visualization_step: {str(e)}")

# Hàm mới: Vẽ animation xe chạy dọc đường đi cuối cùng
def draw_truck_animation(map_data, path, speed=5):
    """Vẽ animation xe chạy dọc theo đường đi cuối cùng.
    
    Args:
        map_data: Đối tượng Map chứa thông tin bản đồ
        path: Danh sách các vị trí trên đường đi cuối cùng
        speed: Tốc độ animation (1-10)
    """
    # Add validation to filter out obstacles from path
    path = filter_obstacle_cells(map_data, path)
    
    if not path or len(path) < 2:
        st.warning("⚠️ Không có đường đi để hiển thị animation!")
        map_display.draw_map(map_data)
        return
    
    # Lấy vị trí hiện tại của xe từ session state hoặc đặt về đầu đường
    if "truck_position_index" not in st.session_state:
        st.session_state.truck_position_index = 0
    
    # Hiển thị thanh tiến trình
    total_steps = len(path) - 1
    current_step = st.session_state.truck_position_index
    progress = float(current_step) / total_steps if total_steps > 0 else 0
    st.progress(progress, text=f"Vị trí: {current_step}/{total_steps}")
    
    # Tạo các vị trí trên đường đi đã đi qua và vị trí hiện tại
    visited_positions = path[:current_step+1]
    current_position = path[current_step]
    
    # Vẽ bản đồ với vị trí xe
    map_display.draw_map(
            map_data=map_data,
        visited=visited_positions,
        current_pos=current_position,
        path=path  # Hiển thị toàn bộ đường đi
    )
    
    # Nếu đang chạy animation và chưa đến cuối đường
    if st.session_state.get("is_playing", False) and current_step < total_steps:
        # Đợi theo tốc độ đã chọn
        time.sleep(1.0 / speed)
        # Di chuyển xe đến vị trí tiếp theo
        st.session_state.truck_position_index += 1
        # Rerun để cập nhật UI
        st.rerun()

def save_algorithm_stats(algorithm_name: str, stats: dict):
    """Lưu thống kê thuật toán vào file JSON
    
    Args:
        algorithm_name: Tên thuật toán
        stats: Dictionary chứa thống kê
        
    Returns:
        str: Đường dẫn đến file đã lưu hoặc None nếu có lỗi
    """
    # Lấy các tham số cấu hình từ session state
    initial_money = st.session_state.get('initial_money', 2000.0)
    initial_fuel = st.session_state.get('initial_fuel', 20.0)
    
    # Tạo tên file an toàn bằng cách thay thế ký tự đặc biệt
    safe_algo_name = algorithm_name.replace('*', 'star').replace('/', '_').replace('\\', '_')
    
    # Tạo thư mục statistics nếu chưa tồn tại
    stats_dir = os.path.join(os.path.dirname(__file__), '..', 'statistics')
    os.makedirs(stats_dir, exist_ok=True)
    
    # Tạo tên file với timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"stats_{safe_algo_name}_{timestamp}.json"
    filepath = os.path.join(stats_dir, filename)
    
    # Chuẩn bị dữ liệu với comments tiếng Việt
    data = {
        "timestamp": {  # Thời gian chạy thuật toán
            "value": timestamp,
            "comment": "Thời gian chạy thuật toán"
        },
        "algorithm": {  # Tên thuật toán sử dụng
            "value": algorithm_name,
            "comment": "Tên thuật toán sử dụng"
        },
        "map_size": {  # Kích thước bản đồ
            "value": stats.get("map_size", ""),
            "comment": "Kích thước bản đồ (rows x cols)"
        },
        "search_process": {  # Thông tin quá trình tìm kiếm
            "steps": {
                "value": stats.get("steps", 0),
                "comment": "Số bước thực hiện"
            },
            "visited_cells": {
                "value": stats.get("visited_cells", 0),
                "comment": "Số ô đã thăm"
            },
            "path_length": {
                "value": stats.get("path_length", 0),
                "comment": "Độ dài đường đi tìm được"
            }
        },
        "fuel_info": {  # Thông tin nhiên liệu
            "initial_fuel": {
                "value": stats.get("initial_fuel", initial_fuel),
                "comment": "Nhiên liệu ban đầu (L)"
            },
            "remaining_fuel": {
                "value": stats.get("fuel", 0),
                "comment": "Nhiên liệu còn lại (L)"
            },
            "fuel_consumed": {
                "value": stats.get("fuel_consumed", 0),
                "comment": "Nhiên liệu đã tiêu thụ (L)"
            }
        },
        "costs": {  # Chi phí hành trình
            "total_cost": {
                "value": stats.get("fuel_cost", 0) + stats.get("toll_cost", 0),
                "comment": "Tổng chi phí (đ)"
            },
            "fuel_cost": {
                "value": stats.get("fuel_cost", 0),
                "comment": "Chi phí nhiên liệu (đ)"
            },
            "toll_cost": {
                "value": stats.get("toll_cost", 0),
                "comment": "Chi phí trạm thu phí (đ)"
            },
            "initial_money": {
                "value": initial_money,
                "comment": "Số tiền ban đầu (đ)"
            },
            "remaining_money": {
                "value": stats.get("money", 0),
                "comment": "Số tiền còn lại (đ)"
            }
        },
        "performance": {  # Phân tích hiệu suất
            "execution_time": {
                "value": stats.get("execution_time", 0),
                "comment": "Thời gian thực thi (giây)"
            },
            "memory_usage": {
                "value": stats.get("memory_usage", 0),
                "comment": "Bộ nhớ sử dụng (MB)"
            }
        },
        "feasibility": {  # Tính khả thi
            "is_feasible": {
                "value": stats.get("is_feasible", False),
                "comment": "Đường đi có khả thi không"
            },
            "reason": {
                "value": stats.get("reason", ""),
                "comment": "Lý do nếu không khả thi"
            }
        }
    }
    
    try:
        # Ghi file JSON với encoding utf-8 và indent
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print(f"Đã lưu thống kê vào file: {filename}")
        return filepath
    except Exception as e:
        print(f"Lỗi khi lưu file thống kê: {str(e)}")
        return None

def run_algorithm(algorithm_name: str, map_data: np.ndarray, start: Tuple[int, int], 
                 goal: Tuple[int, int]) -> Dict:
    """Chạy một thuật toán và trả về kết quả."""
    # Lấy grid từ map_data một cách nhất quán
    grid = get_grid_from_map_data(map_data)
    
    # Kiểm tra điểm bắt đầu và kết thúc hợp lệ
    if is_obstacle_cell(grid, start):
        st.error(f"❌ Điểm bắt đầu {start} nằm trên ô chướng ngại vật hoặc ngoài biên!")
        return None
        
    if is_obstacle_cell(grid, goal):
        st.error(f"❌ Điểm đích {goal} nằm trên ô chướng ngại vật hoặc ngoài biên!")
        return None
    
    # Lấy các tham số cấu hình từ session_state
    initial_money = st.session_state.get('initial_money', 2000.0)
    max_fuel = st.session_state.get('max_fuel', 20.0)
    fuel_per_move = st.session_state.get('fuel_per_move', 0.4)
    gas_station_cost = st.session_state.get('gas_station_cost', 30.0)
    toll_base_cost = st.session_state.get('toll_base_cost', 150.0)
    initial_fuel = st.session_state.get('initial_fuel', max_fuel)
    
    # Khởi tạo thuật toán
    if algorithm_name == "BFS":
        algorithm = BFS(grid, initial_money, max_fuel, fuel_per_move, 
                     gas_station_cost, toll_base_cost, initial_fuel)
    elif algorithm_name == "DFS":
        algorithm = DFS(grid, initial_money, max_fuel, fuel_per_move, 
                     gas_station_cost, toll_base_cost, initial_fuel)
    elif algorithm_name == "A*":
        algorithm = AStar(grid, initial_money, max_fuel, fuel_per_move, 
                       gas_station_cost, toll_base_cost, initial_fuel)
    elif algorithm_name == "Greedy":
        # Thêm debug log
        print(f"Khởi tạo Greedy với: initial_money={initial_money}, max_fuel={max_fuel}, initial_fuel={initial_fuel}")
        algorithm = GreedySearch(grid, initial_money, max_fuel, fuel_per_move, 
                             gas_station_cost, toll_base_cost, initial_fuel)
    elif algorithm_name == "Simulated Annealing":
        # Get parameters from session state if available
        initial_temp = st.session_state.get('initial_temp', 100.0)
        cooling_rate = st.session_state.get('cooling_rate', 0.95)
        steps_per_temp = st.session_state.get('steps_per_temp', 50)
        algorithm = SimulatedAnnealing(grid, 
                                    initial_temperature=initial_temp, 
                                    cooling_rate=cooling_rate, 
                                    steps_per_temp=steps_per_temp,
                                    initial_money=initial_money,
                                    max_fuel=max_fuel,
                                    fuel_per_move=fuel_per_move,
                                    gas_station_cost=gas_station_cost,
                                    toll_base_cost=toll_base_cost,
                                    initial_fuel=initial_fuel)
    elif algorithm_name == "Local Beam Search":
        # Get parameters from session state
        beam_width = st.session_state.get('beam_width', 10)
        use_stochastic = st.session_state.get('use_stochastic', True)
        algorithm = LocalBeamSearch(grid, beam_width=beam_width,
                                initial_money=initial_money,
                                max_fuel=max_fuel,
                                fuel_per_move=fuel_per_move,
                                gas_station_cost=gas_station_cost,
                                toll_base_cost=toll_base_cost,
                                initial_fuel=initial_fuel)
        algorithm.use_stochastic = use_stochastic
    elif algorithm_name == "Genetic Algorithm":
        # Get parameters from session state
        pop_size = st.session_state.get('pop_size', 50)
        crossover_rate = st.session_state.get('crossover_rate', 0.8)
        mutation_rate = st.session_state.get('mutation_rate', 0.2)
        generations = st.session_state.get('generations', 100)
        algorithm = GeneticAlgorithm(grid, 
                                  population_size=pop_size,
                                  crossover_rate=crossover_rate,
                                  mutation_rate=mutation_rate,
                                  generations=generations,
                                  initial_money=initial_money,
                                  max_fuel=max_fuel,
                                  fuel_per_move=fuel_per_move,
                                  gas_station_cost=gas_station_cost,
                                  toll_base_cost=toll_base_cost,
                                  initial_fuel=initial_fuel)
    else:
        st.error(f"Thuật toán {algorithm_name} không được hỗ trợ!")
        return None
    
    # Bắt đầu đo thời gian
    start_time = time.perf_counter()
    
    # Chạy thuật toán
    raw_path = algorithm.search(start, goal)
    
    # THAY ĐỔI QUAN TRỌNG: Luôn xác thực lại đường đi với validate_path_no_obstacles
    print(f"XÁC THỰC TRIỆT ĐỂ: Thuật toán {algorithm_name} trả về đường đi có {len(raw_path) if raw_path else 0} điểm")
    print(f"Thực hiện xác thực đường đi...")
    
    path = algorithm.validate_path_no_obstacles(raw_path) if raw_path else []
    
    if not path:
        print(f"LỖI NGHIÊM TRỌNG: Thuật toán {algorithm_name} không tìm được đường đi hợp lệ!")
        if raw_path:
            print(f"Đường đi gốc có {len(raw_path)} điểm, nhưng validate_path_no_obstacles trả về danh sách rỗng")
            st.error(f"⚠️ Không thể tạo đường đi hợp lệ! Đường đi gốc có {len(raw_path)} điểm nhưng không vượt qua kiểm tra tính hợp lệ.")
        else:
            print(f"Thuật toán không tìm thấy đường đi nào")
            st.error("⚠️ Thuật toán không tìm thấy đường đi nào!")
            
    elif len(path) < len(raw_path):
        print(f"CẢNH BÁO: Đường đi đã được sửa đổi từ {len(raw_path)} xuống {len(path)} điểm sau khi xác thực")
        st.warning(f"⚠️ Đường đi đã được sửa đổi từ {len(raw_path)} xuống {len(path)} điểm sau khi xác thực")
    
    # Cập nhật đường đi của thuật toán với đường đi đã được xác thực
    if path:
        algorithm.current_path = path
    else:
        algorithm.current_path = []
    
    # Kết thúc đo thời gian
    end_time = time.perf_counter()
    execution_time = end_time - start_time
    
    # Lấy thống kê
    stats = algorithm.get_statistics()
    
    # Thêm thông tin về hiệu suất
    stats["execution_time"] = execution_time
    stats["memory_usage"] = len(algorithm.get_visited()) * 16  # Ước tính bộ nhớ sử dụng (bytes) - mỗi vị trí là tuple 2 số
    
    # Đánh giá tính khả thi và chất lượng giải pháp
    if path and stats["fuel"] > 0:
        stats["success_rate"] = 1.0
        stats["solution_quality"] = stats["path_length"]  # Độ dài đường đi thực tế
        stats["is_feasible"] = True
        stats["reason"] = "Đường đi khả thi"
    else:
        stats["success_rate"] = 0.0
        stats["solution_quality"] = float('inf')
        stats["is_feasible"] = False
        stats["reason"] = "Không tìm thấy đường đi khả thi"
    
    # Lấy danh sách các ô đã thăm theo thứ tự thời gian cho animation
    visited_list = algorithm.get_visited()
    
    # Đảm bảo không có trùng lặp trong visited_list
    visited_unique = []
    visited_set = set()
    for pos in visited_list:
        if pos not in visited_set:
            visited_unique.append(pos)
            visited_set.add(pos)
    
    # Lọc bỏ chướng ngại vật khỏi danh sách đã thăm (không dùng các hàm thuật toán ở đây)
    clean_visited = filter_obstacle_cells(map_data, visited_unique)
    
    # Chuẩn bị trạng thái cho cả hai chế độ hiển thị
    # 1. Quá trình tìm đường
    # 2. Xe đi theo đường đã tìm được
    exploration_states = [(pos, 0) for pos in clean_visited]  # Trạng thái cho chế độ tìm đường
    
    # Tạo trạng thái di chuyển xe dựa trên đường đi cuối cùng
    truck_states = []
    if path:
        # Giả lập fuel giảm dần theo từng bước đi
        current_fuel = initial_fuel
        for i, pos in enumerate(path):
            if i > 0:  # Không tính vị trí đầu tiên
                current_fuel -= fuel_per_move
            truck_states.append((pos, current_fuel))
    
    # Sau khi có kết quả, lưu thống kê
    if stats:
        map_size = (grid.shape[0], grid.shape[1])
        stats_file = save_algorithm_stats(algorithm_name, stats)
        st.session_state.last_stats_file = stats_file
    
    return {
        "path": path,  # Đã được xác thực triệt để
        "visited": clean_visited,  # Đã lọc bỏ chướng ngại vật
        "exploration_states": exploration_states,  # Trạng thái cho chế độ tìm đường
        "truck_states": truck_states,  # Trạng thái cho chế độ xe di chuyển
        "stats": stats
    }

def render_routing_visualization():
    """Render tab định tuyến và tối ưu hệ thống."""
    st.markdown("""
    <div style="text-align: center; padding: 10px; margin-bottom: 20px; background: linear-gradient(135deg, #1a5276 0%, #2980b9 100%); border-radius: 10px; box-shadow: 0 4px 10px rgba(0,0,0,0.15);">
        <h2 style="color: white; margin: 0;">🗺️ Định Tuyến & Tối Ưu Hệ Thống</h2>
        <p style="color: white; opacity: 0.9; margin-top: 5px;">Mô phỏng và đánh giá các thuật toán tìm đường tối ưu</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Kiểm tra xem đã có bản đồ chưa
    if "map" not in st.session_state:
        st.warning("⚠️ Vui lòng tạo bản đồ trước khi sử dụng tính năng này!")
        return
    
    # Kiểm tra vị trí bắt đầu và điểm đích
    if "start_pos" not in st.session_state or st.session_state.start_pos is None:
        st.warning("⚠️ Vui lòng thiết lập vị trí bắt đầu của xe!")
        return
    
    if "end_pos" not in st.session_state or st.session_state.end_pos is None:
        st.warning("⚠️ Vui lòng thiết lập điểm đích!")
        return
    
    # Tạo layout hai cột chính: Cấu hình bên trái, Bản đồ + điều khiển bên phải
    config_col, visual_col = st.columns([1, 2])
    
    with config_col:
        # Phần cấu hình thuật toán và điều khiển
        st.markdown("""
        <div style="background-color: #f8f9fa; padding: 10px; border-radius: 5px; border-left: 5px solid #27ae60; margin-bottom: 15px;">
            <h3 style="margin: 0; color: #2c3e50;">⚙️ Cấu hình thuật toán</h3>
        </div>
        """, unsafe_allow_html=True)
        
        # Chọn thuật toán
        algorithm_options = ["BFS", "DFS", "A*", "Greedy", "Local Beam Search", "Simulated Annealing", "Genetic Algorithm"]
        algorithm_name = st.selectbox("Chọn thuật toán:", algorithm_options)
        
        # Lưu thuật toán đã chọn vào session state
        st.session_state.algorithm = algorithm_name
        
        # Hiển thị mô tả thuật toán
        algorithm_descriptions = {
            "BFS": "Tìm kiếm theo chiều rộng, đảm bảo tìm đường đi ngắn nhất về số bước.",
            "DFS": "Tìm kiếm theo chiều sâu, phù hợp cho không gian tìm kiếm sâu.",
            "A*": "Tìm kiếm theo A*, kết hợp cả chi phí thực tế và heuristic.",
            "Greedy": "Luôn chọn bước đi tốt nhất theo đánh giá heuristic.",
            "Local Beam Search": "Theo dõi k trạng thái cùng lúc thay vì một trạng thái duy nhất.",
            "Simulated Annealing": "Mô phỏng quá trình luyện kim, cho phép chấp nhận giải pháp tệ hơn với xác suất giảm dần theo thời gian.",
            "Genetic Algorithm": "Mô phỏng quá trình tiến hóa tự nhiên, sử dụng quần thể, chọn lọc, lai ghép và đột biến."
        }
        st.info(f"**{algorithm_name}**: {algorithm_descriptions.get(algorithm_name, 'Không có mô tả.')}")
        
        # Tạo các tab cho các nhóm cấu hình
        tab1, tab2, tab3 = st.tabs(["🚚 Phương tiện", "🛣️ Chi phí", "🧪 Tham số thuật toán"])
        
        with tab1:
            # Cấu hình phương tiện (xăng)
            st.markdown("##### 🛢️ Nhiên liệu")
            col1, col2 = st.columns(2)
            with col1:
                max_fuel = st.slider("Dung tích bình xăng (L):", 
                              min_value=10.0, max_value=50.0, value=20.0, step=1.0)
                st.session_state.max_fuel = max_fuel
            
            with col2:
                initial_fuel = st.slider("Nhiên liệu ban đầu (L):", 
                                 min_value=5.0, max_value=max_fuel, value=max_fuel, step=1.0)
                st.session_state.initial_fuel = initial_fuel
            
            fuel_per_move = st.slider("Mức tiêu thụ nhiên liệu (L/ô):", 
                               min_value=0.1, max_value=1.0, value=0.4, step=0.1)
            st.session_state.fuel_per_move = fuel_per_move
        
        with tab2:
            # Cấu hình chi phí
            st.markdown("##### 💰 Chi phí")
            initial_money = st.slider("Số tiền ban đầu (đ):", 
                              min_value=1000.0, max_value=5000.0, value=2000.0, step=100.0)
            st.session_state.initial_money = initial_money
            
            col1, col2 = st.columns(2)
            with col1:
                gas_station_cost = st.slider("Chi phí đổ xăng (đ/L):", 
                                     min_value=10.0, max_value=100.0, value=30.0, step=5.0)
                st.session_state.gas_station_cost = gas_station_cost
            
            with col2:
                toll_base_cost = st.slider("Chi phí trạm thu phí (đ):", 
                                   min_value=50.0, max_value=300.0, value=150.0, step=10.0)
                st.session_state.toll_base_cost = toll_base_cost
        
        with tab3:
            # Cấu hình tham số thuật toán
            st.markdown("##### 🔧 Tham số riêng của thuật toán")
            
            if algorithm_name == "Local Beam Search":
                beam_width = st.slider("Beam Width:", min_value=2, max_value=50, value=10, step=1)
                st.session_state.beam_width = beam_width
                
                use_stochastic = st.checkbox("Sử dụng Stochastic Beam Search", value=True)
                st.session_state.use_stochastic = use_stochastic
            
            elif algorithm_name == "Simulated Annealing":
                initial_temp = st.slider("Nhiệt độ ban đầu:", min_value=10.0, max_value=500.0, value=100.0, step=10.0)
                st.session_state.initial_temp = initial_temp
                
                cooling_rate = st.slider("Tốc độ làm lạnh:", min_value=0.7, max_value=0.99, value=0.95, step=0.01)
                st.session_state.cooling_rate = cooling_rate
                
                steps_per_temp = st.slider("Số bước trên mỗi nhiệt độ:", min_value=10, max_value=100, value=50, step=10)
                st.session_state.steps_per_temp = steps_per_temp
            
            elif algorithm_name == "Genetic Algorithm":
                pop_size = st.slider("Kích thước quần thể:", min_value=10, max_value=100, value=50, step=10)
                st.session_state.pop_size = pop_size
                
                col1, col2 = st.columns(2)
                with col1:
                    crossover_rate = st.slider("Tỷ lệ lai ghép:", min_value=0.5, max_value=1.0, value=0.8, step=0.05)
                    st.session_state.crossover_rate = crossover_rate
                
                with col2:
                    mutation_rate = st.slider("Tỷ lệ đột biến:", min_value=0.05, max_value=0.5, value=0.2, step=0.05)
                    st.session_state.mutation_rate = mutation_rate
                
                generations = st.slider("Số thế hệ:", min_value=10, max_value=200, value=100, step=10)
                st.session_state.generations = generations
            
            else:
                st.info(f"Thuật toán {algorithm_name} không có tham số bổ sung để cấu hình.")
        
        # Nút tìm đường với thiết kế đẹp hơn
        st.markdown("")  # Tạo khoảng cách
        search_button = st.button("🔍 Tìm đường", use_container_width=True, type="primary")
        
        if search_button:
            with st.spinner("🔄 Đang tìm đường..."):
                try:
                    result = run_algorithm(
                        algorithm_name,
                        st.session_state.map,
                        st.session_state.start_pos,
                        st.session_state.end_pos
                    )
                    if result:
                        st.session_state.current_result = result
                        st.session_state.visualization_step = 0
                        st.session_state.truck_position_index = 0
                        st.session_state.is_playing = False
                        
                        # Kiểm tra tính khả thi của đường đi
                        if result["stats"]["is_feasible"]:
                            st.success("✅ Đã tìm thấy đường đi khả thi!")
                        else:
                            st.warning("⚠️ Đã tìm được một phần đường đi nhưng không thể đến đích!")
                    else:
                        st.error("❌ Không thể tìm được đường đi!")
                except Exception as e:
                    st.error(f"❌ Lỗi khi thực thi thuật toán: {str(e)}")
                    return
    
    with visual_col:
        # Phần hiển thị trực quan có 2 vùng: Map và điều khiển
        # Vùng bản đồ
        st.markdown("""
        <div style="background-color: #f8f9fa; padding: 10px; border-radius: 5px; border-left: 5px solid #2980b9; margin-bottom: 15px;">
            <h3 style="margin: 0; color: #2c3e50;">🗺️ Bản đồ mô phỏng</h3>
        </div>
        """, unsafe_allow_html=True)
        
        # Container cho bản đồ và trực quan hóa
        map_container = st.empty()
        
        # Hiển thị bản đồ ban đầu
        with map_container:
            map_display.draw_map(st.session_state.map)
        
        # Vùng điều khiển trực quan hóa ngay bên dưới bản đồ
        if "current_result" in st.session_state:
            # Lấy dữ liệu từ kết quả
            stats = st.session_state.current_result["stats"]
            path = st.session_state.current_result["path"]
            
            # Thông tin cơ bản về đường đi - đặt ngay dưới bản đồ
            basic_info_cols = st.columns(4)
            with basic_info_cols[0]:
                st.metric("Thuật toán", st.session_state.algorithm)
            with basic_info_cols[1]:
                st.metric("Độ dài đường đi", stats["path_length"])
            with basic_info_cols[2]:
                st.metric("Thời gian chạy", f"{stats['execution_time']*1000:.2f}ms")
            with basic_info_cols[3]:
                is_feasible = stats.get("is_feasible", False)
                if is_feasible:
                    st.metric("Trạng thái", "✅ Khả thi")
                else:
                    st.metric("Trạng thái", "⚠️ Không khả thi", delta="Hạn chế")
            
            # Chọn chế độ hiển thị ngay dưới thông tin cơ bản
            st.markdown("##### 🎬 Chọn chế độ minh họa:")
            visualization_mode = st.radio(
                "",
                ["1. Quá trình tìm đường", "2. Quá trình xe di chuyển trên đường đi cuối cùng"],
                horizontal=True
            )
            
            # Hiển thị thông tin trạng thái của quá trình minh họa - dạng đơn giản
            if visualization_mode == "1. Quá trình tìm đường":
                visited = st.session_state.current_result["visited"]
                total_steps = len(visited)
                current_step = st.session_state.visualization_step if "visualization_step" in st.session_state else 0
                progress_percentage = int((current_step / total_steps * 100) if total_steps > 0 else 0)
                st.markdown(f"**Quá trình tìm đường:** Đã thăm {current_step}/{total_steps} ô ({progress_percentage}%)")
            else:
                if path:
                    total_steps = len(path) - 1
                    current_step = st.session_state.truck_position_index if "truck_position_index" in st.session_state else 0
                    progress_percentage = int((current_step / total_steps * 100) if total_steps > 0 else 0)
                    st.markdown(f"**Xe di chuyển:** Bước {current_step}/{total_steps} ({progress_percentage}%)")
                else:
                    st.warning("Không có đường đi")
            
            # Các nút điều khiển đặt trên cùng một hàng - SỬA LỖI LỒNG CỘT
            btn_cols = st.columns(4)
            with btn_cols[0]:
                if st.button("⏮️ Bắt đầu", use_container_width=True):
                    st.session_state.visualization_step = 0
                    st.session_state.truck_position_index = 0
                    st.session_state.is_playing = False
                    st.rerun()
            
            with btn_cols[1]:
                play_text = "⏸️ Tạm dừng" if st.session_state.get("is_playing", False) else "▶️ Chạy" 
                if st.button(play_text, use_container_width=True):
                    st.session_state.is_playing = not st.session_state.get("is_playing", False)
                    st.rerun()
            
            with btn_cols[2]:
                if st.button("⏭️ Kết thúc", use_container_width=True):
                    if visualization_mode == "1. Quá trình tìm đường":
                        st.session_state.visualization_step = len(st.session_state.current_result["visited"])
                    else:
                        st.session_state.truck_position_index = len(path) - 1 if path else 0
                    st.session_state.is_playing = False
                    st.rerun()
            
            with btn_cols[3]:
                if st.button("🔄 Làm mới", use_container_width=True):
                    st.session_state.visualization_step = 0
                    st.session_state.truck_position_index = 0
                    st.session_state.is_playing = False
                    st.rerun()
        
        # Điều khiển tốc độ
        speed = st.slider(
            "Tốc độ hiển thị:",
            min_value=1,
            max_value=10,
            value=5,
            help="Điều chỉnh tốc độ hiển thị (1: chậm nhất, 10: nhanh nhất)"
        )
        
        # Thanh tiến trình nằm ngay dưới điều khiển
        if "current_result" in st.session_state:
            if visualization_mode == "1. Quá trình tìm đường":
                if "visualization_step" not in st.session_state:
                    st.session_state.visualization_step = 0
                
                visited = st.session_state.current_result["visited"]
                total_steps = len(visited)
                current_step = st.session_state.visualization_step
                progress = float(current_step) / total_steps if total_steps > 0 else 0
                st.progress(progress, text=f"Bước {current_step}/{total_steps}")
            else:
                if path and len(path) >= 2:
                    if "truck_position_index" not in st.session_state:
                        st.session_state.truck_position_index = 0
                    
                    total_steps = len(path) - 1
                    current_step = st.session_state.truck_position_index
                    progress = float(current_step) / total_steps if total_steps > 0 else 0
                    st.progress(progress, text=f"Vị trí xe: {current_step}/{total_steps}")
            
            # Xử lý trực quan hóa theo chế độ đã chọn
            if visualization_mode == "1. Quá trình tìm đường":
                if st.session_state.is_playing and current_step < total_steps:
                    # Hiển thị bản đồ với các ô đã thăm
                    current_visited = visited[:current_step + 1]
                    current_pos = visited[current_step]
                    
                    # Lấy các ô hàng xóm của vị trí hiện tại
                    current_neighbors = []
                    if hasattr(st.session_state.map, 'get_neighbors'):
                        current_neighbors = st.session_state.map.get_neighbors(current_pos)
                    
                    # Vẽ bước hiện tại
                    with map_container:
                        draw_visualization_step(
                            st.session_state.map,
                            current_visited,
                            current_pos,
                            path if current_step == total_steps - 1 else None,
                            current_neighbors
                        )
                    
                    # Tăng bước và đợi
                    time.sleep(1.0 / speed)
                    st.session_state.visualization_step += 1
                    st.rerun()
                else:
                    # Hiển thị trạng thái hiện tại
                    if current_step < total_steps:
                        current_visited = visited[:current_step + 1]
                        current_pos = visited[current_step]
                        current_neighbors = []
                        if hasattr(st.session_state.map, 'get_neighbors'):
                            current_neighbors = st.session_state.map.get_neighbors(current_pos)
                        
                        with map_container:
                            draw_visualization_step(
                                st.session_state.map,
                                current_visited,
                                current_pos,
                                path if current_step == total_steps - 1 else None,
                                current_neighbors
                            )
                    else:
                        # Hiển thị kết quả cuối cùng với đường đi
                        with map_container:
                            draw_visualization_step(
                                st.session_state.map,
                                visited,
                                None,
                                path
                            )
            else:
                # Chế độ 2: Hiển thị quá trình xe di chuyển trên đường đi cuối cùng
                if not path or len(path) < 2:
                    st.warning("⚠️ Không có đường đi để hiển thị!")
                else:
                    # Xử lý animation xe di chuyển
                    if st.session_state.is_playing and current_step < total_steps:
                        # Hiển thị bản đồ với vị trí xe
                        current_pos = path[current_step]
                        visited_positions = path[:current_step+1]
                        
                        with map_container:
                            map_display.draw_map(
                                map_data=st.session_state.map,
                                visited=visited_positions,
                                current_pos=current_pos,
                                path=path
                            )
                        
                        # Tăng bước và đợi
                        time.sleep(1.0 / speed)
                        st.session_state.truck_position_index += 1
                        st.rerun()
                    else:
                        # Hiển thị trạng thái hiện tại
                        if current_step <= total_steps:
                            current_pos = path[current_step]
                            visited_positions = path[:current_step+1]
                            
                            with map_container:
                                map_display.draw_map(
                                    map_data=st.session_state.map,
                                    visited=visited_positions,
                                    current_pos=current_pos,
                                    path=path
                                )
    
    # Hiển thị thống kê chi tiết ở phần dưới cùng sau khi có kết quả
    if "current_result" in st.session_state:
        # Tạo một vùng phân tách
        st.markdown("""
        <hr style="height:3px;border:none;background-color:#3498db;margin:30px 0;opacity:0.3;">
        """, unsafe_allow_html=True)
        
        with st.expander("📊 Xem thống kê chi tiết", expanded=False):
            stats = st.session_state.current_result["stats"]
            stat_tabs = st.tabs(["Quá trình tìm kiếm", "Nhiên liệu", "Chi phí & Tiền", "Hiệu suất"])
            
            with stat_tabs[0]:
                # Thông tin về quá trình tìm kiếm
                search_cols = st.columns(3)
                with search_cols[0]:
                    st.metric("Số bước thực hiện", stats["steps"])
                with search_cols[1]:
                    st.metric("Số ô đã thăm", stats["visited"])
                with search_cols[2]:
                    st.metric("Độ dài đường đi", stats["path_length"])
            
            with stat_tabs[1]:
                # Thông tin về nhiên liệu
                fuel_cols = st.columns(3)
                with fuel_cols[0]:
                    initial_fuel = st.session_state.get('initial_fuel', 20.0)
                    st.metric("Xăng ban đầu", f"{initial_fuel:.1f}L")
                with fuel_cols[1]:
                    st.metric("Xăng đã tiêu thụ", f"{stats.get('fuel_consumed', 0):.1f}L")
                with fuel_cols[2]:
                    st.metric("Xăng còn lại", f"{stats.get('fuel', 0):.1f}L")
            
            with stat_tabs[2]:
                # Thông tin về chi phí
                cost_cols = st.columns(2)
                with cost_cols[0]:
                    # Chi phí
                    st.markdown("##### Chi phí:")
                    st.metric("Chi phí nhiên liệu", f"{stats.get('fuel_cost', 0):.1f}đ")
                    st.metric("Chi phí trạm thu phí", f"{stats.get('toll_cost', 0):.1f}đ")
                    total_cost = stats.get('fuel_cost', 0) + stats.get('toll_cost', 0)
                    st.metric("Tổng chi phí", f"{total_cost:.1f}đ")
                
                with cost_cols[1]:
                    # Tiền
                    st.markdown("##### Số tiền:")
                    initial_money = st.session_state.get('initial_money', 2000.0)
                    st.metric("Tiền ban đầu", f"{initial_money:.1f}đ")
                    money_spent = initial_money - stats.get('money', 0)
                    st.metric("Tiền đã chi tiêu", f"{money_spent:.1f}đ")
                    st.metric("Tiền còn lại", f"{stats.get('money', 0):.1f}đ")
            
            with stat_tabs[3]:
                # Phân tích hiệu suất
                perf_cols = st.columns(2)
                with perf_cols[0]:
                    execution_time_ms = stats['execution_time'] * 1000
                    st.metric("⏱️ Thời gian chạy", f"{execution_time_ms:.2f}ms")
                    memory_kb = stats['memory_usage'] / 1024
                    st.metric("💾 Bộ nhớ sử dụng", f"{memory_kb:.2f}KB")
                
                with perf_cols[1]:
                    success_percent = stats['success_rate'] * 100
                    st.metric("🎯 Tỷ lệ thành công", f"{success_percent:.0f}%")
                    if stats['solution_quality'] != float('inf'):
                        st.metric("⭐ Chất lượng giải pháp", stats['solution_quality'])
                    else:
                        st.metric("⭐ Chất lượng giải pháp", "Không có")
            
            # Thông báo về việc lưu thống kê ở phía dưới
            if hasattr(st.session_state, 'last_stats_file') and st.session_state.last_stats_file:
                filename = os.path.basename(st.session_state.last_stats_file)
                st.success(f"✅ Đã lưu thống kê vào file: {filename}") 
