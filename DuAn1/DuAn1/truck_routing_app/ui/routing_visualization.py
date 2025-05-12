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
from core.rl_environment import TruckRoutingEnv  # Import RL environment
from core.algorithms.rl_DQNAgent import DQNAgentTrainer  # Import RL agent
from ui import map_display
from core.and_or_search_logic.problem_definition import AndOrProblem
from core.and_or_search_logic.search_algorithm import solve_and_or_problem, FAILURE, NO_PLAN
import sys

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
    
    # Chỉ hiển thị đường đi mũi tên khi đã đến đích
    display_path = path if current_step == total_steps else None
    
    # Tạo start_pos custom để xử lý ẩn icon xe tải ở vị trí bắt đầu
    custom_start_pos = None
    # Chỉ hiển thị xe tải ở vị trí bắt đầu khi KHÔNG đang chạy animation
    # và chỉ ở trạng thái ban đầu (step=0) hoặc đã hoàn thành (step=total_steps)
    if not st.session_state.get("is_playing", False) and (current_step == 0 or current_step == total_steps):
        custom_start_pos = st.session_state.start_pos  # Sử dụng vị trí bắt đầu từ session state
    
    # Vẽ bản đồ với vị trí xe
    map_display.draw_map(
        map_data=map_data,
        start_pos=custom_start_pos,
        visited=visited_positions,
        current_pos=current_position,
        path=display_path  # Chỉ hiển thị đường đi khi đến đích
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
    initial_money = st.session_state.get('initial_money', 1500.0)
    max_fuel = st.session_state.get('max_fuel', 70.0)
    fuel_per_move = st.session_state.get('fuel_per_move', 0.4)
    gas_station_cost = st.session_state.get('gas_station_cost', 30.0)
    toll_base_cost = st.session_state.get('toll_base_cost', 150.0)
    initial_fuel = st.session_state.get('initial_fuel', max_fuel)
    
    # Xử lý riêng cho thuật toán RL
    if algorithm_name == "Học Tăng Cường (RL)":
        try:
            # Kiểm tra xem có mô hình được chọn không
            if "rl_model" not in st.session_state or not st.session_state.rl_model:
                st.error("❌ Chưa chọn mô hình học tăng cường!")
                return None
            
            # Tạo môi trường RL với bản đồ và tham số hiện tại
            rl_env = TruckRoutingEnv(
                map_object=map_data,
                initial_fuel=initial_fuel,
                initial_money=initial_money,
                fuel_per_move=fuel_per_move,
                gas_station_cost=gas_station_cost,
                toll_base_cost=toll_base_cost,
                max_steps_per_episode=2 * grid.shape[0] * grid.shape[1]
            )
            
            # Điều chỉnh tham số dựa trên chiến lược ưu tiên
            priority_strategy = st.session_state.get('rl_priority_strategy', "Cân bằng (mặc định)")
            
            # Áp dụng các điều chỉnh phần thưởng dựa trên chiến lược (điều này sẽ được thực hiện đúng cách nếu environment hỗ trợ)
            if hasattr(rl_env, 'set_reward_weights'):
                if priority_strategy == "Tiết kiệm chi phí":
                    rl_env.set_reward_weights(cost_weight=2.0, time_weight=0.5, safety_weight=1.0)
                elif priority_strategy == "Nhanh nhất":
                    rl_env.set_reward_weights(cost_weight=0.5, time_weight=2.0, safety_weight=0.5)
                elif priority_strategy == "An toàn nhiên liệu":
                    rl_env.set_reward_weights(cost_weight=0.5, time_weight=0.5, safety_weight=2.0)
                else:  # Cân bằng
                    rl_env.set_reward_weights(cost_weight=1.0, time_weight=1.0, safety_weight=1.0)
            
            # Tải model RL
            model_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "saved_models", st.session_state.rl_model)
            
            # Tạo agent và tải model
            agent = DQNAgentTrainer(rl_env)
            agent.load_model(model_path)
            
            # Bắt đầu đo thời gian
            start_time = time.perf_counter()
            
            # Chạy episode và thu thập thông tin
            observation, _ = rl_env.reset()
            path = [rl_env.current_pos]  # Đường đi bắt đầu từ vị trí hiện tại
            visited = [rl_env.current_pos]  # Danh sách các vị trí đã thăm
            terminated = False
            truncated = False
            total_reward = 0
            fuel_consumed = 0
            money_spent = 0
            total_toll_cost = 0
            total_refuel_cost = 0
            refuel_count = 0
            toll_count = 0
            
            # Thực hiện episode
            while not (terminated or truncated):
                # Dự đoán hành động từ agent
                action = agent.predict_action(observation)
                
                # Thực hiện hành động
                next_observation, reward, terminated, truncated, info = rl_env.step(action)
                
                # Cập nhật tổng phần thưởng
                total_reward += reward
                
                # Cập nhật vị trí vào đường đi nếu đã di chuyển
                if rl_env.current_pos not in path:
                    path.append(rl_env.current_pos)
                
                # Thêm vào danh sách đã thăm (để animation)
                if rl_env.current_pos not in visited:
                    visited.append(rl_env.current_pos)
                
                # Cập nhật các số liệu thống kê
                if action <= 3:  # Các hành động di chuyển
                    fuel_consumed += fuel_per_move
                
                if "toll_paid" in info:
                    money_spent += info["toll_paid"]
                    total_toll_cost += info["toll_paid"]
                    toll_count += 1
                
                if "refuel_cost" in info:
                    money_spent += info["refuel_cost"]
                    total_refuel_cost += info["refuel_cost"]
                    refuel_count += 1
                
                # Cập nhật observation
                observation = next_observation
            
            # Kết thúc đo thời gian
            end_time = time.perf_counter()
            execution_time = end_time - start_time
            
            # Tạo trạng thái cho đường đi và animation
            exploration_states = [(pos, 0) for pos in visited]
            
            # Dùng path để tạo truck_states
            truck_states = []
            current_fuel = initial_fuel
            for i, pos in enumerate(path):
                if i > 0:  # Không tính vị trí đầu tiên
                    current_fuel -= fuel_per_move
                truck_states.append((pos, current_fuel))
            
            # Tạo thống kê
            success = rl_env.current_pos == goal
            
            stats = {
                "success_rate": 1.0 if success else 0.0,
                "execution_time": execution_time,
                "path_length": len(path) - 1 if path else 0,  # Trừ vị trí bắt đầu
                "total_reward": total_reward,
                "fuel": observation["fuel"][0] if "fuel" in observation else 0,
                "money": observation["money"][0] if "money" in observation else 0,
                "fuel_consumed": fuel_consumed,
                "money_spent": money_spent,
                "toll_cost": total_toll_cost,
                "refuel_cost": total_refuel_cost,
                "refuel_count": refuel_count,
                "toll_count": toll_count,
                "visited_cells": len(visited),
                "steps": len(visited),
                "memory_usage": sys.getsizeof(visited) + sys.getsizeof(path),
                "is_feasible": success,
                "reason": "Đến đích thành công" if success else "Không thể đến đích"
            }
            
            # Trả về kết quả
            return {
                "path": path,
                "visited": visited,
                "exploration_states": exploration_states,
                "truck_states": truck_states,
                "stats": stats
            }
            
        except Exception as e:
            st.error(f"❌ Lỗi khi chạy thuật toán RL: {str(e)}")
            import traceback
            print(traceback.format_exc())
            return None
    
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
    # 2. Xe đi theo đường đi cuối cùng
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
        algorithm_options = ["BFS", "DFS", "A*", "Greedy", "Local Beam Search", "Simulated Annealing", "Genetic Algorithm", "Học Tăng Cường (RL)"]
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
            "Genetic Algorithm": "Mô phỏng quá trình tiến hóa tự nhiên, sử dụng quần thể, chọn lọc, lai ghép và đột biến.",
            "Học Tăng Cường (RL)": "Sử dụng học tăng cường (Deep Q-Network) để tự học cách tìm đường tối ưu dựa trên kinh nghiệm."
        }
        st.info(f"**{algorithm_name}**: {algorithm_descriptions.get(algorithm_name, 'Không có mô tả.')}")
        
        # Tạo các tab cho các nhóm cấu hình
        tab1, tab2, tab3 = st.tabs(["🚚 Phương tiện", "🛣️ Chi phí", "🧪 Tham số thuật toán"])
        
        with tab1:
            # Cấu hình phương tiện (xăng)
            st.markdown("##### 🛢️ Nhiên liệu")
            col1, col2 = st.columns(2)
            with col1:
                st.slider("Dung tích bình xăng (L):", 
                              min_value=10.0, max_value=50.0, 
                              value=st.session_state.get('max_fuel', 20.0), 
                              step=1.0,
                              key='max_fuel')
            
            with col2:
                # Ensure initial_fuel's max_value is dynamically tied to max_fuel
                current_max_fuel = st.session_state.get('max_fuel', 20.0)
                st.slider("Nhiên liệu ban đầu (L):", 
                                 min_value=5.0, max_value=current_max_fuel, 
                                 value=st.session_state.get('initial_fuel', current_max_fuel), 
                                 step=1.0,
                                 key='initial_fuel')
            
            st.slider("Mức tiêu thụ nhiên liệu (L/ô):", 
                               min_value=0.1, max_value=1.0, 
                               value=st.session_state.get('fuel_per_move', 0.4), 
                               step=0.1,
                               key='fuel_per_move')
        
        with tab2:
            # Cấu hình chi phí
            st.markdown("##### 💰 Chi phí")
            st.slider("Số tiền ban đầu (đ):", 
                              min_value=1000.0, max_value=5000.0, 
                              value=st.session_state.get('initial_money', 2000.0), 
                              step=100.0,
                              key='initial_money')
            
            col1, col2 = st.columns(2)
            with col1:
                st.slider("Chi phí đổ xăng (đ/L):", 
                                     min_value=10.0, max_value=100.0, 
                                     value=st.session_state.get('gas_station_cost', 30.0), 
                                     step=5.0,
                                     key='gas_station_cost')
            
            with col2:
                st.slider("Chi phí trạm thu phí (đ):", 
                                   min_value=50.0, max_value=300.0, 
                                   value=st.session_state.get('toll_base_cost', 150.0), 
                                   step=10.0,
                                   key='toll_base_cost')
        
        with tab3:
            # Cấu hình tham số thuật toán
            st.markdown("##### 🔧 Tham số riêng của thuật toán")
            
            if algorithm_name == "Local Beam Search":
                st.slider("Beam Width:", min_value=2, max_value=50, 
                            value=st.session_state.get('beam_width', 10), 
                            step=1,
                            key='beam_width')
                
                st.checkbox("Sử dụng Stochastic Beam Search", 
                                value=st.session_state.get('use_stochastic', True),
                                key='use_stochastic')
            
            elif algorithm_name == "Simulated Annealing":
                st.slider("Nhiệt độ ban đầu:", min_value=10.0, max_value=500.0, 
                            value=st.session_state.get('initial_temp', 100.0), 
                            step=10.0,
                            key='initial_temp')
                
                st.slider("Tốc độ làm lạnh:", min_value=0.7, max_value=0.99, 
                            value=st.session_state.get('cooling_rate', 0.95), 
                            step=0.01,
                            key='cooling_rate')
                
                st.slider("Số bước trên mỗi nhiệt độ:", min_value=10, max_value=100, 
                            value=st.session_state.get('steps_per_temp', 50), 
                            step=10,
                            key='steps_per_temp')
            
            elif algorithm_name == "Genetic Algorithm":
                st.slider("Kích thước quần thể:", min_value=10, max_value=100, 
                            value=st.session_state.get('pop_size', 50), 
                            step=10,
                            key='pop_size')
                
                col1, col2 = st.columns(2)
                with col1:
                    st.slider("Tỷ lệ lai ghép:", min_value=0.5, max_value=1.0, 
                                value=st.session_state.get('crossover_rate', 0.8), 
                                step=0.05,
                                key='crossover_rate')
                
                with col2:
                    st.slider("Tỷ lệ đột biến:", min_value=0.05, max_value=0.5, 
                                value=st.session_state.get('mutation_rate', 0.2), 
                                step=0.05,
                                key='mutation_rate')
                
                st.slider("Số thế hệ:", min_value=10, max_value=200, 
                            value=st.session_state.get('generations', 100), 
                            step=10,
                            key='generations')
            
            elif algorithm_name == "Học Tăng Cường (RL)":
                # Cấu hình đặc biệt cho RL
                st.markdown("##### 🧠 Mô hình Học Tăng Cường")
                
                # Chọn mô hình đã huấn luyện
                # Tạo một dropdown để chọn mô hình từ thư mục saved_models
                import os
                
                # Đường dẫn đến thư mục saved_models
                models_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "saved_models")
                
                # Kiểm tra xem thư mục có tồn tại không
                if not os.path.exists(models_dir):
                    os.makedirs(models_dir, exist_ok=True)
                    st.warning("⚠️ Thư mục saved_models chưa tồn tại. Đã tạo thư mục mới.")
                
                # Lấy danh sách mô hình trong thư mục
                model_files = [f.replace(".zip", "") for f in os.listdir(models_dir) if f.endswith(".zip")]
                
                if not model_files:
                    st.warning("⚠️ Không tìm thấy mô hình học tăng cường! Vui lòng huấn luyện mô hình trước.")
                    # Thêm link để mở ứng dụng rl_test.py
                    st.markdown("""
                    📝 Bạn có thể huấn luyện mô hình mới bằng cách chạy ứng dụng `rl_test.py`.
                    """)
                else:
                    # Nếu chưa có model được chọn, đặt mô hình đầu tiên là mặc định
                    default_model = st.session_state.get('rl_model', model_files[0] if model_files else None)
                    selected_model = st.selectbox(
                        "Chọn mô hình RL:", 
                        model_files,
                        index=model_files.index(default_model) if default_model in model_files else 0
                    )
                    
                    # Lưu mô hình được chọn vào session state
                    st.session_state.rl_model = selected_model
                    
                    # Hiển thị đường dẫn đầy đủ
                    model_path = os.path.join(models_dir, selected_model)
                    st.info(f"📁 Đường dẫn mô hình: {model_path}")
                
                # Chọn chiến lược ưu tiên (từ hàm phần thưởng)
                st.markdown("##### 🎯 Chiến lược ưu tiên")
                priority_strategy = st.selectbox(
                    "Chiến lược:",
                    ["Cân bằng (mặc định)", "Tiết kiệm chi phí", "Nhanh nhất", "An toàn nhiên liệu"],
                    index=0
                )
                
                # Lưu chiến lược được chọn vào session state
                st.session_state.rl_priority_strategy = priority_strategy
                
                # Hiển thị mô tả chiến lược
                strategy_descriptions = {
                    "Cân bằng (mặc định)": "Cân bằng giữa thời gian, chi phí và an toàn.",
                    "Tiết kiệm chi phí": "Ưu tiên tiết kiệm tiền, tránh trạm thu phí khi có thể.",
                    "Nhanh nhất": "Ưu tiên đường đi ngắn nhất, không quan tâm chi phí.",
                    "An toàn nhiên liệu": "Luôn đảm bảo mức nhiên liệu an toàn, ưu tiên ghé trạm xăng."
                }
                
                st.info(strategy_descriptions[priority_strategy])
            
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
        
        # CSS cho bản đồ và animation (giống với map_display.py)
        st.markdown("""
        <style>
        /* Reset styles để loại bỏ background từ mọi phần tử */
        .map-container, .map-container *, .map-container *:before, .map-container *:after {
            background: transparent !important;
            background-color: transparent !important;
            box-shadow: none !important;
            border: none !important;
        }
        
        .map-container {
            display: flex;
            justify-content: center;
            margin: 20px 0;
            padding: 25px;
            border-radius: 20px;
            transition: all 0.5s ease;
        }
        
        .map-container table {
            border-collapse: collapse;
            border-radius: 15px;
            overflow: hidden;
            transform: perspective(1200px) rotateX(2deg);
            transition: all 0.5s ease;
        }
        
        .map-container:hover table {
            transform: perspective(1200px) rotateX(0deg);
        }
        
        .map-container td {
            width: 64px;
            height: 64px;
            text-align: center;
            padding: 0;
            position: relative;
            border: none;
            transition: all 0.3s ease;
            overflow: hidden;
        }
        
        .map-container td > div {
            width: 100%;
            height: 100%;
            display: flex;
            align-items: center;
            justify-content: center;
            position: relative;
            transition: all 0.3s ease;
        }
        
        .visited-overlay {
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            bottom: 0;
            background-color: rgba(100, 181, 246, 0.05) !important;
            z-index: 1;
            animation: fadeIn 0.7s ease;
        }
        
        .neighbor-overlay {
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            bottom: 0;
            background-color: rgba(255, 215, 0, 0.05) !important;
            z-index: 2;
            animation: pulseGlow 1.5s infinite;
        }
        
        .current-overlay {
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            bottom: 0;
            background-color: rgba(255, 69, 0, 0.08) !important;
            z-index: 3;
            animation: highlightPulse 1.2s infinite;
        }
        
        .path-overlay {
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            bottom: 0;
            background: rgba(76, 175, 80, 0.05) !important;
            z-index: 2;
            animation: pathGlow 3s infinite;
        }
        
        .obstacle-in-path-overlay {
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            bottom: 0;
            background-color: rgba(220, 53, 69, 0.2) !important;
            z-index: 10 !important;
            animation: errorBlink 0.8s infinite;
        }
        
        @keyframes fadeIn {
            from { opacity: 0; transform: scale(0.95); }
            to { opacity: 1; transform: scale(1); }
        }
        
        @keyframes pulseGlow {
            0% { opacity: 0.4; }
            50% { opacity: 0.2; }
            100% { opacity: 0.4; }
        }
        
        @keyframes highlightPulse {
            0% { background-color: rgba(255, 69, 0, 0.08) !important; }
            50% { background-color: rgba(255, 69, 0, 0.15) !important; }
            100% { background-color: rgba(255, 69, 0, 0.08) !important; }
        }
        
        @keyframes pathGlow {
            0% { opacity: 0.4; }
            50% { opacity: 0.7; }
            100% { opacity: 0.4; }
        }
        
        @keyframes errorBlink {
            0% { background-color: rgba(220, 53, 69, 0.2) !important; }
            50% { background-color: rgba(220, 53, 69, 0.35) !important; }
            100% { background-color: rgba(220, 53, 69, 0.2) !important; }
        }
        
        .cell-content {
            position: relative;
            z-index: 4;
            font-size: 32px;
            line-height: 1;
            display: flex;
            align-items: center;
            justify-content: center;
            width: 100%;
            height: 100%;
            transition: all 0.3s ease;
        }
        
        .cell-content:hover {
            transform: scale(1.1);
        }
        
        /* Hiệu ứng khi di chuột qua bản đồ */
        .map-container tr {
            transition: all 0.3s ease;
        }
        
        .map-container tr:hover {
            transform: translateY(-2px);
        }
        
        /* Xóa các đường kẻ giữa các ô */
        .map-container td::after {
            display: none;
        }
        
        .current-pos-cell .cell-content {
            animation: pulseTruck 1.2s infinite ease-in-out;
            transform-origin: center;
            z-index: 5;
        }
        
        @keyframes pulseTruck {
            0% { transform: scale(1); }
            50% { transform: scale(1.15); }
            100% { transform: scale(1); }
        }
        
        /* Xe tải luôn hiển thị rõ ràng */
        .truck-icon {
            font-size: 40px !important;
            filter: drop-shadow(0 2px 5px rgba(0,0,0,0.1));
            color: #FF5722;
        }
        </style>
        """, unsafe_allow_html=True)
        
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
                            None,  # Không hiển thị đường đi khi đang tìm đường
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
                            # Chỉ hiển thị đường đi ở bước cuối cùng
                            display_path = None
                            draw_visualization_step(
                                st.session_state.map,
                                current_visited,
                                current_pos,
                                display_path,
                                current_neighbors
                            )
                    else:
                        # Hiển thị kết quả cuối cùng với đường đi
                        with map_container:
                            draw_visualization_step(
                                st.session_state.map,
                                visited,
                                None,
                                path  # Chỉ hiển thị đường đi ở bước cuối cùng
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
                            # Không hiển thị xe tải ở vị trí bắt đầu khi animation đang chạy
                            map_display.draw_map(
                                map_data=st.session_state.map,
                                start_pos=None,  # Không hiển thị xe tải ở vị trí bắt đầu khi đang di chuyển
                                visited=visited_positions,
                                current_pos=current_pos,
                                # Không hiển thị đường đi mũi tên khi xe đang di chuyển
                                path=None
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
                                # Chỉ hiển thị đường đi mũi tên khi đã đến đích
                                display_path = path if current_step == total_steps else None
                                
                                # Tạo start_pos custom để xử lý ẩn icon xe tải ở vị trí bắt đầu
                                custom_start_pos = None
                                # Chỉ hiển thị xe tải ở vị trí bắt đầu khi KHÔNG đang chạy animation
                                # và chỉ ở trạng thái ban đầu (step=0) hoặc đã hoàn thành (step=total_steps)
                                if not st.session_state.get("is_playing", False) and (current_step == 0 or current_step == total_steps):
                                    custom_start_pos = st.session_state.start_pos  # Sử dụng vị trí bắt đầu từ session state
                                
                                map_display.draw_map(
                                    map_data=st.session_state.map,
                                    start_pos=custom_start_pos,
                                    visited=visited_positions,
                                    current_pos=current_pos,
                                    path=display_path
                                )
    
    # Hiển thị thống kê chi tiết ở phần dưới cùng sau khi có kết quả
    if "current_result" in st.session_state:
        # Tạo một vùng phân tách
        st.markdown("""
        <hr style="height:3px;border:none;background-color:#3498db;margin:30px 0;opacity:0.3;">
        """, unsafe_allow_html=True)
        
        with st.expander("📊 Xem thống kê chi tiết", expanded=False):
            stats = st.session_state.current_result["stats"]
            
            # Kiểm tra nếu đang sử dụng thuật toán RL thì thêm tab cho RL
            if st.session_state.algorithm == "Học Tăng Cường (RL)":
                stat_tabs = st.tabs(["Quá trình tìm kiếm", "Nhiên liệu", "Chi phí & Tiền", "Hiệu suất", "RL Metrics"])
            else:
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
            
            # Tab hiển thị chỉ số RL nếu sử dụng thuật toán RL
            if st.session_state.algorithm == "Học Tăng Cường (RL)" and len(stat_tabs) > 4:
                with stat_tabs[4]:
                    st.markdown("##### 🧠 Chỉ số Học Tăng Cường")
                    
                    # Hiển thị các thông số đặc trưng của RL
                    rl_cols1 = st.columns(3)
                    with rl_cols1[0]:
                        if "total_reward" in stats:
                            st.metric("Tổng phần thưởng", f"{stats['total_reward']:.2f}")
                        else:
                            st.metric("Tổng phần thưởng", "N/A")
                    
                    with rl_cols1[1]:
                        if "refuel_count" in stats:
                            st.metric("Số lần đổ xăng", stats['refuel_count'])
                        else:
                            st.metric("Số lần đổ xăng", "0")
                    
                    with rl_cols1[2]:
                        if "toll_count" in stats:
                            st.metric("Số trạm thu phí đã qua", stats['toll_count'])
                        else:
                            st.metric("Số trạm thu phí đã qua", "0")
                    
                    # Thông tin về chiến lược và model
                    st.markdown("##### 🎯 Thông tin model")
                    rl_cols2 = st.columns(2)
                    with rl_cols2[0]:
                        priority_strategy = st.session_state.get('rl_priority_strategy', "Cân bằng (mặc định)")
                        st.info(f"**Chiến lược ưu tiên**: {priority_strategy}")
                    
                    with rl_cols2[1]:
                        if "rl_model" in st.session_state:
                            st.info(f"**Model đã sử dụng**: {st.session_state.rl_model}")
                        else:
                            st.info("**Model đã sử dụng**: Không xác định")
                    
                    # Hiển thị ghi chú về khả năng thích ứng
                    if priority_strategy == "Tiết kiệm chi phí":
                        st.success("💡 Agent ưu tiên tránh trạm thu phí khi có thể và tối ưu hóa lượng nhiên liệu sử dụng.")
                    elif priority_strategy == "Nhanh nhất":
                        st.success("💡 Agent ưu tiên tìm đường ngắn nhất, có thể chấp nhận chi phí cao hơn.")
                    elif priority_strategy == "An toàn nhiên liệu":
                        st.success("💡 Agent duy trì mức nhiên liệu an toàn và ghé trạm xăng thường xuyên hơn.")
                    else:
                        st.success("💡 Agent cân bằng giữa thời gian, chi phí và an toàn.")
            
            # Thông báo về việc lưu thống kê ở phía dưới
            if hasattr(st.session_state, 'last_stats_file') and st.session_state.last_stats_file:
                filename = os.path.basename(st.session_state.last_stats_file)
                st.success(f"✅ Đã lưu thống kê vào file: {filename}")

    st.markdown("---") # Phân cách
    render_and_or_sandbox_section() # Gọi phần thử nghiệm AND-OR

# Helper function to format plan for Streamlit display
def format_plan_for_streamlit(plan, indent_level=0, current_depth=0, max_depth=15):
    # DEBUG: Print the plan being processed at current level
    # print(f"DEBUG: format_plan_for_streamlit(indent={indent_level}, depth={current_depth}) received plan: {plan}")

    base_indent = "  " * indent_level

    if current_depth > max_depth:
        return f"{base_indent}... (Chi tiết kế hoạch quá sâu, đã được cắt bớt tại đây)"

    if plan == FAILURE:
        # print(f"DEBUG: Plan is FAILURE")
        return f"{base_indent}Thất bại: Không tìm thấy kế hoạch."
    if plan == NO_PLAN:
        # print(f"DEBUG: Plan is NO_PLAN")
        return f"{base_indent}Mục tiêu đạt được (không cần hành động thêm)."

    if not isinstance(plan, dict):
        # print(f"DEBUG: Plan is not a dict: {type(plan)}")
        return f"{base_indent}{str(plan)}"

    plan_type = plan.get("type")
    # print(f"DEBUG: Plan type: {plan_type}")
    output_lines = []

    if plan_type == "OR_PLAN_STEP":
        action = plan.get('action')
        sub_plan = plan.get('sub_plan')
        output_lines.append(f"{base_indent}NẾU TRẠNG THÁI CHO PHÉP, LÀM: {action}")
        if sub_plan is not None:
            # Recursive call increments current_depth
            output_lines.append(format_plan_for_streamlit(sub_plan, indent_level + 1, current_depth + 1, max_depth))
    
    elif plan_type == "AND_PLAN_CONDITIONAL":
        output_lines.append(f"{base_indent}MONG ĐỢI một trong các kết quả sau:")
        contingencies = plan.get('contingencies', {})
        if not contingencies:
             output_lines.append(f"{base_indent}  (Không có tình huống dự phòng nào được định nghĩa)")
        for desc, contingent_plan in contingencies.items():
            output_lines.append(f"{base_indent}  - NẾU ({desc}):")
            if contingent_plan is not None:
                # Recursive call increments current_depth
                output_lines.append(format_plan_for_streamlit(contingent_plan, indent_level + 2, current_depth + 1, max_depth))
            
    elif plan_type == "AND_PLAN_SINGLE_OUTCOME":
        desc = plan.get('description')
        actual_plan = plan.get('plan')
        output_lines.append(f"{base_indent}KẾT QUẢ MONG ĐỢI ({desc}):")
        if actual_plan is not None:
            # Recursive call increments current_depth
            output_lines.append(format_plan_for_streamlit(actual_plan, indent_level + 1, current_depth + 1, max_depth))
    
    else:
        # print(f"DEBUG: Unknown plan type or structure for plan: {plan}")
        output_lines.append(f"{base_indent}Cấu trúc kế hoạch không xác định: {str(plan)}")
        
    # print(f"DEBUG: output_lines before join (indent={indent_level}): {output_lines}")
    final_output = "\n".join(line for line in output_lines if line is not None and line.strip() != "")
    # print(f"DEBUG: final_output after join (indent={indent_level}): repr='{repr(final_output)}'")
    return final_output

def render_and_or_sandbox_section():
    st.header("Tìm Kiếm AND-OR Dự Phòng trên Bản Đồ Hiện Tại")
    st.markdown("""
    Thực hiện thuật toán AND-OR search trên bản đồ và với điểm bắt đầu/kết thúc bạn đã chọn.
    Thuật toán tìm kế hoạch đảm bảo, tính đến khả năng xe hỏng (10% sau mỗi lần đến một ô mới) và có thể sửa chữa.
    Lưu ý: Thuật toán này có thể chạy chậm trên bản đồ lớn do khám phá không gian trạng thái phức tạp.
    """)

    # Check if map, start_pos, and end_pos are available in session_state
    if "map" not in st.session_state or st.session_state.map is None:
        st.warning("⚠️ Vui lòng tạo bản đồ trước.")
        return
    if "start_pos" not in st.session_state or st.session_state.start_pos is None:
        st.warning("⚠️ Vui lòng thiết lập vị trí bắt đầu trên bản đồ.")
        return
    if "end_pos" not in st.session_state or st.session_state.end_pos is None:
        st.warning("⚠️ Vui lòng thiết lập điểm đích trên bản đồ.")
        return

    # Display the current start and end points for confirmation
    st.info(f"Điểm xuất phát hiện tại: {st.session_state.start_pos}, Điểm đích hiện tại: {st.session_state.end_pos}")

    if st.button("Bắt đầu Tìm Kế Hoạch AND-OR trên Bản Đồ", key="and_or_find_plan_on_map_button"):
        map_data = st.session_state.map
        # Ensure map_data.grid is the actual numpy grid, or adjust as needed
        # Example: grid = map_data.grid if hasattr(map_data, 'grid') else map_data
        # For now, assuming map_data directly has a .grid attribute.
        # Based on your get_grid_from_map_data, it seems map_data might be an object with a .grid attribute.
        grid = getattr(map_data, 'grid', map_data) # Safely get .grid or use map_data itself
        if not isinstance(grid, np.ndarray):
            st.error("Lỗi: Dữ liệu bản đồ không phải là một numpy array hợp lệ.")
            return
            
        start_coord = st.session_state.start_pos # Should be (x,y)
        dest_coord = st.session_state.end_pos   # Should be (x,y)

        # Pass the OBSTACLE_CELL definition to problem if it's not hardcoded there
        # For now, AndOrProblem hardcodes self.OBSTACLE_CELL = -1
        # If your global OBSTACLE_CELL is different, this needs to be reconciled.
        problem = AndOrProblem(map_grid=grid,
                               start_coord=start_coord,
                               final_dest_coord=dest_coord)
        
        with st.spinner(f"Đang tìm kiếm kế hoạch AND-OR từ {start_coord} đến {dest_coord}..."):
            solution_plan = solve_and_or_problem(problem)
        
        # DEBUG: Xác nhận thuật toán đã chạy xong
        st.info("DEBUG: solve_and_or_problem đã hoàn thành.") 

        if solution_plan == FAILURE:
            st.error("Không tìm thấy kế hoạch dự phòng đảm bảo trên bản đồ này.")
        else:
            st.success("Đã tìm thấy kế hoạch dự phòng đảm bảo!")
            
            # Bước 1: Định dạng kế hoạch (đây có thể là phần tốn thời gian)
            with st.spinner("Đang định dạng kế hoạch..."):
                plan_details = format_plan_for_streamlit(solution_plan)
            
            # Bước 2: Lấy độ dài thực tế của chuỗi đã định dạng
            actual_display_length = len(plan_details)
            st.write(f"Thông tin gỡ lỗi: Độ dài thực tế của chi tiết kế hoạch đã định dạng: {actual_display_length} ký tự.")

            # Bước 3: Hiển thị kế hoạch, có cảnh báo và cắt bớt nếu cần
            if actual_display_length > 100000: 
                 st.warning(f"Chi tiết kế hoạch rất lớn ({actual_display_length} ký tự). Việc hiển thị có thể làm chậm trình duyệt.")

            with st.spinner("Đang chuẩn bị hiển thị chi tiết kế hoạch..."):
                st.markdown("#### Chi tiết Kế Hoạch:")
                
                TRUNCATION_THRESHOLD = 200000 
                display_key = "and_or_map_plan_details_area"

                if actual_display_length > TRUNCATION_THRESHOLD:
                    st.info(f"Chi tiết kế hoạch quá dài ({actual_display_length} ký tự). Nội dung sau đây đã được cắt bớt để đảm bảo hiệu suất.")
                    truncated_details = plan_details[:TRUNCATION_THRESHOLD] + "\n\n... (NỘI DUNG ĐÃ ĐƯỢC CẮT BỚT DO QUÁ DÀI)"
                    st.text_area("Kế hoạch AND-OR (đã cắt bớt):", value=truncated_details, height=400, key=display_key)
                else:
                    st.text_area("Kế hoạch AND-OR:", value=plan_details, height=400, key=display_key)

# Make sure to import necessary components at the top of the file
# from core.and_or_search_logic.problem_definition import AndOrProblem
# from core.and_or_search_logic.search_algorithm import solve_and_or_problem, FAILURE, NO_PLAN
# import streamlit as st
# (These imports should be added at the top if not already present) 
