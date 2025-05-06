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
from core.algorithms.base_search import BaseSearch
from core.algorithms.bfs import BFS
from core.algorithms.dfs import DFS
from core.algorithms.astar import AStar
from core.algorithms.greedy import GreedySearch
from core.algorithms.hill_climbing import HillClimbing
from core.algorithms.local_beam import LocalBeamSearch
from ui.map_display import draw_map, draw_route

def draw_visualization_step(map_data, visited, current_pos, path=None, current_neighbors=None):
    """Vẽ một bước của thuật toán với các hiệu ứng trực quan."""
    try:
        # Vẽ bản đồ với các hiệu ứng trực quan
        draw_map(
            map_data=map_data,
            visited=visited,
            current_neighbors=current_neighbors,
            current_pos=current_pos,
            path=path
        )
    except Exception as e:
        st.error(f"Lỗi khi vẽ bước trực quan: {str(e)}")

def save_algorithm_stats(algorithm_name: str, stats: dict):
    """Lưu thống kê thuật toán vào file JSON
    
    Args:
        algorithm_name: Tên thuật toán
        stats: Dictionary chứa thống kê
        
    Returns:
        str: Đường dẫn đến file đã lưu hoặc None nếu có lỗi
    """
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
                "value": stats.get("initial_fuel", 0),
                "comment": "Nhiên liệu ban đầu (L)"
            },
            "remaining_fuel": {
                "value": stats.get("remaining_fuel", 0),
                "comment": "Nhiên liệu còn lại (L)"
            },
            "fuel_consumed": {
                "value": stats.get("fuel_consumed", 0),
                "comment": "Nhiên liệu đã tiêu thụ (L)"
            }
        },
        "costs": {  # Chi phí hành trình
            "total_cost": {
                "value": stats.get("total_cost", 0),
                "comment": "Tổng chi phí (đ)"
            },
            "fuel_cost": {
                "value": stats.get("fuel_cost", 0),
                "comment": "Chi phí nhiên liệu (đ)"
            },
            "toll_cost": {
                "value": stats.get("toll_cost", 0),
                "comment": "Chi phí trạm thu phí (đ)"
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
    # Lấy grid từ map_data
    grid = map_data.grid if hasattr(map_data, 'grid') else map_data
    
    # Khởi tạo thuật toán
    if algorithm_name == "BFS":
        algorithm = BFS(grid)
    elif algorithm_name == "DFS":
        algorithm = DFS(grid)
    elif algorithm_name == "A*":
        algorithm = AStar(grid)
    elif algorithm_name == "Greedy":
        algorithm = GreedySearch(grid)
    elif algorithm_name == "Hill Climbing":
        algorithm = HillClimbing(grid)
    elif algorithm_name == "Local Beam Search":
        algorithm = LocalBeamSearch(grid)
    else:
        st.error(f"Thuật toán {algorithm_name} không được hỗ trợ!")
        return None
    
    # Bắt đầu đo thời gian
    start_time = time.perf_counter()
    
    # Chạy thuật toán
    path = algorithm.search(start, goal)
    
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
    else:
        stats["success_rate"] = 0.0
        stats["solution_quality"] = float('inf')
    
    # Xử lý thêm cho BFS và DFS để hiển thị thông tin về tính khả thi
    if algorithm_name in ["BFS", "DFS"] and path:
        # Kiểm tra nếu nhiên liệu về 0, có thể đường đi không khả thi
        if stats["fuel"] <= 0:
            stats["is_feasible"] = False
            stats["reason"] = "Hết nhiên liệu trên đường đi"
            stats["success_rate"] = 0.0  # Cập nhật lại tỷ lệ thành công
        else:
            stats["is_feasible"] = True
            stats["reason"] = "Đường đi khả thi"
    else:
        # Các thuật toán khác đã xét ràng buộc trong quá trình tìm kiếm
        stats["is_feasible"] = True if path and stats["fuel"] > 0 else False
        stats["reason"] = "Đường đi khả thi" if path and stats["fuel"] > 0 else "Không tìm thấy đường đi khả thi"
    
    # Lấy danh sách các ô đã thăm theo thứ tự thời gian cho animation
    visited_list = algorithm.get_visited()
    
    # Đảm bảo không có trùng lặp trong visited_list
    visited_unique = []
    visited_set = set()
    for pos in visited_list:
        if pos not in visited_set:
            visited_unique.append(pos)
            visited_set.add(pos)
    
    # Sau khi có kết quả, lưu thống kê
    if stats:
        map_size = (grid.shape[0], grid.shape[1])
        stats_file = save_algorithm_stats(algorithm_name, stats)
        st.session_state.last_stats_file = stats_file
    
    return {
        "path": path,
        "visited": visited_unique,
        "stats": stats
    }

def render_routing_visualization():
    """Render tab định tuyến và tối ưu hệ thống."""
    st.markdown("## 🗺️ Định Tuyến & Tối Ưu Hệ Thống")
    
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
    
    # Tạo layout chính
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Container cho bản đồ và trực quan
        map_container = st.empty()
        
        # Hiển thị bản đồ ban đầu
        with map_container:
            draw_map(st.session_state.map)
    
    with col2:
        # Chọn thuật toán
        st.markdown("### 🔍 Chọn thuật toán")
        algorithm = st.selectbox(
            "Thuật toán tìm đường:",
            ["BFS", "DFS", "A*", "Greedy", "Hill Climbing", "Local Beam Search"],
            help="Chọn thuật toán để tìm đường đi tối ưu"
        )
        
        # Lưu thuật toán đã chọn vào session state
        st.session_state.algorithm = algorithm
        
        # Hiển thị mô tả thuật toán
        algorithm_descriptions = {
            "BFS": "Tìm kiếm theo chiều rộng, đảm bảo tìm được đường đi ngắn nhất.",
            "DFS": "Tìm kiếm theo chiều sâu, thích hợp cho không gian tìm kiếm lớn.",
            "A*": "Kết hợp tìm kiếm tốt nhất và heuristic, tối ưu và hiệu quả. Tự động ưu tiên tìm trạm xăng khi sắp hết nhiên liệu.",
            "Greedy": "Luôn chọn bước đi có vẻ tốt nhất tại thời điểm hiện tại.",
            "Hill Climbing": "Tìm kiếm cục bộ, luôn di chuyển theo hướng tốt hơn.",
            "Local Beam Search": "Duy trì nhiều trạng thái cùng lúc, tăng khả năng tìm kiếm."
        }
        st.info(algorithm_descriptions[algorithm])
        
        # Nút tìm đường
        if st.button("🔍 Tìm đường", use_container_width=True):
            with st.spinner("🔄 Đang tìm đường..."):
                try:
                    result = run_algorithm(
                        algorithm,
                        st.session_state.map,
                        st.session_state.start_pos,
                        st.session_state.end_pos
                    )
                    if result:
                        st.session_state.current_result = result
                        st.session_state.visualization_step = 0
                        st.session_state.is_playing = False
                        
                        # Kiểm tra tính khả thi của đường đi
                        if result["stats"]["is_feasible"]:
                            st.success("✅ Đã tìm thấy đường đi khả thi!")
                        else:
                            st.warning("⚠️ Đã tìm được một phần đường đi nhưng không thể đến đích (hết nhiên liệu)!")
                    else:
                        st.error("❌ Không thể tìm được đường đi!")
                except Exception as e:
                    st.error(f"❌ Lỗi khi thực thi thuật toán: {str(e)}")
                    return
    
    # Hiển thị kết quả nếu có
    if "current_result" in st.session_state:
        st.markdown("### 📊 Kết quả tìm đường")
        
        # Hiển thị thống kê
        stats = st.session_state.current_result["stats"]
        path = st.session_state.current_result["path"]
        algorithm = st.session_state.algorithm if "algorithm" in st.session_state else ""
        
        # Kiểm tra tính khả thi của đường đi
        is_feasible = stats.get("is_feasible", False)
        reason = stats.get("reason", "")
        
        # Hiển thị thông báo trạng thái
        if is_feasible:
            st.success("✅ Đường đi khả thi đến đích")
        else:
            if path:
                st.warning(f"⚠️ {reason} - Hiển thị đường đi một phần")
            else:
                st.error("❌ Không tìm thấy đường đi")
        
        # Hiển thị thống kê chi tiết
        st.markdown("#### 📈 Thống kê chi tiết:")
        
        # Thông tin về quá trình tìm kiếm
        st.markdown("**Quá trình tìm kiếm:**")
        search_cols = st.columns(3)
        with search_cols[0]:
            st.metric("Số bước thực hiện", stats["steps"], help="Số bước thuật toán đã thực hiện để tìm đường")
        with search_cols[1]:
            st.metric("Số ô đã thăm", stats["visited"], help="Tổng số ô đã được duyệt trong quá trình tìm kiếm")
        with search_cols[2]:
            st.metric("Độ dài đường đi", stats["path_length"], help="Số bước di chuyển trên đường đi tìm được")

        # Thông tin về nhiên liệu
        st.markdown("**Thông tin nhiên liệu:**")
        fuel_cols = st.columns(3)
        with fuel_cols[0]:
            initial_fuel = BaseSearch.MAX_FUEL  # Lấy giá trị từ BaseSearch
            st.metric("Xăng ban đầu", f"{initial_fuel:.1f}l", help="Lượng xăng khi bắt đầu hành trình")
        with fuel_cols[1]:
            st.metric("Xăng đã tiêu thụ", f"{(initial_fuel - stats['fuel']):.1f}l", help="Lượng xăng đã sử dụng trong hành trình")
        with fuel_cols[2]:
            st.metric("Xăng còn lại", f"{stats['fuel']:.1f}l", help="Lượng xăng còn lại khi kết thúc")

        # Thông tin về chi phí
        st.markdown("**Chi phí hành trình:**")
        cost_cols = st.columns(3)
        with cost_cols[0]:
            st.metric("Chi phí nhiên liệu", f"{stats['fuel_cost']:.1f}đ", help="Chi phí đổ xăng tại các trạm")
        with cost_cols[1]:
            st.metric("Chi phí trạm thu phí", f"{stats['toll_cost']:.1f}đ", help="Tổng chi phí qua các trạm thu phí")
        with cost_cols[2]:
            st.metric("Tổng chi phí", f"{stats['total_cost']:.1f}đ", help="Tổng chi phí = Chi phí nhiên liệu + Chi phí trạm thu phí")

        # Phân tích hiệu suất
        st.markdown("#### 🔍 Phân tích hiệu suất:")
        
        # Thời gian và bộ nhớ
        perf_cols1 = st.columns(2)
        with perf_cols1[0]:
            execution_time_ms = stats['execution_time'] * 1000  # Chuyển đổi sang milliseconds
            st.metric("⏱️ Thời gian chạy", f"{execution_time_ms:.2f}ms", help="Thời gian thực thi thuật toán (milliseconds)")
        with perf_cols1[1]:
            memory_kb = stats['memory_usage'] / 1024
            st.metric("💾 Bộ nhớ sử dụng", f"{memory_kb:.2f}KB", help="Ước tính bộ nhớ sử dụng cho việc lưu trữ các ô đã thăm")

        # Tỷ lệ thành công và chất lượng giải pháp
        perf_cols2 = st.columns(2)
        with perf_cols2[0]:
            success_percent = stats['success_rate'] * 100
            success_text = "Tìm được đường đi khả thi" if success_percent == 100 else "Không tìm được đường đi khả thi"
            st.metric("🎯 Tỷ lệ thành công", f"{success_percent:.0f}%", help=success_text)
        with perf_cols2[1]:
            if stats['solution_quality'] != float('inf'):
                quality_text = f"Độ dài đường đi: {stats['solution_quality']} bước"
                st.metric("⭐ Chất lượng giải pháp", stats['solution_quality'], help=quality_text)
            else:
                st.metric("⭐ Chất lượng giải pháp", "Không có", help="Không tìm được đường đi khả thi")
        
        # Hiển thị bản đồ với đường đi (dù là một phần) và các ô đã thăm
        with map_container:
            draw_map(
                st.session_state.map,
                visited=st.session_state.current_result["visited"],
                path=path  # Hiển thị đường đi một phần nếu có
            )
        
        # Điều khiển trực quan
        st.markdown("### 🎬 Trực quan hóa thuật toán")
        
        # Điều khiển tốc độ
        speed = st.slider(
            "Tốc độ hiển thị:",
            min_value=1,
            max_value=10,
            value=5,
            help="Điều chỉnh tốc độ hiển thị (1: chậm nhất, 10: nhanh nhất)"
        )
        
        # Nút điều khiển
        control_cols = st.columns(4)
        with control_cols[0]:
            if st.button("⏮️ Về đầu", use_container_width=True):
                st.session_state.visualization_step = 0
                st.session_state.is_playing = False
        with control_cols[1]:
            if st.button("▶️ Chạy/Tạm dừng", use_container_width=True):
                st.session_state.is_playing = not st.session_state.is_playing
        with control_cols[2]:
            if st.button("⏭️ Kết thúc", use_container_width=True):
                st.session_state.visualization_step = len(st.session_state.current_result["visited"])
                st.session_state.is_playing = False
        with control_cols[3]:
            if st.button("🔄 Làm mới", use_container_width=True):
                st.session_state.visualization_step = 0
                st.session_state.is_playing = False
        
        # Thanh tiến trình
        if "visualization_step" not in st.session_state:
            st.session_state.visualization_step = 0
        
        total_steps = len(st.session_state.current_result["visited"])
        current_step = st.session_state.visualization_step
        progress = float(current_step) / total_steps if total_steps > 0 else 0
        st.progress(progress, text=f"Bước {current_step}/{total_steps}")
        
        # Xử lý animation
        visited = st.session_state.current_result["visited"]
        path = st.session_state.current_result["path"]
        
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
        
        # Thông báo về việc lưu thống kê
        if hasattr(st.session_state, 'last_stats_file') and st.session_state.last_stats_file:
            filename = os.path.basename(st.session_state.last_stats_file)
            st.success(f"✅ Đã lưu thống kê vào: {filename}") 