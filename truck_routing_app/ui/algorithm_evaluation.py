"""
Module for Algorithm Evaluation Page.
Allows users to load and compare statistics from different algorithm runs.
"""
import streamlit as st
import os
import json
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go # For Radar Chart
from typing import List, Dict, Any
import numpy as np # For statistical calculations

# Định nghĩa đường dẫn tới thư mục chứa file thống kê
# Giả sử file này (algorithm_evaluation.py) nằm trong thư mục ui
# và thư mục statistics nằm cùng cấp với thư mục core
STATS_DIR = os.path.join(os.path.dirname(__file__), "..", "statistics")

def extract_algorithm_from_file(file_path: str) -> str:
    """Trích xuất tên thuật toán từ file thống kê với xử lý nhiều định dạng."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Thử nhiều cấu trúc file có thể có
        # Cấu trúc 1: {"algorithm": {"value": "AlgorithmName"}}
        if isinstance(data, dict) and "algorithm" in data:
            if isinstance(data["algorithm"], dict) and "value" in data["algorithm"]:
                return data["algorithm"]["value"]
            elif isinstance(data["algorithm"], str):
                return data["algorithm"]
        
        # Cấu trúc 2: {"algorithm": "AlgorithmName"}
        if isinstance(data, dict) and "algorithm" in data and isinstance(data["algorithm"], str):
            return data["algorithm"]
            
        # Cấu trúc 3: Thuật toán ở trong path của file
        algorithm_name = os.path.basename(file_path)
        if "_" in algorithm_name:
            parts = algorithm_name.split("_")
            for part in parts:
                if "A*" in part or "dijkstra" in part.lower() or "bfs" in part.lower() or "dfs" in part.lower():
                    return part
        
        # Không tìm thấy thuật toán
        return "Unknown"
    except Exception as e:
        print(f"Lỗi khi đọc thuật toán từ file {file_path}: {e}")
        return "Unknown"

def get_stat_files() -> List[str]:
    """Lấy danh sách các file JSON trong thư mục statistics."""
    if not os.path.exists(STATS_DIR):
        return []
    files = [f for f in os.listdir(STATS_DIR) if f.endswith('.json')]
    return sorted(files, reverse=True) # Sắp xếp file mới nhất lên đầu

def load_json_data(filename: str) -> Dict[str, Any]:
    """Tải dữ liệu từ một file JSON."""
    filepath = os.path.join(STATS_DIR, filename)
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data
    except Exception as e:
        st.error(f"Lỗi khi tải file {filename}: {e}")
        return {}

def extract_metrics_from_data(data: Dict[str, Any], filename: str) -> Dict[str, Any]:
    """Trích xuất các số liệu quan trọng từ dữ liệu JSON đã tải."""
    if not data:
        return {"filename": filename, "error": "No data"}
    
    get_value = lambda dp, key_path, default=None: \
        dp.get(key_path[0], {}).get(key_path[1], {}).get("value", default) if len(key_path) == 2 and isinstance(dp.get(key_path[0]), dict) and isinstance(dp.get(key_path[0],{}).get(key_path[1]), dict) else \
        dp.get(key_path[0], {}).get("value", default) if len(key_path) == 1 and isinstance(dp.get(key_path[0]), dict) else \
        default

    metrics = {
        "filename": filename,
        "algorithm": get_value(data, ("algorithm",), "N/A"),
        "timestamp": get_value(data, ("timestamp",), "N/A"),
        "map_size": str(get_value(data, ("map_size",), "N/A")), # Ensure map_size is a string
        "path_length": float(get_value(data, ("search_process", "path_length"), 0.0)),
        "execution_time_ms": round(float(get_value(data, ("performance", "execution_time"), 0.0)) * 1000, 2),
        "total_cost": float(get_value(data, ("costs", "total_cost"), 0.0)),
        "fuel_consumed": float(get_value(data, ("fuel_info", "fuel_consumed"), 0.0)),
        "remaining_fuel": float(get_value(data, ("fuel_info", "remaining_fuel"), 0.0)),
        "is_feasible": bool(get_value(data, ("feasibility", "is_feasible"), False)),
        "reason_infeasible": str(get_value(data, ("feasibility", "reason"), "")),
        "steps": int(get_value(data, ("search_process", "steps"), 0)),
        "visited_cells": int(get_value(data, ("search_process", "visited_cells"), 0)),
        "initial_fuel": float(get_value(data, ("fuel_info", "initial_fuel"), 0.0)),
        "fuel_cost": float(get_value(data, ("costs", "fuel_cost"), 0.0)),
        "toll_cost": float(get_value(data, ("costs", "toll_cost"), 0.0)),
        "initial_money": float(get_value(data, ("costs", "initial_money"), 0.0)),
        "remaining_money": float(get_value(data, ("costs", "remaining_money"), 0.0)),
        "memory_usage_kb": round(float(get_value(data, ("performance", "memory_usage"), 0.0)) / 1024, 2)
    }
    return metrics

def aggregate_metrics(df_all_runs: pd.DataFrame) -> pd.DataFrame:
    """Gom nhóm và tính toán các số liệu thống kê tổng hợp cho mỗi thuật toán."""
    if df_all_runs.empty:
        return pd.DataFrame()

    # Bảng thống kê chuyên nghiệp hơn
    metrics_of_interest = {
        'execution_time_ms': 'Thời Gian (ms)', 
        'path_length': 'Độ Dài Đường Đi',
        'total_cost': 'Tổng Chi Phí (đ)', 
        'fuel_consumed': 'Nhiên Liệu (L)',
        'visited_cells': 'Số Ô Thăm',
        'steps': 'Số Bước',
        'memory_usage_kb': 'Bộ Nhớ (KB)',
    }

    # Chỉ giữ lại các hàng có dữ liệu hợp lệ
    df = df_all_runs.dropna(subset=['algorithm'])
    
    # Tạo DataFrame tổng hợp
    result = []
    
    # Xử lý cho từng thuật toán
    for algo in df['algorithm'].unique():
        algo_data = df[df['algorithm'] == algo]
        
        # Tính tỷ lệ khả thi
        feasibility_rate = algo_data['is_feasible'].fillna(False).mean() * 100
        
        # Các thông số trung bình
        row = {'algorithm': algo, 'Số Lần Chạy': len(algo_data), 'Tỷ Lệ Khả Thi (%)': round(feasibility_rate, 1)}
        
        # Thêm các chỉ số thống kê
        for metric, display_name in metrics_of_interest.items():
            if metric in algo_data.columns and algo_data[metric].notna().any():
                # Lọc bỏ giá trị inf và nan trước khi tính toán
                valid_data = algo_data[metric].replace([np.inf, -np.inf], np.nan).dropna()
                if len(valid_data) > 0:
                    row[f"{display_name} (Min)"] = round(valid_data.min(), 2)
                    row[f"{display_name} (TB)"] = round(valid_data.mean(), 2) 
                    row[f"{display_name} (Max)"] = round(valid_data.max(), 2)
                    
                    # Thêm độ lệch chuẩn nếu có đủ số liệu
                    if len(valid_data) >= 2:
                        row[f"{display_name} (Độ Lệch)"] = round(valid_data.std(), 2)
        
        # Tính điểm hiệu năng tổng hợp (càng thấp càng tốt)
        # Điểm dựa trên thứ hạng của thời gian, độ dài đường đi, chi phí và nhiên liệu
        performance_metrics = ['execution_time_ms', 'path_length', 'total_cost', 'fuel_consumed']
        valid_metrics = [metric for metric in performance_metrics if metric in algo_data.columns]
        
        if valid_metrics:
            valid_means = []
            for metric in valid_metrics:
                # Lọc bỏ giá trị inf và nan trước khi tính giá trị trung bình
                valid_data = algo_data[metric].replace([np.inf, -np.inf], np.nan).dropna()
                if len(valid_data) > 0:
                    valid_means.append(valid_data.mean())
            
            # Chỉ tính điểm nếu có ít nhất một chỉ số hợp lệ
            if valid_means:
                row['Điểm Hiệu Năng'] = round(sum(valid_means) / len(valid_means), 1)
        
        result.append(row)
    
    # Tạo DataFrame từ các hàng đã tính toán
    df_result = pd.DataFrame(result)
    
    # Sắp xếp theo số lần chạy (giảm dần) và điểm hiệu năng (tăng dần)
    if 'Điểm Hiệu Năng' in df_result.columns and not df_result['Điểm Hiệu Năng'].isna().all():
        df_result = df_result.sort_values(['Số Lần Chạy', 'Điểm Hiệu Năng'], ascending=[False, True])
    else:
        df_result = df_result.sort_values('Số Lần Chạy', ascending=False)
        
    return df_result

def ensure_all_algorithms_in_table(df, all_algorithms):
    """Đảm bảo tất cả các thuật toán đều có trong bảng, thêm các hàng trống nếu cần."""
    existing_algos = set(df['algorithm'].unique()) if 'algorithm' in df.columns else set()
    missing_algos = all_algorithms - existing_algos
    
    if not missing_algos:
        return df
    
    # Tạo DataFrame mới với các thuật toán còn thiếu
    missing_rows = []
    for algo in missing_algos:
        new_row = {'algorithm': algo}
        for col in df.columns:
            if col != 'algorithm':
                new_row[col] = None
        missing_rows.append(new_row)
    
    if not missing_rows:
        return df
    
    # Nối với DataFrame gốc
    missing_df = pd.DataFrame(missing_rows)
    return pd.concat([df, missing_df], ignore_index=True)

def scan_all_sources_for_algorithms() -> set:
    """Quét các nguồn chính xác để xác định thuật toán thực sự được sử dụng trong dự án."""
    actual_algorithms = set()
    
    # 1. Ưu tiên tìm kiếm trong thư mục core/algorithms - nơi thuật toán được triển khai
    core_algorithms_dir = os.path.join(os.path.dirname(__file__), "..", "core", "algorithms")
    if os.path.exists(core_algorithms_dir):
        print(f"Đang quét thư mục thuật toán: {core_algorithms_dir}")
        for file in os.listdir(core_algorithms_dir):
            if file.endswith('.py') and not file.startswith('__'):
                # Đọc nội dung file để tìm tên thuật toán
                try:
                    file_path = os.path.join(core_algorithms_dir, file)
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                        
                        # Tìm tên thuật toán từ tên class, docstring hoặc biến tên thuật toán
                        if 'class' in content and ('Algorithm' in file or 'algorithm' in file):
                            # Tìm tên class
                            import re
                            class_matches = re.findall(r'class\s+(\w+)', content)
                            for class_name in class_matches:
                                # Chuyển CamelCase thành tên thân thiện
                                if 'Algorithm' in class_name:
                                    algo_name = class_name.replace('Algorithm', '')
                                    # Chèn khoảng trắng trước mỗi chữ hoa
                                    algo_name = re.sub(r'([A-Z])', r' \1', algo_name).strip()
                                    actual_algorithms.add(algo_name)
                                    
                        # Tìm trong các biến định nghĩa
                        if 'ALGORITHM_NAME' in content or 'algorithm_name' in content:
                            name_matches = re.findall(r'ALGORITHM_NAME\s*=\s*[\'"](.+?)[\'"]', content)
                            name_matches.extend(re.findall(r'algorithm_name\s*=\s*[\'"](.+?)[\'"]', content))
                            for name in name_matches:
                                actual_algorithms.add(name)
                        
                        # Nếu là A*, DFS, BFS, Dijkstra trong tên file, thêm trực tiếp
                        algo_keywords = ['astar', 'a_star', 'a*', 'dfs', 'bfs', 'dijkstra']
                        file_lower = file.lower()
                        for keyword in algo_keywords:
                            if keyword in file_lower:
                                # Định dạng tên
                                if keyword == 'astar' or keyword == 'a_star':
                                    actual_algorithms.add('A*')
                                elif keyword == 'dfs':
                                    actual_algorithms.add('DFS')
                                elif keyword == 'bfs':
                                    actual_algorithms.add('BFS')
                                elif keyword == 'dijkstra':
                                    actual_algorithms.add('Dijkstra')
                                break
                    
                    # Nếu không tìm được bằng các phương pháp trên, lấy từ tên file
                    if not any(algo.lower() in ' '.join(actual_algorithms).lower() for algo in ['A*', 'DFS', 'BFS', 'Dijkstra']):
                        algo_name = file.replace('.py', '').replace('_algorithm', '').replace('_', ' ')
                        actual_algorithms.add(algo_name.title())
                        
                except Exception as e:
                    st.warning(f"Lỗi khi quét thuật toán từ file {file}: {e}")
    
    # 2. Kiểm tra các file thống kê đã chạy (bằng chứng thực sự đã sử dụng)
    if os.path.exists(STATS_DIR):
        stat_files = [f for f in os.listdir(STATS_DIR) if f.endswith('.json')]
        for file in stat_files:
            file_path = os.path.join(STATS_DIR, file)
            try:
                # Đọc thuật toán từ file json
                algo = extract_algorithm_from_file(file_path)
                if algo and algo != "Unknown" and algo != "N/A":
                    actual_algorithms.add(algo)
            except:
                pass
    
    # 3. Loại bỏ các thuật toán thừa
    # Nếu có Genetic Algorithm thì loại bỏ Genetic
    if "Genetic Algorithm" in actual_algorithms and "Genetic" in actual_algorithms:
        actual_algorithms.remove("Genetic")
    
    # Loại bỏ Greedy Best-First nếu không được triển khai rõ ràng
    if "Greedy Best-First" in actual_algorithms and not os.path.exists(os.path.join(core_algorithms_dir, "greedy_best_first.py")):
        actual_algorithms.remove("Greedy Best-First")
    
    # 4. Nếu không có thuật toán nào được tìm thấy, đề xuất một tập thuật toán phổ biến
    if not actual_algorithms:
        st.warning("Không thể tìm thấy thuật toán trong mã nguồn. Đề xuất các thuật toán phổ biến.")
        actual_algorithms = {"A*", "Dijkstra", "BFS", "DFS"}
    
    return actual_algorithms

def render_evaluation_page():
    """Render trang đánh giá thuật toán."""
    st.markdown("""
    <div style="text-align: center; padding: 10px; margin-bottom: 20px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); border-radius: 10px; box-shadow: 0 4px 10px rgba(0,0,0,0.15);">
        <h1 style="color: white; margin: 0;">📊 Đánh Giá Đầy Đủ Hiệu Suất Các Thuật Toán</h1>
        <p style="color: white; opacity: 0.9; margin-top: 5px;">So sánh và phân tích toàn diện kết quả chạy của tất cả các thuật toán tìm đường.</p>
    </div>
    """, unsafe_allow_html=True)

    # Thêm thông tin mô tả cách sử dụng
    with st.expander("ℹ️ Hướng dẫn sử dụng"):
        st.markdown("""
        ### Cách Sử Dụng Trang Đánh Giá
        
        Trang này tự động tải và hiển thị **tất cả các thuật toán** có trong thư mục thống kê. Không cần phải chọn thủ công các thuật toán.
        
        **Các tính năng chính:**
        
        1. **Tổng Quan & So Sánh Nhanh**: Hiển thị bảng so sánh đầy đủ các thuật toán với kết quả mới nhất và biểu đồ trực quan.
        
        2. **Phân Tích Tổng Hợp**: Phân tích thống kê về hiệu suất của các thuật toán qua nhiều lần chạy.
        
        3. **So Sánh Đa Chiều**: Biểu đồ radar để so sánh nhiều chỉ số khác nhau giữa các thuật toán.
        
        4. **Chi Tiết Từng Lần Chạy**: Xem chi tiết từng lần chạy của mọi thuật toán.
        
        **Lưu ý**: Nếu bạn không thấy đầy đủ các thuật toán, có thể các thuật toán đó chưa được chạy hoặc chưa lưu thống kê.
        """)

    stat_files = get_stat_files()

    if not stat_files:
        st.info("ℹ️ Không tìm thấy file thống kê nào trong thư mục `statistics`. Vui lòng chạy thuật toán để tạo dữ liệu.")
        return

    # QUAN TRỌNG: Tự động tải TẤT CẢ file thống kê mà không thông qua multiselect
    selected_files = stat_files

    with st.sidebar:
        st.header("⚙️ Thông Tin Phân Tích")
        st.info("Hệ thống tự động cập nhật theo lần chạy gần nhất của từng thuật toán")
        
        # Hiển thị thông tin về số lượng file đã tải
        st.write(f"📊 Đã tải {len(selected_files)} file thống kê")
        
        # Đặt auto_refresh để tự động cập nhật (nếu có file mới)
        auto_refresh = st.checkbox("Tự động cập nhật", value=True, help="Tự động cập nhật dữ liệu khi có thay đổi")
        
        if auto_refresh:
            st.empty()
            time_placeholder = st.empty()
            from datetime import datetime
            time_placeholder.info(f"Cập nhật lần cuối: {datetime.now().strftime('%H:%M:%S')}")
            
            # Thêm nút làm mới thủ công
            if st.button("Cập nhật thủ công"):
                time_placeholder.info(f"Cập nhật lần cuối: {datetime.now().strftime('%H:%M:%S')}")
                st.experimental_rerun()
    
    # Sử dụng st.cache_data với ttl ngắn hơn để đảm bảo dữ liệu luôn mới nhất
    @st.cache_data(ttl=30) # Chỉ cache trong 30 giây
    def load_metrics_data(files):
        metrics_data = []
        for idx, filename in enumerate(files):
            raw_data = load_json_data(filename)
            if raw_data:
                metrics = extract_metrics_from_data(raw_data, filename)
                if "error" not in metrics:
                    metrics_data.append(metrics)
        return metrics_data
    
    all_metrics_data = load_metrics_data(selected_files)
    
    if not all_metrics_data:
        st.error("❌ Không thể tải hoặc xử lý dữ liệu từ các file đã chọn.")
        return

    # Chuyển đổi thành dataframe và sắp xếp
    df_all_runs = pd.DataFrame(all_metrics_data)
    
    # Tạo một bản sao và thêm cột datetime để sắp xếp theo thời gian
    df_full_compare = df_all_runs.copy()
    df_full_compare['timestamp_dt'] = pd.to_datetime(df_full_compare['timestamp'], errors='coerce')
    
    # Xác định các thuật toán duy nhất từ TẤT CẢ file thống kê
    # Đảm bảo tìm tất cả thuật toán từ mọi file, kể cả các file cũ
    all_algorithms = set()
    
    # Đầu tiên, thử quét tất cả các nguồn có thể để tìm thuật toán
    with st.spinner('Đang tìm kiếm các thuật toán thực tế...'):
        known_algorithms = scan_all_sources_for_algorithms()
        all_algorithms.update(known_algorithms)
    
    # Thứ hai, lấy từ dữ liệu đã tải (để đảm bảo không bỏ sót)
    if df_full_compare is not None and 'algorithm' in df_full_compare.columns:
        for algo in df_full_compare['algorithm'].unique():
            if algo and str(algo) != "nan" and algo != "Unknown":
                all_algorithms.add(algo)
    
    # Nếu vẫn không tìm thấy thuật toán, hiển thị cảnh báo
    if not all_algorithms:
        st.warning("Không thể tự động xác định thuật toán từ file thống kê và mã nguồn. Có thể bạn cần chạy các thuật toán trước để tạo dữ liệu thống kê.")
        all_algorithms = {"Unknown"}
    
    # Lưu danh sách thuật toán có thứ tự
    algorithms = sorted(list(all_algorithms))
    
    # Tự động chọn tất cả thuật toán mặc định
    selected_algorithms = algorithms
    
    # Lấy kết quả mới nhất cho mỗi thuật toán
    df_latest_by_algo = df_full_compare.sort_values('timestamp_dt', ascending=False).groupby('algorithm').first().reset_index()
    
    # Đảm bảo DataFrame chứa đầy đủ tất cả các thuật toán
    df_latest_by_algo = ensure_all_algorithms_in_table(df_latest_by_algo, set(algorithms))
    
    # Tính toán thống kê tổng hợp
    df_aggregated_stats = aggregate_metrics(df_all_runs.copy())

    # --- Tạo Tabs --- 
    tab1, tab2, tab3, tab4 = st.tabs(["🚀 Tổng Quan & So Sánh Nhanh", "📈 Phân Tích Tổng Hợp", "🎯 So Sánh Đa Chiều (Radar)", "📝 Chi Tiết Từng Lần Chạy"])
    
    # Hiển thị thông tin tổng quan về dữ liệu đã tải
    st.sidebar.markdown("---")
    st.sidebar.markdown(f"**🔍 Thông tin thuật toán:**")
    
    # Phân loại thuật toán tìm được
    common_path_algos = [algo for algo in all_algorithms if algo in ["A*", "Dijkstra", "BFS", "DFS"]]
    custom_algos = [algo for algo in all_algorithms if algo not in ["A*", "Dijkstra", "BFS", "DFS"]]
    
    # Hiển thị các thuật toán phổ biến
    if common_path_algos:
        st.sidebar.markdown("**Thuật toán tìm đường cơ bản:**")
        for algo in sorted(common_path_algos):
            st.sidebar.markdown(f"- {algo}")
            
    # Hiển thị các thuật toán tùy chỉnh
    if custom_algos:
        st.sidebar.markdown("**Thuật toán tùy chỉnh:**")
        for algo in sorted(custom_algos):
            st.sidebar.markdown(f"- {algo}")
    
    st.sidebar.markdown(f"- **Tổng số thuật toán**: {len(all_algorithms)}")
    st.sidebar.markdown(f"- **Số file thống kê**: {len(selected_files)}")
    
    if len(all_algorithms) == 0:
        st.sidebar.warning("Không tìm thấy thuật toán nào trong hệ thống.")
    elif len(all_algorithms) <= 4:
        st.sidebar.info("Số lượng thuật toán khá ít. Nếu bạn đã thêm thuật toán mới, hãy đảm bảo rằng chúng được triển khai trong thư mục core/algorithms.")
    
    # Thêm bộ lọc thuật toán (tùy chọn)
    st.sidebar.markdown("---")
    st.sidebar.markdown("**🔎 Lọc Thuật Toán**")
    if st.sidebar.checkbox("Tùy chọn lọc thuật toán", value=False):
        if len(algorithms) > 0:
            selected_algorithms = st.sidebar.multiselect(
                "Chọn thuật toán để so sánh:",
                options=algorithms,
                default=algorithms
            )
            # Đảm bảo luôn có ít nhất một thuật toán được chọn
            if not selected_algorithms:
                st.sidebar.warning("Hãy chọn ít nhất một thuật toán để so sánh")
                selected_algorithms = algorithms
        else:
            st.sidebar.warning("Không tìm thấy thuật toán nào để lọc")
            
    # Thêm thông tin về cách phát hiện thuật toán
    st.sidebar.markdown("---")
    with st.sidebar.expander("ℹ️ Cách phát hiện thuật toán"):
        st.markdown("""
        **Thuật toán được phát hiện từ:**
        
        1. **Mã nguồn** - Quét thư mục `core/algorithms` để tìm các thuật toán được triển khai
        
        2. **File thống kê** - Phân tích từ các file JSON trong thư mục `statistics`
        
        *Nếu bạn thêm thuật toán mới, hãy đảm bảo triển khai nó trong thư mục `core/algorithms` và chạy nó ít nhất một lần để tạo file thống kê.*
        """)

    with tab1: # Tổng Quan & So Sánh Nhanh
        st.markdown("### ⏱️ Bảng So Sánh Đầy Đủ Các Thuật Toán")
        
        if not selected_algorithms:
            st.info("Vui lòng chọn ít nhất một thuật toán để hiển thị so sánh.")
        else:
            # Hiển thị bảng so sánh các lần chạy mới nhất
            st.markdown("#### 📋 Kết Quả Thuật Toán Mới Nhất")
            
            # Lọc theo thuật toán đã chọn
            filtered_df = df_latest_by_algo[df_latest_by_algo['algorithm'].isin(selected_algorithms)]
            
            # Các cột quan trọng cần hiển thị
            display_cols = [
                'algorithm', 'timestamp', 'path_length', 'execution_time_ms', 
                'total_cost', 'fuel_consumed', 'is_feasible', 'map_size'
            ]
            
            # Chỉ hiển thị các cột có trong dữ liệu
            display_cols = [col for col in display_cols if col in filtered_df.columns]
            
            if display_cols:
                # Đổi tên cột cho dễ đọc
                column_rename_map = {
                    'algorithm': 'Thuật Toán',
                    'timestamp': 'Thời Gian',
                    'path_length': 'Độ Dài',
                    'execution_time_ms': 'TG Thực Thi (ms)',
                    'total_cost': 'Tổng Phí (đ)',
                    'fuel_consumed': 'Xăng Tiêu Thụ (L)',
                    'is_feasible': 'Khả Thi',
                    'map_size': 'Kích Thước Bản Đồ'
                }
                
                # Chỉ đổi tên các cột có trong dữ liệu
                rename_cols = {col: column_rename_map[col] for col in display_cols if col in column_rename_map}
                
                # Tạo bảng hiển thị
                df_display = filtered_df[display_cols].rename(columns=rename_cols).copy()
                
                # Định dạng cột Khả Thi
                if 'Khả Thi' in df_display.columns:
                    df_display["Khả Thi"] = df_display["Khả Thi"].apply(lambda x: "✅ Có" if x else "❌ Không")
                
                # Hiển thị bảng với định dạng màu sắc
                st.dataframe(
                    df_display.style
                    .highlight_max(axis=0, subset=['Độ Dài', 'Tổng Phí (đ)', 'Xăng Tiêu Thụ (L)'], color='#FADBD8')  # Light red
                    .highlight_min(axis=0, subset=['TG Thực Thi (ms)'], color='#D5F5E3')  # Light green
                    .set_properties(**{'text-align': 'left'})
                    .set_table_styles([dict(selector='th', props=[('text-align', 'left')])]),
                    use_container_width=True
                )
                
                # Thêm bảng xếp hạng thuật toán
                st.markdown("#### 🏆 Bảng Xếp Hạng Thuật Toán")
                
                # Tạo bảng xếp hạng thuật toán nếu có đủ dữ liệu
                if len(filtered_df) > 1:
                    # Xác định các tiêu chí cần xếp hạng
                    ranking_metrics = {
                        'execution_time_ms': 'Xếp Hạng Thời Gian',
                        'path_length': 'Xếp Hạng Độ Dài',
                        'total_cost': 'Xếp Hạng Chi Phí',
                        'fuel_consumed': 'Xếp Hạng Nhiên Liệu'
                    }
                    
                    # Tạo DataFrame xếp hạng mới
                    ranking_df = filtered_df[['algorithm']].copy()
                    
                    # Thêm xếp hạng cho từng tiêu chí
                    for metric, rank_name in ranking_metrics.items():
                        if metric in filtered_df.columns:
                            # Xếp hạng từ thấp đến cao
                            ranking_df[rank_name] = filtered_df[metric].rank(method='min')
                    
                    # Tính điểm tổng hợp (trung bình xếp hạng)
                    rank_columns = [col for col in ranking_df.columns if 'Xếp Hạng' in col]
                    if rank_columns:
                        # Tính trung bình xếp hạng một cách an toàn
                        ranking_df['Điểm Tổng Hợp'] = ranking_df[rank_columns].fillna(ranking_df[rank_columns].mean()).mean(axis=1).round(2)
                        
                        try:
                            # Xếp hạng chung an toàn hơn
                            ranking_df['Xếp Hạng Chung'] = ranking_df['Điểm Tổng Hợp'].rank(method='min').fillna(9999)
                            # Chuyển đổi sang kiểu int an toàn
                            ranking_df['Xếp Hạng Chung'] = ranking_df['Xếp Hạng Chung'].astype(int)
                            
                            # Sắp xếp theo xếp hạng chung
                            ranking_df = ranking_df.sort_values('Xếp Hạng Chung')
                            
                            # Đổi tên cột thuật toán nếu chưa đổi
                            if 'algorithm' in ranking_df.columns:
                                ranking_df = ranking_df.rename(columns={'algorithm': 'Thuật Toán'})
                        except Exception as e:
                            st.warning(f"Lỗi khi tính xếp hạng chung: {e}")
                            ranking_df['Xếp Hạng Chung'] = 0
                        
                        # Hiển thị bảng xếp hạng với định dạng màu
                        try:
                            st.dataframe(
                                ranking_df.style
                                .background_gradient(cmap='viridis_r', subset=['Điểm Tổng Hợp'])
                                .highlight_min(axis=0, subset=['Điểm Tổng Hợp'], color='#D5F5E3')
                                .set_properties(**{'text-align': 'center'})
                                .set_table_styles([dict(selector='th', props=[('text-align', 'center')])]),
                                use_container_width=True
                            )
                        except Exception as e:
                            st.warning(f"Lỗi khi hiển thị bảng xếp hạng: {e}")
                            st.dataframe(ranking_df, use_container_width=True)
                        
                        # Hiển thị biểu đồ xếp hạng
                        st.markdown("#### 📊 Biểu Đồ Xếp Hạng Thuật Toán")
                        
                        # Tạo biểu đồ cột cho xếp hạng tổng hợp
                        try:
                            fig_ranking = px.bar(
                                ranking_df,
                                x='Thuật Toán',
                                y='Điểm Tổng Hợp',
                                color='Thuật Toán',
                                text='Xếp Hạng Chung',
                                labels={
                                    'Điểm Tổng Hợp': 'Điểm Tổng Hợp (thấp hơn = tốt hơn)',
                                    'Thuật Toán': 'Thuật Toán'
                                },
                                title='Xếp hạng tổng hợp các thuật toán (thấp hơn = tốt hơn)'
                            )
                            
                            st.plotly_chart(fig_ranking, use_container_width=True)
                        except Exception as e:
                            st.error(f"Không thể tạo biểu đồ xếp hạng: {e}")
                    else:
                        st.info("Không đủ dữ liệu để xếp hạng thuật toán.")
                else:
                    st.info("Cần ít nhất 2 thuật toán để so sánh xếp hạng.")
            else:
                st.error("Không có đủ dữ liệu để hiển thị bảng so sánh thuật toán.")
                
            # Hiển thị biểu đồ so sánh các chỉ số chính
            st.markdown("### 📊 Biểu Đồ So Sánh Chỉ Số")
            
            # Các chỉ số quan trọng cần so sánh
            key_metrics = {
                'execution_time_ms': 'Thời Gian Thực Thi (ms)',
                'path_length': 'Độ Dài Đường Đi',
                'total_cost': 'Tổng Chi Phí (đ)',
                'fuel_consumed': 'Nhiên Liệu Tiêu Thụ (L)'
            }
            
            # Tạo layout 2x2 cho biểu đồ
            metric_cols = [col for col in key_metrics.keys() if col in filtered_df.columns]
            
            # Chia thành các hàng, mỗi hàng 2 cột
            for i in range(0, len(metric_cols), 2):
                cols = st.columns(2)
                
                for j in range(2):
                    if i + j < len(metric_cols):
                        metric = metric_cols[i + j]
                        with cols[j]:
                            st.markdown(f"**{key_metrics[metric]}**")
                            fig = px.bar(
                                filtered_df,
                                x='algorithm',
                                y=metric,
                                color='algorithm',
                                text_auto=True,
                                labels={
                                    'algorithm': 'Thuật Toán',
                                    metric: key_metrics[metric]
                                }
                            )
                            st.plotly_chart(fig, use_container_width=True)

    with tab2: # Phân Tích Tổng Hợp
        st.markdown("### 🔬 Phân Tích Tổng Hợp Hiệu Suất Thuật Toán")
        if not selected_algorithms:
            st.info("Vui lòng chọn ít nhất một thuật toán để hiển thị phân tích tổng hợp.")
        elif not df_aggregated_stats.empty:
            # Lọc dữ liệu thống kê theo thuật toán đã chọn
            filtered_stats = df_aggregated_stats[df_aggregated_stats['algorithm'].isin(selected_algorithms)]
            
            # Chia tab phân tích
            summary_tabs = st.tabs(["📊 Điểm Tổng Hợp", "📈 Biểu Đồ So Sánh", "📋 Bảng Thống Kê Chi Tiết"])
            
            with summary_tabs[0]:
                # Hiển thị điểm tổng hợp nếu có
                if 'Điểm Hiệu Năng' in filtered_stats.columns:
                    st.markdown("#### 🏆 Xếp Hạng Hiệu Năng Tổng Hợp")
                    
                    # Tạo bảng xếp hạng
                    ranking_df = filtered_stats[['algorithm', 'Điểm Hiệu Năng', 'Tỷ Lệ Khả Thi (%)', 'Số Lần Chạy']].copy()
                    
                    # Đảm bảo các cột đều có dữ liệu
                    for col in ['Điểm Hiệu Năng', 'Tỷ Lệ Khả Thi (%)']:
                        if col in ranking_df.columns:
                            ranking_df[col] = ranking_df[col].fillna(0)
                    
                    # Xếp hạng an toàn không gây lỗi NaN
                    try:
                        if 'Điểm Hiệu Năng' in ranking_df.columns:
                            ranking_df['Xếp Hạng'] = ranking_df['Điểm Hiệu Năng'].rank(method='min')
                            # Chuyển đổi an toàn sang kiểu int
                            ranking_df['Xếp Hạng'] = ranking_df['Xếp Hạng'].fillna(9999).astype(int)
                            ranking_df = ranking_df.sort_values('Xếp Hạng')
                    except Exception as e:
                        st.warning(f"Lỗi khi tính xếp hạng: {e}")
                        ranking_df['Xếp Hạng'] = 0
                    
                    # Dữ liệu cho biểu đồ xếp hạng
                    try:
                        fig_ranking = px.bar(ranking_df, 
                                           x='algorithm', 
                                           y='Điểm Hiệu Năng', 
                                           color='Tỷ Lệ Khả Thi (%)',
                                           hover_data=['Xếp Hạng', 'Số Lần Chạy'],
                                           color_continuous_scale='Viridis',
                                           labels={'Điểm Hiệu Năng': 'Điểm (thấp hơn = tốt hơn)', 'algorithm': 'Thuật Toán'})
                        
                        fig_ranking.update_layout(title='Điểm Hiệu Năng Tổng Hợp (thấp hơn = tốt hơn)')
                        st.plotly_chart(fig_ranking, use_container_width=True)
                    except Exception as e:
                        st.error(f"Không thể hiển thị biểu đồ: {e}")
                    
                    # Hiển thị bảng xếp hạng
                    st.dataframe(ranking_df.rename(columns={'algorithm': 'Thuật Toán'}), use_container_width=True)
                    
                    # Giải thích cách tính điểm
                    with st.expander("ℹ️ Cách tính điểm hiệu năng"):
                        st.markdown("""
                        **Điểm hiệu năng tổng hợp** được tính bằng cách:
                        
                        1. Lấy giá trị trung bình của các thông số: thời gian thực thi, độ dài đường đi, tổng chi phí và nhiên liệu tiêu thụ.
                        2. Tính tổng các giá trị đó.
                        3. Điểm càng thấp càng tốt.
                        
                        Thuật toán có **điểm thấp nhất** là thuật toán có hiệu năng tổng thể tốt nhất.
                        """)
                else:
                    st.info("Không đủ dữ liệu để tính điểm hiệu năng tổng hợp.")
            
            with summary_tabs[1]:
                st.markdown("#### 📊 So Sánh Các Chỉ Số Chính")
                
                # Hiển thị biểu đồ so sánh theo từng tiêu chí
                metrics_to_compare = [
                    ('Thời Gian (ms) (TB)', 'Thời gian thực thi (ms)'),
                    ('Độ Dài Đường Đi (TB)', 'Độ dài đường đi'),
                    ('Tổng Chi Phí (đ) (TB)', 'Chi phí (đ)'),
                    ('Nhiên Liệu (L) (TB)', 'Nhiên liệu tiêu thụ (L)')
                ]
                
                for metric, title in metrics_to_compare:
                    if metric in filtered_stats.columns:
                        st.subheader(f"{title}")
                        try:
                            fig = px.bar(filtered_stats, 
                                      x='algorithm', 
                                      y=metric, 
                                      color='algorithm',
                                      text_auto=True,
                                      labels={'algorithm': 'Thuật Toán'})
                            st.plotly_chart(fig, use_container_width=True)
                        except Exception as e:
                            st.warning(f"Không thể hiển thị biểu đồ cho {title}: {e}")
            
            with summary_tabs[2]:
                st.markdown("#### 📋 Bảng Thống Kê Chi Tiết")
                
                # Hiển thị bảng thống kê chi tiết với định dạng đẹp
                st.dataframe(filtered_stats.style
                            .set_sticky(axis="index")
                            .background_gradient(cmap='viridis_r', subset=pd.IndexSlice[:, [col for col in filtered_stats.columns if 'TB' in col]])
                                               .set_properties(**{'text-align': 'right'})
                            .set_table_styles([dict(selector='th', props=[('text-align', 'center')])]),
                            use_container_width=True)
        else:
            st.info("Không đủ dữ liệu để tạo phân tích tổng hợp.")

    with tab3: # So Sánh Đa Chiều (Radar Chart)
        st.markdown("### 🎯 So Sánh Đa Chiều (Biểu Đồ Radar)")
        if not selected_algorithms:
            st.info("Vui lòng chọn ít nhất một thuật toán để hiển thị biểu đồ radar.")
        elif not df_aggregated_stats.empty and "Số Lần Chạy" in df_aggregated_stats.columns:
            # Lọc dữ liệu thống kê theo thuật toán đã chọn
            filtered_stats = df_aggregated_stats[df_aggregated_stats['algorithm'].isin(selected_algorithms)]
            
            # Chọn các số liệu để vẽ radar (chỉ lấy giá trị trung bình - TB)
            radar_metrics_map = {
                'Độ Dài Đường Đi (TB)': 'Độ Dài TB',
                'Thời Gian (ms) (TB)': 'TG Thực Thi TB (ms)',
                'Tổng Chi Phí (đ) (TB)': 'Tổng Phí TB (đ)',
                'Nhiên Liệu (L) (TB)': 'Xăng Tiêu Thụ TB (L)',
                'Tỷ Lệ Khả Thi (%)': 'Khả Thi (%)'
            }
            available_radar_cols = [col for col in radar_metrics_map.keys() if col in filtered_stats.columns]
            
            if len(available_radar_cols) > 2 and len(selected_algorithms) >= 2: # Cần ít nhất 3 metrics và 2 thuật toán cho Radar chart
                selected_radar_metrics_display = st.multiselect(
                    "Chọn các chỉ số cho biểu đồ Radar (chọn ít nhất 3):",
                    options=[radar_metrics_map[col] for col in available_radar_cols],
                    default=[radar_metrics_map[col] for col in available_radar_cols[:min(5, len(available_radar_cols))]]
                )

                if len(selected_radar_metrics_display) >= 3:
                    # Lấy lại tên cột gốc từ tên hiển thị
                    selected_radar_metrics_original = [key for key, value in radar_metrics_map.items() if value in selected_radar_metrics_display]
                    
                    df_radar = filtered_stats[["algorithm"] + selected_radar_metrics_original].copy()
                    
                    # Xử lý các giá trị NaN trong dữ liệu radar
                    for col in selected_radar_metrics_original:
                        if col in df_radar.columns:
                            # Thay thế NaN bằng 0 hoặc giá trị trung bình
                            df_radar[col] = df_radar[col].fillna(df_radar[col].mean() if df_radar[col].notna().any() else 0)
                    
                    # Tạo radar chart an toàn
                    try:
                        fig_radar = go.Figure()
                        algorithms_for_radar = df_radar["algorithm"].unique()

                        for algo in algorithms_for_radar:
                            algo_data = df_radar[df_radar["algorithm"] == algo]
                            if not algo_data.empty:
                                values = []
                                for metric_key in selected_radar_metrics_original: # Đổi tên biến để tránh nhầm lẫn
                                    if metric_key in algo_data:
                                        # Lấy giá trị đầu tiên hoặc 0 nếu không có hoặc là NaN
                                        value = algo_data[metric_key].iloc[0]
                                        values.append(0 if pd.isna(value) else value)
                                    else:
                                        values.append(0)
                                
                                if len(values) == len(selected_radar_metrics_original):
                                    fig_radar.add_trace(go.Scatterpolar(
                                        r=values,
                                        theta=[radar_metrics_map[col] for col in selected_radar_metrics_original],
                                        fill='toself',
                                        name=algo
                                    ))
                        
                        fig_radar.update_layout(
                            polar=dict(
                                radialaxis=dict(
                                    visible=True,
                                )
                            ),
                            showlegend=True,
                            title="So sánh đa chiều hiệu suất thuật toán"
                        )
                        st.plotly_chart(fig_radar, use_container_width=True)
                    except Exception as e:
                        st.error(f"Không thể tạo biểu đồ radar: {e}")
                else:
                    st.warning("Vui lòng chọn ít nhất 3 chỉ số cho biểu đồ Radar.")
            else:
                min_required = ""
                if len(available_radar_cols) <= 2:
                    min_required += "3 chỉ số"
                if len(selected_algorithms) < 2:
                    if min_required:
                        min_required += " và "
                    min_required += "2 thuật toán"
                st.info(f"Không đủ dữ liệu để tạo biểu đồ Radar. Cần ít nhất {min_required}.")
        else:
            st.info("Không đủ dữ liệu để tạo biểu đồ Radar. Hãy chọn nhiều file hơn hoặc chạy thêm thuật toán.")

    with tab4: # Chi Tiết Từng Lần Chạy
        st.markdown("### 📝 Chi Tiết Từng Lần Chạy")
        if not selected_algorithms:
            st.info("Vui lòng chọn ít nhất một thuật toán để xem chi tiết các lần chạy.")
        elif not df_all_runs.empty:
            # Lọc dữ liệu theo thuật toán đã chọn
            filtered_runs = df_all_runs[df_all_runs['algorithm'].isin(selected_algorithms)]
            
            if filtered_runs.empty:
                st.info("Không có dữ liệu chi tiết cho các thuật toán đã chọn.")
            else:
                # Sắp xếp theo thuật toán và thời gian (nếu có)
                try:
                    filtered_runs['timestamp_dt'] = pd.to_datetime(filtered_runs['timestamp'], errors='coerce')
                    sorted_runs = filtered_runs.sort_values(['algorithm', 'timestamp_dt'], ascending=[True, False])
                except:
                    sorted_runs = filtered_runs.sort_values('algorithm')
                    
                # Tạo tabs cho từng thuật toán
                algo_tabs = st.tabs([f"🔹 {algo}" for algo in sorted_runs['algorithm'].unique()])
                
                for i, algo in enumerate(sorted_runs['algorithm'].unique()):
                    with algo_tabs[i]:
                        algo_runs = sorted_runs[sorted_runs['algorithm'] == algo]
                        
                        for idx, row in algo_runs.iterrows():
                            with st.expander(f"Chi tiết cho: {row['filename']} (Thời gian: {row['timestamp']})"):
                                st.markdown(f"#### Thông Số Chính - `{row['algorithm']}`")
                                cols_metrics = st.columns(4)
                                cols_metrics[0].metric("Độ Dài Đường Đi", f"{row['path_length']:.0f} bước")
                                cols_metrics[1].metric("Thời Gian TH", f"{row['execution_time_ms']:.2f} ms")
                                cols_metrics[2].metric("Tổng Chi Phí", f"{row['total_cost']:.0f} đ")
                                cols_metrics[3].metric("Khả Thi", "✅ Có" if row['is_feasible'] else f"❌ Không ({row['reason_infeasible']})")

                                st.markdown("##### ⛽ Thông Tin Nhiên Liệu")
                                fuel_cols = st.columns(3)
                                fuel_cols[0].metric("Xăng Ban Đầu", f"{row['initial_fuel']:.1f}L")
                                fuel_cols[1].metric("Xăng Tiêu Thụ", f"{row['fuel_consumed']:.1f}L")
                                fuel_cols[2].metric("Xăng Còn Lại", f"{row['remaining_fuel']:.1f}L")
                                
                                st.markdown("##### 💰 Thông Tin Tài Chính")
                                money_cols = st.columns(3)
                                money_cols[0].metric("Tiền Ban Đầu", f"{row['initial_money']:.0f}đ")
                                money_cols[1].metric("Chi Phí Xăng", f"{row['fuel_cost']:.0f}đ")
                                money_cols[2].metric("Chi Phí Trạm", f"{row['toll_cost']:.0f}đ")
                                st.metric("Tiền Còn Lại", f"{row['remaining_money']:.0f}đ")

                                st.markdown("##### ⚙️ Thông Số Tìm Kiếm & Hiệu Suất")
                                perf_cols = st.columns(3)
                                perf_cols[0].metric("Số Bước Tìm Kiếm", f"{row['steps']}")
                                perf_cols[1].metric("Số Ô Đã Thăm", f"{row['visited_cells']}")
                                perf_cols[2].metric("Bộ Nhớ Ước Tính", f"{row['memory_usage_kb']:.2f} KB")
                                
                                st.markdown(f"**Kích thước bản đồ:** {row['map_size']}")
                                st.caption(f"File: {row['filename']} - Thời gian ghi nhận: {row['timestamp']}")
        else:
            st.info("Không có dữ liệu chi tiết để hiển thị.")

if __name__ == '__main__':
    # This part is for testing the page independently if needed
    st.set_page_config(layout="wide", page_title="Đánh Giá Thuật Toán")
    render_evaluation_page() 