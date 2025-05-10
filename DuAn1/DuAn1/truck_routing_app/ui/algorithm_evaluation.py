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

    # Chỉ tính toán cho các cột số
    numeric_cols = df_all_runs.select_dtypes(include=np.number).columns
    
    # Các hàm tổng hợp
    agg_funcs = {
        col: ['mean', 'min', 'max', 'std', 'count'] for col in numeric_cols 
        if col not in ['is_feasible'] # is_feasible sẽ được xử lý riêng
    }
    agg_funcs['is_feasible'] = [lambda x: x.mean() * 100] # Tính tỷ lệ khả thi (%)
    agg_funcs['filename'] = ['count'] # Đếm số lần chạy

    df_aggregated = df_all_runs.groupby("algorithm").agg(agg_funcs)
    
    # Đổi tên cột cho dễ đọc hơn
    new_column_names = []
    for col_L1, col_L2 in df_aggregated.columns:
        if col_L1 == 'filename' and col_L2 == 'count':
            new_column_names.append("Số Lần Chạy")
        elif col_L1 == 'is_feasible' and col_L2 == '<lambda_0>':
            new_column_names.append("Tỷ Lệ Khả Thi (%)")
        else:
            new_column_names.append(f"{col_L1.replace('_', ' ').title()} ({col_L2.title()})")
    df_aggregated.columns = new_column_names
    
    # Làm tròn các giá trị số
    for col in df_aggregated.columns:
        if df_aggregated[col].dtype == 'float64':
            df_aggregated[col] = df_aggregated[col].round(2)
            
    return df_aggregated.reset_index()

def render_evaluation_page():
    """Render trang đánh giá thuật toán."""
    st.markdown("""
    <div style="text-align: center; padding: 10px; margin-bottom: 20px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); border-radius: 10px; box-shadow: 0 4px 10px rgba(0,0,0,0.15);">
        <h1 style="color: white; margin: 0;">📊 Đánh Giá Hiệu Suất Thuật Toán</h1>
        <p style="color: white; opacity: 0.9; margin-top: 5px;">So sánh và phân tích kết quả chạy của các thuật toán tìm đường.</p>
    </div>
    """, unsafe_allow_html=True)

    stat_files = get_stat_files()

    if not stat_files:
        st.info("ℹ️ Không tìm thấy file thống kê nào trong thư mục `statistics`. Vui lòng chạy thuật toán để tạo dữ liệu.")
        return

    with st.sidebar:
        st.header("⚙️ Tùy Chọn Đánh Giá")
        selected_files = st.multiselect(
            "Chọn các file thống kê để phân tích:",
            options=stat_files,
            default=stat_files[:min(5, len(stat_files))] # Mặc định chọn 5 file
        )

    if not selected_files:
        st.info("ℹ️ Vui lòng chọn ít nhất một file thống kê từ sidebar để xem chi tiết.")
        return

    all_metrics_data = []
    for filename in selected_files:
        raw_data = load_json_data(filename)
        if raw_data:
            metrics = extract_metrics_from_data(raw_data, filename)
            if "error" not in metrics:
                all_metrics_data.append(metrics)
    
    if not all_metrics_data:
        st.error("❌ Không thể tải hoặc xử lý dữ liệu từ các file đã chọn.")
        return

    df_all_runs = pd.DataFrame(all_metrics_data)
    df_aggregated_stats = aggregate_metrics(df_all_runs.copy()) # Sử dụng copy để tránh thay đổi df_all_runs

    # --- Tạo Tabs --- 
    tab1, tab2, tab3, tab4 = st.tabs(["🚀 Tổng Quan & So Sánh Nhanh", "📈 Phân Tích Tổng Hợp", "🎯 So Sánh Đa Chiều (Radar)", "📝 Chi Tiết Từng Lần Chạy"])

    with tab1: # Tổng Quan & So Sánh Nhanh
        st.markdown("### ⏱️ Bảng So Sánh Các Lần Chạy Gần Nhất")
        columns_to_display = [
            "algorithm", "timestamp", "map_size", "path_length", "execution_time_ms", 
            "total_cost", "fuel_consumed", "is_feasible"
        ]
        column_rename_map = {
            "algorithm": "Thuật Toán", "timestamp": "Thời Gian", "map_size": "Map",
            "path_length": "Độ Dài", "execution_time_ms": "TG Thực Thi (ms)",
            "total_cost": "Tổng Phí (đ)", "fuel_consumed": "Xăng (L)",
            "is_feasible": "Khả Thi?"
        }
        df_quick_compare = df_all_runs[columns_to_display].rename(columns=column_rename_map).copy()
        # Định dạng cột Khả Thi
        df_quick_compare["Khả Thi?"] = df_quick_compare["Khả Thi?"].apply(lambda x: "✅" if x else "❌")
        st.dataframe(df_quick_compare.style.highlight_max(axis=0, subset=['Độ Dài', 'Tổng Phí (đ)', 'Xăng (L)'], color='#FADBD8') # Light red
                                     .highlight_min(axis=0, subset=['TG Thực Thi (ms)'], color='#D5F5E3') # Light green
                                     .set_properties(**{'text-align': 'left'})
                                     .set_table_styles([dict(selector='th', props=[('text-align', 'left')])]))
        
        st.markdown("### 📊 Biểu Đồ So Sánh Nhanh (Dựa trên các lần chạy được chọn)")
        # ... (Giữ lại các biểu đồ cột hiện tại, nhưng dùng df_all_runs)
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Thời Gian Thực Thi (ms)**")
            fig_time = px.bar(df_all_runs, x="algorithm", y="execution_time_ms", color="algorithm", text_auto=True,
                              labels={"execution_time_ms": "Thời gian (ms)", "algorithm": "Thuật Toán"}, hover_data=["filename"])
            st.plotly_chart(fig_time, use_container_width=True)
        with col2:
            st.markdown("**Độ Dài Đường Đi**")
            fig_path = px.bar(df_all_runs, x="algorithm", y="path_length", color="algorithm", text_auto=True,
                              labels={"path_length": "Độ dài", "algorithm": "Thuật Toán"}, hover_data=["filename"])
            st.plotly_chart(fig_path, use_container_width=True)
        # ...(Thêm biểu đồ cho total_cost và fuel_consumed tương tự) ...
        col3, col4 = st.columns(2)
        with col3:
            st.markdown("**Tổng Chi Phí (đ)**")
            fig_cost = px.bar(df_all_runs, x="algorithm", y="total_cost", color="algorithm", text_auto=True,
                              labels={"total_cost": "Chi phí (đ)", "algorithm": "Thuật Toán"}, hover_data=["filename"])
            st.plotly_chart(fig_cost, use_container_width=True)
        with col4:
            st.markdown("**Xăng Tiêu Thụ (L)**")
            fig_fuel = px.bar(df_all_runs, x="algorithm", y="fuel_consumed", color="algorithm", text_auto=True,
                              labels={"fuel_consumed": "Xăng tiêu thụ (L)", "algorithm": "Thuật Toán"}, hover_data=["filename"])
            st.plotly_chart(fig_fuel, use_container_width=True)

    with tab2: # Phân Tích Tổng Hợp
        st.markdown("### 🔬 Phân Tích Tổng Hợp Hiệu Suất Thuật Toán")
        if not df_aggregated_stats.empty:
            st.dataframe(df_aggregated_stats.style.set_sticky(axis="index")
                                               .background_gradient(cmap='viridis_r', subset=pd.IndexSlice[:, [col for col in df_aggregated_stats.columns if 'Mean' in col or 'Median' in col or 'Tỷ Lệ' in col]])
                                               .set_properties(**{'text-align': 'right'})
                                               .set_table_styles([dict(selector='th', props=[('text-align', 'center')])]))
        else:
            st.info("Không đủ dữ liệu để tạo phân tích tổng hợp.")

    with tab3: # So Sánh Đa Chiều (Radar Chart)
        st.markdown("### 🎯 So Sánh Đa Chiều (Biểu Đồ Radar)")
        if not df_aggregated_stats.empty and "Số Lần Chạy" in df_aggregated_stats.columns:
            # Chọn các số liệu để vẽ radar (chỉ lấy giá trị trung bình - Mean)
            radar_metrics_map = {
                'Path Length (Mean)': 'Độ Dài TB',
                'Execution Time Ms (Mean)': 'TG Thực Thi TB (ms)',
                'Total Cost (Mean)': 'Tổng Phí TB (đ)',
                'Fuel Consumed (Mean)': 'Xăng Tiêu Thụ TB (L)',
                'Tỷ Lệ Khả Thi (%)': 'Khả Thi (%)'
            }
            available_radar_cols = [col for col in radar_metrics_map.keys() if col in df_aggregated_stats.columns]
            
            if len(available_radar_cols) > 2: # Cần ít nhất 3 metrics cho Radar chart
                selected_radar_metrics_display = st.multiselect(
                    "Chọn các chỉ số cho biểu đồ Radar (chọn ít nhất 3):",
                    options=[radar_metrics_map[col] for col in available_radar_cols],
                    default=[radar_metrics_map[col] for col in available_radar_cols[:min(5, len(available_radar_cols))]]
                )

                if len(selected_radar_metrics_display) >= 3:
                    # Lấy lại tên cột gốc từ tên hiển thị
                    selected_radar_metrics_original = [key for key, value in radar_metrics_map.items() if value in selected_radar_metrics_display]
                    
                    df_radar = df_aggregated_stats[["algorithm"] + selected_radar_metrics_original].copy()
                    
                    # Chuẩn hóa dữ liệu cho Radar chart (0-1) nếu cần, hoặc để Plotly tự xử lý thang đo
                    # Ở đây, ta sẽ để Plotly tự xử lý thang đo cho từng trục
                    
                    fig_radar = go.Figure()
                    algorithms_for_radar = df_radar["algorithm"].unique()

                    for algo in algorithms_for_radar:
                        algo_data = df_radar[df_radar["algorithm"] == algo]
                        values = algo_data[selected_radar_metrics_original].iloc[0].values.flatten().tolist()
                        fig_radar.add_trace(go.Scatterpolar(
                            r=values,
                            theta=[radar_metrics_map[col] for col in selected_radar_metrics_original], # Sử dụng tên hiển thị
                            fill='toself',
                            name=algo
                        ))
                    
                    fig_radar.update_layout(
                        polar=dict(
                            radialaxis=dict(
                                visible=True,
                                # range=[0, max_r_value] # Có thể set range nếu muốn chuẩn hóa
                            )
                        ),
                        showlegend=True,
                        title="So sánh đa chiều hiệu suất thuật toán"
                    )
                    st.plotly_chart(fig_radar, use_container_width=True)
                else:
                    st.warning("Vui lòng chọn ít nhất 3 chỉ số cho biểu đồ Radar.")
            else:
                st.info("Không đủ chỉ số (cần ít nhất 3) hoặc thuật toán để vẽ biểu đồ Radar.")
        else:
            st.info("Không đủ dữ liệu để tạo biểu đồ Radar. Hãy chọn nhiều file hơn hoặc chạy thêm thuật toán.")

    with tab4: # Chi Tiết Từng Lần Chạy
        st.markdown("### 📝 Chi Tiết Từng Lần Chạy")
        # ... (Giữ lại phần expander hiện tại, dùng df_all_runs)
        if not df_all_runs.empty:
            for idx, row in df_all_runs.iterrows():
                with st.expander(f"Chi tiết cho: {row['filename']} (Thuật toán: {row['algorithm']})"):
                    # ... (Code hiển thị metric chi tiết như cũ) ...
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