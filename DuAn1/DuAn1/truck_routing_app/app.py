"""
Main application file for the Truck Routing System.
This file serves as the entry point for the Streamlit application.
"""

import streamlit as st
from ui.home import render_home
from ui.map_config import render_map_config
from ui.routing_visualization import render_routing_visualization
from ui.algorithm_evaluation import render_evaluation_page

def main():
    # Thiết lập cấu hình trang
    st.set_page_config(
        page_title="Hệ Thống Định Tuyến Phân Phối Hàng Hóa",
        page_icon="🚚",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # CSS chung cho toàn bộ ứng dụng
    st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Roboto:wght@300;400;500;700&display=swap');
        
        html, body, [class*="css"] {
            font-family: 'Roboto', sans-serif;
        }
        
        .main .block-container {
            padding-top: 1rem;
        }
        
        .stApp {
            background-color: #f8f9fa;
        }
        
        h1, h2, h3 {
            color: #2c3e50;
            font-weight: 500;
        }
        
        .app-header {
            background: linear-gradient(135deg, #1a5276 0%, #2980b9 100%);
            color: white;
            padding: 1.5rem;
            border-radius: 10px;
            margin-bottom: 20px;
            text-align: center;
            box-shadow: 0 4px 10px rgba(0,0,0,0.15);
        }
        
        .app-title {
            font-size: 2.5rem;
            font-weight: bold;
            margin-bottom: 0.5rem;
            text-shadow: 1px 1px 2px rgba(0,0,0,0.2);
        }
        
        .app-subtitle {
            font-size: 1.2rem;
            opacity: 0.9;
        }
        
        /* Cải thiện sidebar */
        .css-1d391kg {
            background-color: #f1f3f5;
        }
        
        /* Nút đẹp hơn */
        .stButton>button {
            background-color: #2980b9;
            color: white;
            border: none;
            border-radius: 4px;
            padding: 0.5rem 1rem;
            font-weight: 500;
            transition: all 0.3s;
        }
        
        .stButton>button:hover {
            background-color: #3498db;
            box-shadow: 0 2px 5px rgba(0,0,0,0.2);
        }
        
        /* Định dạng expander */
        .streamlit-expanderHeader {
            background-color: #f1f3f5;
            border-radius: 4px;
        }
        
        /* Định dạng các widget */
        .stSlider, .stSelectbox {
            padding-bottom: 1rem;
        }
    </style>
    
    <div class="app-header">
        <div class="app-title">🚚 Hệ Thống Định Tuyến Phân Phối Hàng Hóa</div>
        <div class="app-subtitle">Mô phỏng và tối ưu hóa tuyến đường vận chuyển</div>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar navigation
    with st.sidebar:
        st.image("https://cdn-icons-png.flaticon.com/512/2539/2539144.png", width=100)
        
        st.markdown("## Điều hướng")
        page = st.radio(
            "Chọn trang:",
            ["Trang chủ", "Tạo Bản Đồ & Cấu Hình", "Định Tuyến & Tối Ưu Hệ Thống", "Đánh Giá Thuật Toán"]
        )
        
        # Thông tin ứng dụng
        st.markdown("---")
        st.markdown("### Thông tin ứng dụng")
        st.markdown("Phiên bản: 1.0.0")
        st.markdown("© 2025 - Tất cả quyền được bảo lưu")
        
        # Thêm hướng dẫn nhanh
        st.markdown("---")
        with st.expander("ℹ️ Hướng dẫn nhanh"):
            st.markdown("""
            **Cách sử dụng ứng dụng:**
            
            1. **Tạo bản đồ mới**:
               - Chọn kích thước bản đồ
               - Điều chỉnh tỷ lệ các loại ô
               - Nhấn "Tạo bản đồ"
               
            2. **Thiết lập vị trí xe**:
               - Chọn tọa độ vị trí bắt đầu (phải là ô đường)
               - Nhấn "Thiết lập vị trí bắt đầu"
               
            3. **Lưu bản đồ** để sử dụng lại sau này
            """)
    
    # Render selected page
    if page == "Trang chủ":
        render_home()
    elif page == "Tạo Bản Đồ & Cấu Hình":
        render_map_config()
    elif page == "Định Tuyến & Tối Ưu Hệ Thống":
        render_routing_visualization()
    elif page == "Đánh Giá Thuật Toán":
        render_evaluation_page()

if __name__ == "__main__":
    main()