import streamlit as st

def render_home():
    st.title("🚛 Hệ Thống Định Tuyến Phân Phối Hàng Hóa")
    
    # Phần giới thiệu
    st.header("Giới thiệu")
    st.markdown("""
    Hệ thống định tuyến phân phối hàng hóa là một ứng dụng mô phỏng quá trình di chuyển của xe tải 
    trên bản đồ dạng lưới với giao diện trực quan và sinh động. Hệ thống tích hợp nhiều thuật toán 
    định tuyến và tối ưu hóa để tìm ra lộ trình tối ưu nhất cho xe tải.
    """)
    
    # Phần mục tiêu
    st.header("Mục tiêu")
    st.markdown("""
    - Mô phỏng quá trình di chuyển xe tải trên bản đồ dạng lưới
    - Tìm tuyến đường tối ưu dựa trên nhiều yếu tố (khoảng cách, phí cầu, tiêu hao nhiên liệu)
    - Cung cấp giao diện trực quan để người dùng dễ dàng theo dõi và điều khiển
    - Tích hợp các thuật toán tối ưu hóa và học máy để cải thiện hiệu suất định tuyến
    """)
    
    # Phần cách thức hoạt động
    st.header("Cách thức hoạt động")
    st.markdown("""
    1. **Tạo và cấu hình bản đồ:**
       - Nhập kích thước bản đồ
       - Thiết lập các điểm đặc biệt (trạm thu phí, điểm đổ xăng)
       - Vẽ bản đồ thủ công hoặc tạo ngẫu nhiên
    
    2. **Định tuyến và tối ưu:**
       - Sử dụng các thuật toán tìm đường truyền thống (BFS, DFS, A*)
       - Áp dụng các thuật toán metaheuristic (Genetic Algorithm, Simulated Annealing)
       - **Học tăng cường (RL)** cho tuyến đường thích ứng thông minh
       - Áp dụng các ràng buộc (nhiên liệu, phí cầu)
       - Tối ưu hóa tuyến đường
    
    3. **Mô phỏng và phân tích:**
       - Hiển thị quá trình di chuyển
       - Phân tích hiệu suất
       - Xuất báo cáo và thống kê
    """)
    
    # Thêm mục Tính năng nổi bật với RL như một điểm nhấn
    st.header("✨ Tính năng nổi bật")
    st.markdown("""
    ### 🧠 Định tuyến với Học Tăng Cường (Reinforcement Learning)
    
    Hệ thống tích hợp các agent Học Tăng Cường (RL) tiên tiến:
    
    - **Thích ứng thông minh**: Agent RL có khả năng thích ứng với các loại bản đồ khác nhau mà không cần huấn luyện lại
    - **Chiến lược tùy chỉnh**: Lựa chọn chiến lược ưu tiên (tiết kiệm chi phí, nhanh nhất, an toàn nhiên liệu)
    - **Ra quyết định tối ưu**: Agent học cách cân bằng giữa chi phí, thời gian và tài nguyên
    - **Khả năng đổ xăng**: Tự động quyết định khi nào cần ghé trạm xăng dựa trên tình hình thực tế
    
    Dùng tab "RL Nâng cao" trong ứng dụng kiểm tra RL (rl_test.py) để huấn luyện và tinh chỉnh các agent.
    """)
    
    # Phần hướng dẫn sử dụng
    st.header("Hướng dẫn sử dụng")
    st.markdown("""
    1. **Bắt đầu:**
       - Chọn tab "Tạo Bản Đồ & Cấu Hình"
       - Nhập các thông số cần thiết
       - Tạo bản đồ theo ý muốn
    
    2. **Định tuyến:**
       - Chọn thuật toán định tuyến (bao gồm Học Tăng Cường)
       - Thiết lập các tham số
       - Chạy mô phỏng
    
    3. **Phân tích:**
       - Xem kết quả định tuyến
       - Phân tích hiệu suất
       - So sánh các thuật toán khác nhau
       - Xem các chỉ số RL chi tiết (khi sử dụng agent RL)
    """)
    
    # Phần thông tin liên hệ
    st.header("Liên hệ")
    st.markdown("""
    Nếu có bất kỳ câu hỏi hoặc góp ý nào, vui lòng liên hệ:
    - Email: vut210225@gmail.com
    - GitHub: [Link to your repository]
    """) 