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
       - Sử dụng các thuật toán tìm đường (BFS, DFS, A*)
       - Áp dụng các ràng buộc (nhiên liệu, phí cầu)
       - Tối ưu hóa tuyến đường
    
    3. **Mô phỏng và phân tích:**
       - Hiển thị quá trình di chuyển
       - Phân tích hiệu suất
       - Xuất báo cáo và thống kê
    """)
    
    # Phần hướng dẫn sử dụng
    st.header("Hướng dẫn sử dụng")
    st.markdown("""
    1. **Bắt đầu:**
       - Chọn tab "Tạo Bản Đồ & Cấu Hình"
       - Nhập các thông số cần thiết
       - Tạo bản đồ theo ý muốn
    
    2. **Định tuyến:**
       - Chọn thuật toán định tuyến
       - Thiết lập các tham số
       - Chạy mô phỏng
    
    3. **Phân tích:**
       - Xem kết quả định tuyến
       - Phân tích hiệu suất
       - Xuất báo cáo
    """)
    
    # Phần thông tin liên hệ
    st.header("Liên hệ")
    st.markdown("""
    Nếu có bất kỳ câu hỏi hoặc góp ý nào, vui lòng liên hệ:
    - Email: vut210225@gmail.com
    - GitHub: [Link to your repository]
    """) 