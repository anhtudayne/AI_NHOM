import streamlit as st

def render_home():
    st.title("🚛 Hệ Thống Định Tuyến Phân Phối Hàng Hóa")
    
    # Phần giới thiệu
    st.header("Giới thiệu")
    st.markdown("""
    Chào mừng bạn đến với Hệ Thống Định Tuyến Phân Phối Hàng Hóa! Đây là một ứng dụng Streamlit
    mô phỏng và tối ưu hóa quá trình vận chuyển hàng hóa bằng xe tải trên bản đồ dạng lưới.
    Hệ thống cho phép người dùng tạo bản đồ tùy chỉnh, thiết lập các tham số vận hành
    (nhiên liệu, chi phí), lựa chọn từ nhiều thuật toán định tuyến khác nhau - bao gồm các phương pháp
    tìm kiếm cổ điển, metaheuristic và học tăng cường (Reinforcement Learning) - để tìm ra
    lộ trình hiệu quả nhất. Quá trình di chuyển của xe tải được mô phỏng trực quan,
    cùng với các công cụ phân tích hiệu suất chi tiết.
    """)
    
    # Phần mục tiêu
    st.header("Mục tiêu")
    st.markdown("""
    - **Mô phỏng:** Tạo môi trường bản đồ dạng lưới linh hoạt và trực quan hóa quá trình di chuyển của xe tải.
    - **Định tuyến:** Cung cấp và so sánh hiệu quả của nhiều thuật toán tìm đường:
        - Thuật toán tìm kiếm cơ bản (BFS, DFS, UCS, IDS, IDA*)
        - Thuật toán tìm kiếm dựa trên heuristics (Greedy, A*, Local Beam Search)
        - Thuật toán metaheuristic (Simulated Annealing, Genetic Algorithm)
        - Học tăng cường (Deep Q-Network - DQN)
    - **Tối ưu hóa:** Tìm kiếm lộ trình tối ưu dựa trên các yếu tố như khoảng cách, chi phí nhiên liệu, phí cầu đường, và thời gian.
    - **Tương tác:** Cung cấp giao diện người dùng thân thiện để dễ dàng cấu hình, chạy mô phỏng và phân tích kết quả.
    - **Nghiên cứu & Phát triển:** Tích hợp các kỹ thuật AI tiên tiến (đặc biệt là RL) để giải quyết bài toán định tuyến phức tạp và thích ứng.
    """)
    
    # Phần các thuật toán hỗ trợ
    st.header("Các thuật toán hỗ trợ")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.subheader("Tìm kiếm cơ bản & Heuristic")
        st.markdown("""
        - Breadth-First Search (BFS)
        - Depth-First Search (DFS)
        - Uniform Cost Search (UCS)
        - Iterative Deepening Search (IDS)
        - Iterative Deepening A* (IDA*)
        - Greedy Best-First Search
        - A* Search (bao gồm biến thể tối ưu nhiên liệu)
        - Local Beam Search
        """)
    with col2:
        st.subheader("Metaheuristic")
        st.markdown("""
        - Simulated Annealing (SA)
        - Genetic Algorithm (GA)
        """)
    with col3:
        st.subheader("🧠 Học Tăng Cường")
        st.markdown("""
        - **Deep Q-Network (DQN):** Agent RL có khả năng học và thích ứng với các môi trường và mục tiêu khác nhau (chi phí, nhiên liệu, thời gian).
        - **Khả năng tùy chỉnh:** Huấn luyện, tinh chỉnh siêu tham số và đánh giá agent trong môi trường `Train/`.
        - **Thông minh:** Tự động ra quyết định về việc đổ xăng, tránh vật cản và tối ưu lộ trình dựa trên kinh nghiệm học được.
        """)
    
    # Phần cách thức hoạt động
    st.header("Cách thức hoạt động")
    st.markdown("""
    1.  **Tạo Bản Đồ & Cấu Hình:**
        -   Truy cập tab "Tạo Bản Đồ & Cấu Hình".
        -   Nhập kích thước bản đồ mong muốn.
        -   Thiết lập tỷ lệ các loại ô (đường đi, vật cản, trạm xăng, trạm thu phí).
        -   Tạo bản đồ ngẫu nhiên hoặc vẽ thủ công.
        -   Thiết lập điểm bắt đầu và kết thúc cho xe tải.
        -   Lưu/Tải cấu hình bản đồ (tùy chọn).

    2.  **Định Tuyến & Tối Ưu Hệ Thống:**
        -   Truy cập tab "Định Tuyến & Tối Ưu Hệ Thống".
        -   Chọn bản đồ đã tạo hoặc tải lên.
        -   Cấu hình các tham số cho xe tải (nhiên liệu ban đầu, mức tiêu thụ, tiền ban đầu).
        -   Chọn thuật toán định tuyến mong muốn từ danh sách (bao gồm cả agent DQN đã huấn luyện).
        -   Thiết lập các tham số riêng cho thuật toán (nếu có).
        -   Chạy mô phỏng để xem xe tải di chuyển và tìm đường.

    3.  **Đánh Giá Thuật Toán:**
        -   Truy cập tab "Đánh Giá Thuật Toán".
        -   Chọn các thuật toán bạn muốn so sánh.
        -   Thiết lập các cấu hình bản đồ và tham số để chạy thử nghiệm.
        -   Chạy đánh giá để xem bảng so sánh chi tiết về hiệu suất (số bước, chi phí, nhiên liệu, thời gian).

    4.  **(Nâng cao) Huấn luyện Agent RL:**
        -   Sử dụng các scripts trong thư mục `Train/` (ví dụ: `auto_train_rl.py`, `rl_test.py`) để:
            -   Huấn luyện agent DQN trên các bản đồ khác nhau.
            -   Tinh chỉnh siêu tham số (hyperparameter tuning).
            -   Lưu và tải các mô hình agent đã huấn luyện.
            -   Đánh giá hiệu quả của agent trong môi trường `TruckRoutingEnv`.
    """)
    
    # Phần liên hệ (Giữ nguyên hoặc cập nhật nếu cần)
    st.header("Liên hệ")
    st.markdown("""
    Nếu có bất kỳ câu hỏi hoặc góp ý nào, vui lòng liên hệ:
    - Email: vut210225@gmail.com
    - GitHub: [Link to your repository] (Hãy cập nhật link GitHub của bạn)
    """) 