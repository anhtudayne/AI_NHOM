# ỨNG DỤNG MÔ PHỎNG ĐƯỜNG ĐI



## 📋 TỔNG QUAN

![Demo](https://github.com/anhtudayne/AI_NHOM/blob/main/truck_routing_app/image/demo.gif?raw=true)


Ứng dụng mô phỏng đường đi là một công cụ mạnh mẽ được phát triển bởi nhóm AI_NHOM để mô phỏng, trực quan hóa và tối ưu hóa các tuyến đường vận chuyển. Được xây dựng trên nền tảng Streamlit với backend Python, ứng dụng này cung cấp một môi trường nghiên cứu và thử nghiệm toàn diện cho việc phân tích các thuật toán tìm đường.

Dự án tập trung vào việc giải quyết bài toán định tuyến trong logistics với các yếu tố thực tế như:
- **Chi phí nhiên liệu** với mô hình tiêu thụ linh hoạt
- **Phí cầu đường** và trạm thu phí
- **Các chướng ngại vật** và địa hình đa dạng
- **Ràng buộc vận hành** về thời gian và nguồn lực

Người dùng có thể:
- **Thiết kế và tùy chỉnh bản đồ** với đầy đủ các loại địa hình và điểm đặc biệt
- **Thử nghiệm đa dạng thuật toán** từ cơ bản đến nâng cao, bao gồm cả học tăng cường
- **Phân tích chi tiết hiệu suất** của từng giải pháp trên nhiều tiêu chí khác nhau
- **Trực quan hóa trực tiếp** quá trình tìm đường và kết quả cuối cùng

## 🌟 TÍNH NĂNG CHÍNH


### Thiết kế bản đồ
- **Tạo bản đồ tùy chỉnh** với kích thước linh hoạt (từ 10x10 đến 100x100)
- **Điều chỉnh tỷ lệ các loại ô**: đường đi, trạm thu phí, trạm xăng, chướng ngại vật
- **Đặt vị trí xuất phát và đích đến** với các ràng buộc tùy chỉnh
- **Lưu và tải cấu hình bản đồ** để tái sử dụng và so sánh

### Thuật toán tìm đường
- **Thuật toán cơ bản**: BFS, DFS, UCS với nhiều biến thể tối ưu
- **Thuật toán dựa trên heuristic**: A*, IDA*, Greedy Search
- **Giải thuật tiến hóa**: Genetic Algorithm, Simulated Annealing
- **Học tăng cường**: Deep Q-Network (DQN), với khả năng tự học và cải thiện

### Phân tích và đánh giá
- **So sánh chi tiết hiệu suất** các thuật toán trên cùng bản đồ
- **Trực quan hóa quá trình tìm kiếm** từng bước
- **Thống kê chi tiết**: thời gian xử lý, bộ nhớ sử dụng, độ phức tạp
- **Xuất báo cáo** dạng PDF và CSV để phân tích ngoại tuyến

### Mô phỏng thời gian thực
- **Theo dõi tiêu thụ nhiên liệu** theo từng đoạn đường
- **Tính toán chi phí cầu đường** dựa trên các trạm thu phí đã đi qua
- **Mô phỏng tình huống thực tế** như tắc đường, thay đổi chi phí
- **Điều chỉnh tốc độ mô phỏng** để quan sát chi tiết

## 🔍 THUẬT TOÁN



Ứng dụng tích hợp và phân tích nhiều thuật toán tìm đường khác nhau:

| Thuật toán | Mô tả | Ưu điểm | Ứng dụng thực tế |
|------------|-------|---------|------------------|
| BFS (Breadth-First Search) | Tìm kiếm theo chiều rộng, khám phá tất cả các nút cùng độ sâu trước khi đi sâu hơn | Tìm được đường đi ngắn nhất về số bước, đảm bảo tìm ra lời giải nếu tồn tại | Tối ưu cho bản đồ đơn giản, chi phí đồng nhất |
| DFS (Depth-First Search) | Tìm kiếm theo chiều sâu, ưu tiên khám phá càng sâu càng tốt trước khi quay lui | Hiệu quả về bộ nhớ, thích hợp với không gian tìm kiếm lớn | Tìm đường trong mê cung, khám phá không gian rộng |
| A* (A-star) | Thuật toán tìm đường dựa trên heuristic, kết hợp chi phí đã đi và ước lượng chi phí còn lại | Tìm đường đi tối ưu với hiệu suất cao khi có heuristic tốt | Ứng dụng rộng rãi trong GPS, định tuyến thực tế |
| UCS (Uniform Cost Search) | Tìm kiếm dựa trên chi phí tích lũy, mở rộng nút có chi phí thấp nhất | Tìm đường đi có chi phí thấp nhất, tối ưu với chi phí không đồng nhất | Định tuyến với chi phí đa dạng (xăng, phí đường) |
| Genetic Algorithm | Mô phỏng quá trình tiến hóa tự nhiên để tìm giải pháp tối ưu | Tìm giải pháp gần tối ưu cho bài toán phức tạp, khám phá không gian rộng | Tối ưu hóa nhiều ràng buộc, bài toán NP-hard |
| DQN (Deep Q-Network) | Kết hợp mạng nơ-ron sâu với học tăng cường để học chiến lược tối ưu | Học từ trải nghiệm, thích nghi với môi trường thay đổi | Định tuyến động với điều kiện thay đổi liên tục |
| IDA* (Iterative Deepening A*) | Kết hợp tìm kiếm lặp sâu dần với A* | Tiết kiệm bộ nhớ hơn A* nhưng vẫn bảo đảm tìm đường tối ưu | Các hệ thống có giới hạn bộ nhớ |
| Simulated Annealing | Thuật toán tối ưu hóa ngẫu nhiên dựa trên quá trình luyện kim | Tránh tối ưu cục bộ, hiệu quả với không gian tìm kiếm lớn | Tối ưu hóa nhiều tham số, lập lịch phức tạp |

Mỗi thuật toán được triển khai với đầy đủ các tùy chọn tham số và có thể được tinh chỉnh cho phù hợp với từng kịch bản cụ thể.

## 📊 ĐÁNH GIÁ HIỆU SUẤT

![Image 1](https://github.com/anhtudayne/AI_NHOM/blob/main/truck_routing_app/image/1.png?raw=true)
![Image 2](https://github.com/anhtudayne/AI_NHOM/blob/main/truck_routing_app/image/2.png?raw=true)
![Image 3](https://github.com/anhtudayne/AI_NHOM/blob/main/truck_routing_app/image/3.png?raw=true)
![Image 4](https://github.com/anhtudayne/AI_NHOM/blob/main/truck_routing_app/image/4.png?raw=true)
![Image 5](https://github.com/anhtudayne/AI_NHOM/blob/main/truck_routing_app/image/5.png?raw=true)

Ứng dụng cung cấp hệ thống đánh giá hiệu suất toàn diện với các tiêu chí:

### Độ đo hiệu suất
- **Thời gian thực thi**: Đo chính xác thời gian CPU và thời gian thực để tìm đường
- **Số nút đã khám phá**: Đếm số lượng ô đã xem xét trước khi tìm ra đường đi tối ưu
- **Bộ nhớ sử dụng**: Theo dõi lượng bộ nhớ tối đa sử dụng trong quá trình tìm kiếm
- **Chi phí tuyến đường**: Tính toán chi tiết các loại chi phí (nhiên liệu, phí đường, thời gian)

### Phân tích so sánh
- **So sánh trên nhiều loại bản đồ**: Từ đơn giản đến phức tạp, từ thưa thớt đến dày đặc chướng ngại vật
- **Phân tích độ mở rộng**: Đánh giá khả năng xử lý khi kích thước bản đồ tăng lên
- **Đánh giá theo ràng buộc**: Hiệu suất khi thêm các ràng buộc về nhiên liệu, thời gian
- **Khả năng thích ứng**: Phản ứng khi môi trường thay đổi (đường tắc, chi phí biến động)

Tất cả các kết quả đánh giá được lưu trữ và có thể truy xuất để so sánh giữa các lần chạy khác nhau.

## 🔧 CẤU TRÚC DỰ ÁN


```
/truck_routing_app/
├── .streamlit/                  # Cấu hình Streamlit
├── core/                        # Logic nghiệp vụ cốt lõi
│   ├── algorithms/              # Các thuật toán tìm đường
│   │   ├── astar.py             # Thuật toán A*
│   │   ├── astar_fuel.py        # Biến thể A* có xét yếu tố nhiên liệu
│   │   ├── backtracking_csp.py  # Thuật toán backtracking và CSP
│   │   ├── base_pathfinder.py   # Interface chung cho các thuật toán
│   │   ├── base_search.py       # Lớp cơ sở cho các thuật toán tìm kiếm
│   │   ├── bfs.py               # Tìm kiếm theo chiều rộng
│   │   ├── blind.py             # Thuật toán tìm kiếm mù
│   │   ├── custom_buffers.py    # Buffer tùy chỉnh cho RL
│   │   ├── custom_policy.py     # Chính sách tùy chỉnh cho RL
│   │   ├── dfs.py               # Tìm kiếm theo chiều sâu
│   │   ├── genetic_algorithm.py # Giải thuật di truyền
│   │   ├── greedy.py            # Tìm kiếm tham lam
│   │   ├── hyperparameter_tuning.py # Tinh chỉnh siêu tham số
│   │   ├── idastar.py           # A* lặp sâu dần
│   │   ├── ids.py               # Tìm kiếm lặp sâu dần
│   │   ├── local_beam.py        # Tìm kiếm chùm cục bộ
│   │   ├── rl_DQNAgent.py       # Agent học tăng cường DQN
│   │   ├── simulated_annealing.py # Giải thuật luyện kim mô phỏng
│   │   ├── ucs.py               # Tìm kiếm chi phí đồng nhất
│   │   └── __init__.py          # Định nghĩa module
│   ├── and_or_search_logic/     # Logic cho giải thuật tìm kiếm AND/OR
│   │   ├── environment.py       # Môi trường cho tìm kiếm AND/OR
│   │   ├── problem_definition.py # Định nghĩa bài toán
│   │   ├── search_algorithm.py  # Thuật toán tìm kiếm
│   │   └── state_and_actions.py # Các trạng thái và hành động
│   ├── constants.py             # Các hằng số và tham số
│   ├── constraints.py           # Các ràng buộc của bài toán
│   ├── dynamics.py              # Mô tả động lực học của hệ thống
│   ├── map.py                   # Xử lý logic bản đồ
│   ├── rl_environment.py        # Môi trường học tăng cường
│   ├── state.py                 # Định nghĩa trạng thái hệ thống
│   └── __init__.py              # Định nghĩa module
├── evaluation_results/          # Kết quả đánh giá hiệu suất
├── logs/                        # Log hệ thống
├── statistics/                  # Phân tích thống kê
├── Train/                       # Thư mục chứa dữ liệu huấn luyện
├── training_logs/               # Log huấn luyện mô hình
├── ui/                          # Giao diện người dùng
│   ├── algorithm_evaluation.py  # Trang đánh giá thuật toán
│   ├── dashboard.py             # Bảng điều khiển tổng hợp
│   ├── home.py                  # Trang chủ
│   ├── map_config.py            # Cấu hình bản đồ
│   ├── map_display.py           # Hiển thị bản đồ
│   ├── routing_visualization.py # Hiển thị tuyến đường
│   └── __init__.py              # Định nghĩa module
├── app.py                       # Điểm khởi chạy chính
├── __pycache__/                 # Cache Python (tự động tạo)
└── truck_routing_app/           # Module con (có thể chứa phiên bản khác)
```

Cấu trúc dự án được tổ chức theo mô hình module rõ ràng, phân tách logic nghiệp vụ với giao diện người dùng, giúp dễ dàng mở rộng và bảo trì.

## 🖥️ GIAO DIỆN NGƯỜI DÙNG


Ứng dụng được thiết kế với giao diện thân thiện, trực quan và được chia thành các module chức năng chính:

### 1. Trang chủ (Home)
- Giới thiệu tổng quan về ứng dụng và chức năng
- Hướng dẫn sử dụng chi tiết cho người mới
- Tổng quan về các tính năng mới nhất và cập nhật

### 2. Tạo Bản Đồ & Cấu Hình (Map Configuration)
- Công cụ thiết kế bản đồ với nhiều tùy chỉnh
- Điều chỉnh các tham số như kích thước, tỷ lệ các loại ô
- Chọn vị trí xuất phát và đích đến
- Tùy chỉnh các ràng buộc bài toán (nhiên liệu ban đầu, ngân sách)
- Lưu và tải các cấu hình bản đồ đã tạo

### 3. Định Tuyến & Tối Ưu (Routing Visualization)
- Hiển thị bản đồ và tuyến đường trực quan
- Chọn thuật toán và tinh chỉnh tham số
- Theo dõi quá trình tìm kiếm theo thời gian thực
- Phân tích chi tiết về tuyến đường tìm được
- Mô phỏng quá trình di chuyển trên tuyến đường

### 4. Đánh Giá Thuật Toán (Algorithm Evaluation)
- So sánh hiệu suất các thuật toán
- Biểu đồ phân tích đa chiều
- Xuất báo cáo đánh giá chi tiết
- Phân tích điểm mạnh, điểm yếu của từng thuật toán

## 📝 YÊU CẦU CÀI ĐẶT

Để chạy ứng dụng, bạn cần cài đặt các thành phần sau:

### Yêu cầu cơ bản
- Python 3.8 trở lên
- Streamlit 1.15.0 trở lên
- NumPy 1.20.0 trở lên
- Pandas 1.3.0 trở lên

### Thư viện cho thuật toán nâng cao
- PyTorch 1.10.0 trở lên (cho các thành phần học tăng cường)
- scikit-learn 1.0.0 trở lên (cho tinh chỉnh siêu tham số)
- matplotlib 3.5.0 trở lên (cho trực quan hóa)
- plotly 5.5.0 trở lên (cho biểu đồ tương tác)

### Yêu cầu hệ thống
- RAM: Tối thiểu 4GB (khuyến nghị 8GB cho bản đồ lớn)
- CPU: Dual-core trở lên (khuyến nghị quad-core cho mô hình học tăng cường)
- GPU: Không bắt buộc, nhưng khuyến nghị cho huấn luyện DQN

Tất cả các thư viện cần thiết được liệt kê trong file `requirements.txt`.

## ⚙️ HƯỚNG DẪN CÀI ĐẶT

### Cài đặt từ mã nguồn

1. Clone repository:
   ```bash
   git clone https://github.com/AI_NHOM/path-simulation-app.git
   cd path-simulation-app
   ```

2. Tạo và kích hoạt môi trường ảo (khuyến nghị):
   ```bash
   # Sử dụng venv
   python -m venv venv
   
   # Trên Windows
   venv\Scripts\activate
   
   # Trên macOS/Linux
   source venv/bin/activate
   ```

3. Cài đặt các thư viện cần thiết:
   ```bash
   pip install -r requirements.txt
   ```

4. Khởi chạy ứng dụng:
   ```bash
   streamlit run app.py
   ```

### Sử dụng Docker (tùy chọn)

1. Xây dựng Docker image:
   ```bash
   docker build -t path-simulation-app .
   ```

2. Chạy container:
   ```bash
   docker run -p 8501:8501 path-simulation-app
   ```

Ứng dụng sẽ khả dụng tại địa chỉ `http://localhost:8501` trên trình duyệt của bạn.

## 📊 KẾT QUẢ NGHIÊN CỨU


Dự án đã đạt được nhiều kết quả nghiên cứu quan trọng:

### So sánh thuật toán
- Phân tích hiệu suất giữa các thuật toán truyền thống và học tăng cường trên nhiều loại bản đồ
- Chứng minh tính hiệu quả của DQN trong các môi trường động và phức tạp
- Đánh giá chi tiết trade-off giữa thời gian xử lý và chất lượng giải pháp

### Tối ưu hóa tham số
- Xác định các giá trị tối ưu cho hàm heuristic trong A* và các biến thể
- Phân tích ảnh hưởng của các siêu tham số đến hiệu suất của DQN
- Đề xuất chiến lược tinh chỉnh tham số cho các loại bản đồ khác nhau

### Mô hình tiêu thụ nhiên liệu
- Phát triển mô hình dự đoán mức tiêu thụ nhiên liệu dựa trên đặc điểm địa hình
- Tối ưu hóa tuyến đường dựa trên chi phí nhiên liệu và điểm tiếp nhiên liệu
- Đánh giá hiệu quả của các chiến lược tiết kiệm nhiên liệu

### Ứng dụng thực tế
- Mô phỏng các kịch bản logistics thực tế với nhiều ràng buộc
- Phân tích hiệu quả chi phí của các chiến lược định tuyến khác nhau
- Đề xuất mô hình tích hợp cho hệ thống vận tải thông minh





Được phát triển với 💻 và ❤️ bởi Thắng và Tú
