# Reinforcement Learning cho Định Tuyến Xe Tải

Tài liệu này mô tả các tính năng Học Tăng Cường (Reinforcement Learning) nâng cao trong hệ thống Định Tuyến Xe Tải, bao gồm hướng dẫn sử dụng, tích hợp với ứng dụng chính, và các tính năng đặc biệt.

## Tổng quan

Tính năng Học Tăng Cường (RL) cho phép xe tải học cách di chuyển tối ưu trên bản đồ dựa trên kinh nghiệm. Agent RL sử dụng thuật toán Deep Q-Network (DQN) để học cách ra quyết định tốt nhất tại mỗi vị trí, đồng thời cân bằng giữa hiệu quả về thời gian, chi phí và an toàn.

## Tích hợp trong Ứng dụng Chính (Streamlit)

Hệ thống đã tích hợp đầy đủ Agent RL vào ứng dụng Streamlit với các tính năng sau:

### 1. Lựa chọn thuật toán "Học Tăng Cường (RL)"

Trong tab "Định Tuyến & Tối Ưu Hệ Thống", bạn có thể chọn "Học Tăng Cường (RL)" từ danh sách thuật toán. Khi chọn thuật toán này, bạn sẽ thấy thêm các tùy chọn cấu hình đặc biệt dành cho RL.

### 2. Tùy chọn mô hình

Hệ thống sẽ tự động dò tìm và hiển thị danh sách các mô hình RL đã huấn luyện từ thư mục `saved_models`. Bạn có thể chọn mô hình mong muốn từ danh sách này.

### 3. Chiến lược ưu tiên

Bạn có thể chọn một trong các chiến lược ưu tiên sau:

- **Cân bằng (mặc định)**: Cân bằng giữa thời gian, chi phí và an toàn
- **Tiết kiệm chi phí**: Ưu tiên tiết kiệm tiền, tránh trạm thu phí khi có thể
- **Nhanh nhất**: Ưu tiên đường đi ngắn nhất, không quan tâm chi phí
- **An toàn nhiên liệu**: Luôn đảm bảo mức nhiên liệu an toàn, ưu tiên ghé trạm xăng

### 4. Thống kê và đánh giá RL

Khi chạy Agent RL, tab "RL Metrics" sẽ hiển thị trong phần "Xem thống kê chi tiết" với các thông tin như:

- Tổng phần thưởng
- Số lần đổ xăng
- Số trạm thu phí đã qua
- Chiến lược ưu tiên
- Model đã sử dụng

## Huấn luyện và Đánh giá với RL Test App

Để huấn luyện và đánh giá Agent RL một cách chi tiết, bạn nên sử dụng ứng dụng `rl_test.py` với các tính năng nâng cao:

### 1. Huấn luyện cơ bản

Trong tab "Agent RL (DQN)", bạn có thể:
- Điều chỉnh các siêu tham số
- Huấn luyện model
- Lưu và tải model
- Chạy agent trên bản đồ hiện tại

### 2. Tinh chỉnh siêu tham số

Tab "RL Nâng cao" > "Tinh chỉnh Siêu tham số" cho phép:
- Tối ưu hóa tự động các siêu tham số bằng Optuna
- Huấn luyện model với các tham số tốt nhất

### 3. Đánh giá chi tiết

Tab "RL Nâng cao" > "Đánh giá Chi tiết" cung cấp:
- Đánh giá agent trên nhiều bản đồ khác nhau
- Thống kê chi tiết về tỷ lệ thành công, phần thưởng, độ dài đường đi...
- Biểu đồ trực quan hóa hiệu suất

### 4. So sánh thuật toán

Tab "RL Nâng cao" > "So sánh Thuật toán" cho phép:
- So sánh Agent RL với các thuật toán truyền thống như A*, Greedy, Simulated Annealing...
- Hiển thị đường đi của cả hai thuật toán trên cùng một bản đồ
- So sánh các chỉ số như thời gian, chi phí, độ dài đường đi

## Yêu cầu hệ thống

Để sử dụng tính năng RL, bạn cần cài đặt các thư viện sau:

```bash
pip install stable-baselines3 gymnasium tensorboard optuna
```

## Thiết lập môi trường

Để chuẩn bị môi trường cho việc huấn luyện và sử dụng RL:

1. Chạy script thiết lập: `python setup_rl_directories.py`
2. Tạo các thư mục bản đồ từ ứng dụng `rl_test.py` > tab "RL Nâng cao" > "Tạo Thư mục Bản đồ"

## Tính năng nâng cao

### Khả năng thích ứng (Adaptability)

Agent RL được huấn luyện trên nhiều loại bản đồ khác nhau, giúp nó thích ứng với các môi trường mới mà không cần huấn luyện lại.

### Tối ưu hóa đa mục tiêu (Multi-Objective Optimization)

Thông qua việc điều chỉnh các trọng số trong hàm phần thưởng, Agent RL có thể tối ưu hóa đồng thời nhiều mục tiêu như thời gian, chi phí và an toàn.

### Ra quyết định thông minh

Agent RL không chỉ chọn đường đi ngắn nhất mà còn có thể quyết định:
- Khi nào cần đổ xăng
- Có nên đi qua trạm thu phí hay không
- Đường đi nào cân bằng tốt nhất giữa các yếu tố

## Tips để có hiệu quả tốt nhất

1. **Huấn luyện đủ lâu**: Các agent RL cần thời gian để học hành vi tối ưu. Tối thiểu 50,000 bước huấn luyện để có kết quả tốt.

2. **Tinh chỉnh siêu tham số**: Sử dụng tính năng tinh chỉnh siêu tham số với ít nhất 20 lần thử để tìm ra cấu hình tốt nhất.

3. **Đánh giá kỹ lưỡng**: Luôn đánh giá agent trên nhiều bản đồ khác nhau để đảm bảo tính ổn định.

4. **So sánh với thuật toán truyền thống**: Đánh giá chất lượng của agent RL bằng cách so sánh với các thuật toán như A* và Greedy.

## Thông tin thêm

Để biết thêm thông tin chi tiết về cấu trúc và cách thức hoạt động của thuật toán RL, vui lòng tham khảo mã nguồn trong thư mục `core/algorithms/` và các file liên quan đến môi trường RL trong `core/rl_environment.py`. 