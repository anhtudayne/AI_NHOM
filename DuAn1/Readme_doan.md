# Dự Án Định Tuyến Phân Phối Hàng Hóa với Streamlit

## Giới thiệu

Dự án này mô phỏng quá trình di chuyển của xe tải chở hàng trên một bản đồ dạng lưới với giao diện trực quan, sinh động và chuyên nghiệp – hoàn toàn được xây dựng trên nền tảng Streamlit mà không cần thêm CSS hay HTML tùy chỉnh. Người dùng có thể tạo bản đồ bằng tay hoặc sinh ngẫu nhiên, với các ô đặc biệt được biểu diễn bằng các icon sinh động như xe tải, cây xăng và trạm thu phí.

Dự án tích hợp đầy đủ các thuật toán định tuyến và tối ưu:

- **Uninformed Search (BFS/DFS):** Duyệt không gian ban đầu.
- **Informed Search (A\*):** Tìm tuyến đường tối ưu dựa trên hàm heuristic tích hợp các yếu tố: khoảng cách, phí cầu, tiêu hao nhiên liệu.
- **Local Search:** Tối ưu hóa tuyến đường ban đầu bằng các phương pháp hill climbing hoặc simulated annealing.
- **Constraint Satisfaction & Environment Dynamics:** Áp dụng các ràng buộc (ví dụ: xe phải dừng đổ xăng sau khoảng cách quy định, tính phí khi đi qua ô trạm thu phí) và mô phỏng các yếu tố môi trường thay đổi (giao thông, thời tiết, tai nạn, tắc nghẽn,…).
- **Reinforcement Learning (RL):** Huấn luyện agent học từ dữ liệu mô phỏng để tự động cải thiện chiến lược định tuyến qua thời gian.

## Tính năng chính

### 1. Tab Tạo Bản Đồ & Cấu Hình
- **Nhập thông số:** Cho phép người dùng nhập kích thước bản đồ (n x n), tỷ lệ ô có trạm thu phí và ô đổ xăng.
- **Tùy chọn bản đồ:** Lựa chọn vẽ bản đồ thủ công hoặc tạo bản đồ ngẫu nhiên.
- **Hiển thị trực quan:** Bản đồ dạng lưới được hiển thị với các icon sinh động:
  - **Icon xe tải:** Vị trí xe.
  - **Icon cây xăng:** Điểm đổ xăng.
  - **Icon trạm thu phí:** Ô có phí cầu.

### 2. Tab Định Tuyến & Tối Ưu Hệ Thống (Visualization & Comparison)
- **Quá trình định tuyến toàn diện:** Gộp các bước của Uninformed Search, Informed Search (A\*) và Local Search vào một tab duy nhất.
- **Visual Step-by-Step:**  
  - Hiển thị từng bước của thuật toán trên bản đồ với animation minh họa:
    - **Uninformed Search:** Các nút đã thăm và frontier được đánh dấu màu sắc.
    - **A\*:** Các nút đã thăm, đang xét và đường đi tối ưu được phân biệt bằng màu sắc khác nhau kèm chú thích.
    - **Local Search:** So sánh tuyến đường ban đầu và tuyến đường sau tối ưu hóa.
- **Điều khiển tương tác:**  
  - Các nút "Pause," "Play," "Step Forward," "Step Backward" cho phép người dùng kiểm soát quá trình hiển thị từng bước, giúp mục đích giáo dục và phân tích hoạt động của thuật toán.
- **Tùy chỉnh tham số:** Sử dụng các widget như st.button, st.slider để điều chỉnh tham số của từng thuật toán và xem ảnh hưởng theo thời gian thực.

### 3. Tab Constraint Satisfaction & Environment Dynamics
- **Ràng buộc định tuyến:**  
  - Đảm bảo xe phải dừng đổ xăng sau khoảng cách nhất định.
  - Tính phí khi đi qua ô có trạm thu phí.
- **Mô phỏng môi trường động:**  
  - Cho phép người dùng kích hoạt các sự kiện ngẫu nhiên như tắc đường, đóng đường, thời tiết xấu, tai nạn.
  - Mô phỏng giao thông theo giờ cao điểm với mật độ xe tăng, ảnh hưởng đến chi phí và thời gian định tuyến.
  - Cập nhật định tuyến tự động khi môi trường có thay đổi (ví dụ: nếu có tai nạn, hệ thống sẽ tự động tìm tuyến đường mới và hiển thị sự thay đổi ngay lập tức).

### 4. Tab Reinforcement Learning (RL) Simulation & Training
- **Mô hình MDP:**  
  - Trạng thái gồm vị trí xe, lượng nhiên liệu và điều kiện môi trường.
- **Huấn luyện Agent:**  
  - Sử dụng thuật toán RL (Q-Learning, DQN hoặc PPO) để tối ưu hóa chiến lược định tuyến.
  - Hiển thị quá trình huấn luyện thông qua biểu đồ tiến trình, cho thấy sự cải thiện theo số episode.
- **So sánh kết quả:**  
  - So sánh tuyến đường trước và sau huấn luyện RL, đánh giá hiệu quả cải thiện qua các chỉ số: tổng chi phí, khoảng cách, số lần dừng đổ xăng, thời gian tính toán.

### 5. Tab Dashboard & Phân Tích (So sánh & Đánh giá)
- **Biểu đồ tương tác:**  
  - Sử dụng Plotly để tạo biểu đồ đường, biểu đồ cột hiển thị số liệu theo thời gian thực, cho phép người dùng chọn thuật toán và điều chỉnh tham số.
- **Bảng so sánh chi tiết:**  
  - Bảng thống kê với các chỉ số:
  
    | Thuật toán    | Tổng chi phí | Khoảng cách (km) | Số lần dừng | Thời gian tính toán (s) |
    |---------------|--------------|------------------|------------|-------------------------|
    | BFS           | 500          | 100              | 2          | 0.5                     |
    | A\*           | 450          | 95               | 2          | 0.7                     |
    | Local Search  | 430          | 92               | 2          | 1.0                     |
    | RL (PPO)      | 420          | 90               | 2          | 2.0                     |
  
- **Xuất dữ liệu:**  
  - Cho phép xuất dữ liệu kết quả mô phỏng ra file CSV để người dùng có thể phân tích sâu hơn ngoài ứng dụng.

### 6. Tab Cài Đặt & Tùy Chỉnh
- **Cấu hình tham số:**  
  - Thiết lập tham số cho A\* (trọng số khoảng cách, phí cầu, tiêu hao nhiên liệu).
  - Thiết lập tham số cho Local Search (số vòng lặp, nhiệt khởi đầu, ngưỡng dừng).
  - Thiết lập tham số cho RL (learning rate, discount factor, số episode huấn luyện).
- **Lưu & tải cấu hình:**  
  - Cho phép người dùng lưu lại và tải lại các thiết lập dự án.

## Cài đặt và chạy dự án

### Yêu cầu:
- Python 3.7+
- Các thư viện cần thiết:
  - [Streamlit](https://streamlit.io/)
  - [NumPy](https://numpy.org/)
  - [Matplotlib](https://matplotlib.org/) hoặc [Plotly](https://plotly.com/)
  - [NetworkX](https://networkx.org/)
  - [Pillow](https://pillow.readthedocs.io/en/stable/) (để xử lý icon)
  - Các thư viện RL (ví dụ: Stable Baselines3, TensorFlow hoặc PyTorch)

### Cài đặt:
1. Clone repository:
   ```bash
   git clone https://github.com/yourusername/your-project.git
   cd your-project
   ```
2. Cài đặt các thư viện:
   ```bash
   pip install -r requirements.txt
   ```
3. Chạy ứng dụng Streamlit:
   ```bash
   streamlit run app.py
   ```

## Cấu trúc dự án

```
your-project/
├── app.py                     # File chính chạy Streamlit
├── modules/
│   ├── map_generator.py       # Tạo bản đồ và xử lý giao diện
│   ├── uninformed_search.py   # Các hàm BFS/DFS
│   ├── a_star.py              # Hàm tìm đường A*
│   ├── local_search.py        # Các thuật toán tối ưu cục bộ
│   ├── constraints.py         # Xử lý ràng buộc (CSP)
│   ├── environment.py         # Mô phỏng yếu tố môi trường động
│   ├── rl_agent.py            # Mô hình RL và huấn luyện agent
├── assets/                    # Icon và hình ảnh minh họa (xe, cây xăng, trạm thu phí, ...)
├── requirements.txt           # Danh sách các thư viện cần thiết
└── README.md                  # Hướng dẫn sử dụng (file này)
```

## Giao diện & Visual

- **Bản đồ Sinh động:**  
  Tab "Tạo Bản Đồ & Cấu Hình" cho phép nhập thông số và hiển thị bản đồ dạng lưới với các icon trực quan, giúp xác định rõ vị trí xe, điểm đổ xăng và trạm thu phí.

- **Visual các Thuật toán:**  
  Quá trình định tuyến được minh họa theo từng bước với animation tương tác, cho phép người dùng điều khiển (Pause, Play, Step Forward, Step Backward) để theo dõi cách hoạt động của các thuật toán:
  - Uninformed Search (BFS/DFS)
  - Informed Search (A\*)
  - Local Search

- **So sánh & Đánh giá:**  
  Tab Dashboard & Phân Tích cung cấp các biểu đồ tương tác và bảng số liệu chi tiết, đồng thời hỗ trợ xuất dữ liệu kết quả mô phỏng ra file CSV để người dùng có thể phân tích sâu hơn.

## Kết luận

Dự án này cung cấp một hệ thống mô phỏng định tuyến phân phối hàng hóa tích hợp đầy đủ các thuật toán định tuyến, tối ưu hóa và học máy qua RL, với giao diện trực quan và sinh động sử dụng Streamlit. Các tính năng tương tác như điều khiển từng bước của thuật toán, animation minh họa, cập nhật môi trường động và xuất dữ liệu ra file CSV không chỉ giúp người dùng hiểu rõ quá trình định tuyến mà còn hỗ trợ mục đích giáo dục, nghiên cứu và báo cáo.

Nếu có bất kỳ câu hỏi hoặc góp ý nào, xin vui lòng mở issue trên GitHub hoặc liên hệ qua email: vut210225@gmail.com.

