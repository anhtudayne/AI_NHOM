HỆ THỐNG Mô Phỏng Đương Đi
=============================================================

🚚 Chào mừng bạn đến với Hệ Thống Mô Phỏng Định Tuyến Phân Phối Hàng Hóa! Dự án này được xây dựng để cung cấp một công cụ mạnh mẽ cho việc mô phỏng, trực quan hóa và tìm kiếm các tuyến đường vận chuyển tối ưu cho xe tải.

TỔNG QUAN
---------

Đây là một ứng dụng web tương tác được phát triển bằng Streamlit. Mục tiêu chính là cho phép người dùng:
- Thiết kế và tùy chỉnh bản đồ: Xác định các loại địa hình như đường đi, trạm thu phí, trạm xăng và các chướng ngại vật.
- Thử nghiệm đa dạng thuật toán: Áp dụng và so sánh hiệu quả của nhiều thuật toán tìm đường, từ các giải thuật cổ điển đến các phương pháp dựa trên học tăng cường.
- Phân tích chi phí và hiệu quả: Đánh giá các tuyến đường dựa trên các yếu tố quan trọng như tiêu thụ nhiên liệu, phí cầu đường, và các ràng buộc vận hành khác.

Ứng dụng này không chỉ là một công cụ giải quyết bài toán định tuyến mà còn là một môi trường để nghiên cứu và đánh giá sâu hơn về các chiến lược tối ưu hóa trong lĩnh vực logistics.

CẤU TRÚC THƯ MỤC DỰ ÁN
----------------------

Dưới đây là cái nhìn tổng quan về cách tổ chức các thành phần trong dự án:

```
/
├── .streamlit/                  # Chứa các file cấu hình riêng cho Streamlit (ví dụ: secrets, themes).
├── core/                        # Nơi chứa đựng toàn bộ logic nghiệp vụ cốt lõi của hệ thống.
│   ├── algorithms/              # Bộ sưu tập các thuật toán tìm đường và tối ưu hóa.
│   │   ├── astar.py             # Thuật toán A*
│   │   ├── astar_fuel.py        # Biến thể A* có xét yếu tố nhiên liệu
│   │   ├── base_search.py       # Lớp cơ sở hoặc tiện ích chung cho các thuật toán tìm kiếm
│   │   ├── bfs.py               # Tìm kiếm theo chiều rộng (Breadth-First Search)
│   │   ├── dfs.py               # Tìm kiếm theo chiều sâu (Depth-First Search)
│   │   ├── genetic_algorithm.py # Giải thuật di truyền
│   │   ├── greedy.py            # Tìm kiếm tham lam
│   │   ├── hyperparameter_tuning.py # Tinh chỉnh siêu tham số cho mô hình học máy
│   │   ├── idastar.py           # A* lặp sâu dần (Iterative Deepening A*)
│   │   ├── ids.py               # Tìm kiếm lặp sâu dần (Iterative Deepening Search)
│   │   ├── local_beam.py        # Tìm kiếm chùm cục bộ
│   │   ├── rl_DQNAgent.py       # Agent học tăng cường sử dụng Deep Q-Network
│   │   ├── simulated_annealing.py # Giải thuật luyện kim mô phỏng
│   │   └── ucs.py               # Tìm kiếm chi phí đồng nhất (Uniform Cost Search)
│   ├── and_or_search_logic/     # Logic cho giải thuật tìm kiếm trên đồ thị AND/OR.
│   │   ├── environment.py
│   │   ├── problem_definition.py
│   │   ├── search_algorithm.py
│   │   └── state_and_actions.py
│   ├── constants.py             # Định nghĩa các hằng số quan trọng: chi phí, trọng số, loại ô trên bản đồ.
│   ├── constraints.py           # Các ràng buộc của bài toán định tuyến.
│   ├── dynamics.py              # Mô tả động lực học của hệ thống (nếu có, ví dụ: thay đổi trạng thái xe).
│   ├── map.py                   # Xử lý logic bản đồ, bao gồm các loại ô và tương tác.
│   ├── rl_environment.py        # Môi trường mô phỏng cho các tác nhân học tăng cường.
│   └── state.py                 # Định nghĩa cấu trúc trạng thái của hệ thống.
├── evaluation_results/          # Nơi lưu trữ kết quả đánh giá hiệu suất của các thuật toán.
├── hyperparameter_tuning_results/ # Kết quả từ quá trình tinh chỉnh siêu tham số cho các mô hình.
├── logs/                        # Log hệ thống và các sự kiện quan trọng.
├── training_logs/               # Log chi tiết của quá trình huấn luyện các mô hình học máy.
├── truck_routing_app/           # Module con, có thể chứa một phiên bản khác hoặc các thành phần chuyên biệt
│                                # liên quan đến học tăng cường (agents, environments).
│   └── core/
│       ├── agents/
│       └── environments/
├── ui/                          # Các file Python xây dựng giao diện người dùng với Streamlit.
│   ├── algorithm_evaluation.py  # Trang dùng để đánh giá và so sánh các thuật toán.
│   ├── dashboard.py             # Có thể là một bảng điều khiển tổng hợp hoặc thành phần phụ.
│   ├── home.py                  # Trang chủ của ứng dụng.
│   ├── map_config.py            # Giao diện cấu hình và tạo bản đồ.
│   ├── map_display.py           # Thành phần trực quan hóa bản đồ.
│   └── routing_visualization.py # Trang hiển thị và tương tác với tuyến đường đã tối ưu.
├── Train/                       # Thư mục có thể chứa dữ liệu huấn luyện hoặc các kịch bản training.
├── app.py                       # Điểm khởi chạy chính của ứng dụng Streamlit.
└── README.md                    # Chính là tài liệu bạn đang đọc.
```

CHỨC NĂNG CHÍNH
---------------

Ứng dụng được chia thành các module chức năng chính, truy cập qua thanh điều hướng bên cạnh:

1.  Trang chủ (`ui/home.py`):
    -   Cung cấp lời chào mừng và giới thiệu tổng quát về mục đích và khả năng của hệ thống.

2.  Tạo Bản Đồ & Cấu Hình (`ui/map_config.py`):
    -   Tạo bản đồ linh hoạt: Cho phép người dùng định nghĩa kích thước bản đồ (số hàng, số cột).
    -   Phân bố loại ô: Tùy chỉnh tỷ lệ các loại ô trên bản đồ: đường đi (`ROAD`), trạm thu phí (`TOLL`), trạm xăng (`GAS`), và vật cản (`OBSTACLE`).
    -   Xác định điểm xuất phát: Chọn tọa độ khởi hành cho xe tải, đảm bảo vị trí này là một ô đường hợp lệ.
    -   Lưu/Tải bản đồ: Hỗ trợ lưu cấu hình bản đồ hiện tại để sử dụng lại hoặc tải các bản đồ đã tạo trước đó.

3.  Định Tuyến & Tối Ưu Hệ Thống (`ui/routing_visualization.py`):
    -   Trực quan hóa đa dạng: Hiển thị bản đồ một cách rõ ràng cùng với tuyến đường được đề xuất bởi thuật toán.
    -   Lựa chọn thuật toán: Người dùng có thể chọn từ một danh sách phong phú các thuật toán tìm đường đã được tích hợp.
    -   Thông tin chi tiết tuyến đường: Cung cấp các số liệu quan trọng về tuyến đường được tìm thấy, bao gồm tổng chi phí, lượng nhiên liệu tiêu thụ ước tính, quãng đường, và các thông tin khác liên quan đến các trạm dịch vụ.

4.  Đánh Giá Thuật Toán (`ui/algorithm_evaluation.py`):
    -   So sánh hiệu năng: Trang này dành riêng cho việc so sánh các thuật toán với nhau dựa trên nhiều tiêu chí (ví dụ: thời gian xử lý, chi phí tuyến đường, độ phức tạp).
    -   Phân tích kết quả: Giúp người dùng hiểu rõ hơn về ưu nhược điểm của từng phương pháp trong các kịch bản bản đồ và ràng buộc khác nhau.

LOGIC CỐT LÕI (`core/`)
----------------------

Phần trái tim của hệ thống nằm trong thư mục `core/`, nơi xử lý tất cả các tính toán phức tạp:

-   Quản lý Bản đồ và Chi phí (`core/map.py`, `core/constants.py`):
    -   Định nghĩa và quản lý các thực thể trên bản đồ như `CellType` (ROAD, TOLL, GAS, OBSTACLE).
    -   Tính toán và quản lý các loại chi phí: `MovementCosts` (chi phí di chuyển, nhiên liệu), `StationCosts` (chi phí tại trạm xăng, trạm thu phí).
    -   Sử dụng `PathfindingWeights` để xác định trọng số cho các thuật toán tìm đường. Đặc biệt, các trọng số này có thể được điều chỉnh động, ví dụ, trọng số của trạm xăng sẽ thay đổi tùy thuộc vào mức nhiên liệu hiện tại của xe (`calculate_gas_station_weight`), và trọng số trạm thu phí có thể thay đổi dựa trên số trạm đã đi qua (`calculate_toll_station_weight`).

-   Bộ Thuật Toán Tìm Đường (`core/algorithms/`):
    -   Nơi quy tụ một thư viện đa dạng các thuật toán, từ những thuật toán tìm kiếm cơ bản (BFS, DFS, UCS) đến các thuật toán có sử dụng thông tin heuristic (A*, IDA*).
    -   Bao gồm cả các thuật toán tối ưu hóa cục bộ (Greedy, Local Beam Search) và các phương pháp metaheuristic tiên tiến (Simulated Annealing, Genetic Algorithm).
    -   Đặc biệt, có sự hiện diện của tác nhân học tăng cường (Deep Q-Network Agent - `rl_DQNAgent.py`) cho phép hệ thống học hỏi và tìm ra chiến lược định tuyến thông minh.
    -   `base_search.py` đóng vai trò là nền tảng, cung cấp các cấu trúc và hàm dùng chung, giúp việc triển khai các thuật toán mới trở nên nhất quán và dễ dàng hơn.
    -   Chức năng tinh chỉnh siêu tham số (`hyperparameter_tuning.py`) hỗ trợ việc tìm ra bộ thông số tốt nhất cho các mô hình học máy.

-   Học Tăng Cường (`core/rl_environment.py`, `core/algorithms/rl_DQNAgent.py`):
    -   Xây dựng một môi trường (`rl_environment.py`) mô phỏng chi tiết các tương tác của xe tải, làm cơ sở cho việc huấn luyện các tác nhân học tăng cường như DQN Agent.

-   Tìm kiếm AND/OR (`core/and_or_search_logic/`):
    -   Cung cấp một phương pháp tiếp cận khác cho các bài toán có thể được phân rã thành các bài toán con độc lập hoặc phụ thuộc, phù hợp với một số loại vấn đề lập kế hoạch phức tạp.

CÁC THƯ MỤC LƯU TRỮ KẾT QUẢ
-----------------------------

Trong quá trình hoạt động và thử nghiệm, ứng dụng sẽ tạo ra và lưu trữ dữ liệu tại các thư mục sau:
-   `evaluation_results/`: Các báo cáo, số liệu từ việc đánh giá và so sánh thuật toán.
-   `hyperparameter_tuning_results/`: Kết quả của các lần chạy tinh chỉnh siêu tham số.
-   `logs/`: Ghi lại các thông tin gỡ lỗi, cảnh báo và các sự kiện hoạt động của ứng dụng.
-   `training_logs/`: Lưu trữ nhật ký chi tiết từ quá trình huấn luyện các mô hình học máy, đặc biệt là các mô hình RL.

THIẾT LẬP VÀ CHẠY ỨNG DỤNG
--------------------------

Để khởi chạy ứng dụng này trên máy của bạn:

1.  Yêu cầu cần có:
    -   Đảm bảo bạn đã cài đặt Python (phiên bản 3.x khuyến nghị).
    -   Cài đặt Streamlit: `pip install streamlit`.
    -   Các thư viện khác mà dự án yêu cầu (ví dụ: NumPy, Pandas, PyTorch nếu bạn sử dụng các thành phần RL). `app.py` có chứa một đoạn mã để xử lý một vấn đề tương thích tiềm ẩn giữa PyTorch và Streamlit, sẽ tự động được áp dụng nếu PyTorch được cài đặt.

2.  Khởi chạy ứng dụng:
    Mở terminal hoặc command prompt, điều hướng đến thư mục gốc của dự án và chạy lệnh sau:
    ```bash
    streamlit run app.py
    ```
    Ứng dụng sẽ tự động mở trong trình duyệt web của bạn.

ĐÓNG GÓP
--------

Hiện tại, dự án đang được phát triển và hoàn thiện. Nếu bạn có ý tưởng đóng góp hoặc phát hiện lỗi, vui lòng tạo một "issue" trên repository (nếu có) hoặc liên hệ với nhóm phát triển.

GIẤY PHÉP
---------

(Phần này sẽ được cập nhật với thông tin giấy phép cụ thể của dự án. Ví dụ: MIT, Apache 2.0, etc.)

---
Hy vọng tài liệu này sẽ giúp bạn hiểu rõ hơn về dự án. Chúc bạn có những trải nghiệm thú vị với Hệ Thống Định Tuyến Phân Phối Hàng Hóa!
