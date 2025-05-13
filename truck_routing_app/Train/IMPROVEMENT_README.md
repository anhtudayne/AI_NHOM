# RL Agent Improvements

## Cải tiến RL cho Truck Routing

Tài liệu này mô tả các cải tiến đã được thực hiện để nâng cao hiệu suất của agent RL (Reinforcement Learning) trong bài toán định tuyến xe tải.

## Các cải tiến đã thực hiện

### 1. Cải tiến thuật toán DQN

#### Double DQN
- **Mô tả**: Sử dụng hai mạng neural để tách riêng việc chọn hành động và đánh giá hành động, giúp giảm overestimation bias.
- **Lợi ích**: Cải thiện tính ổn định trong quá trình học, đặc biệt trong môi trường phức tạp.
- **Thực hiện**: `use_double_dqn=True` mặc định trong `DQNAgentTrainer.create_model()`.

#### Dueling Network Architecture
- **Mô tả**: Tách biệt việc ước lượng giá trị trạng thái (state value) và lợi thế của hành động (action advantage).
- **Lợi ích**: Học tốt hơn về giá trị của các trạng thái, giúp đánh giá chính xác hơn khi không phải mọi hành động đều quan trọng.
- **Thực hiện**: `use_dueling_network=True` trong `DQNAgentTrainer.create_model()`.

#### Prioritized Experience Replay (PER)
- **Mô tả**: Ưu tiên replay các kinh nghiệm có sai số TD (Temporal Difference) cao.
- **Lợi ích**: Tập trung học hỏi từ những tình huống khó và mới lạ.
- **Thực hiện**: `use_prioritized_replay=True` trong `DQNAgentTrainer.create_model()`.

### 2. Cải tiến khả năng khám phá (Exploration)

#### Adaptive Exploration
- **Mô tả**: Tự động điều chỉnh tỷ lệ khám phá (epsilon) dựa trên hiệu suất học tập.
- **Lợi ích**: Tăng cường khám phá khi agent gặp khó khăn, và khai thác kiến thức đã học khi hiệu suất tốt.
- **Thực hiện**: Thông qua `_create_adaptive_exploration_callback()` trong quá trình huấn luyện.

#### Phát hiện và thoát khỏi vòng lặp (Loop Detection)
- **Mô tả**: Tự động phát hiện khi agent bị mắc kẹt trong cùng một vùng.
- **Lợi ích**: Khuyến khích khám phá các vùng mới của không gian trạng thái.
- **Thực hiện**: Cải tiến hệ thống hình phạt trong môi trường RL.

### 3. Cải tiến hàm phần thưởng (Reward Function)

#### Shaped Rewards
- **Mô tả**: Thiết kế lại hàm phần thưởng để cung cấp tín hiệu học tập có ý nghĩa hơn.
- **Lợi ích**: Giúp agent học nhanh hơn trong không gian trạng thái lớn.
- **Thực hiện**: Bổ sung phần thưởng tiềm năng dựa trên khoảng cách tới đích.

#### Hiệu quả đường đi
- **Mô tả**: Thưởng dựa trên hiệu quả đường đi (so với độ dài tối ưu ước tính).
- **Lợi ích**: Khuyến khích agent tìm đường ngắn hơn, không chỉ đơn giản là đến đích.
- **Thực hiện**: Bổ sung phần thưởng dựa trên tỷ lệ giữa độ dài đường đi ước lượng và thực tế.

### 4. Cải tiến biểu diễn trạng thái (State Representation)

#### Thông tin bổ sung
- **Mô tả**: Bổ sung thông tin vào không gian trạng thái như khoảng cách tới đích, số bước đã đi, và thông tin ô đã thăm.
- **Lợi ích**: Cung cấp thông tin hữu ích hơn để agent đưa ra quyết định.
- **Thực hiện**: Mở rộng không gian observation trong môi trường.

### 5. Cải tiến hiệu suất huấn luyện

#### Early Stopping
- **Mô tả**: Dừng huấn luyện sớm khi không có cải thiện đáng kể.
- **Lợi ích**: Tiết kiệm thời gian huấn luyện và tránh overfitting.
- **Thực hiện**: Thông qua `_create_early_stopping_callback()`.

#### Lưu model tốt nhất
- **Mô tả**: Tự động lưu model có hiệu suất tốt nhất trong quá trình huấn luyện.
- **Lợi ích**: Đảm bảo sử dụng model có hiệu suất tốt nhất, không phải model cuối cùng.
- **Thực hiện**: Trong early stopping callback.

## Hướng dẫn sử dụng

### Chạy huấn luyện nâng cao:

1. Khởi tạo môi trường và bản đồ
2. Nhấn nút "Huấn luyện nâng cao" trong tab "Manual Training"
3. Theo dõi tiến độ và kết quả huấn luyện

### Đánh giá và so sánh:

Để so sánh hiệu suất của các biến thể DQN:

```bash
python Train/test_enhanced_dqn.py --map_sizes 8 10 12 --episodes 30 --timesteps 50000
```

Kết quả đánh giá và biểu đồ so sánh sẽ được lưu trong thư mục `Train/evaluation_results/`.

## Lưu ý

- Double DQN và Early Stopping được bật mặc định để cải thiện hiệu suất
- Dueling Network và Prioritized Experience Replay có thể cần cài đặt thêm thư viện sb3-contrib
- Hãy điều chỉnh các tham số như kích thước mạng và batch size dựa trên phức tạp của bản đồ

## Tài liệu tham khảo

1. [Deep Reinforcement Learning: Double Q-Learning](https://arxiv.org/abs/1509.06461)
2. [Dueling Network Architectures for Deep Reinforcement Learning](https://arxiv.org/abs/1511.06581)
3. [Prioritized Experience Replay](https://arxiv.org/abs/1511.05952)
4. [Reward Shaping in Reinforcement Learning](https://www.jair.org/index.php/jair/article/view/10283)
5. [Stable Baselines3 Documentation](https://stable-baselines3.readthedocs.io/) 