#!/usr/bin/env python
"""
Script để thiết lập các thư mục cần thiết cho việc huấn luyện, đánh giá
và so sánh các thuật toán RL trong ứng dụng định tuyến xe tải.
"""

import os
import sys
import json
from datetime import datetime

# Thêm thư mục cha vào sys.path để có thể import từ các module khác
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

try:
    from core.map import Map
    print("Import module map thành công!")
except ImportError as e:
    print(f"Không thể import module map: {e}")
    sys.exit(1)

def create_directories():
    """Tạo các thư mục cần thiết cho ứng dụng."""
    directories = [
        "./maps",
        "./maps/train",
        "./maps/eval",
        "./maps/test",
        "./saved_models",
        "./rl_models_logs",
        "./hyperparameter_tuning_results",
        "./evaluation_results"
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"Đã tạo thư mục: {directory}")

def create_sample_maps():
    """Tạo các bản đồ mẫu cho huấn luyện, đánh giá và kiểm thử."""
    # Bản đồ huấn luyện (kích thước nhỏ hơn)
    train_dir = "./maps/train"
    for i in range(5):
        map_obj = Map.generate_random(size=10, toll_ratio=0.05, gas_ratio=0.05, brick_ratio=0.2)
        map_obj.save(os.path.join(train_dir, f"train_map_{i}.json"))
    print(f"Đã tạo 5 bản đồ huấn luyện tại {train_dir}")
    
    # Bản đồ đánh giá (kích thước trung bình)
    eval_dir = "./maps/eval"
    for i in range(3):
        map_obj = Map.generate_random(size=10, toll_ratio=0.05, gas_ratio=0.05, brick_ratio=0.2)
        map_obj.save(os.path.join(eval_dir, f"eval_map_{i}.json"))
    print(f"Đã tạo 3 bản đồ đánh giá tại {eval_dir}")
    
    # Bản đồ kiểm thử (đa dạng hơn)
    test_dir = "./maps/test"
    # Map cơ bản
    map_obj = Map.generate_random(size=12, toll_ratio=0.05, gas_ratio=0.05, brick_ratio=0.2)
    map_obj.save(os.path.join(test_dir, "test_map_basic.json"))
    
    # Map với nhiều vật cản
    map_obj = Map.generate_random(size=12, toll_ratio=0.05, gas_ratio=0.05, brick_ratio=0.3)
    map_obj.save(os.path.join(test_dir, "test_map_obstacles.json"))
    
    # Map với nhiều trạm xăng
    map_obj = Map.generate_random(size=12, toll_ratio=0.05, gas_ratio=0.1, brick_ratio=0.2)
    map_obj.save(os.path.join(test_dir, "test_map_gas.json"))
    
    # Map với nhiều trạm thu phí
    map_obj = Map.generate_random(size=12, toll_ratio=0.1, gas_ratio=0.05, brick_ratio=0.2)
    map_obj.save(os.path.join(test_dir, "test_map_tolls.json"))
    
    # Map lớn
    map_obj = Map.generate_random(size=15, toll_ratio=0.05, gas_ratio=0.05, brick_ratio=0.2)
    map_obj.save(os.path.join(test_dir, "test_map_large.json"))
    
    print(f"Đã tạo 5 bản đồ kiểm thử đa dạng tại {test_dir}")

def create_config_file():
    """Tạo file cấu hình với các tham số mặc định."""
    config = {
        "training": {
            "initial_fuel": 5.0,
            "initial_money": 1000.0,
            "fuel_per_move": 0.3,
            "n_timesteps": 50000,
            "default_hyperparams": {
                "learning_rate": 1e-4,
                "buffer_size": 50000,
                "batch_size": 64,
                "gamma": 0.99,
                "exploration_fraction": 0.1,
                "exploration_final_eps": 0.05
            }
        },
        "evaluation": {
            "n_episodes": 10,
            "n_runs": 5
        },
        "directories": {
            "maps_train": "./maps/train",
            "maps_eval": "./maps/eval",
            "maps_test": "./maps/test",
            "saved_models": "./saved_models",
            "logs": "./rl_models_logs",
            "tuning_results": "./hyperparameter_tuning_results",
            "evaluation_results": "./evaluation_results"
        }
    }
    
    config_path = "./rl_config.json"
    with open(config_path, "w") as f:
        json.dump(config, f, indent=4)
    
    print(f"Đã tạo file cấu hình tại {config_path}")

def main():
    """Hàm chính để thiết lập ứng dụng."""
    print("Bắt đầu thiết lập thư mục cho ứng dụng định tuyến xe tải RL...")
    
    # Tạo các thư mục
    create_directories()
    
    # Tạo bản đồ mẫu
    create_sample_maps()
    
    # Tạo file cấu hình
    create_config_file()
    
    print("\nThiết lập hoàn tất! Bây giờ bạn có thể chạy ứng dụng rl_test.py.")

if __name__ == "__main__":
    main() 