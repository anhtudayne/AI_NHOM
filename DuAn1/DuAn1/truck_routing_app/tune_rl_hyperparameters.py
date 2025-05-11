#!/usr/bin/env python
"""
Script để chạy tinh chỉnh siêu tham số của model RL bằng Optuna.
"""

import os
import sys
import json
import argparse
from datetime import datetime

# Thêm thư mục cha vào sys.path để có thể import từ các module khác
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

try:
    from core.map import Map
    from core.algorithms.hyperparameter_tuning import optimize_hyperparameters, train_agent_with_best_params
    print("Import các module thành công!")
except ImportError as e:
    print(f"Không thể import module: {e}")
    print("Đảm bảo bạn đã cài đặt các gói phụ thuộc cần thiết:")
    print("pip install optuna stable-baselines3 torch gymnasium shimmy")
    sys.exit(1)

def parse_arguments():
    """Phân tích đối số dòng lệnh."""
    parser = argparse.ArgumentParser(description="Tinh chỉnh siêu tham số cho model RL bằng Optuna")
    
    parser.add_argument("--train-dir", type=str, default="./maps/train",
                        help="Thư mục chứa bản đồ huấn luyện")
    parser.add_argument("--eval-dir", type=str, default="./maps/eval",
                        help="Thư mục chứa bản đồ đánh giá")
    parser.add_argument("--results-dir", type=str, default="./hyperparameter_tuning_results",
                        help="Thư mục lưu kết quả tinh chỉnh")
    parser.add_argument("--n-trials", type=int, default=50,
                        help="Số lần thử nghiệm")
    parser.add_argument("--n-timesteps", type=int, default=25000,
                        help="Số bước huấn luyện mỗi lần thử")
    parser.add_argument("--n-eval-episodes", type=int, default=5,
                        help="Số episodes đánh giá mỗi lần thử")
    parser.add_argument("--n-jobs", type=int, default=1,
                        help="Số công việc song song")
    parser.add_argument("--train-best", action="store_true",
                        help="Huấn luyện model với tham số tốt nhất sau khi tinh chỉnh")
    parser.add_argument("--final-timesteps", type=int, default=100000,
                        help="Số bước huấn luyện cho model cuối cùng")
    
    return parser.parse_args()

def main():
    """Hàm chính để chạy tinh chỉnh siêu tham số."""
    args = parse_arguments()
    
    # Tạo thư mục kết quả
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = os.path.join(args.results_dir, timestamp)
    os.makedirs(results_dir, exist_ok=True)
    
    # Kiểm tra thư mục bản đồ
    if not os.path.exists(args.train_dir) or len(os.listdir(args.train_dir)) == 0:
        print(f"Lỗi: Thư mục bản đồ huấn luyện '{args.train_dir}' không tồn tại hoặc trống!")
        return
    
    if not os.path.exists(args.eval_dir) or len(os.listdir(args.eval_dir)) == 0:
        print(f"Lỗi: Thư mục bản đồ đánh giá '{args.eval_dir}' không tồn tại hoặc trống!")
        return
    
    # Lưu cấu hình
    config = {
        "train_dir": args.train_dir,
        "eval_dir": args.eval_dir,
        "n_trials": args.n_trials,
        "n_timesteps": args.n_timesteps,
        "n_eval_episodes": args.n_eval_episodes,
        "n_jobs": args.n_jobs,
        "train_best": args.train_best,
        "final_timesteps": args.final_timesteps,
        "timestamp": timestamp
    }
    
    with open(os.path.join(results_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=4)
    
    # Hiển thị thông tin
    print("=" * 50)
    print("TINH CHỈNH SIÊU THAM SỐ CHO MODEL RL")
    print("=" * 50)
    print(f"Thư mục bản đồ huấn luyện: {args.train_dir}")
    print(f"Thư mục bản đồ đánh giá: {args.eval_dir}")
    print(f"Số lần thử nghiệm: {args.n_trials}")
    print(f"Số bước huấn luyện mỗi lần thử: {args.n_timesteps}")
    print(f"Số episodes đánh giá mỗi lần thử: {args.n_eval_episodes}")
    print(f"Số công việc song song: {args.n_jobs}")
    print(f"Thư mục kết quả: {results_dir}")
    print("=" * 50)
    
    try:
        # Bắt đầu tinh chỉnh
        print("Bắt đầu tinh chỉnh siêu tham số...")
        best_params = optimize_hyperparameters(
            train_maps_dir=args.train_dir,
            eval_maps_dir=args.eval_dir,
            n_trials=args.n_trials,
            n_timesteps=args.n_timesteps,
            n_eval_episodes=args.n_eval_episodes,
            n_jobs=args.n_jobs,
            study_name=f"dqn_optimization_{timestamp}"
        )
        
        print("Tinh chỉnh siêu tham số hoàn tất!")
        
        # Huấn luyện model với tham số tốt nhất nếu cần
        if args.train_best:
            print(f"\nBắt đầu huấn luyện model với tham số tốt nhất ({args.final_timesteps} bước)...")
            
            # Tải một bản đồ để huấn luyện
            eval_maps = os.listdir(args.eval_dir)
            map_path = os.path.join(args.eval_dir, eval_maps[0])
            train_map = Map.load(map_path)
            
            # Thư mục lưu model
            save_path = os.path.join("./saved_models", f"best_dqn_agent_{timestamp}")
            
            # Huấn luyện model
            train_agent_with_best_params(
                best_params=best_params,
                train_map=train_map,
                n_timesteps=args.final_timesteps,
                save_path=save_path
            )
            
            print(f"Huấn luyện hoàn tất! Model đã được lưu tại {save_path}.zip")
    
    except Exception as e:
        print(f"Lỗi khi tinh chỉnh siêu tham số: {e}")
        import traceback
        print(traceback.format_exc())

if __name__ == "__main__":
    main() 