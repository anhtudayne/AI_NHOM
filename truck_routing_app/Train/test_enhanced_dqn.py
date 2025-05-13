"""
Script để đánh giá và so sánh hiệu suất của mô hình DQN cải tiến
với DQN cơ bản và các thuật toán khác.
"""

import os
import sys
import argparse
import numpy as np
import json
import matplotlib.pyplot as plt
from datetime import datetime
from pathlib import Path

# Thêm thư mục cha vào sys.path để có thể import từ thư mục core ở cấp cao hơn
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

# Import các module cần thiết
from core.map import Map
from core.rl_environment import TruckRoutingEnv
from core.algorithms.rl_DQNAgent import DQNAgentTrainer
from core.algorithms.astar import AStar
from core.algorithms.greedy import Greedy

# Định nghĩa đường dẫn
_ROOT_DIR = Path(__file__).resolve().parent
MODELS_DIR = _ROOT_DIR / "saved_models"
EVAL_DIR = _ROOT_DIR / "evaluation_results"

# Đảm bảo thư mục tồn tại
MODELS_DIR.mkdir(parents=True, exist_ok=True)
EVAL_DIR.mkdir(parents=True, exist_ok=True)

def create_test_environments(map_sizes=[8, 10, 12, 16]):
    """
    Tạo một tập hợp các môi trường thử nghiệm với các kích thước bản đồ khác nhau.
    
    Args:
        map_sizes (list): Danh sách các kích thước bản đồ cần tạo
    
    Returns:
        dict: Dictionary chứa các môi trường, ánh xạ từ kích thước bản đồ đến môi trường
    """
    environments = {}
    
    for size in map_sizes:
        # Tạo bản đồ ngẫu nhiên
        map_obj = Map(size=size)
        map_obj.generate_random(num_obstacles=int(size*size*0.15), 
                               num_gas_stations=int(size*size*0.05),
                               num_toll_stations=int(size*size*0.05))
        
        # Lưu bản đồ để tái sử dụng
        map_path = os.path.join(EVAL_DIR, f"test_map_{size}x{size}.json")
        map_obj.save_to_file(map_path)
        
        # Tạo môi trường
        env = TruckRoutingEnv(map_obj)
        
        # Lưu môi trường
        environments[size] = env
        
    return environments

def train_and_evaluate_models(environments, n_episodes=30, timesteps=50000, render=False):
    """
    Huấn luyện và đánh giá các biến thể của DQN trên các môi trường đã tạo.
    
    Args:
        environments (dict): Dictionary chứa các môi trường
        n_episodes (int): Số lượng episode đánh giá cho mỗi mô hình
        timesteps (int): Số lượng bước huấn luyện cho mỗi mô hình
        render (bool): Nếu True, hiển thị quá trình thực thi
    
    Returns:
        dict: Kết quả đánh giá
    """
    results = {}
    
    # Các cấu hình DQN cần thử nghiệm
    dqn_configs = [
        {"name": "Basic DQN", "use_double": False, "use_dueling": False, "use_per": False},
        {"name": "Double DQN", "use_double": True, "use_dueling": False, "use_per": False},
        {"name": "Double-Dueling DQN", "use_double": True, "use_dueling": True, "use_per": False},
        {"name": "Enhanced DQN", "use_double": True, "use_dueling": True, "use_per": True}
    ]
    
    # Huấn luyện và đánh giá trên từng môi trường
    for size, env in environments.items():
        print(f"\n=== Evaluating on {size}x{size} map ===")
        
        results[size] = {}
        
        # Lưu trữ thư mục đánh giá riêng cho mỗi kích thước bản đồ
        eval_dir = os.path.join(EVAL_DIR, f"map_{size}x{size}")
        os.makedirs(eval_dir, exist_ok=True)
        
        # Huấn luyện và đánh giá từng biến thể DQN
        for config in dqn_configs:
            print(f"\nTraining {config['name']} on {size}x{size} map...")
            
            # Tạo thư mục log
            log_dir = os.path.join(eval_dir, config['name'].replace(" ", "_"))
            os.makedirs(log_dir, exist_ok=True)
            
            # Tạo trainer
            trainer = DQNAgentTrainer(env, log_dir=log_dir)
            
            # Cấu hình và tạo model
            trainer.create_model(
                learning_rate=0.0001,
                buffer_size=50000,
                learning_starts=5000,
                batch_size=64,
                gamma=0.99,
                tau=0.005,
                train_freq=4,
                gradient_steps=1,
                target_update_interval=1000,
                exploration_fraction=0.2,
                exploration_initial_eps=1.0,
                exploration_final_eps=0.05,
                max_grad_norm=10,
                policy_kwargs={"net_arch": [64, 64]},
                verbose=1,
                use_double_dqn=config["use_double"],
                use_dueling_network=config["use_dueling"],
                use_prioritized_replay=config["use_per"]
            )
            
            # Huấn luyện model
            print(f"Training {config['name']} for {timesteps} timesteps...")
            trainer.train(total_timesteps=timesteps)
            
            # Lưu model
            model_path = os.path.join(log_dir, "final_model")
            trainer.model.save(model_path)
            
            # Đánh giá model
            print(f"Evaluating {config['name']}...")
            eval_results = trainer.evaluate(n_episodes=n_episodes)
            
            # Lưu kết quả đánh giá
            with open(os.path.join(log_dir, "evaluation_results.json"), "w") as f:
                json.dump(eval_results, f, indent=4)
            
            # Lưu kết quả đánh giá vào dictionary
            results[size][config["name"]] = {
                "success_rate": eval_results["success_rate"],
                "avg_reward": eval_results["avg_reward"],
                "avg_episode_length": eval_results["avg_episode_length"],
                "avg_visited_cells": eval_results["avg_visited_cells"]
            }
            
            print(f"  Success Rate: {eval_results['success_rate']:.2f}")
            print(f"  Average Reward: {eval_results['avg_reward']:.2f}")
            print(f"  Average Episode Length: {eval_results['avg_episode_length']:.2f}")
            
        # Thêm đánh giá cho A* và Greedy để so sánh
        print("\nEvaluating classical algorithms...")
        
        # Đánh giá A*
        astar = AStar(env.map_object)
        astar_results = evaluate_classical_algorithm(env, astar, n_episodes)
        results[size]["A*"] = astar_results
        
        # Đánh giá Greedy
        greedy = Greedy(env.map_object)
        greedy_results = evaluate_classical_algorithm(env, greedy, n_episodes)
        results[size]["Greedy"] = greedy_results
    
    return results

def evaluate_classical_algorithm(env, algorithm, n_episodes):
    """
    Đánh giá thuật toán cổ điển (A*, Greedy) trên môi trường.
    
    Args:
        env (TruckRoutingEnv): Môi trường đánh giá
        algorithm: Đối tượng thuật toán (A* hoặc Greedy)
        n_episodes (int): Số lượng episode đánh giá
    
    Returns:
        dict: Kết quả đánh giá
    """
    success_count = 0
    total_reward = 0
    total_steps = 0
    total_visited = 0
    
    for i in range(n_episodes):
        # Reset môi trường với các tham số ngẫu nhiên
        obs, info = env.reset()
        
        # Lấy thông tin từ môi trường
        fuel = env.current_episode_initial_fuel
        money = env.current_episode_initial_money
        fuel_per_move = env.current_episode_fuel_per_move
        
        # Tìm đường đi
        path = algorithm.find_path(
            env.start_pos, 
            env.end_pos,
            max_fuel=fuel,
            money=money,
            fuel_per_move=fuel_per_move
        )
        
        if path is not None and len(path) > 0:
            success_count += 1
            # Tính toán số bước và ô đã thăm
            total_steps += len(path)
            total_visited += len(set(path))
            
            # Ước tính reward (không thể tính chính xác như RL)
            estimated_reward = 100.0  # Phần thưởng cơ bản cho việc tìm thấy đường đi
            total_reward += estimated_reward
    
    # Tính trung bình
    success_rate = success_count / n_episodes if n_episodes > 0 else 0
    avg_reward = total_reward / success_count if success_count > 0 else 0
    avg_steps = total_steps / success_count if success_count > 0 else 0
    avg_visited = total_visited / success_count if success_count > 0 else 0
    
    print(f"  Success Rate: {success_rate:.2f}")
    print(f"  Average Steps: {avg_steps:.2f}")
    
    return {
        "success_rate": success_rate,
        "avg_reward": avg_reward,
        "avg_episode_length": avg_steps,
        "avg_visited_cells": avg_visited
    }

def plot_comparison(results):
    """
    Vẽ đồ thị so sánh kết quả giữa các thuật toán.
    
    Args:
        results (dict): Kết quả đánh giá
    """
    map_sizes = sorted(results.keys())
    algorithms = list(results[map_sizes[0]].keys())
    
    # Vẽ biểu đồ tỷ lệ thành công
    plt.figure(figsize=(12, 8))
    
    for alg in algorithms:
        success_rates = [results[size][alg]["success_rate"] for size in map_sizes]
        plt.plot(map_sizes, success_rates, marker='o', linewidth=2, label=alg)
    
    plt.title("Success Rate by Map Size")
    plt.xlabel("Map Size")
    plt.ylabel("Success Rate")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    
    # Lưu biểu đồ
    plt.savefig(os.path.join(EVAL_DIR, "success_rate_comparison.png"))
    
    # Vẽ biểu đồ độ dài đường đi
    plt.figure(figsize=(12, 8))
    
    for alg in algorithms:
        path_lengths = [results[size][alg]["avg_episode_length"] for size in map_sizes]
        plt.plot(map_sizes, path_lengths, marker='o', linewidth=2, label=alg)
    
    plt.title("Average Path Length by Map Size")
    plt.xlabel("Map Size")
    plt.ylabel("Average Path Length")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    
    # Lưu biểu đồ
    plt.savefig(os.path.join(EVAL_DIR, "path_length_comparison.png"))
    
    # Vẽ biểu đồ số ô đã thăm
    plt.figure(figsize=(12, 8))
    
    for alg in algorithms:
        visited_cells = [results[size][alg]["avg_visited_cells"] for size in map_sizes]
        plt.plot(map_sizes, visited_cells, marker='o', linewidth=2, label=alg)
    
    plt.title("Average Visited Cells by Map Size")
    plt.xlabel("Map Size")
    plt.ylabel("Average Visited Cells")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    
    # Lưu biểu đồ
    plt.savefig(os.path.join(EVAL_DIR, "visited_cells_comparison.png"))
    
    # Hiển thị tất cả biểu đồ
    plt.show()

def save_results(results):
    """
    Lưu kết quả đánh giá vào file JSON.
    
    Args:
        results (dict): Kết quả đánh giá
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_path = os.path.join(EVAL_DIR, f"comparison_results_{timestamp}.json")
    
    with open(results_path, "w") as f:
        json.dump(results, f, indent=4)
    
    print(f"\nResults saved to {results_path}")

def main():
    """Hàm chính chạy đánh giá."""
    parser = argparse.ArgumentParser(description="Evaluate and compare DQN variants.")
    parser.add_argument("--map_sizes", type=int, nargs="+", default=[8, 10, 12],
                        help="Map sizes to evaluate on")
    parser.add_argument("--episodes", type=int, default=20,
                        help="Number of evaluation episodes per algorithm")
    parser.add_argument("--timesteps", type=int, default=50000,
                        help="Number of training timesteps per algorithm")
    parser.add_argument("--render", action="store_true",
                        help="Render the evaluation process")
    
    args = parser.parse_args()
    
    print("=== Enhanced DQN Evaluation ===")
    print(f"Map sizes: {args.map_sizes}")
    print(f"Evaluation episodes: {args.episodes}")
    print(f"Training timesteps: {args.timesteps}")
    
    # Tạo môi trường thử nghiệm
    print("\nCreating test environments...")
    environments = create_test_environments(args.map_sizes)
    
    # Huấn luyện và đánh giá
    results = train_and_evaluate_models(
        environments, 
        n_episodes=args.episodes,
        timesteps=args.timesteps,
        render=args.render
    )
    
    # Lưu kết quả
    save_results(results)
    
    # Vẽ biểu đồ so sánh
    plot_comparison(results)

if __name__ == "__main__":
    main() 