"""
Script to install dependencies required for enhanced RL implementation
"""

import subprocess
import sys
import os
import platform

def check_pip():
    """Kiểm tra và cài đặt pip nếu cần thiết."""
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "--version"])
        return True
    except subprocess.CalledProcessError:
        print("pip is not installed. Installing pip...")
        try:
            subprocess.check_call([sys.executable, "-m", "ensurepip", "--default-pip"])
            return True
        except subprocess.CalledProcessError:
            print("Failed to install pip. Please install pip manually.")
            return False

def install_dependencies():
    """Cài đặt các dependencies cần thiết."""
    if not check_pip():
        return False
    
    print("Installing dependencies for enhanced RL implementation...")
    
    # Danh sách các dependencies cơ bản
    dependencies = [
        "numpy>=1.20.0",
        "matplotlib>=3.5.0",
        "gymnasium>=0.26.0",
        "stable-baselines3>=1.7.0",
        "sb3-contrib>=1.7.0",  # Cho Prioritized Experience Replay
        "torch>=1.13.1",  # PyTorch cho các cải tiến neural network
        "optuna>=3.0.0",  # Cho việc tối ưu hóa hyperparameter
    ]
    
    # Cài đặt từng dependency
    for dep in dependencies:
        print(f"Installing {dep}...")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", dep])
        except subprocess.CalledProcessError:
            print(f"Failed to install {dep}. Please install it manually.")
            
    # Kiểm tra cài đặt PyTorch
    try:
        import torch
        print(f"PyTorch version: {torch.__version__}")
        print(f"CUDA available: {torch.cuda.is_available()}")
    except ImportError:
        print("PyTorch not installed successfully. Please install it manually.")
    
    # Kiểm tra cài đặt Stable Baselines3
    try:
        import stable_baselines3
        print(f"Stable Baselines3 version: {stable_baselines3.__version__}")
    except ImportError:
        print("Stable Baselines3 not installed successfully. Please install it manually.")
    
    # Kiểm tra cài đặt SB3-Contrib
    try:
        import sb3_contrib
        print(f"SB3-Contrib version: {sb3_contrib.__version__}")
    except ImportError:
        print("SB3-Contrib not installed successfully. Please install it manually.")
    
    # Kiểm tra cài đặt Gymnasium
    try:
        import gymnasium
        print(f"Gymnasium version: {gymnasium.__version__}")
    except ImportError:
        print("Gymnasium not installed successfully. Please install it manually.")
        
    print("\nDependencies installation completed.")
    return True

def setup_directories():
    """Thiết lập các thư mục cần thiết."""
    print("Setting up directories...")
    
    directories = [
        "saved_models",
        "training_logs",
        "evaluation_results"
    ]
    
    # Tạo các thư mục nếu chưa tồn tại
    for directory in directories:
        dir_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), directory)
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
            print(f"Created directory: {dir_path}")
        else:
            print(f"Directory already exists: {dir_path}")
    
    print("Directory setup completed.")

def main():
    """Hàm chính để cài đặt dependencies và thiết lập môi trường."""
    print("=== Enhanced RL Implementation Setup ===")
    print(f"Python version: {platform.python_version()}")
    print(f"Operating system: {platform.system()} {platform.version()}")
    
    # Cài đặt dependencies
    print("\n=== Installing Dependencies ===")
    install_dependencies()
    
    # Thiết lập thư mục
    print("\n=== Setting Up Directories ===")
    setup_directories()
    
    print("\n=== Setup Completed ===")
    print("You can now run the enhanced RL implementation.")

if __name__ == "__main__":
    main() 