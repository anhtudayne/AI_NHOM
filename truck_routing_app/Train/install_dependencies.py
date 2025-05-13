#!/usr/bin/env python
"""
Script để cài đặt các gói phụ thuộc cần thiết cho ứng dụng định tuyến xe tải RL.
"""

import subprocess
import sys
import os

def check_pip():
    """Kiểm tra xem pip đã được cài đặt chưa và phiên bản."""
    try:
        import pip
        print(f"Pip đã được cài đặt. Phiên bản: {pip.__version__}")
        return True
    except ImportError:
        print("Pip chưa được cài đặt. Vui lòng cài đặt pip trước.")
        return False

def install_package(package_name):
    """
    Cài đặt một gói Python.
    
    Args:
        package_name: Tên gói cần cài đặt
    
    Returns:
        bool: True nếu cài đặt thành công, False nếu không
    """
    print(f"Đang cài đặt {package_name}...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package_name])
        return True
    except subprocess.CalledProcessError as e:
        print(f"Lỗi khi cài đặt {package_name}: {e}")
        return False

def main():
    """Hàm chính để cài đặt các gói phụ thuộc."""
    print("Bắt đầu cài đặt các gói phụ thuộc...")
    
    if not check_pip():
        print("Không thể tiếp tục vì pip chưa được cài đặt.")
        sys.exit(1)
    
    # Danh sách các gói cần cài đặt
    packages = [
        "numpy",
        "gymnasium==0.28.1",  # Thay thế cho gym
        "stable-baselines3==2.1.0",
        "torch==2.0.1",
        "matplotlib",
        "pandas",
        "tensorboard",
        "shimmy>=2.0"  # Cầu nối giữa OpenAI Gym và Gymnasium cho Stable-Baselines3
    ]
    
    # Cài đặt từng gói
    failed_packages = []
    for package in packages:
        if not install_package(package):
            failed_packages.append(package)
    
    # Báo cáo kết quả
    if not failed_packages:
        print("\nTất cả các gói đã được cài đặt thành công!")
        print("\nBạn có thể chạy ứng dụng bằng lệnh: python rl_test.py")
    else:
        print("\nCác gói sau không thể cài đặt:")
        for package in failed_packages:
            print(f"  - {package}")
        print("\nVui lòng cài đặt thủ công các gói này và thử lại.")
    
    input("\nNhấn Enter để thoát...")

if __name__ == "__main__":
    main() 