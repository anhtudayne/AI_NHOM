"""
Script đơn giản để chạy trực tiếp ứng dụng kiểm tra môi trường RL.
"""

import tkinter as tk
from rl_test import RLTestApp

def main():
    root = tk.Tk()
    app = RLTestApp(root)
    root.mainloop()

if __name__ == "__main__":
    main() 