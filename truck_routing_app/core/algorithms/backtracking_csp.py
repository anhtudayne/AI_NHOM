"""
Backtracking CSP (Constraint Satisfaction Problem) algorithm implementation.

------------------------- ĐỊNH NGHĨA THUẬT TOÁN BACKTRACKING CSP -------------------------
Backtracking CSP là thuật toán tìm đường đi bằng cách thử nghiệm có hệ thống các
giá trị khác nhau cho các biến, và quay lui khi phát hiện không thể thỏa mãn ràng buộc.
Thuật toán áp dụng kỹ thuật chọn ngẫu nhiên 1 trong 4 hướng và quay lui khi gặp ràng buộc.

---------------------- MÔ HÌNH CSP TRONG BÀI TOÁN ĐỊNH TUYẾN ---------------------------
1. BIẾN (VARIABLES):
   - Trạng thái: (pos, fuel, money) - vị trí hiện tại, nhiên liệu còn lại, tiền còn lại

2. MIỀN GIÁ TRỊ (DOMAINS):
   - Vị trí: Các ô lân cận (lên, xuống, trái, phải)
   - Nhiên liệu: Giảm theo di chuyển, đổ đầy tại trạm xăng
   - Tiền: Giảm khi qua trạm thu phí hoặc đổ xăng

3. RÀNG BUỘC (CONSTRAINTS):
   - Không đi vào ô chướng ngại vật
   - Chỉ di chuyển đến các ô kề cạnh
   - Nhiên liệu luôn >= 0
   - Tiền luôn >= 0
   - Trạng thái cuối cùng phải ở vị trí đích

---------------- ĐẶC ĐIỂM THUẬT TOÁN BACKTRACKING CSP --------------------
1. CÁCH TIẾP CẬN:
   - Xây dựng lời giải từng bước, thử các giá trị cho biến hiện tại
   - Quay lui khi phát hiện xung đột (không thỏa mãn ràng buộc)
   - Sử dụng ngẫu nhiên đơn giản để chọn hướng đi tiếp theo

2. NGẪU NHIÊN:
   - Chọn ngẫu nhiên 1 trong 4 hướng để thử
   - Nếu không thỏa mãn ràng buộc, quay lui

3. ĐIỀU KIỆN DỪNG:
   - Thành công: Tìm được đường đi đến đích
   - Thất bại: Hết xăng, hết tiền, hoặc đã thử tất cả khả năng
"""

from typing import List, Tuple, Dict, Set, Optional
import numpy as np
import random
from collections import deque
from .base_search import BaseSearch
import streamlit as st

class BacktrackingCSP(BaseSearch):
    """Backtracking CSP algorithm implementation for truck routing."""
    
    def __init__(self, grid: np.ndarray, initial_money: float = None, 
                 max_fuel: float = None, fuel_per_move: float = None, 
                 gas_station_cost: float = None, toll_base_cost: float = None,
                 initial_fuel: float = None):
        """Initialize BacktrackingCSP with a grid and configuration parameters."""
        super().__init__(grid, initial_money, max_fuel, fuel_per_move, 
                         gas_station_cost, toll_base_cost, initial_fuel)
        self.path_failure_reason = ""
        self.visited = set()
        self.best_path = []
        self.debug_path = []  # Đường dẫn chi tiết cho trực quan hóa
        self.visualization_step = 0
        # visited_positions now stores tuples: (position, is_forward_move)
        # where is_forward_move is True when we're moving forward, False when backtracking
        self.visited_positions = []
        # Thêm set lưu trữ các cạnh đã thử và thất bại
        self.failed_edges = set()  # Set of (from_pos, to_pos) tuples
        
        # Sử dụng seed cố định cho random để kết quả nhất quán
        random.seed(42)
        
    def search(self, start: Tuple[int, int], goal: Tuple[int, int]) -> List[Tuple[int, int]]:
        """Run backtracking CSP search algorithm."""
        self.start = start
        self.goal = goal
        self.visited.clear()
        self.current_path = []
        self.debug_path = []
        self.best_path = []
        self.path_failure_reason = ""
        self.visualization_step = 0
        self.failed_edges.clear()  # Xóa cache cạnh thất bại mỗi lần chạy mới
        
        # Reset random seed cho mỗi lần tìm kiếm để đảm bảo kết quả ổn định
        random.seed(42)
        
        # In thông tin bắt đầu tìm kiếm
        print(f"\nBắt đầu tìm đường từ {start} đến {goal} với Backtracking CSP")
        print(f"Cấu hình: initial_fuel={self.initial_fuel}L, initial_money={self.MAX_MONEY}đ, fuel_per_move={self.FUEL_PER_MOVE}L")
        print(f"Chiến lược: Chọn ngẫu nhiên 1 trong 4 hướng, quay lui nếu không thỏa mãn ràng buộc")
        
        # Kiểm tra xem điểm bắt đầu và đích có phải là chướng ngại vật không
        if 0 <= start[0] < self.grid.shape[1] and 0 <= start[1] < self.grid.shape[0]:
            if self.grid[start[1]][start[0]] == self.OBSTACLE_CELL:
                self.path_failure_reason = f"Điểm bắt đầu {start} là ô chướng ngại vật"
                print(f"Lỗi: {self.path_failure_reason}")
                return []
        else:
            self.path_failure_reason = f"Điểm bắt đầu {start} nằm ngoài bản đồ"
            print(f"Lỗi: {self.path_failure_reason}")
            return []
            
        if 0 <= goal[0] < self.grid.shape[1] and 0 <= goal[1] < self.grid.shape[0]:
            if self.grid[goal[1]][goal[0]] == self.OBSTACLE_CELL:
                self.path_failure_reason = f"Điểm đích {goal} là ô chướng ngại vật"
                print(f"Lỗi: {self.path_failure_reason}")
                return []
        else:
            self.path_failure_reason = f"Điểm đích {goal} nằm ngoài bản đồ"
            print(f"Lỗi: {self.path_failure_reason}")
            return []
        
        # Theo dõi thông tin cho animation
        self.visited_positions = []  # Danh sách vị trí đã thăm để hiển thị
        
        # Trạng thái ban đầu
        initial_state = (start, self.initial_fuel, self.MAX_MONEY)
        
        # Thực hiện backtracking
        result = self.backtrack(initial_state)
        
        if result:
            # Thành công, cập nhật thông tin đường đi
            self.path_length = len(self.current_path) - 1
            print(f"BacktrackingCSP: Tìm thấy đường đi với {self.path_length} bước.")
            
            # Xác thực đường đi trước khi trả về
            if not self.validate_path(self.current_path):
                print("CẢNH BÁO QUAN TRỌNG: Đường đi tìm được không hợp lệ khi xác thực lại!")
                self.path_failure_reason = "Đường đi không hợp lệ khi xác thực lại"
                return []
                
            return self.current_path
        else:
            # Thất bại, báo cáo lý do
            print(f"BacktrackingCSP: Không tìm thấy đường đi. Lý do: {self.path_failure_reason}")
            return self.current_path if self.current_path else []
    
    def backtrack(self, current_state: Tuple[Tuple[int, int], float, float]) -> bool:
        """Thuật toán backtracking với lựa chọn ngẫu nhiên đơn giản."""
        pos, fuel, money = current_state
        
        # Thêm vào danh sách vị trí đã thăm (để hiển thị)
        # Sử dụng tuple (pos, True) để đánh dấu là đang thăm
        self.visited_positions.append((pos, True))
        
        # Lưu trạng thái hiện tại cho trực quan hóa chi tiết
        self.debug_path.append((pos, fuel, money))
        
        # Debug: Hiển thị thông tin về trạng thái hiện tại
        indent = "  " * len(self.debug_path)
        print(f"{indent}Thăm vị trí: {pos}, fuel={fuel:.1f}L, money={money:.1f}đ")
        
        # Điều kiện dừng: đã đến đích
        if pos == self.goal:
            self.current_path = [p for p, _, _ in self.debug_path]
            self.best_path = self.current_path.copy()
            
            # Cập nhật thông tin nhiên liệu và tiền
            self.current_fuel = fuel
            self.current_money = money
            
            print(f"{indent}ĐÃ ĐẾN ĐÍCH! Tìm thấy đường đi với {len(self.current_path)-1} bước.")
            print(f"Đường đi tìm được: {self.current_path}")
            return True
        
        # Thêm vị trí hiện tại vào tập đã thăm
        self.visited.add(pos)
        
        # Cập nhật vị trí hiện tại
        self.current_position = pos
        
        # Lấy các lân cận hợp lệ
        neighbors = self.get_neighbors(pos)
        
        # Lọc bỏ các lân cận đã thử và thất bại để không thử lại
        valid_neighbors = [n for n in neighbors if (pos, n) not in self.failed_edges]
        
        # Xáo trộn ngẫu nhiên các lân cận hợp lệ
        random.shuffle(valid_neighbors)
        print(f"{indent}Các lân cận hợp lệ sau khi xáo trộn: {valid_neighbors}")
        
        # Không có lân cận hợp lệ, quay lui
        if not valid_neighbors:
            self.visited.remove(pos)
            self.debug_path.pop()
            # Đánh dấu là vị trí đã được quay lui khỏi
            self.visited_positions.append((pos, False))
            self.path_failure_reason = f"Không có đường đi hợp lệ từ vị trí {pos}"
            print(f"{indent}Không có lân cận hợp lệ, quay lui từ {pos}")
            return False
        
        # Thử từng lân cận theo thứ tự ngẫu nhiên
        for next_pos in valid_neighbors:
            # Bỏ qua nếu đã thăm
            if next_pos in self.visited:
                print(f"{indent}+ Lân cận {next_pos} đã thăm trước đó, bỏ qua")
                continue
                
            # Tính toán nhiên liệu và tiền sau khi di chuyển
            next_fuel = fuel - self.FUEL_PER_MOVE
            next_money = money
            
            # Kiểm tra nếu hết xăng
            if next_fuel < 0:
                self.path_failure_reason = f"Hết nhiên liệu khi di chuyển đến {next_pos}. Còn {fuel:.1f}L, cần {self.FUEL_PER_MOVE}L."
                print(f"{indent}+ Lân cận {next_pos}: Không đủ nhiên liệu ({fuel:.1f}L < {self.FUEL_PER_MOVE}L), bỏ qua")
                # Thêm cạnh này vào danh sách thất bại để không thử lại
                self.failed_edges.add((pos, next_pos))
                continue  # Thử lân cận khác
            
            # Xử lý loại ô
            cell_type = self.grid[next_pos[1]][next_pos[0]]
            
            # Nếu là trạm xăng
            if cell_type == self.GAS_STATION_CELL:
                fuel_needed = self.MAX_FUEL - next_fuel
                refill_cost = fuel_needed * self.GAS_STATION_COST
                
                # Kiểm tra đủ tiền để đổ xăng
                if next_money < refill_cost:
                    self.path_failure_reason = f"Không đủ tiền để đổ xăng tại {next_pos}. Còn {next_money:.1f}đ, cần {refill_cost:.1f}đ."
                    print(f"{indent}+ Lân cận {next_pos}: Không đủ tiền để đổ xăng ({next_money:.1f}đ < {refill_cost:.1f}đ), bỏ qua")
                    # Thêm cạnh này vào danh sách thất bại để không thử lại
                    self.failed_edges.add((pos, next_pos))
                    continue  # Thử lân cận khác
                
                next_fuel = self.MAX_FUEL
                next_money -= refill_cost
                print(f"{indent}+ Lân cận {next_pos}: Đổ xăng từ {fuel - self.FUEL_PER_MOVE:.1f}L lên {next_fuel:.1f}L, chi phí {refill_cost:.1f}đ")
                
            # Nếu là trạm thu phí
            elif cell_type == self.TOLL_CELL:
                toll_cost = self.calculate_toll_cost()
                
                # Kiểm tra đủ tiền để qua trạm
                if next_money < toll_cost:
                    self.path_failure_reason = f"Không đủ tiền để qua trạm thu phí tại {next_pos}. Còn {next_money:.1f}đ, cần {toll_cost:.1f}đ."
                    print(f"{indent}+ Lân cận {next_pos}: Không đủ tiền để qua trạm thu phí ({next_money:.1f}đ < {toll_cost:.1f}đ), bỏ qua")
                    # Thêm cạnh này vào danh sách thất bại để không thử lại
                    self.failed_edges.add((pos, next_pos))
                    continue  # Thử lân cận khác
                
                next_money -= toll_cost
                print(f"{indent}+ Lân cận {next_pos}: Đi qua trạm thu phí, chi phí {toll_cost:.1f}đ")
            
            # Thông báo đang thử vị trí mới
            print(f"{indent}> THỬ vị trí mới: {next_pos}, fuel={next_fuel:.1f}L, money={next_money:.1f}đ")
            
            # Đệ quy với trạng thái mới
            next_state = (next_pos, next_fuel, next_money)
            if self.backtrack(next_state):
                return True
            
            # Đánh dấu cạnh này đã thử và thất bại
            self.failed_edges.add((pos, next_pos))
            print(f"{indent}< Quay lui từ {next_pos} đến {pos}. Đã đánh dấu cạnh ({pos}->{next_pos}) đã thất bại.")
        
        # Không tìm được đường đi, quay lui
        self.visited.remove(pos)
        self.debug_path.pop()
        # Đánh dấu là vị trí đã được quay lui khỏi
        self.visited_positions.append((pos, False))
        
        # Nếu chưa có lý do thất bại, đặt lý do mặc định
        if not self.path_failure_reason:
            self.path_failure_reason = f"Không tìm thấy đường đi từ {pos} đến {self.goal}"
        
        print(f"{indent}Đã thử tất cả lân cận từ {pos}, quay lui")
        return False
    
    def calculate_distance(self, pos1: Tuple[int, int], pos2: Tuple[int, int]) -> float:
        """Tính khoảng cách Manhattan giữa hai điểm."""
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])
    
    def calculate_toll_cost(self) -> float:
        """Tính chi phí trạm thu phí."""
        # Sử dụng cost từ cấu hình
        return self.TOLL_BASE_COST
    
    def find_nearest_gas_station(self, start_pos: Tuple[int, int]) -> Optional[Tuple[int, int]]:
        """Tìm trạm xăng gần nhất từ vị trí cho trước."""
        # Sử dụng BFS để tìm trạm xăng gần nhất
        queue = deque([start_pos])
        visited_bfs = {start_pos}
        
        while queue:
            pos = queue.popleft()
            x, y = pos
            
            # Kiểm tra nếu là trạm xăng
            if self.grid[y][x] == self.GAS_STATION_CELL:
                return pos
            
            # Thêm các lân cận
            for neighbor in self.get_neighbors(pos):
                if neighbor not in visited_bfs:
                    visited_bfs.add(neighbor)
                    queue.append(neighbor)
        
        return None  # Không tìm thấy trạm xăng
    
    def step(self) -> bool:
        """Thực hiện một bước của thuật toán cho trực quan hóa."""
        # Nếu đã hoàn thành quá trình tìm kiếm hoặc không có dữ liệu để hiển thị
        if self.visualization_step >= len(self.visited_positions) or not self.visited_positions:
            print(f"Kết thúc hiển thị: đã hiển thị {self.visualization_step}/{len(self.visited_positions)} vị trí")
            
            # Hiển thị đường đi tốt nhất đã tìm được khi hoàn thành
            if self.best_path:
                self.current_path = self.best_path
                print(f"Hiển thị đường đi tốt nhất với {len(self.best_path)-1} bước")
                print(f"Chi tiết đường đi tốt nhất: {self.best_path}")
            else:
                print("Không tìm được đường đi tốt nhất")
            
            return True
        
        # Lấy thông tin vị trí và hướng di chuyển
        pos, is_forward = self.visited_positions[self.visualization_step]
        
        # Cập nhật vị trí hiện tại cho trực quan hóa
        self.current_position = pos
        
        # Ghi log về hướng di chuyển
        if is_forward:
            print(f"Hiển thị bước {self.visualization_step+1}: đi tới vị trí {pos}")
        else:
            print(f"Hiển thị bước {self.visualization_step+1}: quay lui từ vị trí {pos}")
        
        # Tăng bước trực quan hóa
        self.visualization_step += 1
        
        # Kiểm tra xem đã hoàn thành chưa
        if self.visualization_step >= len(self.visited_positions):
            # Khôi phục đường đi tốt nhất đã tìm được
            if self.best_path:
                self.current_path = self.best_path
                print(f"Kết thúc hiển thị, hiển thị đường đi tốt nhất: {len(self.best_path)-1} bước")
            return True
        
        return False
    
    def get_neighbors(self, pos: Tuple[int, int]) -> List[Tuple[int, int]]:
        """Lấy các ô lân cận hợp lệ."""
        neighbors = []
        x, y = pos
        
        # Các hướng: lên, phải, xuống, trái
        directions = [(0, -1), (1, 0), (0, 1), (-1, 0)]
        
        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            
            # Kiểm tra có nằm trong bản đồ không
            if 0 <= nx < self.grid.shape[1] and 0 <= ny < self.grid.shape[0]:
                # Kiểm tra không phải ô chướng ngại vật
                if self.grid[ny][nx] != self.OBSTACLE_CELL:
                    neighbors.append((nx, ny))
        
        return neighbors
    
    def add_visited(self, pos):
        """Thêm vị trí vào danh sách đã thăm và cập nhật trực quan hóa."""
        super().add_visited(pos)
        # Thêm vào visited_positions với is_forward=True
        self.visited_positions.append((pos, True))
    
    def validate_path(self, path: List[Tuple[int, int]]) -> bool:
        """Kiểm tra xem đường đi có hợp lệ không.
        
        Một đường đi hợp lệ phải:
        1. Bắt đầu từ điểm xuất phát
        2. Kết thúc ở điểm đích
        3. Các điểm liên tiếp phải kề nhau (chỉ di chuyển lên, xuống, trái, phải)
        4. Không có điểm nào là chướng ngại vật
        """
        if not path:
            print("Đường đi rỗng, không hợp lệ")
            return False
            
        # Kiểm tra điểm đầu và cuối
        if path[0] != self.start:
            print(f"Điểm bắt đầu của đường đi {path[0]} khác với điểm xuất phát {self.start}")
            return False
            
        if path[-1] != self.goal:
            print(f"Điểm cuối của đường đi {path[-1]} khác với điểm đích {self.goal}")
            return False
            
        # Kiểm tra các điểm liên tiếp
        for i in range(len(path) - 1):
            curr_pos = path[i]
            next_pos = path[i + 1]
            
            # Tính khoảng cách Manhattan giữa hai điểm liên tiếp
            distance = abs(curr_pos[0] - next_pos[0]) + abs(curr_pos[1] - next_pos[1])
            
            # Nếu khoảng cách > 1, các điểm không kề nhau
            if distance > 1:
                print(f"Lỗi: Điểm {curr_pos} và {next_pos} không kề nhau (khoảng cách = {distance})")
                return False
                
            # Kiểm tra xem điểm tiếp theo có phải là chướng ngại vật không
            if self.grid[next_pos[1]][next_pos[0]] == self.OBSTACLE_CELL:
                print(f"Lỗi: Điểm {next_pos} là chướng ngại vật")
                return False
                
        # Kiểm tra tài nguyên (nhiên liệu và tiền)
        fuel = self.initial_fuel
        money = self.MAX_MONEY
        
        for i in range(len(path) - 1):
            # Tiêu thụ nhiên liệu cho mỗi bước đi
            fuel -= self.FUEL_PER_MOVE
            
            # Kiểm tra nếu hết nhiên liệu
            if fuel < 0:
                print(f"Lỗi: Hết nhiên liệu tại bước {i+1} (vị trí {path[i+1]})")
                return False
                
            # Kiểm tra loại ô
            next_pos = path[i + 1]
            cell_type = self.grid[next_pos[1]][next_pos[0]]
            
            # Xử lý trạm xăng
            if cell_type == self.GAS_STATION_CELL:
                fuel_needed = self.MAX_FUEL - fuel
                refill_cost = fuel_needed * self.GAS_STATION_COST
                
                # Kiểm tra đủ tiền để đổ xăng
                if money < refill_cost:
                    print(f"Lỗi: Không đủ tiền để đổ xăng tại vị trí {next_pos}. Còn {money:.1f}đ, cần {refill_cost:.1f}đ")
                    return False
                
                # Đổ đầy bình và trừ tiền
                fuel = self.MAX_FUEL
                money -= refill_cost
            
            # Xử lý trạm thu phí
            elif cell_type == self.TOLL_CELL:
                toll_cost = self.calculate_toll_cost()
                
                # Kiểm tra đủ tiền để qua trạm
                if money < toll_cost:
                    print(f"Lỗi: Không đủ tiền để qua trạm thu phí tại vị trí {next_pos}. Còn {money:.1f}đ, cần {toll_cost:.1f}đ")
                    return False
                
                money -= toll_cost
                
        # Nếu đã kiểm tra xong và không có lỗi
        print(f"Đường đi hợp lệ: {len(path)-1} bước, còn {fuel:.1f}L nhiên liệu và {money:.1f}đ")
        return True 