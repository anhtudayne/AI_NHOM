"""
Breadth-First Search algorithm implementation.

------------------------- ĐỊNH NGHĨA THUẬT TOÁN BFS -------------------------
BFS (Breadth-First Search - Tìm kiếm theo chiều rộng) là thuật toán tìm đường 
đi từ điểm bắt đầu đến đích bằng cách duyệt tất cả các nút ở cùng độ sâu 
(khoảng cách từ nút gốc) trước khi di chuyển đến các nút ở độ sâu tiếp theo.

---------------------- NGUYÊN LÝ CƠ BẢN CỦA BFS ---------------------------
1. Sử dụng cấu trúc dữ liệu HÀNG ĐỢI (queue) theo nguyên tắc FIFO (First In First Out)
2. Ban đầu, đặt nút bắt đầu vào hàng đợi
3. Lặp lại cho đến khi hàng đợi rỗng:
   - Lấy phần tử đầu tiên ra khỏi hàng đợi
   - Kiểm tra xem đó có phải là đích không
   - Nếu không, thêm tất cả các nút kề chưa thăm vào hàng đợi
4. Khi tìm thấy đích, dùng cơ chế truy vết để tạo đường đi

---------------- TRIỂN KHAI BFS TRONG CHƯƠNG TRÌNH NÀY --------------------
BFS trong chương trình này được MỞ RỘNG với các tính năng nâng cao:

1. KHÔNG GIAN TRẠNG THÁI ĐA CHIỀU:
   - Không chỉ xét tọa độ (x,y), mà còn quan tâm đến:
   - Nhiên liệu còn lại
   - Chi phí di chuyển
   - Số tiền còn lại

2. ĐÁNH GIÁ TÍNH KHẢ THI TRONG KHI TÌM KIẾM:
   - BFS được cập nhật để kiểm tra ngay trong quá trình tìm kiếm:
   - Kiểm tra nhiên liệu và tự động đổ xăng khi đi qua trạm xăng
   - Kiểm tra tiền khi đi qua trạm thu phí
   - Trả về đường đi tới vị trí xa nhất có thể đi được nếu không thể đến đích

3. THỐNG KÊ VÀ CHI TIẾT ĐƯỜNG ĐI:
   - Tính toán chi tiết chi phí đường đi
   - Tính lượng nhiên liệu tiêu thụ và đổ thêm
   - Đếm số trạm thu phí đã đi qua
"""

from typing import List, Tuple, Dict, Set
import numpy as np
from collections import deque
from .base_search import BaseSearch, SearchState

class BFS(BaseSearch):
    """Breadth-First Search algorithm implementation."""
    
    def __init__(self, grid: np.ndarray, initial_money: float = None, 
                 max_fuel: float = None, fuel_per_move: float = None, 
                 gas_station_cost: float = None, toll_base_cost: float = None,
                 initial_fuel: float = None):
        """Initialize BFS with a grid and configuration parameters."""
        super().__init__(grid, initial_money, max_fuel, fuel_per_move, 
                         gas_station_cost, toll_base_cost, initial_fuel)
        self.queue = deque()
        self.parent = {}  # Dictionary để truy vết đường đi
        self.start = None
        self.goal = None
        self.farthest_reachable = None  # Vị trí xa nhất có thể đi được
        self.node_fuel = {}  # Dictionary lưu lượng nhiên liệu tại mỗi nút
        self.node_money = {}  # Dictionary lưu số tiền tại mỗi nút
        self.visited_toll_stations = set()  # Tập hợp các trạm thu phí đã đi qua
        self.path_failure_reason = ""  # Lý do không thể đến đích
    
    def search(self, start: Tuple[int, int], goal: Tuple[int, int]) -> List[Tuple[int, int]]:
        """Thực hiện tìm kiếm BFS với kiểm tra tính khả thi trong quá trình tìm kiếm."""
        self.start = start
        self.goal = goal
        self.queue.clear()
        self.visited_positions.clear()
        self.parent.clear()
        self.visited = []
        self.current_path.clear()
        self.steps = 0
        self.path_length = 0
        self.cost = 0
        self.farthest_reachable = None
        self.node_fuel = {}
        self.node_money = {}
        self.visited_toll_stations = set()
        self.path_failure_reason = ""  # Đặt lại lý do thất bại
        
        # Khởi tạo nhiên liệu và tiền ban đầu
        self.current_fuel = self.initial_fuel  # Sử dụng initial_fuel từ lớp cha
        self.current_money = self.MAX_MONEY
        self.current_total_cost = 0
        self.current_fuel_cost = 0
        self.current_toll_cost = 0
        
        print(f"BFS: Bắt đầu tìm kiếm từ {start} đến {goal} với nhiên liệu ban đầu: {self.current_fuel}L, mức tiêu thụ: {self.FUEL_PER_MOVE}L/ô")
        
        # Lưu thông tin ban đầu tại vị trí xuất phát
        self.node_fuel[start] = self.current_fuel
        self.node_money[start] = self.current_money
        
        # Thêm vị trí bắt đầu vào hàng đợi
        self.queue.append(start)
        self.add_visited(start)
        self.current_position = start
        self.parent[start] = None
        self.farthest_reachable = start  # Ban đầu, vị trí xa nhất là vị trí bắt đầu
        
        # Thực hiện BFS với kiểm tra ràng buộc
        while self.queue:
            self.steps += 1
            current_pos = self.queue.popleft()
            self.current_position = current_pos
            
            # Lấy nhiên liệu và tiền tại vị trí hiện tại
            current_fuel = self.node_fuel[current_pos]
            current_money = self.node_money[current_pos]
            
            print(f"BFS: Đang xét vị trí {current_pos}, nhiên liệu: {current_fuel:.1f}L, tiền: {current_money:.1f}đ")
            
            # Nếu đến đích, truy vết đường đi và trả về
            if current_pos == goal:
                path = self.reconstruct_path(start, goal)
                self.current_path = path
                self.path_length = len(path) - 1
                
                # Cập nhật thông tin nhiên liệu và chi phí
                self.current_fuel = current_fuel
                self.current_money = current_money
                self.calculate_path_fuel_consumption(path)
                
                print(f"BFS: Đường đi đến đích {goal} đã được tìm thấy với {len(path)-1} bước.")
                return path
            
            # Khoảng cách Manhattan từ vị trí hiện tại đến mục tiêu
            manhattan_dist = abs(current_pos[0] - goal[0]) + abs(current_pos[1] - goal[1])
            # Cập nhật vị trí xa nhất dựa trên khoảng cách Manhattan đến đích
            if self.farthest_reachable is None or manhattan_dist < abs(self.farthest_reachable[0] - goal[0]) + abs(self.farthest_reachable[1] - goal[1]):
                self.farthest_reachable = current_pos
            
            # Xử lý các ô lân cận
            for next_pos in self.get_neighbors(current_pos):
                # Kiểm tra xem vị trí đã được thăm chưa
                if next_pos not in self.visited_positions:
                    # Tính toán nhiên liệu tiêu thụ khi di chuyển
                    next_fuel = current_fuel - self.FUEL_PER_MOVE
                    next_money = current_money
                    
                    # Kiểm tra nếu hết nhiên liệu
                    if next_fuel < 0:
                        print(f"BFS: Không thể đi đến {next_pos} vì hết nhiên liệu (cần {self.FUEL_PER_MOVE}L, chỉ còn {current_fuel}L)")
                        self.path_failure_reason = "hết nhiên liệu"
                        continue  # Không đủ nhiên liệu để di chuyển, bỏ qua
                    
                    # Xử lý loại ô
                    cell_type = self.grid[next_pos[1], next_pos[0]]
                    
                    # Nếu là trạm xăng, đổ đầy nhiên liệu
                    if cell_type == self.GAS_STATION_CELL:
                        fuel_needed = self.MAX_FUEL - next_fuel
                        refill_cost = fuel_needed * self.GAS_STATION_COST
                        
                        # Kiểm tra xem có đủ tiền để đổ xăng không
                        if next_money >= refill_cost:
                            print(f"BFS: Đổ xăng tại {next_pos}: {fuel_needed:.1f}L với chi phí {refill_cost:.1f}đ")
                            next_money -= refill_cost
                            next_fuel = self.MAX_FUEL
                        else:
                            print(f"BFS: Không đủ tiền để đổ xăng tại {next_pos} (cần {refill_cost:.1f}đ, chỉ còn {next_money:.1f}đ)")
                    
                    # Nếu là trạm thu phí, trừ tiền phí
                    elif cell_type == self.TOLL_CELL:
                        # Tính phí dựa trên số trạm thu phí đã đi qua
                        toll_stations_visited = len(self.visited_toll_stations)
                        visited_discount = min(0.5, toll_stations_visited * 0.1)
                        toll_cost = self.TOLL_BASE_COST + (self.TOLL_PENALTY * (1.0 - visited_discount))
                        
                        # Kiểm tra xem có đủ tiền để qua trạm thu phí không
                        if next_money < toll_cost:
                            print(f"BFS: Không đủ tiền để qua trạm thu phí tại {next_pos} (cần {toll_cost:.1f}đ, chỉ còn {next_money:.1f}đ)")
                            self.path_failure_reason = "không đủ tiền qua trạm thu phí"
                            continue  # Không đủ tiền để qua trạm, bỏ qua
                        
                        print(f"BFS: Qua trạm thu phí tại {next_pos}: {toll_cost:.1f}đ")
                        next_money -= toll_cost
                        self.visited_toll_stations.add(next_pos)
                    
                    # Lưu thông tin tại vị trí tiếp theo
                    self.node_fuel[next_pos] = next_fuel
                    self.node_money[next_pos] = next_money
                    
                    # Thêm vào hàng đợi và đánh dấu đã thăm
                    self.queue.append(next_pos)
                    self.add_visited(next_pos)
                    self.parent[next_pos] = current_pos
                    print(f"BFS: Đã thêm vị trí {next_pos} vào hàng đợi, nhiên liệu: {next_fuel:.1f}L, tiền: {next_money:.1f}đ")
        
        # Nếu không tìm thấy đường đi đến đích
        if self.farthest_reachable and self.farthest_reachable != start:
            # Xác định lý do không thể đến đích
            reason = self.path_failure_reason
            if not reason:
                reason = "không tìm thấy đường đi phù hợp"
            
            print(f"BFS: Không thể đến đích {goal}, trả về đường đi đến vị trí xa nhất có thể: {self.farthest_reachable}. Lý do: {reason}")
            path = self.reconstruct_path(start, self.farthest_reachable)
            self.current_path = path
            self.path_length = len(path) - 1
            
            # Cập nhật thông tin nhiên liệu và chi phí
            self.current_fuel = self.node_fuel.get(self.farthest_reachable, 0)
            self.current_money = self.node_money.get(self.farthest_reachable, 0)
            self.calculate_path_fuel_consumption(path)
            
            return path
        
        return []  # Không tìm thấy đường đi
    
    def reconstruct_path(self, start: Tuple[int, int], goal: Tuple[int, int]) -> List[Tuple[int, int]]:
        """Truy vết đường đi từ đích về điểm bắt đầu dựa trên parent."""
        path = []
        current = goal
        
        while current is not None:
            path.append(current)
            current = self.parent.get(current)
        
        path = list(reversed(path))  # Đảo ngược để có đường đi từ start đến goal
        
        # Kiểm tra tính khả thi của đường đi dựa trên nhiên liệu
        if len(path) > 0:
            max_steps = int(self.initial_fuel / self.FUEL_PER_MOVE)
            path_steps = len(path) - 1  # Trừ 1 vì path bao gồm cả điểm bắt đầu
            
            if path_steps > max_steps:
                # Đường đi dài hơn số bước có thể đi với lượng nhiên liệu hiện có
                print(f"BFS: Đường đi tới đích dài {path_steps} bước, vượt quá khả năng nhiên liệu ({max_steps} bước với {self.initial_fuel}L)")
                
                # Cắt đường đi theo số bước tối đa có thể đi
                truncated_path = path[:max_steps + 1]
                print(f"BFS: Đường đi được cắt bớt từ {len(path)} xuống {len(truncated_path)} điểm")
                self.path_failure_reason = f"hết nhiên liệu (chỉ đi được tối đa {max_steps} ô với {self.initial_fuel}L)"
                return truncated_path
            elif path_steps == max_steps:
                # Đường đi đúng bằng số bước tối đa có thể đi với lượng nhiên liệu hiện có
                print(f"BFS: Đường đi tới đích sử dụng chính xác {path_steps} bước, bằng với khả năng nhiên liệu tối đa ({self.initial_fuel}L)")
                self.path_failure_reason = f"đường đi sử dụng toàn bộ {self.initial_fuel}L nhiên liệu ban đầu"
            else:
                # Đường đi ngắn hơn số bước tối đa, vẫn còn nhiên liệu dư
                print(f"BFS: Đường đi tới đích sử dụng {path_steps} bước, còn dư {self.initial_fuel - (path_steps * self.FUEL_PER_MOVE):.1f}L nhiên liệu")
        
        return path
    
    def step(self) -> bool:
        """Execute one step of BFS."""
        if not self.queue:
            self.current_position = None
            return True
        
        self.steps += 1
        current_pos = self.queue.popleft()
        self.current_position = current_pos
        
        # Lấy nhiên liệu và tiền tại vị trí hiện tại
        current_fuel = self.node_fuel.get(current_pos, self.current_fuel)
        current_money = self.node_money.get(current_pos, self.current_money)
        
        print(f"BFS step: đang ở vị trí {current_pos}, nhiên liệu: {current_fuel:.1f}L, tiền: {current_money:.1f}đ")
        
        if current_pos == self.goal:
            path = self.reconstruct_path(self.start, self.goal)
            self.current_path = path
            self.path_length = len(path) - 1
            
            # Cập nhật thông tin nhiên liệu và chi phí
            self.current_fuel = current_fuel
            self.current_money = current_money
            self.calculate_path_fuel_consumption(self.current_path)
            
            return True
        
        # Khoảng cách Manhattan từ vị trí hiện tại đến mục tiêu
        manhattan_dist = abs(current_pos[0] - self.goal[0]) + abs(current_pos[1] - self.goal[1])
        # Cập nhật vị trí xa nhất dựa trên khoảng cách Manhattan đến đích
        if self.farthest_reachable is None or manhattan_dist < abs(self.farthest_reachable[0] - self.goal[0]) + abs(self.farthest_reachable[1] - self.goal[1]):
            self.farthest_reachable = current_pos
        
        # Xử lý các ô lân cận
        for next_pos in self.get_neighbors(current_pos):
            # Kiểm tra xem vị trí đã được thăm chưa
            if next_pos not in self.visited_positions:
                # Tính toán nhiên liệu tiêu thụ khi di chuyển
                next_fuel = current_fuel - self.FUEL_PER_MOVE
                next_money = current_money
                
                # Kiểm tra nếu hết nhiên liệu
                if next_fuel < 0:
                    print(f"BFS: Không thể đi đến {next_pos} vì hết nhiên liệu (cần {self.FUEL_PER_MOVE}L, chỉ còn {current_fuel}L)")
                    self.path_failure_reason = "hết nhiên liệu"
                    continue  # Không đủ nhiên liệu để di chuyển, bỏ qua
                
                # Xử lý loại ô
                cell_type = self.grid[next_pos[1], next_pos[0]]
                
                # Nếu là trạm xăng, đổ đầy nhiên liệu
                if cell_type == self.GAS_STATION_CELL:
                    fuel_needed = self.MAX_FUEL - next_fuel
                    refill_cost = fuel_needed * self.GAS_STATION_COST
                    
                    # Kiểm tra xem có đủ tiền để đổ xăng không
                    if next_money >= refill_cost:
                        print(f"BFS: Đổ xăng tại {next_pos}: {fuel_needed:.1f}L với chi phí {refill_cost:.1f}đ")
                        next_money -= refill_cost
                        next_fuel = self.MAX_FUEL
                    else:
                        print(f"BFS: Không đủ tiền để đổ xăng tại {next_pos} (cần {refill_cost:.1f}đ, chỉ còn {next_money:.1f}đ)")
                
                # Nếu là trạm thu phí, trừ tiền phí
                elif cell_type == self.TOLL_CELL:
                    # Tính phí dựa trên số trạm thu phí đã đi qua
                    toll_stations_visited = len(self.visited_toll_stations)
                    visited_discount = min(0.5, toll_stations_visited * 0.1)
                    toll_cost = self.TOLL_BASE_COST + (self.TOLL_PENALTY * (1.0 - visited_discount))
                    
                    # Kiểm tra xem có đủ tiền để qua trạm thu phí không
                    if next_money < toll_cost:
                        print(f"BFS: Không đủ tiền để qua trạm thu phí tại {next_pos} (cần {toll_cost:.1f}đ, chỉ còn {next_money:.1f}đ)")
                        self.path_failure_reason = "không đủ tiền qua trạm thu phí"
                        continue  # Không đủ tiền để qua trạm, bỏ qua
                    
                    print(f"BFS: Qua trạm thu phí tại {next_pos}: {toll_cost:.1f}đ")
                    next_money -= toll_cost
                    self.visited_toll_stations.add(next_pos)
                
                # Lưu thông tin tại vị trí tiếp theo
                self.node_fuel[next_pos] = next_fuel
                self.node_money[next_pos] = next_money
                
                # Thêm vào hàng đợi và đánh dấu đã thăm
                self.queue.append(next_pos)
                self.add_visited(next_pos)
                self.parent[next_pos] = current_pos
                print(f"BFS: Đã thêm vị trí {next_pos} vào hàng đợi, nhiên liệu: {next_fuel:.1f}L, tiền: {next_money:.1f}đ")
        
        # Nếu hàng đợi rỗng và chưa đến đích
        if not self.queue and self.farthest_reachable and self.farthest_reachable != self.start:
            # Xác định lý do không thể đến đích
            reason = self.path_failure_reason
            if not reason:
                reason = "không tìm thấy đường đi phù hợp"
            
            print(f"BFS: Không thể đến đích {self.goal}, trả về đường đi đến vị trí xa nhất có thể: {self.farthest_reachable}. Lý do: {reason}")
            path = self.reconstruct_path(self.start, self.farthest_reachable)
            self.current_path = path
            self.path_length = len(path) - 1
            
            # Cập nhật thông tin nhiên liệu và chi phí
            self.current_fuel = self.node_fuel.get(self.farthest_reachable, 0)
            self.current_money = self.node_money.get(self.farthest_reachable, 0)
            self.calculate_path_fuel_consumption(path)
            
            return True
        
        return False 