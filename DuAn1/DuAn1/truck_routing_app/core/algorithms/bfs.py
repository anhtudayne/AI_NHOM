"""
Breadth-First Search algorithm implementation.
"""

from typing import List, Tuple, Dict
import numpy as np
from collections import deque
from .base_search import BaseSearch, SearchState

class BFS(BaseSearch):
    """Breadth-First Search algorithm implementation."""
    
    def __init__(self, grid: np.ndarray):
        """Initialize BFS with a grid."""
        super().__init__(grid)
        self.queue = deque()
        self.parent = {}  # Dictionary để truy vết đường đi
        self.start = None
        self.goal = None
    
    def search(self, start: Tuple[int, int], goal: Tuple[int, int]) -> List[Tuple[int, int]]:
        """Thực hiện tìm kiếm BFS đơn giản từ start đến goal.
        Chỉ tìm đường đi hình học, không quan tâm đến ràng buộc nhiên liệu và chi phí."""
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
        self.current_fuel = self.MAX_FUEL
        self.current_total_cost = 0
        self.current_fuel_cost = 0
        self.current_toll_cost = 0
        
        # Thêm vị trí bắt đầu vào hàng đợi
        self.queue.append(start)
        self.add_visited(start)
        self.current_position = start
        self.parent[start] = None
        
        # Thực hiện BFS
        while self.queue:
            self.steps += 1
            current_pos = self.queue.popleft()
            self.current_position = current_pos
            
            # Nếu đến đích, truy vết đường đi và trả về
            if current_pos == goal:
                path = self.reconstruct_path(start, goal)
                
                # Đánh giá đường đi để tính toán nhiên liệu, chi phí và kiểm tra tính khả thi
                evaluation_result = self.evaluate_path(path)
                
                if evaluation_result["is_feasible"]:
                    self.current_path = path
                    self.path_length = len(path) - 1
                    self.cost = evaluation_result["total_cost"]
                    self.current_fuel = evaluation_result["fuel_remaining"]
                    self.current_total_cost = evaluation_result["total_cost"]
                    self.current_fuel_cost = evaluation_result["fuel_cost"]
                    self.current_toll_cost = evaluation_result["toll_cost"]
                    return path
                else:
                    # Đường đi không khả thi (hết xăng hoặc chi phí quá cao)
                    # Tìm điểm dừng cuối cùng trước khi hết xăng
                    last_feasible_index = 0
                    current_fuel = self.MAX_FUEL
                    for i in range(len(path) - 1):
                        current_fuel -= self.FUEL_PER_MOVE
                        if current_fuel < 0:
                            break
                        last_feasible_index = i + 1
                    
                    # Lấy đường đi đến điểm cuối cùng có thể đến được
                    partial_path = path[:last_feasible_index + 1]
                    self.current_path = partial_path
                    self.path_length = len(partial_path) - 1
                    self.current_fuel = 0
                    self.current_total_cost = evaluation_result["total_cost"]
                    self.current_fuel_cost = evaluation_result["fuel_cost"]
                    self.current_toll_cost = evaluation_result["toll_cost"]
                    return partial_path  # Trả về đường đi một phần
            
            # Xử lý các ô lân cận
            for next_pos in self.get_neighbors(current_pos):
                # Kiểm tra vật cản (loại 3)
                if self.grid[next_pos[1], next_pos[0]] == 3:
                    continue
                    
                # Kiểm tra xem vị trí đã được thăm chưa
                if next_pos not in self.visited_positions:
                    self.queue.append(next_pos)
                    self.add_visited(next_pos)
                    self.parent[next_pos] = current_pos
        
        return []  # No path found
    
    def reconstruct_path(self, start: Tuple[int, int], goal: Tuple[int, int]) -> List[Tuple[int, int]]:
        """Truy vết đường đi từ đích về điểm bắt đầu dựa trên parent."""
        path = []
        current = goal
        
        while current is not None:
            path.append(current)
            current = self.parent.get(current)
        
        return list(reversed(path))  # Đảo ngược để có đường đi từ start đến goal
    
    def evaluate_path(self, path: List[Tuple[int, int]]) -> Dict:
        """Đánh giá đường đi để tính toán nhiên liệu, chi phí và kiểm tra tính khả thi."""
        # Khởi tạo các giá trị
        current_fuel = self.MAX_FUEL
        total_cost = 0.0
        fuel_cost = 0.0
        toll_cost = 0.0
        toll_stations_visited = set()
        is_feasible = True
        reason = ""
        
        # Lặp qua từng cặp vị trí liên tiếp trên đường đi
        for i in range(len(path) - 1):
            # Trừ nhiên liệu cho bước di chuyển
            current_fuel -= self.FUEL_PER_MOVE
            
            # Kiểm tra hết xăng
            if current_fuel < 0:
                is_feasible = False
                reason = f"Hết nhiên liệu tại bước thứ {i+1}"
                current_fuel = 0
                break
            
            # Xử lý vị trí hiện tại
            current_pos = path[i + 1]
            cell_type = self.grid[current_pos[1], current_pos[0]]
            
            # Xử lý trạm xăng
            if cell_type == 2:  # Trạm xăng
                if current_fuel < self.MAX_FUEL:  # Chỉ đổ xăng nếu bình chưa đầy
                    fuel_cost += self.GAS_STATION_COST
                    current_fuel = self.MAX_FUEL
            
            # Xử lý trạm thu phí
            elif cell_type == 1:  # Trạm thu phí
                if current_pos not in toll_stations_visited:
                    # Giảm phạt dựa trên số trạm đã đi qua (tối đa giảm 50%)
                    visited_discount = min(0.5, len(toll_stations_visited) * 0.1)
                    toll_cost += self.TOLL_BASE_COST + (self.TOLL_PENALTY * (1.0 - visited_discount))
                    toll_stations_visited.add(current_pos)
        
        # Tính tổng chi phí
        total_cost = fuel_cost + toll_cost
        
        # Kiểm tra chi phí tối đa
        if total_cost > self.MAX_TOTAL_COST:
            is_feasible = False
            reason = "Tổng chi phí vượt quá giới hạn cho phép"
        
        return {
            "is_feasible": is_feasible,
            "reason": reason,
            "fuel_remaining": current_fuel,
            "total_cost": total_cost,
            "fuel_cost": fuel_cost,
            "toll_cost": toll_cost
        }
    
    def step(self) -> bool:
        """Execute one step of BFS."""
        if not self.queue:
            self.current_position = None
            return True
        
        self.steps += 1
        current_pos = self.queue.popleft()
        self.current_position = current_pos
        
        if current_pos == self.goal:
            path = self.reconstruct_path(self.start, self.goal)
            
            # Đánh giá đường đi
            evaluation_result = self.evaluate_path(path)
            
            self.current_path = path
            self.path_length = len(path) - 1
            self.current_fuel = evaluation_result["fuel_remaining"]
            self.current_total_cost = evaluation_result["total_cost"]
            self.current_fuel_cost = evaluation_result["fuel_cost"]
            self.current_toll_cost = evaluation_result["toll_cost"]
            
            return True
        
        for next_pos in self.get_neighbors(current_pos):
            # Kiểm tra vật cản
            if self.grid[next_pos[1], next_pos[0]] == 3:
                continue
                
            # Kiểm tra xem vị trí đã được thăm chưa
            if next_pos not in self.visited_positions:
                self.queue.append(next_pos)
                self.add_visited(next_pos)
                self.parent[next_pos] = current_pos
        
        return False 