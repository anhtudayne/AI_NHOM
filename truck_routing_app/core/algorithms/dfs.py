"""
Depth-First Search algorithm implementation.
"""

from typing import List, Tuple, Dict
import numpy as np
from .base_search import BaseSearch, SearchState

class DFS(BaseSearch):
    """Depth-First Search algorithm implementation."""
    
    # Các hằng số chi phí và ràng buộc
    TOLL_COST = 5.0         # Chi phí qua trạm thu phí (đ)
    TOLL_PENALTY = 1000.0   # Phạt cho việc đi qua trạm thu phí
    GAS_STATION_COST = 30.0 # Chi phí đổ xăng (đ)
    FUEL_PER_MOVE = 0.5     # Nhiên liệu tiêu thụ mỗi bước (L)
    MAX_TOTAL_COST = 5000.0 # Chi phí tối đa cho phép
    ROAD_WEIGHT = 1.0       # Chi phí cơ bản cho mỗi bước di chuyển
    
    def __init__(self, grid: np.ndarray, initial_money: float = None, 
                 max_fuel: float = None, fuel_per_move: float = None, 
                 gas_station_cost: float = None, toll_base_cost: float = None,
                 initial_fuel: float = None):
        """Initialize DFS with a grid and configuration parameters."""
        super().__init__(grid, initial_money, max_fuel, fuel_per_move, 
                         gas_station_cost, toll_base_cost, initial_fuel)
        self.stack = []
        self.parent = {}  # Dictionary để truy vết đường đi
        self.start = None
        self.goal = None
    
    def search(self, start: Tuple[int, int], goal: Tuple[int, int]) -> List[Tuple[int, int]]:
        """Thực hiện tìm kiếm DFS đơn giản từ start đến goal.
        Chỉ tìm đường đi hình học, không quan tâm đến ràng buộc nhiên liệu và chi phí."""
        self.start = start
        self.goal = goal
        self.stack.clear()
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
        
        # Thêm vị trí bắt đầu vào stack
        self.stack.append(start)
        self.add_visited(start)
        self.current_position = start
        self.parent[start] = None
        
        # Thực hiện DFS
        while self.stack:
            self.steps += 1
            current_pos = self.stack.pop()
            self.current_position = current_pos
            
            # Nếu đến đích, truy vết đường đi và trả về
            if current_pos == goal:
                raw_path = self.reconstruct_path(start, goal)
                
                # First, validate and clean this path
                validated_path = self.validate_path_no_obstacles(raw_path)
                
                if not validated_path or len(validated_path) < 2:
                    # Path became invalid or too short after validation, DFS should backtrack/continue
                    print(f"DFS: Path to goal {goal} became invalid after validation. Continuing search.")
                    # continue # In DFS, popping continues the loop, effectively backtracking if other branches exist
                else:
                    # Second, check overall feasibility of the validated path
                    is_still_feasible, reason = self.is_path_feasible(validated_path, self.MAX_FUEL)
                    if is_still_feasible:
                        self.current_path = validated_path
                        self.path_length = len(self.current_path) - 1
                        
                        # Recalculate all statistics on the final, validated path
                        self.calculate_path_fuel_consumption(self.current_path)
                        # self.cost, self.current_fuel, etc. are updated by the above call.
                        
                        print(f"DFS: Valid and feasible path to {goal} found.")
                        return self.current_path # Goal found and path is fully validated
                    else:
                        print(f"DFS: Path to goal {goal} not feasible after validation: {reason}. Continuing search.")
                        # Path not feasible, DFS should backtrack/continue searching
                        # continue 
            
            # Xử lý các ô lân cận
            for next_pos in reversed(self.get_neighbors(current_pos)):
                # Kiểm tra xem vị trí đã được thăm chưa
                if next_pos not in self.visited_positions:
                    self.stack.append(next_pos)
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
                    toll_cost += self.TOLL_COST + self.TOLL_PENALTY
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
        """Execute one step of DFS."""
        if not self.stack:
            self.current_position = None
            return True
        
        self.steps += 1
        current_pos = self.stack.pop()
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
            
            # Tính toán lượng nhiên liệu tiêu thụ cho đường đi
            self.calculate_path_fuel_consumption(self.current_path)
            
            return True
        
        for next_pos in reversed(self.get_neighbors(current_pos)):
            # Kiểm tra xem vị trí đã được thăm chưa
            if next_pos not in self.visited_positions:
                self.stack.append(next_pos)
                self.add_visited(next_pos)
                self.parent[next_pos] = current_pos
        
        return False 