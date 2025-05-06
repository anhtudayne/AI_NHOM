"""
A* search algorithm implementation with fuel and cost constraints.
"""

from typing import List, Tuple, Dict, Set, Optional
import heapq
import numpy as np
from .base_search import BaseSearch, SearchState

class AStar(BaseSearch):
    """A* search algorithm implementation with optimized heuristics."""
    
    def __init__(self, grid: np.ndarray):
        """Initialize A* with a grid."""
        super().__init__(grid)
        self.open_set = []  # Priority queue for nodes to explore
        self.closed_set = set()  # Set of explored nodes
        self.g_score = {}  # Cost from start to current node
        self.f_score = {}  # Estimated total cost (g_score + heuristic)
        self.parent = {}  # Parent pointers for path reconstruction
        self.start = None
        self.goal = None
        
        # Thresholds for decision making
        self.LOW_FUEL_THRESHOLD = 1.0  # Ngưỡng nhiên liệu thấp (1 lít)
        self.TOLL_DISTANCE_THRESHOLD = 3  # Số bước tối thiểu để cân nhắc đi qua trạm thu phí
    
    def calculate_heuristic(self, pos: Tuple[int, int], goal: Tuple[int, int], 
                          current_fuel: float, current_cost: float) -> float:
        """Tính toán heuristic cho A*.
        
        Args:
            pos: Vị trí hiện tại
            goal: Vị trí đích
            current_fuel: Lượng nhiên liệu hiện tại
            current_cost: Chi phí tích lũy hiện tại
        
        Returns:
            float: Giá trị heuristic
        """
        # Khoảng cách Manhattan đến đích
        distance_to_goal = abs(pos[0] - goal[0]) + abs(pos[1] - goal[1])
        
        # Chi phí cơ bản dựa trên khoảng cách
        base_cost = distance_to_goal * self.ROAD_WEIGHT
        
        # Ước tính số lần cần đổ xăng
        fuel_needed = distance_to_goal * self.FUEL_PER_MOVE
        estimated_refills = max(0, (fuel_needed - current_fuel) / self.MAX_FUEL)
        refill_cost = estimated_refills * self.GAS_STATION_COST
        
        # Phạt nếu nhiên liệu thấp và không gần trạm xăng
        fuel_penalty = 0
        if current_fuel < self.LOW_FUEL_THRESHOLD:
            nearest_gas = self.find_nearest_gas_station(pos)
            if nearest_gas:
                distance_to_gas = abs(pos[0] - nearest_gas[0]) + abs(pos[1] - nearest_gas[1])
                fuel_penalty = distance_to_gas * 1000  # Phạt nặng nếu xa trạm xăng khi sắp hết nhiên liệu
        
        # Tổng hợp các thành phần
        return base_cost + refill_cost + fuel_penalty
    
    def find_nearest_gas_station(self, pos: Tuple[int, int]) -> Optional[Tuple[int, int]]:
        """Tìm trạm xăng gần nhất từ vị trí hiện tại."""
        min_distance = float('inf')
        nearest_gas = None
        
        for y in range(self.grid.shape[0]):
            for x in range(self.grid.shape[1]):
                if self.grid[y, x] == 2:  # Trạm xăng
                    distance = abs(x - pos[0]) + abs(y - pos[1])
                    if distance < min_distance:
                        min_distance = distance
                        nearest_gas = (x, y)
        
        return nearest_gas
    
    def get_neighbors_with_priority(self, pos: Tuple[int, int], current_fuel: float) -> List[Tuple[int, int]]:
        """Lấy các ô lân cận với thứ tự ưu tiên dựa trên nhiên liệu và loại ô."""
        neighbors = self.get_neighbors(pos)
        prioritized = []
        
        for next_pos in neighbors:
            if self.grid[next_pos[1], next_pos[0]] == 3:  # Bỏ qua vật cản
                continue
                
            # Tính điểm ưu tiên (số càng thấp càng ưu tiên cao)
            priority = 0
            cell_type = self.grid[next_pos[1], next_pos[0]]
            
            if current_fuel < self.LOW_FUEL_THRESHOLD:
                # Ưu tiên trạm xăng khi nhiên liệu thấp
                if cell_type == 2:  # Trạm xăng
                    priority = 0
                elif cell_type == 0:  # Đường trống
                    priority = 1
                elif cell_type == 1:  # Trạm thu phí
                    priority = 2
            else:
                # Ưu tiên đường trống khi đủ nhiên liệu
                if cell_type == 0:  # Đường trống
                    priority = 0
                elif cell_type == 2:  # Trạm xăng
                    priority = 1
                elif cell_type == 1:  # Trạm thu phí
                    priority = 2
            
            prioritized.append((priority, next_pos))
        
        # Sắp xếp theo ưu tiên và trả về chỉ vị trí
        return [pos for _, pos in sorted(prioritized)]
    
    def search(self, start: Tuple[int, int], goal: Tuple[int, int]) -> List[Tuple[int, int]]:
        """Thực hiện tìm kiếm A* từ điểm start đến goal."""
        self.start = start
        self.goal = goal
        
        # Khởi tạo các tham số
        self.open_set = []
        self.closed_set = set()
        self.g_score = {}
        self.f_score = {}
        self.parent = {}
        self.visited_positions.clear()
        self.visited = []
        self.current_path.clear()
        self.steps = 0
        
        # Khởi tạo trạng thái ban đầu
        initial_state = SearchState(
            position=start,
            fuel=self.MAX_FUEL,
            total_cost=0,
            path=[start],
            visited_gas_stations=set(),
            toll_stations_visited=set()
        )
        
        # Khởi tạo node bắt đầu
        self.g_score[start] = 0
        self.f_score[start] = self.calculate_heuristic(start, goal, self.MAX_FUEL, 0)
        heapq.heappush(self.open_set, (self.f_score[start], initial_state))
        self.add_visited(start)
        
        while self.open_set:
            self.steps += 1
            _, current_state = heapq.heappop(self.open_set)
            current_pos = current_state.position
            self.current_position = current_pos
            
            if current_pos == goal:
                self.current_path = current_state.path
                self.path_length = len(self.current_path) - 1
                self.current_fuel = current_state.fuel
                self.current_total_cost = current_state.total_cost
                
                # Tính toán chi phí nhiên liệu và trạm thu phí
                toll_cost = len(current_state.toll_stations_visited) * self.TOLL_BASE_COST
                self.current_toll_cost = toll_cost
                self.current_fuel_cost = current_state.total_cost - toll_cost
                
                return self.current_path
            
            self.closed_set.add(current_state.get_state_key())
            
            # Xử lý các ô lân cận theo thứ tự ưu tiên
            for next_pos in self.get_neighbors_with_priority(current_pos, current_state.fuel):
                if next_pos in self.closed_set:
                    continue
                
                # Tính chi phí và nhiên liệu mới
                new_fuel, move_cost = self.calculate_cost(current_state, next_pos)
                
                if new_fuel < 0:
                    continue
                
                # Tạo trạng thái mới
                new_state = SearchState(
                    position=next_pos,
                    fuel=new_fuel,
                    total_cost=current_state.total_cost + move_cost,
                    path=current_state.path + [next_pos],
                    visited_gas_stations=current_state.visited_gas_stations.copy(),
                    toll_stations_visited=current_state.toll_stations_visited.copy()
                )
                
                # Cập nhật các trạm đã ghé
                cell_type = self.grid[next_pos[1], next_pos[0]]
                if cell_type == 2:  # Trạm xăng
                    new_state.visited_gas_stations.add(next_pos)
                elif cell_type == 1:  # Trạm thu phí
                    new_state.toll_stations_visited.add(next_pos)
                
                # Kiểm tra và cập nhật chi phí
                state_key = new_state.get_state_key()
                tentative_g_score = current_state.total_cost + move_cost
                
                if state_key not in self.g_score or tentative_g_score < self.g_score[state_key]:
                    self.parent[next_pos] = current_pos
                    self.g_score[state_key] = tentative_g_score
                    f_score = tentative_g_score + self.calculate_heuristic(
                        next_pos, goal, new_fuel, new_state.total_cost
                    )
                    self.f_score[state_key] = f_score
                    
                    heapq.heappush(self.open_set, (f_score, new_state))
                    self.add_visited(next_pos)
        
        return []  # Không tìm thấy đường đi
    
    def get_neighbors(self, pos: Tuple[int, int]) -> List[Tuple[int, int]]:
        """Lấy danh sách các vị trí kề có thể đi được."""
        x, y = pos
        neighbors = []
        
        # Các hướng di chuyển có thể
        for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
            new_x, new_y = x + dx, y + dy
            
            # Kiểm tra giới hạn bản đồ
            if 0 <= new_x < self.cols and 0 <= new_y < self.rows:
                # Kiểm tra không phải vật cản
                if self.grid[new_y, new_x] != 3:
                    neighbors.append((new_x, new_y))
                    
        return neighbors
    
    def step(self) -> bool:
        """Execute one step of A*."""
        if not self.open_set:  # Kiểm tra list rỗng
            self.current_position = None
            return True
        
        self.steps += 1
        _, current_state = heapq.heappop(self.open_set)
        self.current_position = current_state.position
        
        if current_state.position == self.goal:
            self.current_path = current_state.path
            self.path_length = len(self.current_path) - 1
            self.current_fuel = current_state.fuel
            self.current_total_cost = current_state.total_cost
            
            # Tính toán chi phí nhiên liệu và trạm thu phí
            toll_cost = len(current_state.toll_stations_visited) * self.TOLL_BASE_COST
            self.current_toll_cost = toll_cost
            self.current_fuel_cost = current_state.total_cost - toll_cost
            
            self.current_position = None
            return True
        
        # Thêm trạng thái hiện tại vào tập đóng
        self.closed_set.add(current_state.get_state_key())
        
        for next_pos in self.get_neighbors_with_priority(current_state.position, current_state.fuel):
            # Tính toán chi phí và nhiên liệu cho bước tiếp theo
            new_fuel, move_cost = self.calculate_cost(current_state, next_pos)
            
            if new_fuel < 0:
                continue
            
            # Tạo trạng thái mới
            new_state = SearchState(
                position=next_pos,
                fuel=new_fuel,
                total_cost=current_state.total_cost + move_cost,
                path=current_state.path + [next_pos],
                visited_gas_stations=current_state.visited_gas_stations.copy(),
                toll_stations_visited=current_state.toll_stations_visited.copy()
            )
            
            # Cập nhật các trạm đã ghé
            cell_type = self.grid[next_pos[1], next_pos[0]]
            if cell_type == 2:  # Trạm xăng
                new_state.visited_gas_stations.add(next_pos)
            elif cell_type == 1:  # Trạm thu phí
                new_state.toll_stations_visited.add(next_pos)
            
            # Kiểm tra và cập nhật chi phí
            state_key = new_state.get_state_key()
            if state_key in self.closed_set:
                continue
            
            tentative_g_score = current_state.total_cost + move_cost
            if state_key not in self.g_score or tentative_g_score < self.g_score[state_key]:
                self.g_score[state_key] = tentative_g_score
                f_score = tentative_g_score + self.calculate_heuristic(
                    next_pos, self.goal, new_fuel, new_state.total_cost
                )
                self.f_score[state_key] = f_score
                heapq.heappush(self.open_set, (f_score, new_state))
                self.add_visited(next_pos)
        
        return False 