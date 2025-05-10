"""
A* search algorithm implementation with fuel and cost constraints.
"""

from typing import List, Tuple, Dict, Set, Optional
import heapq
import numpy as np
from .base_search import BaseSearch, SearchState

class AStar(BaseSearch):
    """A* search algorithm implementation with optimized heuristics."""
    
    def __init__(self, grid: np.ndarray, initial_money: float = None, 
                 max_fuel: float = None, fuel_per_move: float = None, 
                 gas_station_cost: float = None, toll_base_cost: float = None,
                 initial_fuel: float = None):
        """Initialize A* with a grid and configuration parameters."""
        super().__init__(grid, initial_money, max_fuel, fuel_per_move, 
                         gas_station_cost, toll_base_cost, initial_fuel)
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
        """Lấy các vị trí lân cận và sắp xếp theo ưu tiên."""
        neighbors = self.get_neighbors(pos)
        prioritized = []
        
        # Xác định ưu tiên cho các ô dựa vào loại và mức nhiên liệu
        for next_pos in neighbors:
            cell_type = self.grid[next_pos[1], next_pos[0]]
            
            # Ưu tiên thấp (3): Đường thông thường
            # Ưu tiên trung bình (1): Trạm xăng khi nhiên liệu thấp
            # Ưu tiên cao (0): Trạm xăng khi sắp hết xăng
            # Ưu tiên thấp (2): Trạm thu phí
            
            priority = 3  # Mặc định: đường thường
            
            if cell_type == 2:  # Trạm xăng
                if current_fuel < self.LOW_FUEL_THRESHOLD:  # Nhiên liệu rất thấp
                    priority = 0  # Ưu tiên cao nhất - cần đổ xăng gấp
                elif current_fuel < self.MAX_FUEL * 0.5:  # Nhiên liệu thấp
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
        
        # Kiểm tra tính hợp lệ của điểm đầu và cuối
        if self.grid[start[1], start[0]] == self.OBSTACLE_CELL:
            print(f"ERROR: Điểm xuất phát {start} là ô chướng ngại vật!")
            return []
        
        if self.grid[goal[1], goal[0]] == self.OBSTACLE_CELL:
            print(f"ERROR: Điểm đích {goal} là ô chướng ngại vật!")
            return []
        
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
            money=self.MAX_MONEY if self.current_money is None else self.current_money,
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
                # Original path from search state
                raw_path = current_state.path
                
                # First, validate and clean this path (removes obstacles, tries to fix continuity)
                validated_path = self.validate_path_no_obstacles(raw_path)
                
                # If validation results in an empty or unusable path, return empty
                if not validated_path or len(validated_path) < 2: # Path needs at least start and end
                    print(f"ERROR: Đường đi sau khi validate trở thành không hợp lệ hoặc quá ngắn.")
                    return []

                # Second, check overall feasibility of the validated path (obstacles, fuel, cost, continuity)
                # We use self.MAX_FUEL as initial fuel for this check, assuming a full tank at start of this specific path segment check.
                # The actual fuel tracking is handled by calculate_path_fuel_consumption.
                is_still_feasible, reason = self.is_path_feasible(validated_path, self.MAX_FUEL)
                if not is_still_feasible:
                    print(f"ERROR: Đường đi sau khi validate không khả thi: {reason}")
                    return []
                
                # If all checks pass, this is our definitive path
                self.current_path = validated_path
                self.path_length = len(self.current_path) - 1 if self.current_path else 0 # Ensure current_path is not empty
                
                # Crucially, recalculate all costs and fuel based on this *final, validated* path
                # as validate_path_no_obstacles might have altered it.
                # calculate_path_fuel_consumption updates:
                # self.fuel_consumed, self.fuel_refilled, self.current_fuel, 
                # self.current_money, self.current_fuel_cost, self.current_toll_cost,
                # self.current_total_cost, self.cost
                self.calculate_path_fuel_consumption(self.current_path) 
                
                # The original current_state.total_cost might not be accurate anymore if path changed.
                # self.cost (updated by calculate_path_fuel_consumption) is now the definitive total cost.
                                
                return self.current_path
            
            self.closed_set.add(current_state.get_state_key())
            
            # Xử lý các ô lân cận theo thứ tự ưu tiên
            for next_pos in self.get_neighbors_with_priority(current_pos, current_state.fuel):
                # Kiểm tra chắc chắn rằng vị trí tiếp theo không phải là chướng ngại vật
                cell_value = self.grid[next_pos[1], next_pos[0]]
                if cell_value == self.OBSTACLE_CELL or int(cell_value) == int(self.OBSTACLE_CELL) or cell_value < 0:
                    continue  # Bỏ qua ô chướng ngại vật
                
                # Kiểm tra xem đã thăm trạng thái này chưa
                state_key = (next_pos, round(current_state.fuel, 2))
                if state_key in self.closed_set:
                    continue
                
                # Tính chi phí và nhiên liệu mới
                new_fuel, move_cost, new_money = self.calculate_cost(current_state, next_pos)
                
                if new_fuel < 0 or new_money < 0:
                    continue
                
                # Tạo trạng thái mới
                new_path = current_state.path + [next_pos]
                
                # Kiểm tra đường đi mới có liên tục không
                prev_pos = current_state.position
                if abs(prev_pos[0] - next_pos[0]) + abs(prev_pos[1] - next_pos[1]) > 1:
                    # Đường đi không liên tục, bỏ qua
                    print(f"WARNING: Phát hiện đường đi không liên tục từ {prev_pos} đến {next_pos}. Bỏ qua.")
                    continue
                
                new_state = SearchState(
                    position=next_pos,
                    fuel=new_fuel,
                    total_cost=current_state.total_cost + move_cost,
                    money=new_money,
                    path=new_path,
                    visited_gas_stations=current_state.visited_gas_stations.copy(),
                    toll_stations_visited=current_state.toll_stations_visited.copy()
                )
                
                # Cập nhật các trạm đã ghé
                if cell_value == self.GAS_STATION_CELL:  # Trạm xăng
                    new_state.visited_gas_stations.add(next_pos)
                elif cell_value == self.TOLL_CELL:  # Trạm thu phí
                    new_state.toll_stations_visited.add(next_pos)
                
                # Kiểm tra và cập nhật chi phí
                tentative_g_score = current_state.total_cost + move_cost
                new_state_key = new_state.get_state_key()
                
                if new_state_key not in self.g_score or tentative_g_score < self.g_score[new_state_key]:
                    self.parent[next_pos] = current_pos
                    self.g_score[new_state_key] = tentative_g_score
                    f_score = tentative_g_score + self.calculate_heuristic(
                        next_pos, goal, new_fuel, new_state.total_cost
                    )
                    self.f_score[new_state_key] = f_score
                    
                    heapq.heappush(self.open_set, (f_score, new_state))
                    self.add_visited(next_pos)
        
        print("WARNING: Không tìm thấy đường đi từ", start, "đến", goal)
        return []  # Không tìm thấy đường đi
    
    def get_neighbors(self, pos: Tuple[int, int]) -> List[Tuple[int, int]]:
        """Lấy danh sách các vị trí kề có thể đi được."""
        # Sử dụng phương thức cha từ BaseSearch (đã lọc bỏ ô chướng ngại vật)
        return super().get_neighbors(pos)
    
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
            
            # Tính toán lượng nhiên liệu tiêu thụ cho đường đi
            self.calculate_path_fuel_consumption(self.current_path)
            
            # Kiểm tra tính khả thi của đường đi
            is_feasible, reason = self.is_path_feasible(self.current_path, self.MAX_FUEL)
            if not is_feasible:
                print(f"ERROR: Đường đi tìm được không khả thi: {reason}")
                self.current_path = []  # Đặt đường đi rỗng nếu không khả thi
                self.current_position = None
                return True
            
            self.current_position = None
            return True
        
        # Thêm trạng thái hiện tại vào tập đóng
        self.closed_set.add(current_state.get_state_key())
        
        for next_pos in self.get_neighbors_with_priority(current_state.position, current_state.fuel):
            # Kiểm tra chắc chắn rằng vị trí tiếp theo không phải là chướng ngại vật
            cell_value = self.grid[next_pos[1], next_pos[0]]
            if cell_value == self.OBSTACLE_CELL or int(cell_value) == int(self.OBSTACLE_CELL) or cell_value < 0:
                continue  # Bỏ qua ô chướng ngại vật
            
            # Kiểm tra xem đã thăm trạng thái này chưa
            state_key = (next_pos, round(current_state.fuel, 2))
            if state_key in self.closed_set:
                continue
            
            # Tính toán chi phí và nhiên liệu cho bước tiếp theo
            new_fuel, move_cost, new_money = self.calculate_cost(current_state, next_pos)
            
            if new_fuel < 0 or new_money < 0:
                continue
            
            # Tạo trạng thái mới
            new_path = current_state.path + [next_pos]
            
            # Kiểm tra đường đi mới có liên tục không
            prev_pos = current_state.position
            if abs(prev_pos[0] - next_pos[0]) + abs(prev_pos[1] - next_pos[1]) > 1:
                # Đường đi không liên tục, bỏ qua
                print(f"WARNING: Phát hiện đường đi không liên tục từ {prev_pos} đến {next_pos}. Bỏ qua.")
                continue
            
            new_state = SearchState(
                position=next_pos,
                fuel=new_fuel,
                total_cost=current_state.total_cost + move_cost,
                money=new_money,
                path=new_path,
                visited_gas_stations=current_state.visited_gas_stations.copy(),
                toll_stations_visited=current_state.toll_stations_visited.copy()
            )
            
            # Cập nhật các trạm đã ghé
            if cell_value == self.GAS_STATION_CELL:  # Trạm xăng
                new_state.visited_gas_stations.add(next_pos)
            elif cell_value == self.TOLL_CELL:  # Trạm thu phí
                new_state.toll_stations_visited.add(next_pos)
            
            # Kiểm tra và cập nhật chi phí
            new_state_key = new_state.get_state_key()
            if new_state_key in self.closed_set:
                continue
            
            tentative_g_score = current_state.total_cost + move_cost
            if new_state_key not in self.g_score or tentative_g_score < self.g_score[new_state_key]:
                self.g_score[new_state_key] = tentative_g_score
                f_score = tentative_g_score + self.calculate_heuristic(
                    next_pos, self.goal, new_fuel, new_state.total_cost
                )
                self.f_score[new_state_key] = f_score
                heapq.heappush(self.open_set, (f_score, new_state))
                self.add_visited(next_pos)
        
        return False 