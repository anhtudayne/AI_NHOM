"""
Greedy Search algorithm implementation.
"""

from typing import List, Tuple, Dict, Set, Optional
from collections import deque
import math
import numpy as np
from .base_search import BaseSearch, SearchState
from queue import PriorityQueue
import heapq

class GreedySearch(BaseSearch):
    """Greedy Search algorithm implementation."""
    
    def __init__(self, grid: np.ndarray, initial_money: float = None, 
                 max_fuel: float = None, fuel_per_move: float = None, 
                 gas_station_cost: float = None, toll_base_cost: float = None,
                 initial_fuel: float = None):
        """Initialize Greedy Search with a grid and optional parameters."""
        super().__init__(grid, initial_money, max_fuel, fuel_per_move, 
                        gas_station_cost, toll_base_cost, initial_fuel)
        self.initial_fuel = initial_fuel  # Store initial_fuel as instance attribute
        self.frontier = []  # Priority queue for greedy search
        self.parent = {}  # Dictionary to track path
        self.visited_states = set()  # Track visited states
        self.open_set = []  # Danh sách hàng đợi ưu tiên (sử dụng với heapq)
        self.closed_set = set()
        self.fuel = {}     # Lưu trữ nhiên liệu còn lại tại mỗi điểm
        self.total_cost = {}  # Lưu trữ tổng chi phí tại mỗi điểm
        self.fuel_cost = {}   # Lưu trữ chi phí nhiên liệu tại mỗi điểm
        self.toll_cost = {}   # Lưu trữ chi phí thu phí tại mỗi điểm
        self.visited_gas_stations = set()
        self.toll_stations_visited = set()
        self.start = None
        self.goal = None
        self.current_position = None
        self.distance_from_gas_station = {}  # Lưu khoảng cách từ mỗi ô đến trạm xăng gần nhất
    
    def heuristic(self, pos: Tuple[int, int], goal: Tuple[int, int]) -> float:
        """Calculate Manhattan distance heuristic."""
        x1, y1 = pos
        x2, y2 = goal
        return abs(x1 - x2) + abs(y1 - y2)
    
    def search(self, start: Tuple[int, int], goal: Tuple[int, int]) -> List[Tuple[int, int]]:
        """Execute Greedy Search from start to goal."""
        self.frontier = []  # Priority queue for states
        self.visited_states = set()  # Set to keep track of visited states
        
        # Create initial state
        initial_state = SearchState(
            position=start,
            fuel=self.MAX_FUEL,
            total_cost=0.0,
            money=self.MAX_MONEY,
            path=[start],
            visited_gas_stations=set(),
            toll_stations_visited=set()
        )
        
        # Add initial state to frontier with initial priority based on heuristic
        initial_priority = self.heuristic(start, goal)
        heapq.heappush(self.frontier, (initial_priority, 0, initial_state))
        
        # Reset algorithm state
        self.visited.clear()
        self.current_path.clear()
        self.steps = 0
        self.path_length = 0
        self.cost = 0
        self.current_position = start
        self.add_visited(start)
        
        # Dictionary to store best state for each position
        best_state_at_position = {}
        
        # Main loop
        while self.frontier and self.steps < 1000:  # Limit steps to prevent infinite loops
            self.steps += 1
            
            # Get state with lowest priority (best heuristic value)
            _, _, current_state = heapq.heappop(self.frontier)
            current_pos = current_state.position
            self.current_position = current_pos
            
            # Check if we reached the goal
            if current_pos == goal:
                raw_path = current_state.path
                
                # First, validate and clean this path
                validated_path = self.validate_path_no_obstacles(raw_path)
                
                if not validated_path or len(validated_path) < 2:
                    print(f"GREEDY: Đường đi sau khi validate (goal) trở thành không hợp lệ hoặc quá ngắn.")
                    return []

                # Second, check overall feasibility of the validated path
                is_still_feasible, reason = self.is_path_feasible(validated_path, self.MAX_FUEL)
                if not is_still_feasible:
                    print(f"GREEDY: Đường đi sau khi validate (goal) không khả thi: {reason}")
                    return []
                
                self.current_path = validated_path
                self.path_length = len(self.current_path) - 1
                # self.cost will be updated by calculate_path_fuel_consumption
                
                # Recalculate statistics on the final, validated path
                self.calculate_path_fuel_consumption(self.current_path)
                # self.cost is updated by the call above to reflect the true cost of the validated path
                
                return self.current_path
                
            # Skip if we've already visited this state with a lower cost
            state_key = current_state.get_state_key()
            if state_key in self.visited_states:
                continue
                
            self.visited_states.add(state_key)
            
            # Store best state for this position if it's the best we've seen
            if current_pos not in best_state_at_position or current_state.total_cost < best_state_at_position[current_pos].total_cost:
                best_state_at_position[current_pos] = current_state
            
            # Explore neighbors
            for next_pos in self.get_neighbors(current_pos):
                # Calculate cost to move to the next position
                new_fuel, move_cost, new_money = self.calculate_cost(current_state, next_pos)
                
                # Skip if not feasible (not enough fuel or money)
                if new_fuel < 0 or new_money < 0:
                    continue
                
                # Create new state
                new_state = SearchState(
                    position=next_pos,
                    fuel=new_fuel,
                    total_cost=current_state.total_cost + move_cost,
                    money=new_money,
                    path=current_state.path + [next_pos],
                    visited_gas_stations=current_state.visited_gas_stations.copy(),
                    toll_stations_visited=current_state.toll_stations_visited.copy()
                )
                
                # Update visited stations sets
                if self.grid[next_pos[1], next_pos[0]] == 2:  # Gas station
                    new_state.visited_gas_stations.add(next_pos)
                elif self.grid[next_pos[1], next_pos[0]] == 1:  # Toll station
                    new_state.toll_stations_visited.add(next_pos)
                
                # Add new state to frontier with priority based on heuristic
                # For gas stations, reduce the heuristic value when fuel is low
                priority = self.heuristic(next_pos, goal)
        
                # If fuel is low and next position is a gas station, prioritize visiting it
                if self.grid[next_pos[1], next_pos[0]] == 2 and new_state.fuel < self.MAX_FUEL * 0.3:
                    priority *= 0.5  # Reduce priority to make it more attractive
                
                heapq.heappush(self.frontier, (priority, self.steps, new_state))
                self.add_visited(next_pos)
        
        # If we didn't find a path to the goal, try to find the closest we got
        closest_state = None
        min_distance = float('inf')
        
        for pos, state in best_state_at_position.items():
            distance = abs(pos[0] - goal[0]) + abs(pos[1] - goal[1])
            if distance < min_distance:
                min_distance = distance
                closest_state = state
        
        if closest_state:
            raw_path = closest_state.path
            
            # First, validate and clean this path
            validated_path = self.validate_path_no_obstacles(raw_path)

            if not validated_path or len(validated_path) < 2:
                print(f"GREEDY: Đường đi sau khi validate (closest) trở thành không hợp lệ hoặc quá ngắn.")
                return []

            # Second, check overall feasibility of the validated path
            is_still_feasible, reason = self.is_path_feasible(validated_path, self.MAX_FUEL)
            if not is_still_feasible:
                print(f"GREEDY: Đường đi sau khi validate (closest) không khả thi: {reason}")
                return []

            self.current_path = validated_path
            self.path_length = len(self.current_path) - 1
            # self.cost will be updated by calculate_path_fuel_consumption

            # Recalculate statistics on the final, validated path
            self.calculate_path_fuel_consumption(self.current_path)
            # self.cost is updated by the call above
            
            return self.current_path  # Return closest path we found
        
        return []  # No path found
    
    def calculate_cost(self, current: SearchState, next_pos: Tuple[int, int]) -> Tuple[float, float, float]:
        """Tính chi phí cho việc di chuyển đến ô tiếp theo.
            
        Returns: 
            Tuple[float, float, float]: (fuel sau khi di chuyển, chi phí di chuyển, tiền sau khi di chuyển)
        """
        # Nhiên liệu giảm khi di chuyển
        new_fuel = current.fuel - self.FUEL_PER_MOVE
        move_cost = 0.0
        new_money = current.money
        
        # Xét loại ô tiếp theo
        cell_type = self.grid[next_pos[1], next_pos[0]]
        
        # Lưu ý: new_fuel tại đây là nhiên liệu *sau khi* tiêu thụ, *trước khi* đổ
        fuel_after_consumption = new_fuel
        
        if cell_type == 2:  # Trạm xăng
            # Luôn nạp đầy nhiên liệu khi đến trạm xăng nếu cần
            fuel_needed = self.MAX_FUEL - fuel_after_consumption
            if fuel_needed > 0:
                # Tính chi phí đổ xăng dựa trên lượng cần đổ
                if fuel_after_consumption < self.LOW_FUEL_THRESHOLD:
                    discount = 0.7  # Giảm 30% chi phí khi nhiên liệu thấp
                else:
                    discount = 1.0
                    
                gas_cost = self.GAS_STATION_COST * (fuel_needed / self.MAX_FUEL) * discount
                move_cost = gas_cost
                
                # Kiểm tra nếu đủ tiền để đổ xăng
                if new_money >= gas_cost:
                    new_money -= gas_cost
                    new_fuel = self.MAX_FUEL  # Đổ đầy xăng
                else:
                    # Không đủ tiền, đổ được bao nhiêu thì đổ
                    affordable_ratio = new_money / gas_cost
                    new_fuel = fuel_after_consumption + (fuel_needed * affordable_ratio)
                    new_money = 0  # Dùng hết tiền
            else:
                # Không cần đổ xăng, giữ nguyên nhiên liệu
                new_fuel = fuel_after_consumption
                
        elif cell_type == 1:  # Trạm thu phí
            # Chỉ tính phí nếu chưa đi qua trạm này
            if next_pos not in current.toll_stations_visited:
                # Giảm phạt dựa trên số trạm đã đi qua (tối đa giảm 50%)
                visited_discount = min(0.5, len(current.toll_stations_visited) * 0.1)
                
                # Tính chi phí trạm thu phí
                toll_cost = self.TOLL_BASE_COST + (self.TOLL_PENALTY * (1.0 - visited_discount))
                move_cost = toll_cost
                
                # Kiểm tra nếu đủ tiền để trả phí
                if new_money >= toll_cost:
                    new_money -= toll_cost
        
        return new_fuel, move_cost, new_money
    
    def calculate_path_fuel_consumption(self, path: List[Tuple[int, int]]) -> None:
        """Tính toán lượng nhiên liệu tiêu thụ và đổ thêm cho đường đi cuối cùng."""
        if not path or len(path) < 2:
            self.fuel_consumed = 0.0
            self.fuel_refilled = 0.0
            return
            
        # Khởi tạo các biến
        current_fuel = self.MAX_FUEL  # Bắt đầu với bình đầy
        total_fuel_consumed = 0.0
        total_fuel_refilled = 0.0
        
        print(f"[DEBUG] GreedySearch.calculate_path_fuel_consumption: starting fuel={current_fuel}, max_fuel={self.MAX_FUEL}, path_length={len(path)}")
            
        # Duyệt từng bước trên đường đi
        for i in range(len(path) - 1):
            current_pos = path[i]
            next_pos = path[i + 1]
            
            # Tiêu thụ nhiên liệu cho bước di chuyển này
            fuel_for_step = self.FUEL_PER_MOVE
            total_fuel_consumed += fuel_for_step
            current_fuel -= fuel_for_step
            
            # Kiểm tra xem đã đến trạm xăng chưa
            cell_type = self.grid[next_pos[1], next_pos[0]]
            if cell_type == 2 and next_pos in self.visited_gas_stations:  # Trạm xăng và đã ghé thăm
                # Tính lượng nhiên liệu cần đổ thêm
                fuel_needed = self.MAX_FUEL - current_fuel
                if fuel_needed > 0:
                    # Lượng nhiên liệu đã đổ thêm
                    total_fuel_refilled += fuel_needed
                    current_fuel = self.MAX_FUEL  # Đổ đầy bình
                    print(f"[DEBUG] Refueling at {next_pos}: +{fuel_needed:.1f}L, now at {current_fuel:.1f}L")
        
        # Cập nhật các thuộc tính của thuật toán
        self.fuel_consumed = total_fuel_consumed
        self.fuel_refilled = total_fuel_refilled
        self.current_fuel = current_fuel  # Cập nhật nhiên liệu còn lại sau khi đi xong
        
        print(f"[DEBUG] Final fuel stats: consumed={self.fuel_consumed:.1f}L, refilled={self.fuel_refilled:.1f}L, remaining={self.current_fuel:.1f}L")
        
    def get_neighbors(self, pos: Tuple[int, int]) -> List[Tuple[int, int]]:
        """Lấy danh sách các vị trí kề có thể đi được."""
        # Sử dụng phương thức cha từ BaseSearch (đã lọc bỏ ô chướng ngại vật)
        return super().get_neighbors(pos)
    
    def can_reach_gas_station(self, pos: Tuple[int, int], fuel: float) -> bool:
        """Kiểm tra từ vị trí hiện tại với lượng nhiên liệu hiện có có thể đến được trạm xăng không.
        
        Args:
            pos: Vị trí hiện tại
            fuel: Lượng nhiên liệu hiện có
            
        Returns:
            bool: True nếu có thể đến được trạm xăng, False nếu không
        """
        # Tạo trạng thái tạm thời để kiểm tra
        temp_state = SearchState(
            position=pos,
            fuel=fuel,
            total_cost=0,
            money=self.current_money,  # Đảm bảo truyền tiền hiện tại
            path=[],
            visited_gas_stations=set(),
            toll_stations_visited=set()
        )
        
        # Tìm trạm xăng gần nhất
        nearest_gas = self.find_nearest_reachable_gas_station(temp_state)
        
        # Có thể đến được trạm xăng nếu tìm thấy một trạm
        return nearest_gas is not None
    
    def find_nearest_reachable_gas_station(self, state: SearchState) -> Optional[Tuple[int, int]]:
        """Tìm trạm xăng gần nhất có thể đến được từ trạng thái hiện tại."""
        min_distance = float('inf')
        nearest_gas = None
        
        for x in range(self.cols):
            for y in range(self.rows):
                if self.grid[y, x] == 2:  # Trạm xăng
                    distance = self.heuristic(state.position, (x, y))
                    if distance < min_distance:
                        min_distance = distance
                        nearest_gas = (x, y)
        
        return nearest_gas 
    
    def step(self) -> bool:
        """Execute one step of Greedy Search. Returns True if finished."""
        # Nếu chưa khởi tạo frontier
        if not hasattr(self, 'frontier') or not self.frontier or not hasattr(self, 'goal'):
            self.current_position = None
            return True
            
        # Lấy trạng thái với giá trị heuristic thấp nhất
        if not self.frontier:
            self.current_position = None
            return True
        
        self.steps += 1
        _, state_id, current_state = heapq.heappop(self.frontier)
        current_pos = current_state.position
        self.current_position = current_pos
        
        # Kiểm tra xem đã đến đích chưa
        if current_pos == self.goal:
            # Đã tìm thấy đích, cập nhật đường đi và thống kê
            self.current_path = current_state.path
            self.path_length = len(self.current_path) - 1
            self.cost = current_state.total_cost
            self.current_fuel = current_state.fuel
            self.current_money = current_state.money
            self.current_total_cost = current_state.total_cost
        
            # Lấy danh sách trạm xăng và trạm thu phí đã thăm từ trạng thái
            self.visited_gas_stations = current_state.visited_gas_stations
            self.toll_stations_visited = current_state.toll_stations_visited
            
            # Tính chi phí nhiên liệu và trạm thu phí
            self.current_fuel_cost = 0
            self.current_toll_cost = 0
            
            # Duyệt qua từng bước trong đường đi
            for i in range(len(self.current_path) - 1):
                pos1 = self.current_path[i]
                pos2 = self.current_path[i + 1]
                
                # Kiểm tra loại ô
                cell_type = self.grid[pos2[1], pos2[0]]
                
                # Tính chi phí cho trạm thu phí
                if cell_type == 1 and pos2 in self.toll_stations_visited:
                    visited_discount = min(0.5, len(self.toll_stations_visited) * 0.1)
                    toll_cost = self.TOLL_BASE_COST + (self.TOLL_PENALTY * (1.0 - visited_discount))
                    self.current_toll_cost += toll_cost
                
                # Tính chi phí cho trạm xăng
                if cell_type == 2 and pos2 in self.visited_gas_stations:
                    # Chi phí đã được tính trong calculate_path_fuel_consumption
                    pass
            
            # Tính tổng chi phí nhiên liệu = tổng chi phí - chi phí trạm thu phí
            self.current_fuel_cost = self.cost - self.current_toll_cost
            
            # Tính toán lượng nhiên liệu tiêu thụ cho đường đi
            self.calculate_path_fuel_consumption(self.current_path)
            
            # Final validation to remove any obstacle cells
            path = self.validate_path_no_obstacles(self.current_path)
            self.current_path = path
            
            return True
        
        # Kiểm tra xem đã thăm trạng thái này chưa
        state_key = current_state.get_state_key()
        if state_key in self.visited_states:
            return False
            
        self.visited_states.add(state_key)
        
        # Duyệt các ô lân cận
        for next_pos in self.get_neighbors(current_pos):
            # Calculate cost to move to the next position
            new_fuel, move_cost, new_money = self.calculate_cost(current_state, next_pos)
                
            # Bỏ qua nếu không khả thi (không đủ nhiên liệu hoặc tiền)
            if new_fuel < 0 or new_money < 0:
                    continue
                
                # Tạo trạng thái mới
            new_state = SearchState(
                position=next_pos,
                fuel=new_fuel,
                total_cost=current_state.total_cost + move_cost,
                money=new_money,
                path=current_state.path + [next_pos],
                visited_gas_stations=current_state.visited_gas_stations.copy(),
                toll_stations_visited=current_state.toll_stations_visited.copy()
            )
                
            # Cập nhật tập trạm đã thăm
            if self.grid[next_pos[1], next_pos[0]] == 2:  # Trạm xăng
                new_state.visited_gas_stations.add(next_pos)
            elif self.grid[next_pos[1], next_pos[0]] == 1:  # Trạm thu phí
                new_state.toll_stations_visited.add(next_pos)
            
            # Thêm trạng thái mới vào frontier với ưu tiên dựa trên heuristic
            # Với trạm xăng, giảm giá trị heuristic khi nhiên liệu thấp
            priority = self.heuristic(next_pos, self.goal)
                
            # Nếu nhiên liệu thấp và vị trí tiếp theo là trạm xăng, ưu tiên thăm
            if self.grid[next_pos[1], next_pos[0]] == 2 and new_state.fuel < self.MAX_FUEL * 0.3:
                priority *= 0.5  # Giảm ưu tiên để làm nó hấp dẫn hơn
            
            heapq.heappush(self.frontier, (priority, self.steps, new_state))
            self.add_visited(next_pos)
        
        # Nếu không còn trạng thái nào để xét, trả về True
        if not self.frontier:
            return True
        
        return False 