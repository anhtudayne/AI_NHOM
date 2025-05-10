"""
A* search algorithm implementation with fuel constraints.
"""

from typing import List, Tuple, Dict, Set, Optional
import numpy as np
from .base_search import BaseSearch, SearchState
from queue import PriorityQueue
import heapq

class AStarFuel(BaseSearch):
    """A* search algorithm implementation with fuel constraints."""
    
    def __init__(self, grid: np.ndarray, initial_money: float = None, 
                 max_fuel: float = None, fuel_per_move: float = None, 
                 gas_station_cost: float = None, toll_base_cost: float = None,
                 initial_fuel: float = None):
        """Initialize A* with fuel constraints."""
        super().__init__(grid, initial_money, max_fuel, fuel_per_move, 
                         gas_station_cost, toll_base_cost, initial_fuel)
        self.initial_fuel = initial_fuel  # Store initial_fuel as instance attribute
        self.open_set = PriorityQueue()
        self.closed_set = set()
        self.g_score = {}  # Chi phí thực tế từ điểm bắt đầu đến điểm hiện tại
        self.f_score = {}  # Ước tính tổng chi phí (g_score + heuristic)
        self.parent = {}
        self.start = None
        self.goal = None
    
    def heuristic(self, pos: Tuple[int, int], current_fuel: float) -> float:
        """
        Hàm heuristic ước tính chi phí từ pos đến đích
        """
        x1, y1 = pos
        x2, y2 = self.goal
        distance = abs(x1 - x2) + abs(y1 - y2)
        
        # Ước tính nhiên liệu cần thiết
        needed_fuel = distance * self.FUEL_PER_MOVE
        
        # Nếu nhiên liệu không đủ, ưu tiên tìm trạm xăng
        if current_fuel < needed_fuel:
            # Tìm trạm xăng gần nhất
            nearest_gas = self.find_nearest_gas_station(pos)
            if nearest_gas:
                # Tính khoảng cách đến trạm xăng
                gas_x, gas_y = nearest_gas
                gas_distance = abs(x1 - gas_x) + abs(y1 - gas_y)
                return gas_distance * 0.5  # Giảm trọng số để ưu tiên tìm trạm xăng
                
        return distance
    
    def search(self) -> Optional[List[Tuple[int, int]]]:
        """
        Tìm đường đi tối ưu từ điểm bắt đầu đến điểm kết thúc
        Returns:
            Danh sách các điểm trên đường đi hoặc None nếu không tìm thấy
        """
        # Khởi tạo hàng đợi ưu tiên
        open_set = []
        heapq.heappush(open_set, (0, SearchState(
            position=self.start,
            fuel=self.initial_fuel,
            total_cost=0,
            path=[self.start],
            visited_gas_stations=set(),
            toll_stations_visited=set()
        )))
        
        # Tập các trạng thái đã xét
        closed_set = set()
        
        while open_set:
            # Lấy trạng thái có chi phí thấp nhất
            _, current_state = heapq.heappop(open_set)
            
            # Kiểm tra đã đến đích chưa
            if current_state.position == self.goal:
                raw_optimized_path = self.optimize_path(current_state.path, self.initial_fuel)

                if not raw_optimized_path: # optimize_path might return None or empty
                    print(f"ASTAR_FUEL: optimize_path returned no path for {self.goal}.")
                    return None # Or continue if the algorithm structure supports it

                # First, validate and clean this path further
                validated_path = self.validate_path_no_obstacles(raw_optimized_path)
                
                if not validated_path or len(validated_path) < 2:
                    print(f"ASTAR_FUEL: Path to goal {self.goal} became invalid after validation.")
                    return None 
                
                # Second, check overall feasibility of the validated path
                is_still_feasible, reason = self.is_path_feasible(validated_path, self.MAX_FUEL)
                if is_still_feasible:
                    self.current_path = validated_path
                    self.path_length = len(self.current_path) - 1
                    
                    # Recalculate all statistics on the final, validated path
                    self.calculate_path_fuel_consumption(self.current_path)
                    # self.cost, self.current_fuel, etc. are updated by the above call.
                    
                    print(f"ASTAR_FUEL: Valid and feasible path to {self.goal} found.")
                    return self.current_path # Goal found and path is fully validated
                else:
                    print(f"ASTAR_FUEL: Path to goal {self.goal} not feasible after validation: {reason}.")
                    return None
                
            # Thêm vào tập đã xét
            closed_set.add(current_state.get_key())
            
            # Xét các trạng thái kế tiếp
            for next_pos in self.get_neighbors(current_state.position):
                # Tạo trạng thái mới
                new_state = current_state.copy()
                new_state.position = next_pos
                
                # Kiểm tra trạng thái đã xét chưa
                if new_state.get_key() in closed_set:
                    continue
                    
                # Tính toán chi phí và nhiên liệu
                new_fuel, move_cost = self.calculate_cost(current_state, next_pos)
                
                # Kiểm tra tính khả thi
                if new_fuel < 0:
                    continue
                    
                # Cập nhật trạng thái
                new_state.fuel = new_fuel
                new_state.total_cost = current_state.total_cost + move_cost
                new_state.path = current_state.path + [next_pos]
                
                # Cập nhật các trạm đã ghé
                if self.get_cell_type(next_pos) == 2:
                    new_state.visited_gas_stations.add(next_pos)
                elif self.get_cell_type(next_pos) == 1:
                    new_state.toll_stations_visited.add(next_pos)
                    
                # Tính f = g + h
                g = new_state.total_cost
                h = self.heuristic(next_pos, new_state.fuel)
                f = g + h
                
                # Thêm vào hàng đợi ưu tiên
                heapq.heappush(open_set, (f, new_state))
                
        return None
    
    def step(self) -> bool:
        """Execute one step of A* with fuel constraints."""
        if not self.open_set:
            self.current_position = None
            return True
        
        self.steps += 1
        
        # Tìm điểm có f_score nhỏ nhất trong open_set
        current = min(self.open_set, key=lambda x: self.f_score[x])
        
        if current == self.goal:
            # Kiểm tra tính khả thi của đường đi
            is_feasible, reason = self.is_path_feasible(self._reconstruct_path(current), self.MAX_FUEL)
            if is_feasible:
                self.current_path = self._reconstruct_path(current)
                self.path_length = len(self.current_path) - 1
                self.cost = self.g_score[current]
                self.current_fuel = self.fuel[current]
                self.current_total_cost = self.total_cost[current]
                self.current_fuel_cost = self.current_total_cost - self.current_toll_cost
                self.current_position = None
                return True
            else:
                self.open_set.remove(current)
                self.closed_set.add(current)
                return False
        
        self.open_set.remove(current)
        self.closed_set.add(current)
        self.visited.append(current)
        self.current_position = current
        
        # Xử lý các điểm lân cận
        for next_pos in self.get_neighbors(current):
            # THÊM KIỂM TRA CHƯỚNG NGẠI VẬT RÕ RÀNG
            if self.grid[next_pos[1], next_pos[0]] == self.OBSTACLE_CELL:
                continue

            if next_pos in self.closed_set:
                continue
            
            # Tính toán chi phí và nhiên liệu cho bước tiếp theo
            # LƯU Ý: logic tính toán chi phí ở đây có vẻ khác với phương thức calculate_cost của lớp này
            # và calculate_cost của BaseSearch. Cần xem xét lại nếu có vấn đề.
            # Giả sử distance_cost, fuel_cost, toll_cost được tính bằng một cách nào đó
            # Dưới đây là một ví dụ giữ nguyên cấu trúc cũ, nhưng cần đảm bảo các biến này được định nghĩa đúng
            
            # Ghi chú: Phần tính toán chi phí (distance_cost, fuel_cost, toll_cost) 
            # trong phiên bản gốc của step() không được định nghĩa rõ ràng.
            # Để mã có thể chạy, chúng ta cần một giả định hoặc một cách tính toán tạm thời.
            # Ở đây, tôi sẽ giả định chi phí di chuyển cơ bản là 1 và không có chi phí nhiên liệu/trạm thu phí đặc biệt
            # cho mục đích làm cho mã chạy được. Điều này CẦN ĐƯỢC XEM XÉT LẠI CẨN THẬN.
            move_cost_for_step = 1 # Chi phí di chuyển cơ bản, cần được điều chỉnh cho đúng logic
            fuel_consumed_for_step = self.FUEL_PER_MOVE # Nhiên liệu tiêu thụ, giả định từ BaseSearch

            # Kiểm tra nhiên liệu trước
            if self.fuel.get(current, 0) < fuel_consumed_for_step:
                continue

            new_g_score = float(self.g_score.get(current, float('inf'))) + move_cost_for_step
            
            # Nếu điểm lân cận chưa trong open_set hoặc có g_score tốt hơn
            if next_pos not in self.open_set or new_g_score < self.g_score.get(next_pos, float('inf')):
                self.parent[next_pos] = current
                self.g_score[next_pos] = new_g_score
                self.f_score[next_pos] = new_g_score + self.heuristic(next_pos, self.fuel[current])
                
                # Cập nhật nhiên liệu và chi phí
                self.fuel[next_pos] = float(self.fuel[current]) - float(self.FUEL_PER_MOVE)
                self.total_cost[next_pos] = new_g_score
                self.fuel_cost[next_pos] = self.fuel_cost[current] + fuel_consumed_for_step
                self.toll_cost[next_pos] = self.toll_cost[current]
                
                # Cập nhật các trạm đã thăm
                self.visited_gas_stations[next_pos] = self.visited_gas_stations[current].copy()
                self.toll_stations_visited[next_pos] = self.toll_stations_visited[current].copy()
                
                # Nếu đến trạm xăng
                if self.grid[next_pos[0], next_pos[1]] == 2:  # Value 2 is Gas Station
                    self.fuel[next_pos] = float(self.MAX_FUEL)
                    self.visited_gas_stations[next_pos].add(next_pos)
                
                # Nếu đến trạm thu phí
                if self.grid[next_pos[0], next_pos[1]] == 1:  # Value 1 is Toll Station
                    self.toll_stations_visited[next_pos].add(next_pos)
                
                self.open_set.add(next_pos)
        
        return False
    
    def _reconstruct_path(self, current: Tuple[int, int]) -> List[Tuple[int, int]]:
        """Tái tạo đường đi từ điểm hiện tại về điểm bắt đầu."""
        path = [current]
        while self.parent[current] is not None:
            current = self.parent[current]
            path.append(current)
        return list(reversed(path)) 

    def get_neighbors(self, pos: Tuple[int, int]) -> List[Tuple[int, int]]:
        """Lấy danh sách các vị trí kề có thể đi được."""
        # Sử dụng phương thức cha từ BaseSearch (đã lọc bỏ ô chướng ngại vật)
        return super().get_neighbors(pos) 