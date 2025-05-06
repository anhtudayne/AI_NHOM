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
    
    def __init__(self, grid: np.ndarray):
        """Initialize A* with fuel constraints."""
        super().__init__(grid)
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
                # Tối ưu hóa đường đi
                return self.optimize_path(current_state.path, self.initial_fuel)
                
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
        for next_pos in self.get_neighbor_states(current):
            if next_pos in self.closed_set:
                continue
            
            # Tính toán chi phí và nhiên liệu cho bước tiếp theo
            distance_cost, fuel_cost, toll_cost = self.calculate_cost(current, next_pos)
            new_g_score = float(self.g_score[current]) + float(distance_cost) + float(fuel_cost) + float(toll_cost)
            
            # Nếu điểm lân cận chưa trong open_set hoặc có g_score tốt hơn
            if next_pos not in self.open_set or new_g_score < self.g_score.get(next_pos, float('inf')):
                self.parent[next_pos] = current
                self.g_score[next_pos] = new_g_score
                self.f_score[next_pos] = new_g_score + self.heuristic(next_pos, self.fuel[current])
                
                # Cập nhật nhiên liệu và chi phí
                self.fuel[next_pos] = float(self.fuel[current]) - float(self.FUEL_PER_MOVE)
                self.total_cost[next_pos] = new_g_score
                self.fuel_cost[next_pos] = self.fuel_cost[current] + fuel_cost
                self.toll_cost[next_pos] = self.toll_cost[current] + toll_cost
                
                # Cập nhật các trạm đã thăm
                self.visited_gas_stations[next_pos] = self.visited_gas_stations[current].copy()
                self.toll_stations_visited[next_pos] = self.toll_stations_visited[current].copy()
                
                # Nếu đến trạm xăng
                if self.grid[next_pos[0], next_pos[1]] == 3:
                    self.fuel[next_pos] = float(self.MAX_FUEL)
                    self.visited_gas_stations[next_pos].add(next_pos)
                
                # Nếu đến trạm thu phí
                if self.grid[next_pos[0], next_pos[1]] == 2:
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
        """
        Lấy danh sách các vị trí kề có thể đi được
        """
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