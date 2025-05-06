"""
Hill Climbing search algorithm implementation.
"""

from typing import List, Tuple, Dict
import numpy as np
from .base_search import BaseSearch, SearchState

class HillClimbing(BaseSearch):
    """Hill Climbing search algorithm implementation."""
    
    def __init__(self, grid: np.ndarray):
        """Initialize Hill Climbing with a grid."""
        super().__init__(grid)
        self.current_state = None
        self.start = None
        self.goal = None
    
    def heuristic(self, pos: Tuple[int, int], goal: Tuple[int, int]) -> float:
        """Tính hàm heuristic (Manhattan distance)."""
        x1, y1 = pos
        x2, y2 = goal
        return float(abs(x1 - x2) + abs(y1 - y2))
    
    def heuristic_with_fuel(self, state: SearchState, goal: Tuple[int, int]) -> float:
        """Tính hàm heuristic có xét đến nhiên liệu và chi phí qua trạm thu phí.
        
        Args:
            state: Trạng thái hiện tại
            goal: Vị trí đích
            
        Returns:
            float: Giá trị heuristic (càng nhỏ càng tốt)
        """
        pos = state.position
        fuel = state.fuel
        
        # Tính toán Manhattan distance cơ bản
        base_heuristic = self.heuristic(pos, goal)
        
        # Hệ số cho các thành phần khác nhau của heuristic
        FUEL_WEIGHT = 2.0        # Trọng số cho yếu tố nhiên liệu
        TOLL_WEIGHT = 0.5        # Trọng số cho trạm thu phí
        
        # Phần giá trị heuristic cho nhiên liệu
        fuel_component = 0
        
        # Nếu nhiên liệu thấp, tìm và ưu tiên trạm xăng
        if fuel < self.LOW_FUEL_THRESHOLD:
            # Tìm trạm xăng gần nhất
            nearest_gas = self.find_nearest_reachable_gas_station(state)
            
            # Nếu tìm thấy trạm xăng trong tầm với, thêm yếu tố khuyến khích đi qua trạm xăng
            if nearest_gas is not None:
                # Chi phí đến trạm xăng
                cost_to_gas = self.heuristic(pos, nearest_gas)
                
                # Chi phí từ trạm xăng đến đích
                cost_from_gas_to_goal = self.heuristic(nearest_gas, goal)
                
                # Ưu tiên đường đi qua trạm xăng bằng cách giảm chi phí ước tính
                fuel_component = FUEL_WEIGHT * (cost_to_gas + 0.7 * cost_from_gas_to_goal - base_heuristic)
        
        # Phần giá trị heuristic cho trạm thu phí
        toll_component = 0
        
        # BFS để tìm các trạm thu phí có thể gặp trên đường đi đến đích
        from collections import deque
        
        queue = deque([pos])
        visited = {pos}
        max_search_depth = min(10, base_heuristic * 1.5)  # Giới hạn độ sâu tìm kiếm
        depth = {pos: 0}
        
        potential_toll_stations = []
        
        while queue:
            current_pos = queue.popleft()
            
            # Nếu đã tìm kiếm quá sâu, dừng lại
            if depth[current_pos] > max_search_depth:
                continue
            
            # Nếu vị trí hiện tại là trạm thu phí và chưa đi qua, thêm vào danh sách
            if self.grid[current_pos[1], current_pos[0]] == 1 and current_pos not in state.toll_stations_visited:
                potential_toll_stations.append((current_pos, depth[current_pos]))
            
            # Kiểm tra các ô lân cận
            for next_pos in self.get_neighbors(current_pos):
                # Bỏ qua ô vật cản
                if self.grid[next_pos[1], next_pos[0]] == 3:
                    continue
                
                # Nếu ô chưa được thăm, thêm vào hàng đợi
                if next_pos not in visited:
                    visited.add(next_pos)
                    queue.append(next_pos)
                    depth[next_pos] = depth[current_pos] + 1
                    
                    # Nếu đã tìm thấy đích, dừng lại
                    if next_pos == goal:
                        queue.clear()
                        break
        
        # Tính toán thành phần heuristic cho trạm thu phí
        for toll_pos, toll_depth in potential_toll_stations:
            # Đánh giá xem trạm thu phí này có nằm trên đường đi tới đích không
            on_path_likelihood = 1.0 - (abs(self.heuristic(toll_pos, goal) + toll_depth - base_heuristic) / base_heuristic)
            on_path_likelihood = max(0, on_path_likelihood)  # Giữ giá trị không âm
            
            # Thêm vào thành phần heuristic với trọng số phù hợp
            toll_component += TOLL_WEIGHT * on_path_likelihood * self.TOLL_PENALTY
        
        # Kết hợp các thành phần để tạo giá trị heuristic cuối cùng
        # Lưu ý: Thêm các thành phần vì base_heuristic càng nhỏ càng tốt
        return base_heuristic + fuel_component + toll_component
    
    def get_neighbor_states(self, current_state: SearchState) -> List[SearchState]:
        """Lấy danh sách các trạng thái lân cận có thể đến được."""
        neighbors = []
        
        for next_pos in self.get_neighbors(current_state.position):
            # Kiểm tra vật cản
            if self.grid[next_pos[1], next_pos[0]] == 3:  # Nếu là vật cản (loại 3)
                continue
                
            # Tính toán chi phí và nhiên liệu cho bước tiếp theo
            new_fuel, move_cost = self.calculate_cost(current_state, next_pos)
            
            # Kiểm tra tính khả thi của bước tiếp theo
            if new_fuel < 0 or current_state.total_cost + move_cost > self.MAX_TOTAL_COST:
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
            
            # Cập nhật các tập đã thăm
            if self.grid[next_pos[1], next_pos[0]] == 2:  # Trạm xăng (loại 2 thay vì 'G')
                new_state.visited_gas_stations.add(next_pos)
            elif self.grid[next_pos[1], next_pos[0]] == 1:  # Trạm thu phí (loại 1 thay vì 'T')
                new_state.toll_stations_visited.add(next_pos)
            
            neighbors.append(new_state)
        
        return neighbors
    
    def search(self, start: Tuple[int, int], goal: Tuple[int, int]) -> List[Tuple[int, int]]:
        """Execute Hill Climbing from start to goal."""
        self.start = start
        self.goal = goal
        self.visited.clear()
        self.current_path.clear()
        self.steps = 0
        self.path_length = 0
        self.cost = 0
        self.current_fuel = self.MAX_FUEL
        self.current_total_cost = 0
        self.current_fuel_cost = 0
        self.current_toll_cost = 0
        
        # Sử dụng kỹ thuật Random Restarts
        return self.search_with_random_restarts(start, goal, num_restarts=25)
    
    def search_with_random_restarts(self, start: Tuple[int, int], goal: Tuple[int, int], 
                                   num_restarts: int = 25) -> List[Tuple[int, int]]:
        """Thực hiện tìm kiếm Hill Climbing với kỹ thuật Random Restarts.
        
        Args:
            start: Vị trí bắt đầu
            goal: Vị trí đích
            num_restarts: Số lần chạy lại với vị trí xuất phát khác nhau
            
        Returns:
            List[Tuple[int, int]]: Đường đi tốt nhất tìm được
        """
        best_path = []
        best_cost = float('inf')
        total_visited = []
        
        # Thực hiện tìm kiếm ban đầu từ vị trí xuất phát gốc
        path, cost, fuel, visited = self.do_hill_climbing(start, goal)
        total_visited.extend(visited)
        
        if path and (not best_path or cost < best_cost):
            best_path = path
            best_cost = cost
            self.current_fuel = fuel
        
        # Thực hiện các lần tìm kiếm bổ sung
        for i in range(num_restarts - 1):
            # Lấy vị trí xuất phát mới từ tập đã thăm (nếu có)
            new_start = start
            if total_visited:
                import random
                # Lấy ngẫu nhiên một vị trí từ tập đã thăm
                # Tránh chọn vị trí vật cản và vị trí đích
                valid_positions = [pos for pos in total_visited 
                                  if self.grid[pos[1], pos[0]] != 3 and pos != goal]
                if valid_positions:
                    new_start = random.choice(valid_positions)
            
            # Thực hiện tìm kiếm từ vị trí xuất phát mới
            path, cost, fuel, visited = self.do_hill_climbing(new_start, goal)
            total_visited.extend(visited)
            
            # Cập nhật kết quả tốt nhất
            if path and (not best_path or cost < best_cost):
                best_path = path
                best_cost = cost
                self.current_fuel = fuel
        
        # Cập nhật thông tin thống kê
        if best_path:
            self.current_path = best_path
            self.path_length = len(best_path) - 1
            self.cost = best_cost
            # current_fuel đã được cập nhật trong quá trình tìm kiếm
            self.current_total_cost = best_cost
            self.current_fuel_cost = best_cost - self.current_toll_cost
        
        # Cập nhật danh sách các ô đã thăm
        self.visited = list(set(total_visited))
        
        return best_path
    
    def do_hill_climbing(self, start: Tuple[int, int], goal: Tuple[int, int]) -> Tuple[List[Tuple[int, int]], float, float, List[Tuple[int, int]]]:
        """Thực hiện một lần tìm kiếm Hill Climbing từ vị trí xuất phát đến đích,
        với hỗ trợ cho sideways moves để thoát khỏi plateaus.
        
        Args:
            start: Vị trí bắt đầu
            goal: Vị trí đích
            
        Returns:
            Tuple: (path, cost, fuel, visited) - đường đi, chi phí, nhiên liệu còn lại, và danh sách các ô đã thăm
        """
        # Tạo trạng thái ban đầu
        current_state = SearchState(
            position=start,
            fuel=self.MAX_FUEL,
            total_cost=0,
            path=[start],
            visited_gas_stations=set(),
            toll_stations_visited=set()
        )
        
        visited = [start]
        
        # Các tham số cho sideways moves
        max_sideways_moves = 10  # Số sideways moves tối đa cho phép
        sideways_count = 0       # Số sideways moves đã thực hiện
        
        # Tập đã thăm (để tránh lặp lại)
        state_keys_visited = {current_state.get_state_key()}
        
        while True:
            if current_state.position == goal:
                # Kiểm tra tính khả thi của đường đi
                is_feasible, reason = self.is_path_feasible(current_state.path, self.MAX_FUEL)
                if is_feasible:
                    return current_state.path, current_state.total_cost, current_state.fuel, visited
                else:
                    break
            
            # Tìm hàng xóm tốt nhất
            best_neighbor = None
            best_score = float('inf')
            best_equal_neighbor = None  # Hàng xóm tốt nhất có giá trị bằng trạng thái hiện tại
            
            for neighbor_state in self.get_neighbor_states(current_state):
                # Bỏ qua các trạng thái đã thăm
                if neighbor_state.get_state_key() in state_keys_visited:
                    continue
                
                # Tính điểm đánh giá với xét đến nhiên liệu
                score = self.heuristic_with_fuel(neighbor_state, goal)
                
                if score < best_score:
                    best_score = score
                    best_neighbor = neighbor_state
                elif score == best_score and best_equal_neighbor is None:
                    best_equal_neighbor = neighbor_state
            
            # Nếu không có hàng xóm tốt hơn hoặc bằng, kết thúc
            if best_neighbor is None and best_equal_neighbor is None:
                break
            
            # Tính điểm đánh giá của trạng thái hiện tại với xét đến nhiên liệu
            current_score = self.heuristic_with_fuel(current_state, goal)
            
            # Nếu hàng xóm tốt nhất tốt hơn trạng thái hiện tại, di chuyển đến đó
            if best_neighbor is not None and best_score < current_score:
                current_state = best_neighbor
                visited.append(current_state.position)
                state_keys_visited.add(current_state.get_state_key())
                sideways_count = 0  # Reset sideways count
            # Nếu có hàng xóm bằng và chưa vượt quá số sideways moves tối đa, thực hiện sideways move
            elif best_equal_neighbor is not None and sideways_count < max_sideways_moves:
                current_state = best_equal_neighbor
                visited.append(current_state.position)
                state_keys_visited.add(current_state.get_state_key())
                sideways_count += 1
            else:
                # Đã đến đỉnh đồi (local maximum) hoặc đã vượt quá số sideways moves tối đa, kết thúc
                break
        
        # Không tìm thấy đường đi
        return [], float('inf'), 0, visited
    
    def step(self) -> bool:
        """Execute one step of Hill Climbing."""
        if self.current_state is None:
            self.current_position = None
            return True
        
        self.steps += 1
        
        if self.current_state.position == self.goal:
            # Kiểm tra tính khả thi của đường đi
            is_feasible, reason = self.is_path_feasible(self.current_state.path, self.MAX_FUEL)
            if is_feasible:
                self.current_path = self.current_state.path
                self.path_length = len(self.current_path) - 1
                self.cost = self.current_state.total_cost
                self.current_fuel = self.current_state.fuel
                self.current_total_cost = self.current_state.total_cost
                self.current_fuel_cost = self.current_state.total_cost - self.current_toll_cost
                self.current_state = None
                return True
            else:
                return False
        
        # Tìm hàng xóm tốt nhất
        best_neighbor = None
        best_score = float('inf')
        
        for neighbor_state in self.get_neighbor_states(self.current_state):
            # Tính điểm đánh giá với xét đến nhiên liệu
            score = self.heuristic_with_fuel(neighbor_state, self.goal)
            
            if score < best_score:
                best_score = score
                best_neighbor = neighbor_state
        
        # Nếu không có hàng xóm tốt hơn, kết thúc
        if best_neighbor is None:
            self.current_state = None
            self.current_position = None
            return True
        
        # Tính điểm đánh giá của trạng thái hiện tại với xét đến nhiên liệu
        current_score = self.heuristic_with_fuel(self.current_state, self.goal)
        
        # Nếu hàng xóm tốt nhất tốt hơn trạng thái hiện tại, di chuyển đến đó
        if best_score < current_score:
            self.current_state = best_neighbor
            self.current_position = self.current_state.position
            self.visited.append(self.current_position)
        else:
            # Đã đến đỉnh đồi (local maximum), kết thúc
            self.current_state = None
            self.current_position = None
            return True
        
        return False 