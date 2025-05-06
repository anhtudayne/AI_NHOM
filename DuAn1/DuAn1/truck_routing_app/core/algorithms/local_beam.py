"""
Local Beam Search algorithm implementation.
"""

from typing import List, Tuple, Dict
import numpy as np
import random
import math
from collections import deque
from .base_search import BaseSearch, SearchState

class LocalBeamSearch(BaseSearch):
    """Local Beam Search algorithm implementation."""
    
    def __init__(self, grid: np.ndarray, beam_width: int = 10):
        """Initialize Local Beam Search with a grid and beam width."""
        super().__init__(grid)
        self.beam_width = beam_width
        self.current_states = []
        self.start = None
        self.goal = None
        # Tham số cho Stochastic Beam Search
        self.temperature = 2.0  # Nhiệt độ cho chọn ngẫu nhiên (càng cao càng ngẫu nhiên)
        self.use_stochastic = True  # Sử dụng Stochastic Beam Search
    
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
    
    def select_states_stochastic(self, states: List[SearchState], goal: Tuple[int, int], k: int) -> List[SearchState]:
        """Chọn k trạng thái từ danh sách states theo phương pháp ngẫu nhiên có trọng số.
        
        Args:
            states: Danh sách các trạng thái
            goal: Vị trí đích
            k: Số trạng thái cần chọn
            
        Returns:
            List[SearchState]: k trạng thái được chọn
        """
        if not states:
            return []
            
        if len(states) <= k:
            return states
        
        # Tính điểm heuristic cho mỗi trạng thái
        scores = [self.heuristic_with_fuel(state, goal) for state in states]
        
        # Chuyển đổi điểm thành xác suất (điểm càng thấp càng tốt)
        # Sử dụng công thức Softmax: P(i) = exp(-score[i]/T) / sum(exp(-score[j]/T))
        min_score = min(scores)
        adjusted_scores = [-(score - min_score) for score in scores]  # Đảo dấu để điểm thấp có xác suất cao hơn
        
        # Áp dụng softmax
        exp_scores = [math.exp(score/self.temperature) for score in adjusted_scores]
        sum_exp_scores = sum(exp_scores)
        probabilities = [exp_score/sum_exp_scores for exp_score in exp_scores]
        
        # Chọn k trạng thái theo xác suất
        selected_indices = random.choices(range(len(states)), probabilities, k=k)
        return [states[i] for i in selected_indices]
    
    def search(self, start: Tuple[int, int], goal: Tuple[int, int]) -> List[Tuple[int, int]]:
        """Execute Local Beam Search from start to goal."""
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
        
        # Tạo trạng thái ban đầu
        initial_state = SearchState(
            position=start,
            fuel=self.MAX_FUEL,
            total_cost=0,
            path=[start],
            visited_gas_stations=set(),
            toll_stations_visited=set()
        )
        
        self.current_states = [initial_state]
        self.visited.append(start)
        self.current_position = start
        self.current_path = [start]
        
        # Tập các trạng thái đã xử lý (để tránh lặp lại)
        processed_states = set()
        
        while self.current_states:
            self.steps += 1
            
            # Kiểm tra xem có trạng thái nào đã đến đích không
            for state in self.current_states:
                if state.position == goal:
                    # Kiểm tra tính khả thi của đường đi
                    is_feasible, reason = self.is_path_feasible(state.path, self.MAX_FUEL)
                    if is_feasible:
                        self.current_path = state.path
                        self.path_length = len(self.current_path) - 1
                        self.cost = state.total_cost
                        self.current_fuel = state.fuel
                        self.current_total_cost = state.total_cost
                        self.current_fuel_cost = state.total_cost - self.current_toll_cost
                        return self.current_path
            
            # Tạo danh sách tất cả các trạng thái kế tiếp
            all_neighbors = []
            for state in self.current_states:
                # Thêm trạng thái này vào tập đã xử lý
                processed_states.add(state.get_state_key())
                
                # Lấy tất cả trạng thái lân cận
                neighbors = self.get_neighbor_states(state)
                
                # Lọc bỏ các trạng thái đã xử lý
                valid_neighbors = [n for n in neighbors if n.get_state_key() not in processed_states]
                all_neighbors.extend(valid_neighbors)
            
            if not all_neighbors:
                break
            
            # Chọn k trạng thái tốt nhất theo phương pháp ngẫu nhiên hoặc đơn thuần
            if self.use_stochastic:
                self.current_states = self.select_states_stochastic(all_neighbors, goal, self.beam_width)
            else:
                # Sắp xếp các trạng thái kế tiếp theo heuristic có xét đến nhiên liệu
                all_neighbors.sort(key=lambda x: self.heuristic_with_fuel(x, goal))
                # Chọn k trạng thái tốt nhất
                self.current_states = all_neighbors[:self.beam_width]
            
            # Cập nhật vị trí hiện tại và danh sách đã thăm
            if self.current_states:
                self.current_position = self.current_states[0].position
                for state in self.current_states:
                    if state.position not in self.visited:
                        self.visited.append(state.position)
        
        return []  # No path found
    
    def step(self) -> bool:
        """Execute one step of Local Beam Search."""
        if not self.current_states:
            self.current_position = None
            return True
        
        self.steps += 1
        
        # Kiểm tra xem có trạng thái nào đã đến đích không
        for state in self.current_states:
            if state.position == self.goal:
                # Kiểm tra tính khả thi của đường đi
                is_feasible, reason = self.is_path_feasible(state.path, self.MAX_FUEL)
                if is_feasible:
                    self.current_path = state.path
                    self.path_length = len(self.current_path) - 1
                    self.cost = state.total_cost
                    self.current_fuel = state.fuel
                    self.current_total_cost = state.total_cost
                    self.current_fuel_cost = state.total_cost - self.current_toll_cost
                    self.current_states = []
                    self.current_position = None
                    return True
        
        # Tập các trạng thái đã xử lý (để tránh lặp lại)
        processed_states = set([state.get_state_key() for state in self.current_states])
        
        # Tạo danh sách tất cả các trạng thái kế tiếp
        all_neighbors = []
        for state in self.current_states:
            neighbors = self.get_neighbor_states(state)
            # Lọc bỏ các trạng thái đã xử lý
            valid_neighbors = [n for n in neighbors if n.get_state_key() not in processed_states]
            all_neighbors.extend(valid_neighbors)
        
        if not all_neighbors:
            self.current_states = []
            self.current_position = None
            return True
        
        # Chọn k trạng thái tốt nhất theo phương pháp ngẫu nhiên hoặc đơn thuần
        if self.use_stochastic:
            self.current_states = self.select_states_stochastic(all_neighbors, self.goal, self.beam_width)
        else:
            # Sắp xếp các trạng thái kế tiếp theo heuristic có xét đến nhiên liệu
            all_neighbors.sort(key=lambda x: self.heuristic_with_fuel(x, self.goal))
            # Chọn k trạng thái tốt nhất
            self.current_states = all_neighbors[:self.beam_width]
        
        # Cập nhật vị trí hiện tại và danh sách đã thăm
        if self.current_states:
            self.current_position = self.current_states[0].position
            for state in self.current_states:
                if state.position not in self.visited:
                    self.visited.append(state.position)
        
        return False 