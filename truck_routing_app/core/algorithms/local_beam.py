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
    
    def __init__(self, grid: np.ndarray, beam_width: int = 10, initial_money: float = None,
                 max_fuel: float = None, fuel_per_move: float = None, 
                 gas_station_cost: float = None, toll_base_cost: float = None,
                 initial_fuel: float = None):
        """Initialize Local Beam Search with a grid, beam width and configuration parameters."""
        super().__init__(grid, initial_money, max_fuel, fuel_per_move, 
                         gas_station_cost, toll_base_cost, initial_fuel)
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
        neighbors = []  # Initialize neighbors list
        
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
                # Kiểm tra vị trí đã thăm
                if next_pos in visited:
                    continue
                
                # Tính chi phí và nhiên liệu mới
                new_fuel, move_cost, new_money = self.calculate_cost(state, next_pos)
                
                # Kiểm tra ràng buộc
                if new_fuel < 0 or new_money < 0:
                    continue
                
                # Tạo trạng thái mới
                new_state = SearchState(
                    position=next_pos,
                    fuel=new_fuel,
                    total_cost=state.total_cost + move_cost,
                    money=new_money,
                    path=state.path + [next_pos],
                    visited_gas_stations=state.visited_gas_stations.copy(),
                    toll_stations_visited=state.toll_stations_visited.copy()
                )
                
                # Cập nhật các tập đã thăm
                if self.grid[next_pos[1], next_pos[0]] == 2:  # Trạm xăng (loại 2)
                    new_state.visited_gas_stations.add(next_pos)
                elif self.grid[next_pos[1], next_pos[0]] == 1:  # Trạm thu phí (loại 1)
                    new_state.toll_stations_visited.add(next_pos)
                
                neighbors.append(new_state)
        
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
            # Kiểm tra vị trí đã thăm
            if next_pos in self.visited:
                continue
                
            # Tính toán chi phí và nhiên liệu cho bước tiếp theo
            new_fuel, move_cost, new_money = self.calculate_cost(current_state, next_pos)
            
            # Kiểm tra tính khả thi của bước tiếp theo
            if new_fuel < 0 or new_money < 0 or current_state.total_cost + move_cost > self.MAX_TOTAL_COST:
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
            
            # Cập nhật các tập đã thăm
            if self.grid[next_pos[1], next_pos[0]] == 2:  # Trạm xăng (loại 2)
                new_state.visited_gas_stations.add(next_pos)
            elif self.grid[next_pos[1], next_pos[0]] == 1:  # Trạm thu phí (loại 1)
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
        """Thực hiện thuật toán Local Beam Search từ vị trí bắt đầu đến đích.
        Theo mã giả cung cấp bởi người dùng.
        """
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
        self.current_money = self.MAX_MONEY
        
        # Khởi tạo chùm tia với trạng thái ban đầu
        initial_state = SearchState(
            position=start,
            fuel=self.MAX_FUEL,
            total_cost=0,
            money=self.MAX_MONEY,
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
        processed_states.add(initial_state.get_state_key())
        
        # Vòng lặp cho đến khi tìm thấy đích hoặc không thể đi tiếp
        while self.current_states and self.steps < 2000:  # Giới hạn số bước
            self.steps += 1
            next_generation_states = []
            goal_found_and_validated = False

            for state in self.current_states:
                # Nếu đã tìm thấy đích
                if state.position == self.goal:
                    raw_path = state.path

                    # First, validate and clean this path
                    validated_path = self.validate_path_no_obstacles(raw_path)

                    if not validated_path or len(validated_path) < 2:
                        print(f"LOCAL BEAM: Path to goal became invalid or too short after validation. Continuing search.")
                        continue # Try other states in beam or next generation

                    # Second, check overall feasibility of the validated path
                    is_still_feasible, reason = self.is_path_feasible(validated_path, self.MAX_FUEL)
                    if not is_still_feasible:
                        print(f"LOCAL BEAM: Path to goal after validation is not feasible: {reason}. Continuing search.")
                        continue # Try other states in beam or next generation
                    
                    # If all checks pass, this is our definitive path
                    self.current_path = validated_path
                    self.path_length = len(self.current_path) - 1
                    
                    # Recalculate all statistics on the final, validated path
                    self.calculate_path_fuel_consumption(self.current_path)
                    # self.cost, self.current_fuel, etc. are updated by the above call.
                    
                    print(f"Local Beam Search: Đã tìm thấy đường đi đến đích sau {self.steps} bước.")
                    goal_found_and_validated = True
                    return self.current_path # Return the first validated path to goal

                # Lấy các trạng thái lân cận
                successors = self.get_neighbor_states(state)
                
                # Thêm vào danh sách các trạng thái kế tiếp
                next_generation_states.extend(successors)
            
            # Nếu không còn trạng thái nào trong chùm tia, dừng lại
            if not self.current_states:
                print("Local Beam Search: Không còn trạng thái nào để khám phá.")
                break
            
            # Lọc bỏ các trạng thái lân cận đã xử lý đầy đủ
            valid_neighbors = []
            for neighbor in next_generation_states:
                if neighbor.get_state_key() not in processed_states:
                    valid_neighbors.append(neighbor)
                    processed_states.add(neighbor.get_state_key())
            
            # Nếu không còn trạng thái lân cận hợp lệ
            if not valid_neighbors:
                break
            
            # Chọn BeamWidth trạng thái tiếp theo theo phương pháp
            if self.use_stochastic:
                # Theo phân phối xác suất (Stochastic)
                self.current_states = self.select_states_stochastic(valid_neighbors, self.goal, self.beam_width)
            else:
                # Chọn BeamWidth trạng thái tốt nhất theo heuristic (Deterministically)
                valid_neighbors.sort(key=lambda x: self.heuristic_with_fuel(x, self.goal))
                self.current_states = valid_neighbors[:self.beam_width]
            
            # Cập nhật các trạng thái đã thăm cho visualization
            for state in self.current_states:
                if state.position not in self.visited:
                    self.visited.append(state.position)
            
            # Cập nhật vị trí hiện tại cho visualization
            if self.current_states:
                self.current_position = self.current_states[0].position
        
        # Nếu vòng lặp kết thúc mà không tìm được đích (và trả về từ bên trong)
        print("Local Beam Search: Không tìm thấy đường đi đến đích hoặc không có đường đi hợp lệ.")
        self.current_path = []
        return []
    
    def step(self) -> bool:
        """Execute one step of Local Beam Search."""
        if not self.start or not self.goal:
            return True  # Finished (not initialized)
        
        # Khởi tạo chùm tia nếu chưa có
        if not hasattr(self, 'processed_states'):
            self.processed_states = set()
            
            if not self.current_states:
                # Tạo trạng thái ban đầu
                initial_state = SearchState(
                    position=self.start,
                    fuel=self.MAX_FUEL,
                    total_cost=0,
                    money=self.MAX_MONEY,
                    path=[self.start],
                    visited_gas_stations=set(),
                    toll_stations_visited=set()
                )
                
                self.current_states = [initial_state]
                self.processed_states.add(initial_state.get_state_key())
                self.visited.append(self.start)
                self.current_position = self.start
        
        # Nếu không còn trạng thái nào để xét
        if not self.current_states:
            self.current_position = None
            return True
        
        self.steps += 1
        
        # Kiểm tra nếu có trạng thái đạt đích
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
                    
                    # Tính toán lượng nhiên liệu tiêu thụ cho đường đi
                    self.calculate_path_fuel_consumption(self.current_path)
                    
                    self.current_states = []
                    self.current_position = None
                    return True
        
        # Tạo danh sách tất cả các trạng thái lân cận
        all_neighbors = []
        
        # Mở rộng tất cả các trạng thái trong chùm tia hiện tại
        for state in self.current_states:
            # Lấy tất cả trạng thái lân cận
            neighbors = self.get_neighbor_states(state)
            all_neighbors.extend(neighbors)
        
        # Nếu không có trạng thái lân cận nào
        if not all_neighbors:
            self.current_states = []
            self.current_position = None
            return True
        
        # Lọc bỏ các trạng thái lân cận đã xử lý
        valid_neighbors = []
        for neighbor in all_neighbors:
            if neighbor.get_state_key() not in self.processed_states:
                valid_neighbors.append(neighbor)
                self.processed_states.add(neighbor.get_state_key())
        
        # Nếu không còn trạng thái lân cận hợp lệ
        if not valid_neighbors:
            self.current_states = []
            self.current_position = None
            return True
        
        # Chọn BeamWidth trạng thái tiếp theo
        if self.use_stochastic:
            # Chọn theo phân phối xác suất (Stochastic)
            self.current_states = self.select_states_stochastic(valid_neighbors, self.goal, self.beam_width)
        else:
            # Chọn BeamWidth trạng thái tốt nhất theo heuristic
            valid_neighbors.sort(key=lambda x: self.heuristic_with_fuel(x, self.goal))
            self.current_states = valid_neighbors[:self.beam_width]
        
        # Cập nhật trạng thái đã thăm và vị trí hiện tại cho visualization
        for state in self.current_states:
            if state.position not in self.visited:
                self.visited.append(state.position)
        
        if self.current_states:
            self.current_position = self.current_states[0].position
            
        return False 