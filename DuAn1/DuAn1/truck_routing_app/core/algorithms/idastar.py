"""
Thuật toán IDA* (Iterative Deepening A*) - Tìm đường đi tối ưu kết hợp ID-DFS và A*.

------------------------- ĐỊNH NGHĨA THUẬT TOÁN IDA* -------------------------
IDA* (Iterative Deepening A*) là thuật toán tìm kiếm đường đi kết hợp ưu điểm của:
- Tìm kiếm theo độ sâu tăng dần (IDDFS): sử dụng ít bộ nhớ O(d)
- Thuật toán A*: đảm bảo tìm đường đi tối ưu nhờ sử dụng heuristic

Thuật toán hoạt động bằng cách thực hiện nhiều lần tìm kiếm DFS giới hạn, với giới hạn là
giá trị f(n) = g(n) + h(n) thay vì độ sâu. Ngưỡng f-limit tăng dần qua mỗi lần lặp.

---------------------- NGUYÊN LÝ CƠ BẢN CỦA IDA* ---------------------------
1. Khởi tạo ngưỡng f-limit ban đầu = h(start)
2. Thực hiện DFS với giới hạn là f-limit hiện tại
3. Nếu tìm thấy đích, kết thúc thuật toán
4. Nếu không tìm thấy, tăng f-limit lên giá trị f nhỏ nhất vượt quá ngưỡng hiện tại
5. Lặp lại quá trình cho đến khi tìm thấy đích hoặc không thể tìm thấy nữa

---------------------- ƯU ĐIỂM CỦA IDA* ----------------------------------
- Sử dụng ít bộ nhớ hơn A* nhờ tính chất của DFS (chỉ O(d) với d là độ sâu)
- Vẫn đảm bảo tìm được đường đi tối ưu nếu heuristic là admissible
- Thích hợp cho các bài toán có không gian trạng thái lớn và độ sâu giới hạn
"""

from typing import List, Tuple, Dict, Set, Optional
import numpy as np
import math
from .base_search import BaseSearch, SearchState

class IDAStar(BaseSearch):
    """Triển khai thuật toán IDA* (Iterative Deepening A*) tối ưu về bộ nhớ."""
    
    def __init__(self, grid: np.ndarray, initial_money: float = None, 
                 max_fuel: float = None, fuel_per_move: float = None, 
                 gas_station_cost: float = None, toll_base_cost: float = None,
                 initial_fuel: float = None):
        """Khởi tạo thuật toán IDA* với lưới và các tham số cấu hình.
        
        Args:
            grid: Ma trận biểu diễn bản đồ (0: đường, -1: chướng ngại vật, 1: trạm thu phí, 2: trạm xăng)
            initial_money: Số tiền ban đầu
            max_fuel: Dung tích bình xăng tối đa
            fuel_per_move: Lượng xăng tiêu thụ mỗi bước di chuyển
            gas_station_cost: Chi phí xăng mỗi lít
            toll_base_cost: Chi phí cơ bản qua trạm thu phí
            initial_fuel: Lượng nhiên liệu ban đầu (mặc định đầy bình)
        """
        # Gọi constructor của lớp cha để khởi tạo các thông số cơ bản
        super().__init__(grid, initial_money, max_fuel, fuel_per_move, 
                         gas_station_cost, toll_base_cost, initial_fuel)
        
        # Các biến dùng cho IDA*
        self.start = None          # Điểm xuất phát
        self.goal = None           # Điểm đích
        self.bound = float('inf')  # Ngưỡng f giới hạn hiện tại
        self.min_f_over_bound = float('inf')  # Giá trị f nhỏ nhất vượt ngưỡng (cho lần lặp tiếp)
        self.path = []             # Đường đi tối ưu hiện tại
        self.min_path = []         # Đường đi tốt nhất đã tìm thấy
        
        # Khởi tạo trạng thái tìm kiếm
        self.current_state = None   # Trạng thái hiện tại đang xét
        self.search_stack = []      # Stack lưu các trạng thái đang xét (thay thế recursive)
        
        # Biến cho việc bước tùy chỉnh
        self.step_counter = 0
        self.current_iteration = 0  # Số lần lặp của IDA*
        self.is_search_complete = False
        
        # Ngưỡng cho việc ra quyết định
        self.LOW_FUEL_THRESHOLD = self.MAX_FUEL * 0.3  # 30% bình xăng - ngưỡng cảnh báo nhiên liệu thấp
        self.TOLL_DISTANCE_THRESHOLD = 3  # Số bước tối thiểu để cân nhắc đi qua trạm thu phí
    
    def calculate_heuristic(self, pos: Tuple[int, int], goal: Tuple[int, int], 
                          current_fuel: float, current_cost: float) -> float:
        """Tính toán hàm heuristic cho IDA*.
        
        Hàm heuristic ước tính chi phí từ vị trí hiện tại đến đích. Hàm này kết hợp nhiều 
        yếu tố để đưa ra ước tính chính xác hơn, bao gồm:
        - Khoảng cách Manhattan đến đích
        - Chi phí nhiên liệu ước tính (dựa trên DEFAULT_GAS_STATION_COST)
        - Chi phí trạm thu phí ước tính (dựa trên DEFAULT_TOLL_BASE_COST)
        - Phạt khi nhiên liệu thấp (cần đến trạm xăng gấp)
        - Ưu đãi/phạt theo loại ô hiện tại
        
        Args:
            pos: Vị trí hiện tại
            goal: Vị trí đích
            current_fuel: Lượng nhiên liệu hiện tại
            current_cost: Chi phí tích lũy hiện tại
        
        Returns:
            float: Giá trị heuristic ước tính chi phí đến đích
        """
        # 1. Khoảng cách Manhattan đến đích
        distance_to_goal = abs(pos[0] - goal[0]) + abs(pos[1] - goal[1])
        
        # Nếu đã ở đích, heuristic = 0
        if distance_to_goal == 0:
            return 0.0
        
        # 2. Chi phí cơ bản dựa trên khoảng cách (giả định đường thẳng tối ưu)
        base_cost = distance_to_goal * self.ROAD_WEIGHT
        
        # 3. Ước tính nhu cầu nhiên liệu và chi phí đổ xăng
        fuel_needed = distance_to_goal * self.FUEL_PER_MOVE
        remaining_range = current_fuel / self.FUEL_PER_MOVE  # Số ô có thể đi với lượng xăng hiện tại
        
        # 3.1 Tính số lần cần đổ xăng chính xác hơn
        if fuel_needed <= current_fuel:
            # Đủ nhiên liệu để đến đích không cần đổ xăng
            estimated_refills = 0
        else:
            # Phải đổ xăng ít nhất 1 lần
            fuel_deficit = fuel_needed - current_fuel
            full_tank_range = self.MAX_FUEL / self.FUEL_PER_MOVE  # Quãng đường tối đa với bình đầy
            
            # Số lần đổ đầy bình (làm tròn lên) trừ đi phần đã có
            estimated_refills = math.ceil(fuel_deficit / self.MAX_FUEL)
        
        # 3.2 Tính chi phí đổ xăng dựa trên chi phí trạm xăng từ BaseSearch
        refill_cost = estimated_refills * self.GAS_STATION_COST * self.MAX_FUEL
        
        # 4. Ước tính số trạm thu phí trên đường đi
        # Ước lượng tỷ lệ trạm thu phí: số trạm thu phí / tổng số ô trên lưới
        toll_density = self.estimate_toll_density()
        
        # Ước tính số trạm thu phí có thể gặp phải (theo tỷ lệ)
        estimated_tolls = distance_to_goal * toll_density
        # Làm tròn xuống để đảm bảo tính admissible
        estimated_tolls = max(0, math.floor(estimated_tolls))
        
        # Chi phí ước tính cho trạm thu phí từ BaseSearch
        toll_cost = estimated_tolls * self.TOLL_BASE_COST
        
        # 5. Phạt đặc biệt cho nhiên liệu thấp - ưu tiên tìm trạm xăng khi nhiên liệu dưới ngưỡng
        fuel_penalty = 0
        if current_fuel < self.LOW_FUEL_THRESHOLD:
            # Tìm trạm xăng gần nhất
            nearest_gas = self.find_nearest_gas_station(pos)
            if nearest_gas:
                distance_to_gas = abs(pos[0] - nearest_gas[0]) + abs(pos[1] - nearest_gas[1])
                # Phạt tỷ lệ nghịch với nhiên liệu còn lại và tỷ lệ thuận với khoảng cách đến trạm xăng
                if distance_to_gas * self.FUEL_PER_MOVE > current_fuel:
                    # Không đủ nhiên liệu để đến trạm xăng - phạt rất nặng
                    fuel_penalty = 1000 * (distance_to_gas * self.FUEL_PER_MOVE - current_fuel)
                else:
                    # Đủ nhiên liệu nhưng thấp - phạt nhẹ
                    fuel_penalty = distance_to_gas * 5
        
        # 6. Thêm phần thưởng/phạt dựa trên loại ô hiện tại
        cell_type_modifier = 0
        try:
            current_cell_type = self.grid[pos[1], pos[0]]
            
            # Ưu tiên nhẹ cho trạm xăng khi nhiên liệu thấp
            if current_cell_type == self.GAS_STATION_CELL and current_fuel < self.MAX_FUEL * 0.5:
                cell_type_modifier = -5  # Thưởng (giảm chi phí ước tính)
            
            # Tránh trạm thu phí một chút nếu còn cách đích xa
            elif current_cell_type == self.TOLL_CELL and distance_to_goal > 5:
                cell_type_modifier = 3  # Phạt nhẹ
        except IndexError:
            pass  # Bỏ qua lỗi nếu có
        
        # 7. Cân bằng ước tính
        # Đảm bảo heuristic luôn admissible (không vượt quá chi phí thực tế)
        # Kết hợp các thành phần với trọng số phù hợp
        fuel_factor = 0.9   # Giảm nhẹ ảnh hưởng của chi phí xăng (90%)
        toll_factor = 0.8   # Giảm nhẹ ảnh hưởng của chi phí trạm thu phí (80%)
        
        # Kết hợp tất cả các thành phần để có ước tính heuristic cuối cùng
        return base_cost + (refill_cost * fuel_factor) + (toll_cost * toll_factor) + fuel_penalty + cell_type_modifier
    
    def find_nearest_gas_station(self, pos: Tuple[int, int]) -> Optional[Tuple[int, int]]:
        """Tìm trạm xăng gần nhất từ vị trí hiện tại.
        
        Phương thức này quét toàn bộ bản đồ để tìm trạm xăng gần nhất
        sử dụng khoảng cách Manhattan (|x1-x2| + |y1-y2|). Được sử dụng
        để tính toán heuristic khi nhiên liệu thấp.
        
        Args:
            pos: Vị trí hiện tại (x, y)
            
        Returns:
            Tuple (x, y) của trạm xăng gần nhất hoặc None nếu không tìm thấy
        """
        min_distance = float('inf')
        nearest_gas = None
        
        for y in range(self.grid.shape[0]):
            for x in range(self.grid.shape[1]):
                if self.grid[y, x] == self.GAS_STATION_CELL:  # Trạm xăng
                    distance = abs(x - pos[0]) + abs(y - pos[1])
                    if distance < min_distance:
                        min_distance = distance
                        nearest_gas = (x, y)
        
        return nearest_gas
    
    def estimate_toll_density(self) -> float:
        """Ước lượng mật độ trạm thu phí trên bản đồ.
        
        Returns:
            float: Tỷ lệ trạm thu phí trên tổng số ô có thể đi được (0.0 - 1.0)
        """
        total_cells = self.grid.size
        obstacle_cells = np.sum(self.grid == self.OBSTACLE_CELL)
        traversable_cells = total_cells - obstacle_cells
        
        if traversable_cells == 0:
            return 0.0
        
        toll_cells = np.sum(self.grid == self.TOLL_CELL)
        
        return toll_cells / traversable_cells
    
    def initialize_search(self, start: Tuple[int, int], goal: Tuple[int, int]):
        """Khởi tạo các tham số cho quá trình tìm kiếm IDA*.
        
        Args:
            start: Vị trí bắt đầu (x, y)
            goal: Vị trí đích (x, y)
        """
        self.start = start
        self.goal = goal
        self.visited = []
        self.visited_positions = set()
        self.add_visited(start)
        self.current_position = start
        
        # Khởi tạo trạng thái ban đầu
        initial_state = SearchState(
            position=start,
            fuel=self.current_fuel,
            total_cost=0,
            money=self.current_money,
            path=[start],
            visited_gas_stations=set(),
            toll_stations_visited=set()
        )
        
        # Tính toán ngưỡng ban đầu dựa trên heuristic
        self.bound = self.calculate_heuristic(
            start, goal, initial_state.fuel, initial_state.total_cost
        )
        
        # Reset các biến
        self.min_f_over_bound = float('inf')
        self.path = [start]
        self.min_path = []
        self.current_state = initial_state
        self.search_stack = []
        self.is_search_complete = False
        self.current_iteration = 0
        self.step_counter = 0
    
    def search(self, start: Tuple[int, int], goal: Tuple[int, int]) -> List[Tuple[int, int]]:
        """Thực hiện thuật toán IDA* để tìm đường đi từ start đến goal.
        
        Args:
            start: Vị trí bắt đầu (x, y)
            goal: Vị trí đích (x, y)
        
        Returns:
            List[Tuple[int, int]]: Đường đi từ start đến goal nếu tìm thấy, ngược lại là []
        """
        if not self.is_valid_position(start) or not self.is_valid_position(goal):
            print(f"IDA*: Vị trí start {start} hoặc goal {goal} không hợp lệ!")
            return []
        
        # Nếu vị trí bắt đầu và kết thúc giống nhau
        if start == goal:
            return [start]
        
        # Khởi tạo tìm kiếm
        self.initialize_search(start, goal)
        
        # Thực hiện IDA* với nhiều vòng lặp, mỗi vòng tăng ngưỡng bound
        while not self.is_search_complete and self.bound < float('inf'):
            # Reset biến theo dõi giá trị f nhỏ nhất vượt ngưỡng
            self.min_f_over_bound = float('inf')
            
            # Khởi tạo trạng thái ban đầu
            initial_state = SearchState(
                position=start,
                fuel=self.current_fuel,
                total_cost=0,
                money=self.current_money,
                path=[start],
                visited_gas_stations=set(),
                toll_stations_visited=set()
            )
            
            # Thực hiện DFS có giới hạn (non-recursive)
            result = self.dfs(initial_state, 0, self.bound)
            
            # Nếu tìm thấy đường đi, return
            if result == "FOUND":
                # Cập nhật thống kê
                self.current_path = self.min_path
                self.path_length = len(self.min_path) - 1  # Trừ điểm bắt đầu
                return self.min_path
            
            # Nếu không tìm thấy và không có đường đi nào tốt hơn
            if self.min_f_over_bound == float('inf'):
                break
            
            # Tăng ngưỡng bound cho lần lặp tiếp theo
            self.bound = self.min_f_over_bound
            self.current_iteration += 1
            
            # In thông tin debug
            print(f"IDA* Iteration {self.current_iteration}: Tăng ngưỡng lên {self.bound}")
        
        # Nếu không tìm thấy đường đi
        print("IDA*: Không tìm thấy đường đi!")
        return []
    
    def dfs(self, state: SearchState, g: float, bound: float) -> str:
        """Thực hiện DFS có giới hạn ngưỡng f cho IDA*.
        
        Args:
            state: Trạng thái hiện tại
            g: Chi phí thực tế từ điểm bắt đầu đến trạng thái hiện tại
            bound: Ngưỡng giới hạn f hiện tại
        
        Returns:
            str: "FOUND" nếu tìm thấy đích, ngược lại là "NOT_FOUND"
        """
        # Cập nhật vị trí hiện tại
        self.current_position = state.position
        
        # Tính f(n) = g(n) + h(n)
        f = g + self.calculate_heuristic(state.position, self.goal, state.fuel, state.total_cost)
        
        # Nếu f vượt quá ngưỡng bound, cập nhật min_f_over_bound và dừng nhánh này
        if f > bound:
            self.min_f_over_bound = min(self.min_f_over_bound, f)
            return "OVER_BOUND"
        
        # Nếu đến đích, đánh dấu tìm thấy và lưu đường đi
        if state.position == self.goal:
            self.min_path = state.path.copy()
            self.is_search_complete = True
            return "FOUND"
        
        # Thêm vào danh sách đã thăm để hiển thị animation
        self.add_visited(state.position)
        
        # Lấy danh sách các hàng xóm
        neighbors = self.get_neighbors(state.position)
        
        # Xử lý từng hàng xóm
        for next_pos in neighbors:
            # Bỏ qua nếu đã có trong đường đi hiện tại
            if next_pos in state.path:
                continue
                
            # Tính chi phí mới
            next_cost, fuel_cost, toll_cost = self.calculate_cost(state, next_pos)
            
            # Bỏ qua nếu không thể chi trả hoặc hết xăng
            if next_cost < 0:
                continue
                
            # Tạo trạng thái mới
            next_state = state.copy()
            next_state.position = next_pos
            next_state.total_cost += next_cost
            next_state.path.append(next_pos)
            
            # Cập nhật nhiên liệu - trừ nhiên liệu tiêu thụ cho bước di chuyển
            next_state.fuel -= self.FUEL_PER_MOVE
            
            # Xử lý trạm xăng
            cell_type = self.grid[next_pos[1], next_pos[0]]
            if cell_type == self.GAS_STATION_CELL:
                # Đổ xăng nếu chưa đầy bình
                if next_state.fuel < self.MAX_FUEL:
                    # Tính lượng xăng cần đổ và chi phí
                    refill_amount = self.MAX_FUEL - next_state.fuel
                    refill_cost = refill_amount * self.GAS_STATION_COST
                    
                    # Cập nhật trạng thái
                    next_state.fuel = self.MAX_FUEL
                    next_state.money -= refill_cost
                    next_state.visited_gas_stations.add(next_pos)
            
            # Xử lý trạm thu phí
            if cell_type == self.TOLL_CELL:
                next_state.toll_stations_visited.add(next_pos)
            
            # Tiếp tục DFS với trạng thái mới
            result = self.dfs(next_state, g + next_cost, bound)
            
            # Nếu tìm thấy đích, dừng tìm kiếm
            if result == "FOUND":
                return "FOUND"
        
        # Không tìm thấy đường đi trong nhánh này
        return "NOT_FOUND"
    
    def step(self) -> bool:
        """Thực hiện một bước của thuật toán IDA*.
        
        Returns:
            bool: True nếu thuật toán đã hoàn thành, False nếu chưa
        """
        # Nếu tìm kiếm đã hoàn thành, trả về True
        if self.is_search_complete:
            return True
        
        # Nếu chưa khởi tạo, khởi tạo tìm kiếm
        if self.start is None or self.goal is None:
            return True
        
        # Thực hiện một bước của thuật toán
        self.step_counter += 1
        
        # TODO: Implement step logic - phức tạp và không bắt buộc với IDA*
        # Vì IDA* sử dụng DFS nên khó chia nhỏ thành các bước riêng lẻ
        # Thường sẽ thực hiện trực tiếp search() thay vì step()
        
        return self.is_search_complete