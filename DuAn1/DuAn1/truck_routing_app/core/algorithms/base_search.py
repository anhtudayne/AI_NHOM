"""
Base search algorithm module.
Provides abstract base classes for implementing various search algorithms.
"""

from abc import ABC, abstractmethod
from typing import List, Tuple, Dict, Optional, Set
from dataclasses import dataclass
from queue import PriorityQueue
import numpy as np
import math
from collections import deque

# Định nghĩa các hằng số loại ô
OBSTACLE_CELL = -1    # Ô chướng ngại vật
ROAD_CELL = 0         # Ô đường thường
TOLL_CELL = 1         # Ô trạm thu phí
GAS_STATION_CELL = 2  # Ô trạm xăng

@dataclass
class SearchState:
    """Trạng thái tìm kiếm bao gồm vị trí, nhiên liệu và chi phí"""
    position: Tuple[int, int]  # (x, y)
    fuel: float               # Lượng nhiên liệu còn lại (L)
    total_cost: float        # Tổng chi phí (đ)
    money: float             # Số tiền còn lại (đ)
    path: List[Tuple[int, int]]  # Đường đi đã qua
    visited_gas_stations: Set[Tuple[int, int]]  # Các trạm xăng đã thăm
    toll_stations_visited: Set[Tuple[int, int]]  # Các trạm thu phí đã qua
    
    def __lt__(self, other):
        # So sánh để sử dụng trong PriorityQueue
        # Ưu tiên theo tổng chi phí thấp hơn
        return self.total_cost < other.total_cost
    
    def __eq__(self, other):
        # So sánh bằng dựa trên vị trí và lượng nhiên liệu
        if not isinstance(other, SearchState):
            return False
        return (self.position == other.position and 
                abs(self.fuel - other.fuel) < 0.01)  # So sánh float với độ chính xác
    
    def __hash__(self):
        # Hash dựa trên vị trí và lượng nhiên liệu (làm tròn để tránh vấn đề với float)
        return hash((self.position, round(self.fuel, 2)))
    
    def get_state_key(self) -> Tuple[Tuple[int, int], float]:
        """Tạo khóa duy nhất cho trạng thái dựa trên vị trí và nhiên liệu"""
        return (self.position, round(self.fuel, 2))
    
    def copy(self) -> 'SearchState':
        """Tạo bản sao của trạng thái hiện tại"""
        return SearchState(
            position=self.position,
            fuel=self.fuel,
            total_cost=self.total_cost,
            money=self.money,
            path=self.path.copy(),
            visited_gas_stations=self.visited_gas_stations.copy(),
            toll_stations_visited=self.toll_stations_visited.copy()
        )
    
    def can_reach(self, target: Tuple[int, int], base_search: 'BaseSearch') -> bool:
        """Kiểm tra xem có thể đến được target không"""
        needed_fuel = base_search.estimate_fuel_needed(self.position, target)
        return self.fuel >= needed_fuel

class BaseSearch(ABC):
    """Abstract base class for search algorithms."""
    
    # Định nghĩa các loại ô
    OBSTACLE_CELL = -1    # Ô chướng ngại vật
    ROAD_CELL = 0         # Ô đường thường
    TOLL_CELL = 1         # Ô trạm thu phí
    GAS_STATION_CELL = 2  # Ô trạm xăng
    
    # Các giá trị mặc định cho hằng số chi phí và trọng số
    DEFAULT_FUEL_PER_MOVE = 0.4      # Nhiên liệu tiêu thụ mỗi bước (L)
    DEFAULT_MAX_FUEL = 20.0          # Mặc định dung tích bình xăng (L)
    DEFAULT_GAS_STATION_COST = 30.0  # Chi phí đổ xăng cơ bản (đ)
    DEFAULT_TOLL_BASE_COST = 150.0   # Chi phí cơ bản qua trạm thu phí (đ) - đã bao gồm cả phí phạt
    DEFAULT_TOLL_PENALTY = 0.0      # Phạt ban đầu khi qua trạm thu phí (đ) - đã bao gồm trong TOLL_BASE_COST
    DEFAULT_MAX_TOTAL_COST = 5000.0  # Chi phí tối đa cho phép (đ)
    DEFAULT_LOW_FUEL_THRESHOLD = 1.0  # Ngưỡng xăng thấp (L)
    DEFAULT_MAX_MONEY = 2000.0       # Số tiền tối đa ban đầu (đ)
    
    # Trọng số cho các loại ô
    ROAD_WEIGHT = 1.0        # Đường thông thường - chi phí thấp nhất
    TOLL_WEIGHT = 8.0        # Trạm thu phí - chi phí cao nhất do có phí và phạt
    GAS_WEIGHT = 3.0         # Trạm xăng - chi phí trung bình, cần thiết khi hết xăng
    OBSTACLE_WEIGHT = float('inf')  # Vật cản - không thể đi qua
    
    def __init__(self, grid: np.ndarray, initial_money: float = None, 
                 max_fuel: float = None, fuel_per_move: float = None, 
                 gas_station_cost: float = None, toll_base_cost: float = None,
                 initial_fuel: float = None):
        """Initialize the search algorithm with a grid and configuration parameters.
        
        Args:
            grid: The grid representation of the map
            initial_money: Optional initial money amount
            max_fuel: Optional maximum fuel capacity
            fuel_per_move: Optional fuel consumption per move
            gas_station_cost: Optional cost for refueling at gas stations per liter
            toll_base_cost: Optional base cost for passing through a toll station
            initial_fuel: Optional initial fuel amount (defaults to max_fuel)
        """
        self.grid = grid
        self.rows, self.cols = grid.shape
        self.visited = []  # Danh sách các vị trí đã thăm theo thứ tự
        self.visited_positions = set()  # Tập hợp các vị trí đã thăm (không trùng lặp)
        self.current_path = []
        self.current_position = None
        self.steps = 0
        self.path_length = 0
        self.cost = 0
        
        # Cài đặt các thông số có thể tùy chỉnh
        self.MAX_FUEL = max_fuel if max_fuel is not None else self.DEFAULT_MAX_FUEL
        self.FUEL_PER_MOVE = fuel_per_move if fuel_per_move is not None else self.DEFAULT_FUEL_PER_MOVE
        self.GAS_STATION_COST = gas_station_cost if gas_station_cost is not None else self.DEFAULT_GAS_STATION_COST
        self.TOLL_BASE_COST = toll_base_cost if toll_base_cost is not None else self.DEFAULT_TOLL_BASE_COST
        self.TOLL_PENALTY = self.DEFAULT_TOLL_PENALTY
        self.MAX_TOTAL_COST = self.DEFAULT_MAX_TOTAL_COST
        self.LOW_FUEL_THRESHOLD = self.DEFAULT_LOW_FUEL_THRESHOLD
        self.MAX_MONEY = initial_money if initial_money is not None else self.DEFAULT_MAX_MONEY
        
        # Nhiên liệu ban đầu
        self.current_fuel = initial_fuel if initial_fuel is not None else self.MAX_FUEL
        
        self.fuel_consumed = 0.0  # Lượng nhiên liệu tiêu thụ cho đường đi cuối cùng
        self.fuel_refilled = 0.0  # Lượng nhiên liệu đã đổ thêm tại các trạm xăng
        self.current_total_cost = 0  # Tổng chi phí (bao gồm cả trạm thu phí)
        self.current_fuel_cost = 0  # Chi phí xăng
        self.current_toll_cost = 0  # Chi phí trạm thu phí
        self.toll_stations_visited = set()  # Các trạm thu phí đã qua
        self.current_money = initial_money if initial_money is not None else self.MAX_MONEY  # Số tiền ban đầu
    
    def add_visited(self, pos: Tuple[int, int]):
        """Thêm một vị trí vào danh sách đã thăm, NẾU hợp lệ."""
        # KIỂM TRA BỔ SUNG: Chỉ thêm nếu không phải là ô chướng ngại vật và hợp lệ
        # 1. Kiểm tra trong giới hạn lưới
        if not (0 <= pos[0] < self.grid.shape[1] and 0 <= pos[1] < self.grid.shape[0]):
            # print(f"DEBUG: add_visited called with out-of-bounds pos {pos}. Skipping.")
            return
        
        # 2. Kiểm tra loại ô và chướng ngại vật
        try:
            cell_value = self.grid[pos[1], pos[0]]
            # Sử dụng các hằng số đã định nghĩa trong lớp BaseSearch
            if cell_value == self.OBSTACLE_CELL or \
               int(cell_value) == int(self.OBSTACLE_CELL) or \
               cell_value < 0 or \
               cell_value not in [self.ROAD_CELL, self.TOLL_CELL, self.GAS_STATION_CELL]:
                # print(f"DEBUG: add_visited called with invalid/obstacle pos {pos} (value: {cell_value}). Skipping.")
                return
        except IndexError:
            # print(f"DEBUG: add_visited called with pos {pos} causing IndexError. Skipping.")
            return 
        except Exception as e: # Bắt các lỗi tiềm ẩn khác khi truy cập/kiểm tra giá trị ô
            # print(f"DEBUG: add_visited: Error checking cell {pos}: {e}. Skipping.")
            return

        if pos not in self.visited_positions:
            self.visited_positions.add(pos)
            self.visited.append(pos)
    
    def get_visited(self) -> List[Tuple[int, int]]:
        """Get the list of visited positions in order."""
        return self.visited
    
    def get_visited_set(self) -> Set[Tuple[int, int]]:
        """Get the set of unique visited positions."""
        return self.visited_positions
    
    def get_statistics(self) -> Dict:
        """Get statistics about the algorithm's execution."""
        return {
            "steps": self.steps,
            "visited": len(self.visited_positions),  # Số ô đã thăm (không trùng lặp)
            "path_length": self.path_length,
            "cost": self.cost,
            "fuel": self.current_fuel,  # Nhiên liệu còn lại
            "fuel_consumed": self.fuel_consumed,  # Nhiên liệu tiêu thụ cho đường đi
            "fuel_refilled": self.fuel_refilled,  # Nhiên liệu đã đổ thêm tại các trạm xăng
            "total_cost": self.current_total_cost,
            "fuel_cost": self.current_fuel_cost,
            "toll_cost": self.current_toll_cost,
            "money": self.current_money  # Số tiền còn lại
        }
    
    @abstractmethod
    def search(self, start: Tuple[int, int], goal: Tuple[int, int]) -> List[Tuple[int, int]]:
        """Execute search algorithm from start to goal."""
        pass
    
    @abstractmethod
    def step(self) -> bool:
        """Execute one step of the algorithm. Returns True if finished."""
        pass
    
    def get_current_path(self) -> List[Tuple[int, int]]:
        """Get the current path found by the algorithm."""
        return self.current_path
    
    def get_current_position(self) -> Optional[Tuple[int, int]]:
        """Get the current position of the algorithm."""
        return self.current_position
    
    def is_finished(self) -> bool:
        """Check if the algorithm has finished."""
        return self.current_position is None
    
    def is_valid_position(self, pos: Tuple[int, int]) -> bool:
        """
        Kiểm tra xem một vị trí có nằm trong lưới không.
        
        Args:
            pos: Tuple (x, y) chỉ vị trí cần kiểm tra
            
        Returns:
            bool: True nếu vị trí nằm trong lưới, False nếu không
        """
        try:
            x, y = pos
            if not (isinstance(x, int) and isinstance(y, int)):
                return False
                
            # Lưu ý: Trong numpy, shape[0] là số hàng (y), shape[1] là số cột (x)
            return 0 <= x < self.grid.shape[1] and 0 <= y < self.grid.shape[0]
        except Exception as e:
            print(f"Error checking position validity for {pos}: {str(e)}")
            return False
    
    def get_neighbors(self, pos: Tuple[int, int]) -> List[Tuple[int, int]]:
        """
        Lấy các ô lân cận hợp lệ (tuyệt đối không bao gồm ô chướng ngại vật).
        Phương thức này áp dụng nhiều lớp kiểm tra để đảm bảo không có ô chướng ngại vật nào được bao gồm.
        """
        x, y = pos
        neighbors = []
        
        # Tạo danh sách các hướng di chuyển có thể (4 hướng: lên, phải, xuống, trái)
        directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
        
        for dx, dy in directions:
            new_x, new_y = x + dx, y + dy
            new_pos = (new_x, new_y)
            
            # KIỂM TRA 1: Đảm bảo ô mới nằm trong lưới
            if not self.is_valid_position(new_pos):
                continue
                
            # KIỂM TRA 2: Đảm bảo ô không phải chướng ngại vật (mức raw)
            try:
                cell_value = self.grid[new_y, new_x]
            except:
                print(f"WARNING: Cannot access grid at {new_pos}")
                continue
                
            if cell_value == self.OBSTACLE_CELL:
                continue
                
            # KIỂM TRA 3: Đảm bảo giá trị ô là hợp lệ (chuyển đổi sang int nếu cần)
            try:
                int_cell_value = int(cell_value)
                if int_cell_value < 0 or int_cell_value not in [self.ROAD_CELL, self.TOLL_CELL, self.GAS_STATION_CELL]:
                    continue
            except:
                # Nếu không thể chuyển đổi sang int, kiểm tra giá trị gốc
                if cell_value < 0 or cell_value not in [self.ROAD_CELL, self.TOLL_CELL, self.GAS_STATION_CELL]:
                    continue
                
            # Nếu vượt qua tất cả các kiểm tra, thêm vào danh sách kết quả
            neighbors.append(new_pos)
            
        return neighbors
    
    def calculate_cost(self, current: SearchState, next_pos: Tuple[int, int]) -> Tuple[float, float, float]:
        """Tính toán chi phí và nhiên liệu khi di chuyển đến ô tiếp theo
        Returns: (new_fuel, move_cost, new_money)"""
        new_fuel = current.fuel - self.FUEL_PER_MOVE
        move_cost = self.ROAD_WEIGHT  # Chi phí cơ bản cho mỗi bước di chuyển
        new_money = current.money  # Khởi tạo tiền mới bằng tiền hiện tại
        
        next_cell_type = self.grid[next_pos[1], next_pos[0]]
        
        if next_cell_type == self.GAS_STATION_CELL:  # Trạm xăng (sử dụng hằng số)
            if new_fuel < self.MAX_FUEL:
                fuel_needed = self.MAX_FUEL - new_fuel
                # Giảm chi phí khi nhiên liệu thấp để khuyến khích ghé trạm xăng
                if new_fuel < self.LOW_FUEL_THRESHOLD:
                    discount = 0.7  # Giảm 30% chi phí khi nhiên liệu thấp
                else:
                    discount = 1.0
                
                # Chi phí đổ xăng tỷ lệ với lượng xăng cần đổ
                gas_cost = (self.GAS_STATION_COST * (fuel_needed / self.MAX_FUEL) * discount)
                move_cost = gas_cost + self.GAS_WEIGHT
                
                # Kiểm tra nếu đủ tiền để đổ xăng
                if new_money >= gas_cost:
                    new_money -= gas_cost
                    new_fuel = self.MAX_FUEL
        
        elif next_cell_type == self.TOLL_CELL:  # Trạm thu phí (sử dụng hằng số)
            if next_pos not in current.toll_stations_visited:
                # Giảm phạt dựa trên số trạm đã đi qua (tối đa giảm 50%)
                visited_discount = min(0.5, len(current.toll_stations_visited) * 0.1)
                
                # Tính chi phí trạm thu phí
                toll_cost = self.TOLL_BASE_COST + (self.TOLL_PENALTY * (1.0 - visited_discount))
                move_cost = toll_cost + self.TOLL_WEIGHT
                
                # Kiểm tra nếu đủ tiền để trả phí
                if new_money >= toll_cost:
                    new_money -= toll_cost
        
        elif next_cell_type == self.OBSTACLE_CELL:  # Vật cản (sử dụng hằng số)
            move_cost = self.OBSTACLE_WEIGHT  # Không thể đi qua
            new_fuel = -1  # Coi như không thể đi qua vì gặp vật cản
        
        # Thêm ưu tiên cho trạm xăng khi nhiên liệu thấp
        if next_cell_type == self.GAS_STATION_CELL and new_fuel < self.LOW_FUEL_THRESHOLD:
            move_cost *= 0.5  # Giảm 50% chi phí để ưu tiên đi qua trạm xăng
        
        return new_fuel, move_cost, new_money
    
    def is_path_feasible(self, path: List[Tuple[int, int]], start_fuel: float) -> Tuple[bool, str]:
        """
        Kiểm tra nghiêm ngặt xem đường đi có khả thi không: 
        - Không có chướng ngại vật
        - Đường đi phải liên tục
        - Đủ nhiên liệu và tiền
        
        Args:
            path: Danh sách các tọa độ (x, y) tạo thành đường đi
            start_fuel: Lượng nhiên liệu ban đầu
            
        Returns:
            Tuple (is_feasible, reason) - (Có khả thi không, Lý do nếu không khả thi)
        """
        # Kiểm tra độ dài tối thiểu của đường đi
        if not path or len(path) < 2:
            return False, "Đường đi không hợp lệ (độ dài < 2)"
            
        # KIỂM TRA 1: Không chứa ô chướng ngại vật và trong giới hạn lưới
        for i, pos in enumerate(path):
            # Kiểm tra tọa độ trong phạm vi lưới
            if not (0 <= pos[0] < self.grid.shape[1] and 0 <= pos[1] < self.grid.shape[0]):
                return False, f"Đường đi chứa tọa độ nằm ngoài lưới tại vị trí {i}: {pos}"
            
            # Lấy giá trị ô và kiểm tra từng loại
            try:
                cell_value = self.grid[pos[1], pos[0]]
                
                # Kiểm tra 1.1: So sánh trực tiếp với OBSTACLE_CELL
                if cell_value == self.OBSTACLE_CELL:
                    return False, f"Đường đi qua vật cản tại vị trí {i}: {pos}"
                
                # Kiểm tra 1.2: So sánh chặt chẽ giá trị số nguyên
                if int(cell_value) == int(self.OBSTACLE_CELL):
                    return False, f"Đường đi qua vật cản (kiểm tra chặt chẽ) tại vị trí {i}: {pos}"
                
                # Kiểm tra 1.3: Đảm bảo ô là một trong các loại cho phép
                if cell_value not in [self.ROAD_CELL, self.TOLL_CELL, self.GAS_STATION_CELL]:
                    return False, f"Đường đi qua ô không hợp lệ loại {cell_value} tại vị trí {i}: {pos}"
                
                # Kiểm tra 1.4: Đảm bảo giá trị không âm
                if cell_value < 0:
                    return False, f"Đường đi qua ô có giá trị âm {cell_value} tại vị trí {i}: {pos}"
                
            except Exception as e:
                return False, f"Lỗi kiểm tra ô tại vị trí {i}: {pos} - {str(e)}"
        
        # KIỂM TRA 2: Tính liên tục của đường đi (các ô phải liền kề)
        for i in range(1, len(path)):
            prev_pos = path[i-1]
            curr_pos = path[i]
            
            # Tính Manhattan distance giữa hai điểm liên tiếp
            manhattan_dist = abs(prev_pos[0] - curr_pos[0]) + abs(prev_pos[1] - curr_pos[1])
            
            # Các ô liền kề phải có Manhattan distance = 1 (chỉ cho phép di chuyển 4 hướng)
            if manhattan_dist > 1:
                return False, f"Đường đi không liên tục giữa vị trí {i-1}: {prev_pos} và vị trí {i}: {curr_pos}"
            
            # Kiểm tra bổ sung cho khoảng cách bằng 1: phải đủ nhiên liệu để di chuyển
            if not self.is_direct_path_feasible(prev_pos, curr_pos, self.FUEL_PER_MOVE):
                return False, f"Không thể di chuyển trực tiếp từ {prev_pos} đến {curr_pos}"
        
        # KIỂM TRA 3: Đảm bảo đủ nhiên liệu và chi phí
        current_fuel = start_fuel
        total_cost = 0
        total_money = self.current_money if hasattr(self, 'current_money') and self.current_money is not None else self.MAX_MONEY
        visited_tolls = set()
        visited_gas_stations = set()
        
        for i in range(1, len(path)):
            # Tính chi phí nhiên liệu cho mỗi bước đi
            current_fuel -= self.FUEL_PER_MOVE
            
            # Kiểm tra nhiên liệu có đủ không
            if current_fuel < 0:
                return False, f"Hết nhiên liệu tại vị trí {i}: {path[i]}"
            
            # Xử lý trạm xăng và thu phí
            pos = path[i]
            cell_type = self.grid[pos[1], pos[0]]
            
            if cell_type == self.GAS_STATION_CELL:  # Trạm xăng
                if pos not in visited_gas_stations:
                    # Chỉ tính chi phí đổ xăng lần đầu đến trạm
                    visited_gas_stations.add(pos)
                    
                    # Đổ xăng nếu còn ít hơn 70% (không đổ nếu còn nhiều)
                    if current_fuel < self.MAX_FUEL * 0.7:
                        refill_amount = self.MAX_FUEL - current_fuel
                        gas_cost = refill_amount * self.GAS_STATION_COST
                        
                        # Kiểm tra đủ tiền không
                        if total_money < gas_cost:
                            return False, f"Không đủ tiền để đổ xăng tại vị trí {i}: {pos} (cần {gas_cost:.2f}, có {total_money:.2f})"
                        
                        total_cost += gas_cost
                        total_money -= gas_cost
                        current_fuel = self.MAX_FUEL
            
            elif cell_type == self.TOLL_CELL:  # Trạm thu phí
                if pos not in visited_tolls:
                    # Tính phí qua trạm thu phí
                    toll_cost = self.TOLL_BASE_COST
                    
                    # Kiểm tra đủ tiền không
                    if total_money < toll_cost:
                        return False, f"Không đủ tiền để qua trạm thu phí tại vị trí {i}: {pos} (cần {toll_cost:.2f}, có {total_money:.2f})"
                    
                    total_cost += toll_cost
                    total_money -= toll_cost
                    visited_tolls.add(pos)
        
        # Kiểm tra tổng chi phí có vượt quá giới hạn không
        if total_cost > self.MAX_TOTAL_COST:
            return False, f"Tổng chi phí {total_cost:.2f} vượt quá giới hạn {self.MAX_TOTAL_COST:.2f}"
        
        # Nếu vượt qua tất cả các kiểm tra, đường đi khả thi
        return True, "Đường đi khả thi"
    
    def calculate_path_fuel_consumption(self, path: List[Tuple[int, int]]) -> None:
        """Tính toán lượng nhiên liệu tiêu thụ và đổ thêm cho đường đi cuối cùng.
        
        Phương thức này tính chi tiết:
        - Lượng nhiên liệu tiêu thụ trên toàn bộ đường đi
        - Lượng nhiên liệu đã đổ thêm tại các trạm xăng
        - Chi phí xăng và trạm thu phí riêng biệt
        - Cập nhật các thuộc tính tương ứng của thuật toán
        """
        if not path or len(path) < 2:
            self.fuel_consumed = 0.0
            self.fuel_refilled = 0.0
            self.current_fuel_cost = 0.0
            self.current_toll_cost = 0.0
            return
            
        # Khởi tạo các biến
        current_fuel = self.MAX_FUEL  # Bắt đầu với bình đầy
        total_fuel_consumed = 0.0
        total_fuel_refilled = 0.0
        total_fuel_cost = 0.0  # Chi phí nhiên liệu
        total_toll_cost = 0.0  # Chi phí trạm thu phí
        visited_gas_stations = set()
        visited_toll_stations = set()
        current_money = self.MAX_MONEY if self.current_money is None else self.current_money
        
        # Duyệt từng bước trên đường đi
        for i in range(len(path) - 1):
            pos1, pos2 = path[i], path[i + 1]
            
            # Tiêu thụ nhiên liệu cho bước di chuyển này
            fuel_for_step = self.FUEL_PER_MOVE
            total_fuel_consumed += fuel_for_step
            current_fuel -= fuel_for_step
            
            # Kiểm tra xem đã đến trạm xăng chưa
            cell_type = self.grid[pos2[1], pos2[0]]
            if cell_type == self.GAS_STATION_CELL and pos2 not in visited_gas_stations:  # Trạm xăng
                # Tính lượng nhiên liệu cần đổ thêm
                fuel_needed = self.MAX_FUEL - current_fuel
                if fuel_needed > 0:
                    # Tính chi phí đổ xăng
                    if current_fuel < self.LOW_FUEL_THRESHOLD:
                        discount = 0.7  # Giảm 30% chi phí khi nhiên liệu thấp
                    else:
                        discount = 1.0
                    
                    # Chi phí đổ xăng tỷ lệ với lượng xăng cần đổ
                    gas_cost = (self.GAS_STATION_COST * (fuel_needed / self.MAX_FUEL) * discount)
                    
                    # Kiểm tra nếu đủ tiền để đổ xăng
                    if current_money >= gas_cost:
                        current_money -= gas_cost
                        total_fuel_cost += gas_cost  # Cập nhật chi phí nhiên liệu
                        total_fuel_refilled += fuel_needed
                        current_fuel = self.MAX_FUEL
                
                # Đánh dấu đã thăm trạm xăng này
                visited_gas_stations.add(pos2)
            
            # Kiểm tra xem đã đến trạm thu phí chưa
            elif cell_type == self.TOLL_CELL and pos2 not in visited_toll_stations:  # Trạm thu phí
                # Giảm phạt dựa trên số trạm đã đi qua (tối đa giảm 50%)
                visited_discount = min(0.5, len(visited_toll_stations) * 0.1)
                
                # Tính chi phí trạm thu phí
                toll_cost = self.TOLL_BASE_COST + (self.TOLL_PENALTY * (1.0 - visited_discount))
                
                # Kiểm tra nếu đủ tiền để trả phí
                if current_money >= toll_cost:
                    current_money -= toll_cost
                    total_toll_cost += toll_cost  # Cập nhật chi phí trạm thu phí
                    visited_toll_stations.add(pos2)
        
        # Cập nhật các thuộc tính của thuật toán
        self.fuel_consumed = total_fuel_consumed
        self.fuel_refilled = total_fuel_refilled
        self.current_fuel = current_fuel  # Cập nhật nhiên liệu còn lại sau khi đi xong
        self.current_money = current_money  # Cập nhật số tiền còn lại
        self.current_fuel_cost = total_fuel_cost  # Cập nhật chi phí nhiên liệu
        self.current_toll_cost = total_toll_cost  # Cập nhật chi phí trạm thu phí
        self.current_total_cost = total_fuel_cost + total_toll_cost  # Cập nhật tổng chi phí
        
        # Cập nhật biến cost tổng
        self.cost = self.current_total_cost
    
    def update_fuel_and_cost(self, pos: Tuple[int, int]):
        """Cập nhật nhiên liệu và chi phí khi di chuyển đến một vị trí."""
        # Giảm xăng khi di chuyển
        self.current_fuel -= self.FUEL_PER_MOVE
        
        cell_type = self.grid[pos[1], pos[0]]
        
        # Xử lý trạm xăng
        if cell_type == self.GAS_STATION_CELL:  # Trạm xăng
            fuel_needed = self.MAX_FUEL - self.current_fuel
            if fuel_needed > 0:
                # Tính chi phí đổ xăng dựa trên lượng xăng cần đổ
                if self.current_fuel < self.LOW_FUEL_THRESHOLD:
                    discount = 0.7
                else:
                    discount = 1.0
                gas_cost = (self.GAS_STATION_COST * (fuel_needed / self.MAX_FUEL) * discount)
                
                # Kiểm tra xem còn đủ tiền để đổ xăng không
                if hasattr(self, 'current_money') and self.current_money is not None:
                    if self.current_money >= gas_cost:
                        self.current_money -= gas_cost
                        self.current_fuel_cost += gas_cost  # Tăng chi phí nhiên liệu
                        self.current_fuel = self.MAX_FUEL
                        print(f"Đổ xăng tại {pos}: {gas_cost:.2f}đ, Nhiên liệu: {self.current_fuel:.2f}L, Tiền còn lại: {self.current_money:.2f}đ")
                else:
                    # Nếu không theo dõi tiền, chỉ cập nhật chi phí và nhiên liệu
                    self.current_fuel_cost += gas_cost  # Tăng chi phí nhiên liệu
                    self.current_fuel = self.MAX_FUEL
                    print(f"Đổ xăng tại {pos}: {gas_cost:.2f}đ, Nhiên liệu: {self.current_fuel:.2f}L")
        
        # Xử lý trạm thu phí
        elif cell_type == self.TOLL_CELL:  # Trạm thu phí
            if pos not in self.toll_stations_visited:
                # Giảm phạt theo số trạm đã qua
                visited_discount = min(0.5, len(self.toll_stations_visited) * 0.1)
                toll_cost = self.TOLL_BASE_COST + (self.TOLL_PENALTY * (1.0 - visited_discount))
                
                # Kiểm tra xem còn đủ tiền để trả phí không
                if hasattr(self, 'current_money') and self.current_money is not None:
                    if self.current_money >= toll_cost:
                        self.current_money -= toll_cost
                        self.current_toll_cost += toll_cost  # Tăng chi phí trạm thu phí
                        self.toll_stations_visited.add(pos)
                        print(f"Qua trạm thu phí tại {pos}: {toll_cost:.2f}đ, Tiền còn lại: {self.current_money:.2f}đ")
                else:
                    # Nếu không theo dõi tiền, chỉ cập nhật chi phí
                    self.current_toll_cost += toll_cost  # Tăng chi phí trạm thu phí
                    self.toll_stations_visited.add(pos)
                    print(f"Qua trạm thu phí tại {pos}: {toll_cost:.2f}đ")
        
        # Cập nhật tổng chi phí
        self.current_total_cost = self.current_fuel_cost + self.current_toll_cost
    
    def has_enough_fuel(self, pos: Tuple[int, int]) -> bool:
        """Kiểm tra xem có đủ xăng để di chuyển đến vị trí không."""
        return self.current_fuel >= self.FUEL_PER_MOVE
    
    def find_nearest_reachable_gas_station(self, current_state: SearchState) -> Optional[Tuple[int, int]]:
        """Tìm trạm xăng gần nhất có thể đến được từ trạng thái hiện tại
        
        Args:
            current_state: Trạng thái hiện tại
            
        Returns:
            Optional[Tuple[int, int]]: Tọa độ trạm xăng gần nhất hoặc None nếu không tìm thấy
        """
        from collections import deque
        
        # Khởi tạo hàng đợi BFS và tập đã thăm
        queue = deque([current_state.position])
        visited = {current_state.position}
        parent = {current_state.position: None}
        distance = {current_state.position: 0}
        max_distance = current_state.fuel / self.FUEL_PER_MOVE  # Số bước tối đa có thể đi
        
        # BFS để tìm trạm xăng gần nhất
        while queue:
            current_pos = queue.popleft()
            
            # Nếu đã đi quá xa so với lượng xăng hiện tại, bỏ qua
            if distance[current_pos] > max_distance:
                continue
            
            # Nếu vị trí hiện tại là trạm xăng, trả về vị trí này
            if self.grid[current_pos[1], current_pos[0]] == self.GAS_STATION_CELL:  # Trạm xăng (loại 2)
                return current_pos
            
            # Kiểm tra các ô lân cận
            for next_pos in self.get_neighbors(current_pos):
                # Nếu ô chưa được thăm, thêm vào hàng đợi
                if next_pos not in visited:
                    visited.add(next_pos)
                    queue.append(next_pos)
                    parent[next_pos] = current_pos
                    distance[next_pos] = distance[current_pos] + 1
        
        # Không tìm thấy trạm xăng trong phạm vi có thể đi được
        return None
    
    def estimate_fuel_needed(self, start: Tuple[int, int], end: Tuple[int, int]) -> float:
        """Ước tính nhiên liệu cần thiết để đi từ start đến end"""
        x1, y1 = start
        x2, y2 = end
        distance = abs(x1 - x2) + abs(y1 - y2)
        return distance * self.FUEL_PER_MOVE

    def is_direct_path_feasible(self, start: Tuple[int, int], end: Tuple[int, int], 
                            current_fuel: float) -> bool:
        """
        Kiểm tra xem có thể đi thẳng từ start đến end không mà không gặp chướng ngại vật.
        
        Phương thức này kiểm tra:
        1. Đủ nhiên liệu để di chuyển
        2. Không có chướng ngại vật trên đường đi
        3. Đường đi nằm trong phạm vi lưới
        """
        # Kiểm tra xem hai điểm có liền kề không (Manhattan distance = 1)
        manhattan_dist = abs(start[0] - end[0]) + abs(start[1] - end[1])
        
        # Nếu di chuyển chỉ 1 ô, kiểm tra đơn giản
        if manhattan_dist == 1:
        # Kiểm tra nhiên liệu
            if current_fuel < self.FUEL_PER_MOVE:
                return False
            
            # Kiểm tra trực tiếp ô đích có phải chướng ngại vật không
            try:
                cell_value = self.grid[end[1], end[0]]
                if cell_value == self.OBSTACLE_CELL or int(cell_value) == int(self.OBSTACLE_CELL) or cell_value < 0:
                    return False
            except:
                return False
            
            return True
        
        # Kiểm tra nhiên liệu cho di chuyển xa hơn
        if current_fuel < manhattan_dist * self.FUEL_PER_MOVE:
                    return False
        
        # Kiểm tra đường thẳng theo Bresenham's line algorithm
        x0, y0 = start
        x1, y1 = end
        
        # Đảm bảo tọa độ nằm trong lưới
        if not (0 <= x0 < self.grid.shape[1] and 0 <= y0 < self.grid.shape[0] and 
                0 <= x1 < self.grid.shape[1] and 0 <= y1 < self.grid.shape[0]):
            return False
        
        # Chỉ xử lý các trường hợp đường thẳng ngang hoặc dọc
        if x0 == x1:  # Đường thẳng đứng
            # Đảm bảo y0 < y1
            if y0 > y1:
                y0, y1 = y1, y0
            
            # Kiểm tra từng ô trên đường thẳng
            for y in range(y0 + 1, y1):
                try:
                    cell_value = self.grid[y, x0]
                    if (cell_value == self.OBSTACLE_CELL or 
                        int(cell_value) == int(self.OBSTACLE_CELL) or 
                        cell_value < 0 or
                        cell_value not in [self.ROAD_CELL, self.TOLL_CELL, self.GAS_STATION_CELL]):
                        return False
                except:
                    return False
                
            return True
            
        elif y0 == y1:  # Đường thẳng ngang
            # Đảm bảo x0 < x1
            if x0 > x1:
                x0, x1 = x1, x0
            
            # Kiểm tra từng ô trên đường thẳng
            for x in range(x0 + 1, x1):
                try:
                    cell_value = self.grid[y0, x]
                    if (cell_value == self.OBSTACLE_CELL or 
                        int(cell_value) == int(self.OBSTACLE_CELL) or 
                        cell_value < 0 or
                        cell_value not in [self.ROAD_CELL, self.TOLL_CELL, self.GAS_STATION_CELL]):
                        return False
                except:
                    return False
                
            return True
            
        # Không hỗ trợ di chuyển theo đường chéo hoặc đường phức tạp
        return False

    def validate_path_no_obstacles(self, path: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
        """
        Xác thực đường đi, loại bỏ các ô chướng ngại vật và đảm bảo tính liên tục.
        Nếu không thể tạo đường đi hợp lệ, trả về danh sách rỗng.
        
        Args:
            path: Danh sách các vị trí trên đường đi
            
        Returns:
            List[Tuple[int, int]]: Đường đi đã được xác thực và sửa chữa, hoặc [] nếu không hợp lệ
        """
        if not path:
            print("WARNING: validate_path_no_obstacles called with empty path")
            return []
            
        # Thống kê ban đầu
        print(f"VALIDATE: Checking path with {len(path)} points for obstacles")
        
        # Bước 1: Loại bỏ các ô chướng ngại vật và ô không hợp lệ BAN ĐẦU
        valid_path_initial_filter = []
        for pos in path:
            # Chặn các vị trí không hợp lệ rõ ràng
            try:
                x, y = pos
                if not (isinstance(x, int) and isinstance(y, int)):
                    print(f"WARNING: Non-integer coordinates at position {pos}")
                    continue
                    
                # Kiểm tra biên
                if not self.is_valid_position(pos):
                    print(f"WARNING: Position {pos} is outside grid boundaries")
                    continue
                
                # Kiểm tra kiểu ô một cách an toàn
                try:
                    cell_value = int(self.grid[pos[1], pos[0]])
                except ValueError:
                    # Nếu không thể chuyển đổi sang int, lấy giá trị gốc
                    cell_value = self.grid[pos[1], pos[0]]
                
                # Kiểm tra xem ô có phải là chướng ngại vật
                if cell_value == self.OBSTACLE_CELL or int(cell_value) == int(self.OBSTACLE_CELL) or cell_value < 0 or cell_value not in [self.ROAD_CELL, self.TOLL_CELL, self.GAS_STATION_CELL]:
                    print(f"WARNING: Obstacle or invalid cell detected at {pos}. Removing.")
                    continue
                
                valid_path_initial_filter.append(pos)
                    
            except Exception as e:
                print(f"ERROR at position {pos}: {str(e)}")
                continue
                
        # Kiểm tra sau bước lọc ban đầu
        if not valid_path_initial_filter or len(valid_path_initial_filter) < 2:
            print("ERROR: Path became invalid (too short or all obstacles) after initial filtering. Rejecting path.")
            return []  # TRẢ VỀ RỖNG nếu đường đi không còn hợp lệ
                
        # Bước 2: Đảm bảo tính liên tục của đường đi và sửa chữa nếu cần
        continuous_path = [valid_path_initial_filter[0]]
            
        for i in range(1, len(valid_path_initial_filter)):
            prev_pos = continuous_path[-1]
            curr_pos = valid_path_initial_filter[i]
                
            manhattan_dist = abs(prev_pos[0] - curr_pos[0]) + abs(prev_pos[1] - curr_pos[1])
                
            if manhattan_dist <= 1:
                continuous_path.append(curr_pos)
            else:
                # Nếu không liền kề, tìm đường nối KHÔNG chướng ngại vật
                print(f"WARNING: Discontinuity detected between {prev_pos} and {curr_pos}. Attempting to repair...")
                mini_path = self.find_mini_path(prev_pos, curr_pos)
                
                if mini_path:
                    # Nối đường đi (bỏ qua điểm đầu của mini_path vì đã có)
                    continuous_path.extend(mini_path[1:])
                    print(f"INFO: Found and inserted connecting path with {len(mini_path)} points.")
                else:
                    # Nếu KHÔNG tìm được đường nối KHÔNG chướng ngại vật -> đường đi không hợp lệ
                    print(f"ERROR: Could not find a valid, obstacle-free path between {prev_pos} and {curr_pos}. Path is invalid.")
                    return []  # TRẢ VỀ RỖNG: Đây là biện pháp triệt để
        
        valid_path_after_repair = continuous_path
        
        # Bước 3: KIỂM TRA CUỐI CÙNG để đảm bảo không còn chướng ngại vật hoặc đứt gãy

        # Kiểm tra 3.1: Không có chướng ngại vật sau sửa chữa
        final_obstacle_check_path = []
        for pos in valid_path_after_repair:
            try:
                cell_value = self.grid[pos[1], pos[0]]
                if cell_value == self.OBSTACLE_CELL or int(cell_value) == int(self.OBSTACLE_CELL) or cell_value < 0 or cell_value not in [self.ROAD_CELL, self.TOLL_CELL, self.GAS_STATION_CELL]:
                    print(f"ERROR: Final validated path contains an obstacle or invalid cell at {pos} after repair. Rejecting path.")
                    return []  # TRẢ VỀ RỖNG nếu phát hiện chướng ngại vật cuối cùng
                final_obstacle_check_path.append(pos)
            except Exception as e:
                print(f"Error during final obstacle check at {pos}: {str(e)}. Rejecting path.")
                return []  # TRẢ VỀ RỖNG nếu có lỗi

        valid_path_final = final_obstacle_check_path

        # Kiểm tra 3.2: Tính liên tục cuối cùng sau sửa chữa
        if valid_path_final and len(valid_path_final) >= 2:
            for i in range(1, len(valid_path_final)):
                prev_pos = valid_path_final[i-1]
                curr_pos = valid_path_final[i]
                if abs(prev_pos[0] - curr_pos[0]) + abs(prev_pos[1] - curr_pos[1]) > 1:
                    print(f"ERROR: Final validated path is still discontinuous between {prev_pos} and {curr_pos}. Rejecting path.")
                    return []  # TRẢ VỀ RỖNG nếu vẫn còn đứt gãy

        elif valid_path_final and len(valid_path_final) < 2:
             # Nếu sau tất cả chỉ còn 0 hoặc 1 điểm, đường đi không hợp lệ
             print("ERROR: Path became too short after all validation and repair. Rejecting path.")
             return []  # TRẢ VỀ RỖNG nếu quá ngắn

        print(f"VALIDATE: Path successfully validated. Final path length: {len(valid_path_final)}.")
        return valid_path_final  # Trả về đường đi đã được xác thực và đảm bảo triệt để

    def find_mini_path(self, start: Tuple[int, int], end: Tuple[int, int]) -> List[Tuple[int, int]]:
        """
        Tìm đường đi ngắn nhất giữa hai điểm không liền kề, tránh ô chướng ngại vật.
        Sử dụng thuật toán BFS đơn giản.
        """
        if start == end:
            return [start]
        
        # Sử dụng BFS để tìm đường đi
        queue = deque([(start, [start])])
        visited = set([start])
        
        while queue:
            (current, path) = queue.popleft()
            
            # Lấy các ô liền kề hợp lệ (không có chướng ngại vật)
            for next_pos in self.get_neighbors(current):
                if next_pos == end:
                    # Đã tìm thấy đích
                    return path + [end]
                
                if next_pos not in visited:
                    visited.add(next_pos)
                    queue.append((next_pos, path + [next_pos]))
                    
                    # Giới hạn tìm kiếm để tránh tràn bộ nhớ
                    if len(visited) > 1000:
                        return None
        
        # Không tìm thấy đường đi
        return None

class BasePathFinder:
    """Lớp cơ sở cho thuật toán tìm đường với ràng buộc nhiên liệu"""
    
    # Định nghĩa các loại ô (THÊM MỚI ĐỂ NHẤT QUÁN)
    OBSTACLE_CELL = -1    # Ô chướng ngại vật
    ROAD_CELL = 0         # Ô đường thường
    TOLL_CELL = 1         # Ô trạm thu phí
    GAS_STATION_CELL = 2  # Ô trạm xăng
    
    # Các hằng số
    FUEL_PER_MOVE = 0.5      # Nhiên liệu tiêu thụ mỗi bước (L)
    MAX_FUEL = 20.0          # Dung tích bình xăng tối đa (L)
    GAS_STATION_COST = 30.0  # Chi phí đổ xăng (đ)
    TOLL_COST = 5.0          # Chi phí qua trạm thu phí (đ)
    TOLL_PENALTY = 1000.0    # Phạt cho việc đi qua trạm thu phí
    
    def __init__(self, grid: List[List[str]]):
        """
        Khởi tạo thuật toán với bản đồ
        grid: List[List[str]] - Bản đồ với các ký tự:
            '.' - Ô trống
            'G' - Trạm xăng
            'T' - Trạm thu phí
            'S' - Điểm bắt đầu
            'E' - Điểm kết thúc
        """
        self.grid = grid
        self.height = len(grid)
        self.width = len(grid[0]) if self.height > 0 else 0
        
        # Tìm tất cả các trạm xăng trên bản đồ
        self.gas_stations = set()
        for y in range(self.height):
            for x in range(self.width):
                if grid[y][x] == 'G':
                    self.gas_stations.add((x, y))
    
    def get_neighbors(self, pos: Tuple[int, int]) -> List[Tuple[int, int]]:
        """Lấy danh sách các ô lân cận có thể di chuyển tới"""
        x, y = pos
        neighbors = []
        for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0)]:  # 4 hướng
            new_x, new_y = x + dx, y + dy
            if 0 <= new_x < self.width and 0 <= new_y < self.height:
                # CRITICAL: Check that the cell is not an obstacle (-1)
                # SỬA ĐỔI: Sử dụng hằng số của lớp
                if self.grid[new_y][new_x] != self.OBSTACLE_CELL:  # Not an obstacle
                    neighbors.append((new_x, new_y))
        return neighbors
    
    def calculate_cost(self, current: SearchState, next_pos: Tuple[int, int]) -> Tuple[float, float]:
        """Tính toán chi phí và nhiên liệu khi di chuyển đến ô tiếp theo
        Returns: (new_fuel, move_cost)"""
        new_fuel = current.fuel - self.FUEL_PER_MOVE
        move_cost = 0.0
        
        next_cell_type = self.grid[next_pos[1], next_pos[0]]  # Lấy giá trị số nguyên
        
        if next_cell_type == self.GAS_STATION_CELL:  # Trạm xăng (thay vì 'G', sử dụng hằng số)
            # Chỉ đổ xăng nếu chưa đổ tại trạm này và nhiên liệu chưa đầy
            if next_pos not in current.visited_gas_stations and new_fuel < self.MAX_FUEL:
                new_fuel = self.MAX_FUEL
                move_cost = self.GAS_STATION_COST
        elif next_cell_type == self.TOLL_CELL:  # Trạm thu phí (thay vì 'T', sử dụng hằng số)
            # Chỉ tính phí trạm thu phí nếu chưa đi qua trạm này
            if next_pos not in current.toll_stations_visited:
                move_cost = self.TOLL_COST + self.TOLL_PENALTY
            
        return new_fuel, move_cost
    
    def is_valid_state(self, state: SearchState) -> bool:
        """Kiểm tra trạng thái có hợp lệ không (đủ nhiên liệu)"""
        return state.fuel >= self.FUEL_PER_MOVE
    
    def find_path(self, start: Tuple[int, int], end: Tuple[int, int]) -> List[Tuple[int, int]]:
        """
        Tìm đường đi từ điểm bắt đầu đến điểm kết thúc
        Phương thức này sẽ được triển khai bởi các lớp con cụ thể
        """
        raise NotImplementedError("Phương thức này cần được triển khai bởi lớp con")

    def optimize_path(self, path: List[Tuple[int, int]], start_fuel: float) -> List[Tuple[int, int]]:
        """
        Tối ưu hóa đường đi bằng cách cân nhắc các yếu tố:
        - Trạm xăng: Đảm bảo đủ nhiên liệu
        - Trạm thu phí: Tối ưu chi phí
        - Nhiên liệu: Đảm bảo an toàn
        """
        if len(path) <= 2:
            return path
            
        optimized = [path[0]]
        current_fuel = start_fuel
        visited_gas_stations = set()
        toll_stations_visited = set()
        total_cost = 0
        
        for i in range(1, len(path) - 1):
            prev_pos = optimized[-1]
            current_pos = path[i]
            next_pos = path[i + 1]
            
            # Lấy loại ô hiện tại
            current_cell_type = self.grid[current_pos[1], current_pos[0]]
            
            # Xử lý trạm xăng
            if current_cell_type == self.GAS_STATION_CELL:  # Trạm xăng
                # Kiểm tra xem có cần nạp nhiên liệu không
                if current_fuel < self.MAX_FUEL * 0.3:  # Nếu nhiên liệu dưới 30%
                    optimized.append(current_pos)
                    visited_gas_stations.add(current_pos)
                    current_fuel = self.MAX_FUEL  # Nạp đầy nhiên liệu
                    total_cost += self.GAS_STATION_COST
                    continue
                    
            # Xử lý trạm thu phí
            elif current_cell_type == self.TOLL_CELL:  # Trạm thu phí
                # Tính chi phí khi đi qua trạm thu phí
                toll_cost = self.TOLL_COST + self.TOLL_PENALTY / (len(toll_stations_visited) + 1)
                
                # Kiểm tra xem có đường đi thay thế không
                if self.has_alternative_path(prev_pos, next_pos, current_fuel, toll_cost):
                    continue
                    
                optimized.append(current_pos)
                toll_stations_visited.add(current_pos)
                total_cost += toll_cost
                continue
                
            # Kiểm tra đường đi trực tiếp
            if self.is_direct_path_feasible(prev_pos, next_pos, current_fuel):
                continue
                
            optimized.append(current_pos)
            current_fuel -= self.FUEL_PER_MOVE
            
        optimized.append(path[-1])
        return optimized
        
    def has_alternative_path(self, start: Tuple[int, int], end: Tuple[int, int], 
                            current_fuel: float, toll_cost: float) -> bool:
        """
        Kiểm tra xem có đường đi thay thế tốt hơn không
        """
        # Tìm đường đi thay thế không qua trạm thu phí
        alternative_path = self.find_alternative_path(start, end)
        if not alternative_path:
            return False
            
        # Tính chi phí của đường đi thay thế
        alt_cost = self.calculate_path_cost(alternative_path, current_fuel)
        
        # So sánh chi phí
        return alt_cost < toll_cost
        
    def find_alternative_path(self, start: Tuple[int, int], end: Tuple[int, int]) -> Optional[List[Tuple[int, int]]]:
        """
        Tìm đường đi thay thế không qua trạm thu phí
        """
        # Sử dụng BFS để tìm đường đi thay thế
        queue = deque([start])
        visited = {start}
        parent = {start: None}
        
        while queue:
            current = queue.popleft()
            
            if current == end:
                # Tạo đường đi từ parent
                path = []
                while current:
                    path.append(current)
                    current = parent[current]
                return path[::-1]
                
            for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                next_pos = (current[0] + dx, current[1] + dy)
                
                if (0 <= next_pos[0] < self.width and 
                    0 <= next_pos[1] < self.height and 
                    next_pos not in visited and 
                    self.grid[next_pos[1], next_pos[0]] != 1):    # Không phải trạm thu phí
                    
                    queue.append(next_pos)
                    visited.add(next_pos)
                    parent[next_pos] = current
                    
        return None
        
    def calculate_path_cost(self, path: List[Tuple[int, int]], start_fuel: float) -> float:
        """
        Tính tổng chi phí của đường đi
        """
        current_fuel = start_fuel
        total_cost = 0
        visited_gas_stations = set()
        toll_stations_visited = set()
        
        for i in range(len(path) - 1):
            current_pos = path[i]
            next_pos = path[i + 1]
            
            state = SearchState(
                position=current_pos,
                fuel=current_fuel,
                total_cost=total_cost,
                path=path[:i+1],
                visited_gas_stations=visited_gas_stations,
                toll_stations_visited=toll_stations_visited
            )
            
            current_fuel, move_cost = self.calculate_cost(state, next_pos)
            total_cost += move_cost
            
            if self.grid[next_pos[1], next_pos[0]] == 2:
                visited_gas_stations.add(next_pos)
            elif self.grid[next_pos[1], next_pos[0]] == 1:
                toll_stations_visited.add(next_pos)
                
        return total_cost
        
    def is_direct_path_feasible(self, start: Tuple[int, int], end: Tuple[int, int], 
                            current_fuel: float) -> bool:
        """
        Kiểm tra xem có thể đi thẳng từ start đến end không mà không gặp chướng ngại vật.
        
        Phương thức này kiểm tra:
        1. Đủ nhiên liệu để di chuyển
        2. Không có chướng ngại vật trên đường đi
        3. Đường đi nằm trong phạm vi lưới
        """
        # Kiểm tra xem hai điểm có liền kề không (Manhattan distance = 1)
        manhattan_dist = abs(start[0] - end[0]) + abs(start[1] - end[1])
        
        # Nếu di chuyển chỉ 1 ô, kiểm tra đơn giản
        if manhattan_dist == 1:
        # Kiểm tra nhiên liệu
            if current_fuel < self.FUEL_PER_MOVE:
                return False
            
            # Kiểm tra trực tiếp ô đích có phải chướng ngại vật không
            try:
                cell_value = self.grid[end[1], end[0]]
                if cell_value == self.OBSTACLE_CELL or int(cell_value) == int(self.OBSTACLE_CELL) or cell_value < 0:
                    return False
            except:
                return False
            
            return True
        
        # Kiểm tra nhiên liệu cho di chuyển xa hơn
        if current_fuel < manhattan_dist * self.FUEL_PER_MOVE:
                    return False
        
        # Kiểm tra đường thẳng theo Bresenham's line algorithm
        x0, y0 = start
        x1, y1 = end
        
        # Đảm bảo tọa độ nằm trong lưới
        if not (0 <= x0 < self.grid.shape[1] and 0 <= y0 < self.grid.shape[0] and 
                0 <= x1 < self.grid.shape[1] and 0 <= y1 < self.grid.shape[0]):
            return False
        
        # Chỉ xử lý các trường hợp đường thẳng ngang hoặc dọc
        if x0 == x1:  # Đường thẳng đứng
            # Đảm bảo y0 < y1
            if y0 > y1:
                y0, y1 = y1, y0
            
            # Kiểm tra từng ô trên đường thẳng
            for y in range(y0 + 1, y1):
                try:
                    cell_value = self.grid[y, x0]
                    if (cell_value == self.OBSTACLE_CELL or 
                        int(cell_value) == int(self.OBSTACLE_CELL) or 
                        cell_value < 0 or
                        cell_value not in [self.ROAD_CELL, self.TOLL_CELL, self.GAS_STATION_CELL]):
                        return False
                except:
                    return False
                
            return True
            
        elif y0 == y1:  # Đường thẳng ngang
            # Đảm bảo x0 < x1
            if x0 > x1:
                x0, x1 = x1, x0
            
            # Kiểm tra từng ô trên đường thẳng
            for x in range(x0 + 1, x1):
                try:
                    cell_value = self.grid[y0, x]
                    if (cell_value == self.OBSTACLE_CELL or 
                        int(cell_value) == int(self.OBSTACLE_CELL) or 
                        cell_value < 0 or
                        cell_value not in [self.ROAD_CELL, self.TOLL_CELL, self.GAS_STATION_CELL]):
                        return False
                except:
                    return False
                
            return True
            
        # Không hỗ trợ di chuyển theo đường chéo hoặc đường phức tạp
        return False