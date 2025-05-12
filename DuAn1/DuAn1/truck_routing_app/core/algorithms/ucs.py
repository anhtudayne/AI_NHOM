"""
Uniform Cost Search algorithm implementation.

------------------------- ĐỊNH NGHĨA THUẬT TOÁN UCS -------------------------
UCS (Uniform Cost Search - Tìm kiếm theo chi phí đồng nhất) là thuật toán tìm đường
đi từ điểm bắt đầu đến đích dựa trên chi phí tích lũy. UCS luôn mở rộng nút có
chi phí tích lũy thấp nhất, do đó đảm bảo tìm được đường đi tối ưu về chi phí.

---------------------- NGUYÊN LÝ CƠ BẢN CỦA UCS ---------------------------
1. Sử dụng cấu trúc dữ liệu HÀNG ĐỢI ƯU TIÊN (priority queue)
2. Ban đầu, đặt nút bắt đầu vào hàng đợi với chi phí 0
3. Lặp lại cho đến khi hàng đợi rỗng:
   - Lấy phần tử có chi phí thấp nhất ra khỏi hàng đợi
   - Kiểm tra xem đó có phải là đích không
   - Nếu không, thêm tất cả các nút kề vào hàng đợi với chi phí tích lũy
4. Khi tìm thấy đích, dùng cơ chế truy vết để tạo đường đi

---------------- TRIỂN KHAI UCS TRONG CHƯƠNG TRÌNH NÀY --------------------
UCS trong chương trình này tính toán chi phí thực tế:
- Chi phí nhiên liệu tiêu thụ khi di chuyển
- Chi phí thu phí tại các trạm thu phí
- Chi phí đổ xăng tại trạm xăng (nếu cần thiết)
- Yếu tố hình phạt (penalty) khi nhiên liệu thấp

Thuật toán đảm bảo tìm đường đi có chi phí tổng thực tế thấp nhất, đồng thời
xem xét các ràng buộc về nhiên liệu và chi phí để đảm bảo đường đi khả thi.
"""

from typing import List, Tuple, Dict, Any
import numpy as np
from queue import PriorityQueue
from .base_search import BaseSearch, SearchState

class UCS(BaseSearch):
    """Uniform Cost Search algorithm implementation."""
    
    def __init__(self, grid: np.ndarray, initial_money: float = None, 
                 max_fuel: float = None, fuel_per_move: float = None, 
                 gas_station_cost: float = None, toll_base_cost: float = None,
                 initial_fuel: float = None):
        """Initialize UCS with a grid and configuration parameters."""
        super().__init__(grid, initial_money, max_fuel, fuel_per_move, 
                         gas_station_cost, toll_base_cost, initial_fuel)
        self.priority_queue = PriorityQueue()
        self.cost_so_far = {}  # Lưu chi phí từ start đến mỗi vị trí
        self.parent = {}  # Dictionary để truy vết đường đi
        self.start = None
        self.goal = None
        # Lưu trạng thái (fuel, money) tại mỗi vị trí
        self.state_at_position = {}
    
    def calculate_cost(self, current_pos: Tuple[int, int], next_pos: Tuple[int, int], current_fuel: float, current_money: float) -> Tuple[float, float, float, float, float]:
        """Tính toán chi phí di chuyển từ current_pos đến next_pos.
        
        Args:
            current_pos: Vị trí hiện tại
            next_pos: Vị trí tiếp theo
            current_fuel: Lượng nhiên liệu hiện có
            current_money: Số tiền hiện có
            
        Returns:
            Tuple (move_cost, fuel_cost, toll_cost, next_fuel, next_money)
        """
        # Giảm nhiên liệu khi di chuyển
        next_fuel = current_fuel - self.FUEL_PER_MOVE
        next_money = current_money
        
        # Chi phí cơ bản cho mỗi bước di chuyển
        move_cost = self.ROAD_WEIGHT
        fuel_cost = 0.0
        toll_cost = 0.0
        
        # Lấy loại ô tiếp theo
        next_cell_type = self.grid[next_pos[1], next_pos[0]]
        
        # Tính chi phí dựa trên loại ô
        if next_cell_type == self.GAS_STATION_CELL:  # Trạm xăng
            if next_fuel < self.MAX_FUEL:
                # Tính lượng xăng cần đổ
                fuel_needed = self.MAX_FUEL - next_fuel
                
                # Giảm chi phí khi nhiên liệu thấp để ưu tiên ghé trạm xăng
                if next_fuel < self.LOW_FUEL_THRESHOLD:
                    discount = 0.7  # Giảm 30% chi phí khi nhiên liệu thấp
                else:
                    discount = 1.0
                
                # Chi phí đổ xăng
                fuel_cost = self.GAS_STATION_COST * fuel_needed * discount
                
                # Cập nhật chi phí di chuyển
                move_cost = fuel_cost + self.GAS_WEIGHT
                
                # Nếu đủ tiền thì đổ xăng
                if next_money >= fuel_cost:
                    next_money -= fuel_cost
                    next_fuel = self.MAX_FUEL
        
        elif next_cell_type == self.TOLL_CELL:  # Trạm thu phí
            # Tính chi phí trạm thu phí
            toll_cost = self.TOLL_BASE_COST
            
            # Cập nhật chi phí di chuyển
            move_cost = toll_cost + self.TOLL_WEIGHT
            
            # Nếu đủ tiền thì trả phí
            if next_money >= toll_cost:
                next_money -= toll_cost
        
        elif next_cell_type == self.OBSTACLE_CELL:  # Vật cản
            move_cost = float('inf')  # Không thể đi qua
            next_fuel = -1  # Đánh dấu không khả thi
        
        # Thêm ưu tiên cho trạm xăng khi nhiên liệu thấp
        if next_cell_type == self.GAS_STATION_CELL and next_fuel < self.LOW_FUEL_THRESHOLD:
            move_cost *= 0.5  # Giảm 50% chi phí để ưu tiên đi qua trạm xăng
        
        return move_cost, fuel_cost, toll_cost, next_fuel, next_money

    def search(self, start: Tuple[int, int], goal: Tuple[int, int]) -> List[Tuple[int, int]]:
        """Thực hiện tìm kiếm UCS từ start đến goal dựa trên chi phí thực tế."""
        self.start = start
        self.goal = goal
        
        # Khởi tạo các biến
        while not self.priority_queue.empty():
            self.priority_queue.get()  # Làm rỗng hàng đợi ưu tiên
        
        self.cost_so_far.clear()
        self.parent.clear()
        self.state_at_position.clear()
        self.visited_positions.clear()
        self.visited = []
        self.current_path.clear()
        self.steps = 0
        self.path_length = 0
        self.cost = 0
        self.current_money = self.MAX_MONEY
        self.current_fuel = self.MAX_FUEL
        self.current_total_cost = 0
        self.current_fuel_cost = 0
        self.current_toll_cost = 0
        
        # Khởi tạo trạng thái start
        initial_state = (self.current_fuel, self.current_money)
        self.priority_queue.put((0, start, initial_state))  # (chi phí, vị trí, trạng thái)
        self.cost_so_far[start] = 0
        self.parent[start] = None
        self.state_at_position[start] = initial_state
        self.add_visited(start)
        self.current_position = start
        
        # Thực hiện UCS
        while not self.priority_queue.empty():
            self.steps += 1
            
            # Lấy vị trí có chi phí thấp nhất từ hàng đợi ưu tiên
            current_cost, current_pos, current_state = self.priority_queue.get()
            current_fuel, current_money = current_state
            self.current_position = current_pos
            
            # Nếu đến đích, truy vết đường đi và trả về
            if current_pos == goal:
                raw_path = self.reconstruct_path(start, goal)
                
                # Xác thực và làm sạch đường đi
                validated_path = self.validate_path_no_obstacles(raw_path)
                
                if not validated_path or len(validated_path) < 2:
                    print(f"UCS: Đường đi đến {goal} không hợp lệ sau khi xác thực. UCS kết thúc tìm kiếm.")
                    self.current_path = []  # Không tìm thấy đường đi hợp lệ
                    return [] 
                
                # Kiểm tra tính khả thi tổng thể của đường đi (nhiên liệu và chi phí)
                is_still_feasible, reason = self.is_path_feasible(validated_path, self.MAX_FUEL)
                if is_still_feasible:
                    self.current_path = validated_path
                    self.path_length = len(self.current_path) - 1
                    self.cost = current_cost  # Lưu chi phí tổng
                    
                    # Tính toán lại tất cả thống kê trên đường đi cuối cùng đã xác thực
                    self.calculate_path_fuel_consumption(self.current_path)
                    
                    print(f"UCS: Tìm thấy đường đi hợp lệ và khả thi đến {goal}.")
                    return self.current_path  # Tìm thấy đích và đường đi hoàn toàn được xác thực
                else:
                    print(f"UCS: Đường đi đến {goal} không khả thi sau khi xác thực: {reason}. UCS kết thúc tìm kiếm.")
                    self.current_path = validated_path  # Lưu lại đường đi không khả thi để hiển thị
                    return self.current_path  # Trả về đường đi để hiển thị, mặc dù không khả thi
            
            # Nếu đã tìm thấy đường đi tốt hơn đến vị trí hiện tại, bỏ qua
            if current_pos in self.cost_so_far and self.cost_so_far[current_pos] < current_cost:
                continue
            
            # Xử lý các ô lân cận
            for next_pos in self.get_neighbors(current_pos):
                # Tính chi phí di chuyển thực tế đến ô tiếp theo
                move_cost, fuel_cost, toll_cost, next_fuel, next_money = self.calculate_cost(
                    current_pos, next_pos, current_fuel, current_money
                )
                
                # Nếu không đủ nhiên liệu hoặc tiền để di chuyển đến ô tiếp theo, bỏ qua
                if next_fuel < 0 or next_money < 0:
                    continue
                
                # Tính tổng chi phí từ điểm bắt đầu đến ô tiếp theo
                new_cost = current_cost + move_cost
                next_state = (next_fuel, next_money)
                
                # Nếu chưa thăm hoặc tìm thấy đường đi tốt hơn
                if next_pos not in self.cost_so_far or new_cost < self.cost_so_far[next_pos]:
                    self.cost_so_far[next_pos] = new_cost
                    self.priority_queue.put((new_cost, next_pos, next_state))
                    self.parent[next_pos] = current_pos
                    self.state_at_position[next_pos] = next_state
                    self.add_visited(next_pos)
        
        return []  # Không tìm thấy đường đi
    
    def reconstruct_path(self, start: Tuple[int, int], goal: Tuple[int, int]) -> List[Tuple[int, int]]:
        """Truy vết đường đi từ đích về điểm bắt đầu dựa trên parent."""
        path = []
        current = goal
        
        while current is not None:
            path.append(current)
            current = self.parent.get(current)
        
        return list(reversed(path))  # Đảo ngược để có đường đi từ start đến goal
    
    def step(self) -> bool:
        """Execute one step of UCS. Used for visualization."""
        if self.priority_queue.empty():
            self.current_position = None
            return True
        
        self.steps += 1
        
        # Lấy vị trí có chi phí thấp nhất từ hàng đợi ưu tiên
        current_cost, current_pos, current_state = self.priority_queue.get()
        current_fuel, current_money = current_state
        self.current_position = current_pos
        
        # Nếu đến đích, truy vết đường đi
        if current_pos == self.goal:
            path = self.reconstruct_path(self.start, self.goal)
            
            # Xác thực đường đi
            validated_path = self.validate_path_no_obstacles(path)
            if not validated_path or len(validated_path) < 2:
                self.current_path = []
                return True
            
            # Kiểm tra tính khả thi
            is_feasible, reason = self.is_path_feasible(validated_path, self.MAX_FUEL)
            self.current_path = validated_path
            self.path_length = len(validated_path) - 1
            self.cost = current_cost
            
            # Tính toán chi tiết về nhiên liệu và chi phí
            self.calculate_path_fuel_consumption(self.current_path)
            
            return True
        
        # Xử lý các ô lân cận
        for next_pos in self.get_neighbors(current_pos):
            # Tính chi phí di chuyển thực tế đến ô tiếp theo            
            move_cost, fuel_cost, toll_cost, next_fuel, next_money = self.calculate_cost(
                current_pos, next_pos, current_fuel, current_money
            )
            
            # Nếu không đủ nhiên liệu hoặc tiền để di chuyển đến ô tiếp theo, bỏ qua
            if next_fuel < 0 or next_money < 0:
                continue
            
            # Tính tổng chi phí
            new_cost = current_cost + move_cost
            next_state = (next_fuel, next_money)
            
            # Nếu chưa thăm hoặc tìm thấy đường đi tốt hơn
            if next_pos not in self.cost_so_far or new_cost < self.cost_so_far[next_pos]:
                self.cost_so_far[next_pos] = new_cost
                self.priority_queue.put((new_cost, next_pos, next_state))
                self.parent[next_pos] = current_pos
                self.state_at_position[next_pos] = next_state
                self.add_visited(next_pos)
        
        return False 