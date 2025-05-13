"""
Iterative Deepening Search algorithm implementation.

------------------------- ĐỊNH NGHĨA THUẬT TOÁN IDS -------------------------
IDS (Iterative Deepening Search - Tìm kiếm sâu dần) là thuật toán kết hợp ưu điểm 
của cả BFS (tìm kiếm chiều rộng) và DFS (tìm kiếm chiều sâu). Thuật toán thực hiện 
tìm kiếm chiều sâu (DFS) nhiều lần với độ sâu giới hạn tăng dần, đảm bảo tìm được 
đường đi ngắn nhất như BFS nhưng tiết kiệm bộ nhớ hơn.

---------------------- NGUYÊN LÝ CƠ BẢN CỦA IDS ---------------------------
1. Bắt đầu với giới hạn độ sâu = 0, thực hiện DFS với giới hạn này
2. Nếu không tìm thấy đích, tăng giới hạn độ sâu lên 1 và thực hiện DFS lại
3. Lặp lại quá trình, tăng dần giới hạn độ sâu cho đến khi tìm thấy đích
4. Đảm bảo tìm được đường đi ngắn nhất (như BFS) nhưng tiết kiệm bộ nhớ (như DFS)

---------------- TRIỂN KHAI IDS TRONG CHƯƠNG TRÌNH NÀY --------------------
IDS trong chương trình này được triển khai với các tính năng nâng cao:

1. KHÔNG GIAN TRẠNG THÁI ĐA CHIỀU:
   - Không chỉ xét tọa độ (x,y), mà còn quan tâm đến:
   - Nhiên liệu còn lại
   - Chi phí di chuyển
   - Số tiền còn lại

2. DEPTH-LIMITED SEARCH:
   - Mỗi lần DFS chỉ thăm đến độ sâu giới hạn cụ thể
   - Tăng dần giới hạn độ sâu cho đến khi tìm thấy đường đi tốt nhất

3. ĐÁNH GIÁ TÍNH KHẢ THI:
   - Sau khi tìm được đường đi, thuật toán kiểm tra:
   - Không đi qua vật cản
   - Đủ nhiên liệu để di chuyển (mỗi bước tiêu hao FUEL_PER_MOVE)
   - Đủ tiền để trả phí trạm thu phí và đổ xăng

4. CÁC GIAI ĐOẠN THỰC HIỆN:
   - Giai đoạn 1: IDS tìm đường đi ngắn nhất về số bước
   - Giai đoạn 2: Xác thực đường đi (validate_path_no_obstacles)
   - Giai đoạn 3: Kiểm tra tính khả thi (is_path_feasible)
   - Giai đoạn 4: Tính toán chi tiết (calculate_path_fuel_consumption)

5. ƯU ĐIỂM VÀ NHƯỢC ĐIỂM:
   - Ưu điểm: Đảm bảo tìm được đường đi ngắn nhất như BFS nhưng tiết kiệm bộ nhớ
   - Nhược điểm: Tính toán lặp lại các nút ở độ sâu nhỏ nhiều lần, có thể mất nhiều 
                thời gian trên các đồ thị lớn
"""

from typing import List, Tuple, Dict, Optional
import numpy as np
from .base_search import BaseSearch, SearchState

class IDS(BaseSearch):
    """Iterative Deepening Search algorithm implementation."""
    
    def __init__(self, grid: np.ndarray, initial_money: float = None, 
                 max_fuel: float = None, fuel_per_move: float = None, 
                 gas_station_cost: float = None, toll_base_cost: float = None,
                 initial_fuel: float = None):
        """Initialize IDS with a grid and configuration parameters."""
        super().__init__(grid, initial_money, max_fuel, fuel_per_move, 
                         gas_station_cost, toll_base_cost, initial_fuel)
        self.start = None
        self.goal = None
        self.parent = {}  # Dictionary để truy vết đường đi
        self.current_depth_limit = 0  # Giới hạn độ sâu hiện tại
        self.max_depth_limit = 100  # Giới hạn độ sâu tối đa
        self.current_path_tracking = []  # Đường đi hiện tại đang xét (dùng cho step())
        self.is_visualization_mode = False  # Chế độ minh họa bước
        
        # Thêm các biến để theo dõi quá trình tìm kiếm
        self.depth_of_nodes = {}  # Độ sâu của mỗi nút
        self.current_iteration = 0  # Lần lặp IDS hiện tại
    
    def search(self, start: Tuple[int, int], goal: Tuple[int, int]) -> List[Tuple[int, int]]:
        """Thực hiện tìm kiếm IDS từ start đến goal."""
        self.start = start
        self.goal = goal
        self.visited_positions.clear()
        self.parent.clear()
        self.visited = []
        self.current_path.clear()
        self.steps = 0
        self.path_length = 0
        self.cost = 0
        self.current_fuel = self.MAX_FUEL
        self.current_total_cost = 0
        self.current_fuel_cost = 0
        self.current_toll_cost = 0
        
        # Xác định giới hạn độ sâu tối đa dựa trên kích thước lưới
        # Giới hạn này có thể lớn hơn kích thước lưới một chút để đảm bảo tìm được đường đi
        self.max_depth_limit = max(self.grid.shape) * 2
        
        # Thực hiện IDS với giới hạn độ sâu tăng dần
        for depth_limit in range(1, self.max_depth_limit + 1):
            self.current_iteration += 1
            self.current_depth_limit = depth_limit
            
            # Khởi tạo lại các biến cho mỗi vòng lặp
            self.depth_of_nodes.clear()
            self.visited_positions.clear()
            self.parent.clear()
            self.visited = []
            
            # Gán độ sâu 0 cho nút bắt đầu
            self.depth_of_nodes[start] = 0
            
            # Thực hiện DFS với giới hạn độ sâu
            result = self.depth_limited_search(start, goal, depth_limit)
            
            # Nếu tìm thấy đường đi, xác thực và kiểm tra tính khả thi
            if result:
                # Xác thực đường đi (loại bỏ vật cản)
                validated_path = self.validate_path_no_obstacles(result)
                
                if not validated_path or len(validated_path) < 2:
                    print(f"IDS (depth={depth_limit}): Đường đi đến {goal} không hợp lệ sau khi xác thực. Tiếp tục tìm kiếm.")
                    continue
                
                # Kiểm tra tính khả thi (nhiên liệu và chi phí)
                is_feasible, reason = self.is_path_feasible(validated_path, self.MAX_FUEL)
                if is_feasible:
                    self.current_path = validated_path
                    self.path_length = len(self.current_path) - 1
                    
                    # Tính toán chi tiết về đường đi
                    self.calculate_path_fuel_consumption(self.current_path)
                    
                    print(f"IDS: Tìm thấy đường đi hợp lệ và khả thi đến {goal} với độ sâu {depth_limit}.")
                    return self.current_path
                else:
                    print(f"IDS (depth={depth_limit}): Đường đi đến {goal} không khả thi: {reason}. Tiếp tục tìm kiếm.")
        
        # Không tìm thấy đường đi trong giới hạn độ sâu tối đa
        print(f"IDS: Không tìm thấy đường đi khả thi đến {goal} trong giới hạn độ sâu {self.max_depth_limit}.")
        return []
    
    def depth_limited_search(self, start: Tuple[int, int], goal: Tuple[int, int], depth_limit: int) -> List[Tuple[int, int]]:
        """Thực hiện DFS với giới hạn độ sâu."""
        # Thêm vị trí bắt đầu vào stack
        stack = [(start, 0)]  # (vị trí, độ sâu hiện tại)
        self.add_visited(start)
        self.parent[start] = None
        
        while stack:
            self.steps += 1
            current_pos, current_depth = stack.pop()
            self.current_position = current_pos
            
            # Nếu đến đích, truy vết đường đi và trả về
            if current_pos == goal:
                return self.reconstruct_path(start, goal)
            
            # Nếu chưa đạt đến giới hạn độ sâu, tiếp tục tìm kiếm
            if current_depth < depth_limit:
                # Xử lý các ô lân cận (duyệt ngược để ưu tiên trái-lên-phải-xuống)
                for next_pos in reversed(self.get_neighbors(current_pos)):
                    # Kiểm tra xem vị trí đã được thăm chưa
                    if next_pos not in self.visited_positions:
                        stack.append((next_pos, current_depth + 1))
                        self.add_visited(next_pos)
                        self.parent[next_pos] = current_pos
                        self.depth_of_nodes[next_pos] = current_depth + 1
        
        # Không tìm thấy đường đi trong giới hạn độ sâu hiện tại
        return []
    
    def reconstruct_path(self, start: Tuple[int, int], goal: Tuple[int, int]) -> List[Tuple[int, int]]:
        """Truy vết đường đi từ đích về điểm bắt đầu dựa trên parent."""
        path = []
        current = goal
        
        while current is not None:
            path.append(current)
            current = self.parent.get(current)
        
        return list(reversed(path))  # Đảo ngược để có đường đi từ start đến goal
    
    def step(self) -> bool:
        """Thực hiện một bước của IDS cho mục đích minh họa.
        Trả về True nếu thuật toán kết thúc, False nếu cần tiếp tục."""
        
        # Nếu đây là lần gọi step() đầu tiên, khởi tạo chế độ minh họa
        if not self.is_visualization_mode:
            self.is_visualization_mode = True
            self.current_depth_limit = 0
            self.current_path_tracking = []
            self.visited_positions.clear()
            self.parent.clear()
            self.visited = []
            
            # Thiết lập các giá trị ban đầu
            self.add_visited(self.start)
            self.parent[self.start] = None
            self.depth_of_nodes[self.start] = 0
            self.current_position = self.start
            
            # Bắt đầu với depth_limit = 1
            self.current_depth_limit = 1
            self.current_path_tracking = [(self.start, 0)]  # (vị trí, độ sâu)
            return False
        
        # Nếu không còn nút nào để khám phá ở giới hạn độ sâu hiện tại
        if not self.current_path_tracking:
            # Tăng giới hạn độ sâu nếu chưa đạt tối đa
            if self.current_depth_limit < self.max_depth_limit:
                self.current_depth_limit += 1
                
                # Xóa thông tin cũ để bắt đầu lại với độ sâu mới
                self.visited_positions.clear()
                self.parent.clear()
                self.visited = []
                
                # Khởi tạo lại với nút bắt đầu
                self.add_visited(self.start)
                self.parent[self.start] = None
                self.depth_of_nodes[self.start] = 0
                self.current_position = self.start
                self.current_path_tracking = [(self.start, 0)]
                
                return False
            else:
                # Đã đạt đến giới hạn độ sâu tối đa, kết thúc thuật toán
                self.current_position = None
                return True
        
        # Lấy nút hiện tại từ đường đi đang theo dõi
        current_pos, current_depth = self.current_path_tracking.pop()
        self.current_position = current_pos
        
        # Tăng số bước
        self.steps += 1
        
        # Nếu đến đích, truy vết và kiểm tra đường đi
        if current_pos == self.goal:
            path = self.reconstruct_path(self.start, self.goal)
            
            # Xác thực đường đi
            validated_path = self.validate_path_no_obstacles(path)
            if not validated_path or len(validated_path) < 2:
                # Đường đi không hợp lệ, tiếp tục tìm kiếm
                return False
            
            # Kiểm tra tính khả thi
            is_feasible, reason = self.is_path_feasible(validated_path, self.MAX_FUEL)
            self.current_path = validated_path
            self.path_length = len(validated_path) - 1
            
            # Tính toán chi tiết về nhiên liệu và chi phí
            self.calculate_path_fuel_consumption(self.current_path)
            
            # Kết thúc thuật toán nếu tìm thấy đường đi khả thi
            if is_feasible:
                return True
        
        # Nếu chưa đạt đến giới hạn độ sâu, mở rộng các nút lân cận
        if current_depth < self.current_depth_limit:
            # Xử lý các ô lân cận (duyệt ngược để ưu tiên trái-lên-phải-xuống)
            for next_pos in reversed(self.get_neighbors(current_pos)):
                # Kiểm tra xem vị trí đã được thăm chưa
                if next_pos not in self.visited_positions:
                    self.current_path_tracking.append((next_pos, current_depth + 1))
                    self.add_visited(next_pos)
                    self.parent[next_pos] = current_pos
                    self.depth_of_nodes[next_pos] = current_depth + 1
        
        return False
    
    def evaluate_path(self, path: List[Tuple[int, int]]) -> Dict:
        """Đánh giá đường đi để tính toán nhiên liệu, chi phí và kiểm tra tính khả thi."""
        # Khởi tạo các giá trị
        current_fuel = self.MAX_FUEL
        total_cost = 0.0
        fuel_cost = 0.0
        toll_cost = 0.0
        toll_stations_visited = set()
        is_feasible = True
        reason = ""
        
        # Lặp qua từng cặp vị trí liên tiếp trên đường đi
        for i in range(len(path) - 1):
            # Trừ nhiên liệu cho bước di chuyển
            current_fuel -= self.FUEL_PER_MOVE
            
            # Kiểm tra hết xăng
            if current_fuel < 0:
                is_feasible = False
                reason = f"Hết nhiên liệu tại bước thứ {i+1}"
                current_fuel = 0
                break
            
            # Xử lý vị trí hiện tại
            current_pos = path[i + 1]
            cell_type = self.grid[current_pos[1], current_pos[0]]
            
            # Xử lý trạm xăng
            if cell_type == self.GAS_STATION_CELL:  # Trạm xăng
                if current_fuel < self.MAX_FUEL:  # Chỉ đổ xăng nếu bình chưa đầy
                    fuel_needed = self.MAX_FUEL - current_fuel
                    fuel_cost += self.GAS_STATION_COST * fuel_needed
                    current_fuel = self.MAX_FUEL
            
            # Xử lý trạm thu phí
            elif cell_type == self.TOLL_CELL:  # Trạm thu phí
                if current_pos not in toll_stations_visited:
                    toll_cost += self.TOLL_BASE_COST
                    toll_stations_visited.add(current_pos)
        
        # Tính tổng chi phí
        total_cost = fuel_cost + toll_cost
        
        # Kiểm tra số tiền còn lại
        current_money = self.MAX_MONEY - total_cost
        if current_money < 0:
            is_feasible = False
            reason = f"Không đủ tiền (cần {total_cost:.2f}đ, có {self.MAX_MONEY:.2f}đ)"
        
        return {
            "is_feasible": is_feasible,
            "reason": reason,
            "fuel_remaining": current_fuel,
            "total_cost": total_cost,
            "fuel_cost": fuel_cost,
            "toll_cost": toll_cost,
            "money_remaining": current_money
        }
    
    def get_statistics(self) -> Dict:
        """Get statistics about the algorithm's execution."""
        stats = super().get_statistics()
        # Bổ sung thông tin riêng của IDS
        stats.update({
            "max_depth_reached": self.current_depth_limit,
            "iterations": self.current_iteration
        })
        return stats

    def set_max_depth_limit(self, max_depth_limit: int):
        """Thiết lập giới hạn độ sâu tối đa cho thuật toán."""
        self.max_depth_limit = max_depth_limit
        return self 