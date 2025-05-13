"""
Thuật toán A* (A-star) - Tìm đường đi tối ưu có xét đến ràng buộc nhiên liệu và chi phí.

------------------------- ĐỊNH NGHĨA THUẬT TOÁN A* -------------------------
A* là thuật toán tìm kiếm theo thông tin (informed search) kết hợp giữa:
- Tìm kiếm theo chi phí (như UCS - Uniform Cost Search): xét chi phí thực tế g(n)
- Tìm kiếm theo đánh giá (như Greedy Best-First Search): xét chi phí ước tính h(n)

Thuật toán sử dụng hàm đánh giá tổng f(n) = g(n) + h(n) trong đó:
- g(n): Chi phí thực tế từ điểm xuất phát đến nút hiện tại n
- h(n): Chi phí ước tính (heuristic) từ nút hiện tại n đến đích

---------------------- NGUYÊN LÝ CƠ BẢN CỦA A* ---------------------------
1. Khởi tạo danh sách mở (open_set) chứa nút bắt đầu
2. Lặp lại cho đến khi tìm thấy đích hoặc danh sách mở rỗng:
   - Lấy nút có f(n) nhỏ nhất từ danh sách mở
   - Nếu nút này là đích, kết thúc tìm kiếm
   - Đánh dấu nút là đã xét (thêm vào closed_set)
   - Với mỗi nút kề chưa xét:
     + Tính chi phí g mới từ điểm xuất phát 
     + Tính giá trị heuristic h đến đích
     + Cập nhật f = g + h và thêm vào danh sách mở

---------------------- A* TRONG CHƯƠNG TRÌNH NÀY -------------------------
A* trong chương trình này tính toán chi phí thực tế:
- Chi phí di chuyển cơ bản (theo loại ô)
- Chi phí xăng (tiêu thụ và nạp thêm)
- Chi phí trạm thu phí
- Các hình phạt/ưu đãi với nhiên liệu thấp/trạm xăng

Thuật toán đảm bảo tìm được đường đi tối ưu về chi phí và khả thi về:
- Nhiên liệu: Đảm bảo không bao giờ hết nhiên liệu giữa đường
- Tiền: Đảm bảo đủ tiền cho xăng và trạm thu phí
- Tính liên tục: Đảm bảo đường đi mượt mà, không đứt quãng

---------------------- CẤU TRÚC DỮ LIỆU CHÍNH ---------------------------
- open_set: Hàng đợi ưu tiên chứa các nút cần xét tiếp
- closed_set: Tập hợp các nút đã xét
- g_score: Từ điển lưu chi phí thực từ điểm xuất phát đến mỗi nút
- f_score: Từ điển lưu tổng chi phí ước tính f(n) = g(n) + h(n)
- parent: Từ điển lưu nút cha để tái tạo đường đi
"""

from typing import List, Tuple, Dict, Set, Optional
import heapq
import numpy as np
from .base_search import BaseSearch, SearchState
import math

class AStar(BaseSearch):
    """Triển khai thuật toán A* với heuristic tối ưu hóa chi phí và nhiên liệu."""
    
    def __init__(self, grid: np.ndarray, initial_money: float = None, 
                 max_fuel: float = None, fuel_per_move: float = None, 
                 gas_station_cost: float = None, toll_base_cost: float = None,
                 initial_fuel: float = None):
        """Khởi tạo thuật toán A* với lưới và các tham số cấu hình.
        
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
        
        # Cấu trúc dữ liệu chính của A*
        self.open_set = []  # Hàng đợi ưu tiên [(f_score, state), ...] cho các nút cần khám phá
        self.closed_set = set()  # Tập hợp các nút đã khám phá {(pos, fuel_rounded), ...}
        self.g_score = {}  # Chi phí từ điểm xuất phát đến nút hiện tại {state_key: cost, ...}
        self.f_score = {}  # Chi phí ước tính tổng = g_score + heuristic {state_key: f_score, ...}
        self.parent = {}  # Con trỏ nút cha để tái tạo đường đi {child_pos: parent_pos, ...}
        
        # Điểm đầu và cuối của đường đi
        self.start = None  # Điểm xuất phát (x, y)
        self.goal = None   # Điểm đích (x, y)
        
        # Ngưỡng cho việc ra quyết định (có thể điều chỉnh)
        self.LOW_FUEL_THRESHOLD = self.MAX_FUEL * 0.3  # 30% bình xăng - ngưỡng cảnh báo nhiên liệu thấp
        self.TOLL_DISTANCE_THRESHOLD = 3  # Số bước tối thiểu để cân nhắc đi qua trạm thu phí
    
    def calculate_heuristic(self, pos: Tuple[int, int], goal: Tuple[int, int], 
                          current_fuel: float, current_cost: float) -> float:
        """Tính toán hàm heuristic cho A*.
        
        Hàm heuristic ước tính chi phí từ vị trí hiện tại đến đích. Hàm này kết hợp nhiều 
        yếu tố để đưa ra ước tính chính xác hơn, bao gồm:
        - Khoảng cách Manhattan đến đích
        - Chi phí nhiên liệu ước tính (dựa trên DEFAULT_GAS_STATION_COST)
        - Chi phí trạm thu phí ước tính (dựa trên DEFAULT_TOLL_BASE_COST)
        - Phạt khi nhiên liệu thấp (cần đến trạm xăng gấp)
        - Ưu đãi/phạt theo loại ô hiện tại
        
        Mỗi thành phần được cân nhắc với trọng số phù hợp để đảm bảo tính admissible
        của heuristic (không ước tính cao hơn chi phí thực tế).
        
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
                if self.grid[y, x] == 2:  # Trạm xăng
                    distance = abs(x - pos[0]) + abs(y - pos[1])
                    if distance < min_distance:
                        min_distance = distance
                        nearest_gas = (x, y)
        
        return nearest_gas
    
    def get_neighbors_with_priority(self, pos: Tuple[int, int], current_fuel: float) -> List[Tuple[int, int]]:
        """Lấy các vị trí lân cận và sắp xếp theo thứ tự ưu tiên.
        
        Phương thức này mở rộng hàm get_neighbors cơ bản bằng cách thêm cơ chế
        ưu tiên dựa vào loại ô và tình trạng nhiên liệu hiện tại. Các ưu tiên bao gồm:
        - Ưu tiên cao nhất: Trạm xăng khi sắp hết nhiên liệu
        - Ưu tiên cao: Trạm xăng khi nhiên liệu thấp
        - Ưu tiên trung bình: Trạm thu phí
        - Ưu tiên thấp: Đường thông thường
        
        Args:
            pos: Vị trí hiện tại (x, y)
            current_fuel: Lượng nhiên liệu hiện tại
            
        Returns:
            Danh sách các vị trí lân cận đã sắp xếp theo thứ tự ưu tiên
        """
        neighbors = self.get_neighbors(pos)
        prioritized = []
        
        # Xác định ưu tiên cho các ô dựa vào loại và mức nhiên liệu
        for next_pos in neighbors:
            cell_type = self.grid[next_pos[1], next_pos[0]]
            
            # Ưu tiên thấp (3): Đường thông thường
            # Ưu tiên trung bình (1): Trạm xăng khi nhiên liệu thấp
            # Ưu tiên cao (0): Trạm xăng khi sắp hết xăng
            # Ưu tiên thấp (2): Trạm thu phí
            
            priority = 3  # Mặc định: đường thường
            
            if cell_type == 2:  # Trạm xăng
                if current_fuel < self.LOW_FUEL_THRESHOLD:  # Nhiên liệu rất thấp
                    priority = 0  # Ưu tiên cao nhất - cần đổ xăng gấp
                elif current_fuel < self.MAX_FUEL * 0.5:  # Nhiên liệu thấp
                    priority = 1
            elif cell_type == 1:  # Trạm thu phí
                priority = 2
            
            prioritized.append((priority, next_pos))
        
        # Sắp xếp theo ưu tiên và trả về chỉ vị trí
        return [pos for _, pos in sorted(prioritized)]
    
    def search(self, start: Tuple[int, int], goal: Tuple[int, int]) -> List[Tuple[int, int]]:
        """Thực hiện tìm kiếm A* từ điểm xuất phát đến đích.
        
        Đây là phương thức chính của thuật toán A*, thực hiện toàn bộ
        quá trình tìm kiếm đường đi từ điểm xuất phát đến đích. Phương thức này:
        1. Khởi tạo các cấu trúc dữ liệu cần thiết
        2. Lặp cho đến khi tìm thấy đích hoặc hết nút để xét
        3. Với mỗi nút, mở rộng các nút kề và cập nhật chi phí
        4. Khi tìm thấy đích, thực hiện kiểm tra tính khả thi của đường đi
        5. Trả về đường đi hoặc danh sách rỗng nếu không tìm thấy
        
        Args:
            start: Vị trí xuất phát (x, y)
            goal: Vị trí đích (x, y)
            
        Returns:
            Danh sách các vị trí tạo thành đường đi từ start đến goal
        """
        # Lưu lại điểm đầu và cuối
        self.start = start
        self.goal = goal
        
        # Kiểm tra tính hợp lệ của điểm đầu và cuối - phải không phải là chướng ngại vật
        if not self.is_valid_position(start) or self.grid[start[1], start[0]] == self.OBSTACLE_CELL:
            print(f"ERROR: Điểm xuất phát {start} không hợp lệ hoặc là ô chướng ngại vật!")
            return []
        
        if not self.is_valid_position(goal) or self.grid[goal[1], goal[0]] == self.OBSTACLE_CELL:
            print(f"ERROR: Điểm đích {goal} không hợp lệ hoặc là ô chướng ngại vật!")
            return []
        
        # Reset lại tất cả cấu trúc dữ liệu
        self.open_set = []  # Sử dụng heapq làm hàng đợi ưu tiên
        self.closed_set = set()
        self.g_score = {}
        self.f_score = {}
        self.parent = {}
        self.visited_positions.clear()
        self.visited = []
        self.current_path.clear()
        self.steps = 0
        
        # Khởi tạo trạng thái ban đầu với nhiên liệu đầy và tiền ban đầu
        initial_state = SearchState(
            position=start,
            fuel=self.MAX_FUEL,  # Bắt đầu với bình xăng đầy
            total_cost=0,        # Chi phí ban đầu = 0
            money=self.MAX_MONEY if self.current_money is None else self.current_money,
            path=[start],        # Đường đi ban đầu chỉ có điểm xuất phát
            visited_gas_stations=set(),  # Chưa đi qua trạm xăng nào
            toll_stations_visited=set()  # Chưa đi qua trạm thu phí nào
        )
        
        # Thêm trạng thái đầu tiên vào open_set với f_score = heuristic
        initial_f_score = self.calculate_heuristic(start, goal, self.MAX_FUEL, 0)
        self.g_score[start] = 0  # g_score tại điểm xuất phát = 0
        self.f_score[start] = initial_f_score
        
        # Thêm vào hàng đợi ưu tiên: (f_score, trạng thái)
        heapq.heappush(self.open_set, (initial_f_score, initial_state))
        
        # Đánh dấu điểm xuất phát đã thăm
        self.add_visited(start)
        
        # Vòng lặp chính của A*
        while self.open_set:
            # Tăng số bước thực hiện
            self.steps += 1
            
            # Lấy trạng thái có f_score thấp nhất từ open_set
            _, current_state = heapq.heappop(self.open_set)
            current_pos = current_state.position
            self.current_position = current_pos  # Cập nhật vị trí hiện tại cho visualization
            
            # Nếu đến đích, xử lý đường đi cuối cùng
            if current_pos == goal:
                # Lấy đường đi từ trạng thái tìm kiếm
                raw_path = current_state.path
                
                # Bước 1: Xác thực đường đi - loại bỏ chướng ngại vật, đảm bảo tính liên tục
                validated_path = self.validate_path_no_obstacles(raw_path)
                
                # Nếu đường đi sau khi kiểm tra không hợp lệ hoặc quá ngắn, trả về rỗng
                if not validated_path or len(validated_path) < 2:  # Đường đi cần ít nhất điểm đầu và cuối
                    print(f"ERROR: Đường đi sau khi xác thực không hợp lệ hoặc quá ngắn.")
                    return []

                # Bước 2: Kiểm tra tính khả thi về mặt nhiên liệu và chi phí
                is_feasible, reason = self.is_path_feasible(validated_path, self.MAX_FUEL)
                if not is_feasible:
                    print(f"ERROR: Đường đi sau khi xác thực không khả thi: {reason}")
                    return []
                
                # Nếu tất cả kiểm tra đều thành công, đây là đường đi chính thức
                self.current_path = validated_path
                self.path_length = len(self.current_path) - 1
                
                # Tính toán chi tiết về nhiên liệu, chi phí và tiền cho đường đi
                self.calculate_path_fuel_consumption(self.current_path) 
                
                # Trả về đường đi đã xác thực
                return self.current_path
            
            # Thêm trạng thái hiện tại vào tập đóng
            self.closed_set.add(current_state.get_state_key())
            
            # Xử lý các ô lân cận theo thứ tự ưu tiên
            for next_pos in self.get_neighbors_with_priority(current_pos, current_state.fuel):
                # Kiểm tra chắc chắn rằng vị trí tiếp theo không phải là chướng ngại vật
                # Sử dụng nhiều cách kiểm tra để đảm bảo an toàn
                cell_value = self.grid[next_pos[1], next_pos[0]]
                if (cell_value == self.OBSTACLE_CELL or 
                    int(cell_value) == int(self.OBSTACLE_CELL) or 
                    cell_value < 0):
                    continue  # Bỏ qua ô chướng ngại vật
                
                # Kiểm tra xem đã thăm trạng thái này chưa (dựa trên vị trí và nhiên liệu)
                state_key = (next_pos, round(current_state.fuel, 2))
                if state_key in self.closed_set:
                    continue  # Bỏ qua nếu đã thăm
                
                # Tính toán chi phí và nhiên liệu cho bước tiếp theo
                # Trả về (new_fuel, move_cost, new_money)
                new_fuel, move_cost, new_money = self.calculate_cost(current_state, next_pos)
                
                # Kiểm tra nếu không đủ nhiên liệu hoặc tiền
                if new_fuel < 0 or new_money < 0:
                    continue  # Bỏ qua vì không khả thi về nhiên liệu hoặc tiền
                
                # Kiểm tra tính liên tục của đường đi
                # Đảm bảo các ô liền kề nhau theo distance Manhattan = 1
                prev_pos = current_state.position
                if abs(prev_pos[0] - next_pos[0]) + abs(prev_pos[1] - next_pos[1]) > 1:
                    print(f"WARNING: Phát hiện đường đi không liên tục từ {prev_pos} đến {next_pos}. Bỏ qua.")
                    continue  # Bỏ qua do không liên tục
                
                # Tạo trạng thái mới
                new_path = current_state.path + [next_pos]  # Thêm vị trí mới vào đường đi
                
                # Tạo trạng thái mới với chi phí và nhiên liệu đã cập nhật
                new_state = SearchState(
                    position=next_pos,
                    fuel=new_fuel,
                    total_cost=current_state.total_cost + move_cost,
                    money=new_money,
                    path=new_path,
                    visited_gas_stations=current_state.visited_gas_stations.copy(),
                    toll_stations_visited=current_state.toll_stations_visited.copy()
                )
                
                # Cập nhật danh sách trạm đã ghé qua
                if cell_value == self.GAS_STATION_CELL:  # Trạm xăng
                    new_state.visited_gas_stations.add(next_pos)
                elif cell_value == self.TOLL_CELL:  # Trạm thu phí
                    new_state.toll_stations_visited.add(next_pos)
                
                # Lấy khóa duy nhất cho trạng thái mới
                new_state_key = new_state.get_state_key()
                
                # Kiểm tra lại nếu đã trong closed set
                if new_state_key in self.closed_set:
                    continue  # Bỏ qua nếu đã thăm
                
                # Tính toán g_score mới: chi phí từ điểm xuất phát đến vị trí hiện tại
                tentative_g_score = current_state.total_cost + move_cost
                
                # Nếu trạng thái chưa có trong g_score hoặc g_score mới nhỏ hơn, cập nhật
                if new_state_key not in self.g_score or tentative_g_score < self.g_score[new_state_key]:
                    # Lưu thông tin cho việc tái tạo đường đi
                    self.parent[next_pos] = current_pos
                    
                    # Cập nhật g_score và f_score
                    self.g_score[new_state_key] = tentative_g_score
                    
                    # f_score = g_score + heuristic
                    f_score = tentative_g_score + self.calculate_heuristic(
                        next_pos, self.goal, new_fuel, new_state.total_cost
                    )
                    self.f_score[new_state_key] = f_score
                    
                    # Thêm vào hàng đợi ưu tiên và đánh dấu đã thăm
                    heapq.heappush(self.open_set, (f_score, new_state))
                    self.add_visited(next_pos)
        
        # Kết thúc thuật toán không tìm thấy đường đi
        print(f"WARNING: Không tìm thấy đường đi từ {start} đến {goal} sau {self.steps} bước")
        return []  # Trả về danh sách rỗng
    
    def get_neighbors(self, pos: Tuple[int, int]) -> List[Tuple[int, int]]:
        """Lấy danh sách các vị trí kề có thể đi được.
        
        Sử dụng phương thức get_neighbors từ lớp BaseSearch để lấy các ô
        lân cận hợp lệ (không phải chướng ngại vật và nằm trong phạm vi bản đồ).
        
        Args:
            pos: Vị trí hiện tại (x, y)
            
        Returns:
            Danh sách các vị trí lân cận có thể đi được
        """
        # Sử dụng phương thức cha từ BaseSearch (đã lọc bỏ ô chướng ngại vật)
        return super().get_neighbors(pos)
    
    def step(self) -> bool:
        """Thực hiện một bước của thuật toán A*.
        
        Phương thức này thực hiện một bước đơn lẻ của thuật toán A*, được sử dụng
        cho việc hiển thị trực quan từng bước. Tương tự như search() nhưng
        chỉ xử lý một nút mỗi lần gọi.
        
        Returns:
            bool: True nếu đã hoàn thành (tìm thấy đích hoặc không còn nút để xét),
                 False nếu cần tiếp tục thực hiện các bước tiếp theo
        """
        # Kiểm tra nếu hàng đợi rỗng - kết thúc tìm kiếm
        if not self.open_set:
            self.current_position = None  # Đặt lại vị trí hiện tại để báo hiệu kết thúc
            return True  # Đã hoàn thành (không tìm thấy đường đi)
        
        # Tăng bộ đếm số bước
        self.steps += 1
        
        # Lấy trạng thái có f_score thấp nhất từ open_set
        _, current_state = heapq.heappop(self.open_set)
        current_pos = current_state.position
        
        # Cập nhật vị trí hiện tại cho hiển thị trực quan
        self.current_position = current_pos
        
        # Kiểm tra xem đã đến đích chưa
        if current_state.position == self.goal:
            # Đã tìm thấy đích - lưu lại các thông tin cần thiết
            self.current_path = current_state.path  # Đường đi từ trạng thái tìm kiếm
            self.path_length = len(self.current_path) - 1  # Độ dài đường đi
            
            # Lưu lại thông tin về nhiên liệu và chi phí
            self.current_fuel = current_state.fuel  # Nhiên liệu còn lại khi đến đích
            self.current_total_cost = current_state.total_cost  # Tổng chi phí
            
            # Tính chi phí riêng cho trạm thu phí và nhiên liệu
            toll_cost = len(current_state.toll_stations_visited) * self.TOLL_BASE_COST
            self.current_toll_cost = toll_cost  # Chi phí trạm thu phí
            self.current_fuel_cost = current_state.total_cost - toll_cost  # Chi phí nhiên liệu
            
            # Tính toán chi tiết về tiêu thụ nhiên liệu cho đường đi cuối cùng
            self.calculate_path_fuel_consumption(self.current_path)
            
            # Kiểm tra tính khả thi của đường đi
            is_feasible, reason = self.is_path_feasible(self.current_path, self.MAX_FUEL)
            if not is_feasible:
                print(f"ERROR: Đường đi tìm được không khả thi: {reason}")
                self.current_path = []  # Đặt đường đi rỗng để báo hiệu không tìm thấy đường đi khả thi
                self.current_position = None
                return True  # Kết thúc tìm kiếm
            
            # Đã tìm thấy đường đi khả thi đến đích
            self.current_position = None  # Đặt lại vị trí hiện tại để báo hiệu kết thúc
            return True  # Đã hoàn thành tìm kiếm
        
        # Đánh dấu trạng thái hiện tại đã xét
        self.closed_set.add(current_state.get_state_key())
        
        # Xử lý các ô lân cận theo thứ tự ưu tiên
        for next_pos in self.get_neighbors_with_priority(current_state.position, current_state.fuel):
            # Kiểm tra chắc chắn rằng vị trí tiếp theo không phải là chướng ngại vật
            try:
                cell_value = self.grid[next_pos[1], next_pos[0]]
                if (cell_value == self.OBSTACLE_CELL or 
                    int(cell_value) == int(self.OBSTACLE_CELL) or 
                    cell_value < 0):
                    continue  # Bỏ qua ô chướng ngại vật
            except IndexError:
                continue  # Bỏ qua nếu nằm ngoài lưới
            
            # Kiểm tra xem đã thăm trạng thái này chưa
            state_key = (next_pos, round(current_state.fuel, 2))
            if state_key in self.closed_set:
                continue  # Bỏ qua trạng thái đã xét
            
            # Tính toán chi phí, nhiên liệu và tiền mới
            new_fuel, move_cost, new_money = self.calculate_cost(current_state, next_pos)
            
            # Kiểm tra tính khả thi về nhiên liệu và tiền
            if new_fuel < 0 or new_money < 0:
                continue  # Bỏ qua vì không đủ nhiên liệu hoặc tiền
            
            # Tạo đường đi mới bằng cách thêm vị trí tiếp theo
            new_path = current_state.path + [next_pos]
            
            # Kiểm tra tính liên tục của đường đi
            prev_pos = current_state.position
            if abs(prev_pos[0] - next_pos[0]) + abs(prev_pos[1] - next_pos[1]) > 1:
                print(f"WARNING: Phát hiện đường đi không liên tục từ {prev_pos} đến {next_pos}. Bỏ qua.")
                continue  # Bỏ qua đường đi không liên tục
            
            # Tạo trạng thái mới với thông tin cập nhật
            new_state = SearchState(
                position=next_pos,
                fuel=new_fuel,
                total_cost=current_state.total_cost + move_cost,
                money=new_money,
                path=new_path,
                visited_gas_stations=current_state.visited_gas_stations.copy(),
                toll_stations_visited=current_state.toll_stations_visited.copy()
            )
            
            # Cập nhật danh sách trạm đã ghé qua
            if cell_value == self.GAS_STATION_CELL:  # Trạm xăng
                new_state.visited_gas_stations.add(next_pos)
            elif cell_value == self.TOLL_CELL:  # Trạm thu phí
                new_state.toll_stations_visited.add(next_pos)
            
            # Lấy khóa duy nhất cho trạng thái mới và kiểm tra lại
            new_state_key = new_state.get_state_key()
            if new_state_key in self.closed_set:
                continue  # Bỏ qua nếu đã thăm
            
            # Tính g_score mới
            tentative_g_score = current_state.total_cost + move_cost
            
            # Cập nhật nếu tìm thấy đường đi tốt hơn đến trạng thái này
            if new_state_key not in self.g_score or tentative_g_score < self.g_score[new_state_key]:
                # Lưu thông tin để tái tạo đường đi
                self.parent[next_pos] = current_pos
                
                # Cập nhật g_score và f_score
                self.g_score[new_state_key] = tentative_g_score
                f_score = tentative_g_score + self.calculate_heuristic(
                    next_pos, self.goal, new_fuel, new_state.total_cost
                )
                self.f_score[new_state_key] = f_score
                
                # Thêm trạng thái mới vào hàng đợi ưu tiên và đánh dấu đã thăm
                heapq.heappush(self.open_set, (f_score, new_state))
                self.add_visited(next_pos)
        
        # Tiếp tục thực hiện các bước tiếp theo
        return False

    def estimate_toll_density(self) -> float:
        """Ước tính mật độ trạm thu phí trên bản đồ.
        
        Phương thức này quét toàn bộ bản đồ để tính tỷ lệ giữa số lượng
        ô trạm thu phí so với tổng số ô có thể đi được (không bao gồm chướng ngại vật).
        Kết quả được lưu trong bộ nhớ đệm (_toll_density) để tránh tính toán lại 
        nhiều lần, giúp cải thiện hiệu suất.
        
        Mật độ này được sử dụng trong hàm heuristic để ước tính số lượng
        trạm thu phí có thể gặp phải trên đường đi đến đích.
        
        Returns:
            float: Tỷ lệ mật độ trạm thu phí (0.0 - 1.0)
        """
        # Sử dụng bộ nhớ đệm để không phải tính lại
        if hasattr(self, '_toll_density'):
            return self._toll_density
        
        # Đếm số ô thuộc loại trạm thu phí và tổng số ô có thể đi được
        toll_count = 0      # Số ô trạm thu phí
        total_cells = 0     # Tổng số ô có thể đi được
        
        # Sử dụng numpy để đếm nhanh chóng
        # Đếm số ô không phải chướng ngại vật
        valid_cells = (self.grid != self.OBSTACLE_CELL)
        total_cells = np.sum(valid_cells)
        
        # Đếm số ô trạm thu phí
        toll_cells = (self.grid == self.TOLL_CELL)
        toll_count = np.sum(toll_cells)
        
        # Tính tỷ lệ và lưu vào bộ nhớ đệm
        if total_cells > 0:
            self._toll_density = toll_count / total_cells
        else:
            self._toll_density = 0
            
        return self._toll_density 