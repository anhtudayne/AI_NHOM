"""
Map module for creating and managing the routing environment.
Defines classes for Map and Node representations, and functions for map generation.
"""

import numpy as np
import json
import os
import datetime
import math
import random
from pathlib import Path

class Map:
    """Base class for the routing environment map."""
    def __init__(self, size):
        """Khởi tạo bản đồ với kích thước size x size"""
        if size < 8:
            size = 8  # Đảm bảo kích thước tối thiểu là 8
        elif size > 15:
            size = 15  # Đảm bảo kích thước tối đa là 15
        self.size = size
        self.grid = np.zeros((size, size), dtype=int)
        self.start_pos = None  # Vị trí bắt đầu của xe
        self.end_pos = None    # Vị trí đích
        
        # Đảm bảo các thuộc tính luôn tồn tại
        if not hasattr(self, 'start_pos'):
            self.start_pos = None
        if not hasattr(self, 'end_pos'):
            self.end_pos = None
    
    def is_obstacle(self, x, y):
        """
        Kiểm tra xem ô tại vị trí (x, y) có phải là vật cản hay không.
        
        Args:
            x (int): Tọa độ x
            y (int): Tọa độ y
            
        Returns:
            bool: True nếu là vật cản, False nếu không
        """
        from .constants import CellType
        
        # Kiểm tra xem vị trí có nằm trong bản đồ không
        if 0 <= x < self.size and 0 <= y < self.size:
            return self.grid[y, x] == CellType.OBSTACLE.value
        
        # Nếu vị trí nằm ngoài bản đồ, coi như vật cản
        return True
    
    def has_path_from_start_to_end(self):
        """
        Kiểm tra có tồn tại đường đi từ start đến end (chỉ đi qua các ô không phải vật cản).
        Trả về True nếu có đường đi, False nếu không.
        """
        if self.start_pos is None or self.end_pos is None:
            return False
        from collections import deque
        visited = set()
        queue = deque([self.start_pos])
        while queue:
            pos = queue.popleft()
            if pos == self.end_pos:
                return True
            visited.add(pos)
            x, y = pos
            for dx, dy in [(-1,0),(1,0),(0,-1),(0,1)]:
                nx, ny = x+dx, y+dy
                if 0 <= nx < self.size and 0 <= ny < self.size:
                    if (nx, ny) not in visited and self.grid[ny][nx] >= 0:
                        queue.append((nx, ny))
        return False

    def find_path_bfs(self, start_pos=None, end_pos=None):
        """
        Tìm đường đi ngắn nhất từ start_pos đến end_pos bằng BFS.
        Chỉ đi qua các ô không phải vật cản.
        Trả về danh sách các tọa độ (path) nếu tìm thấy, ngược lại trả về None.
        """
        _start = start_pos if start_pos else self.start_pos
        _end = end_pos if end_pos else self.end_pos

        if _start is None or _end is None:
            return None
        
        from collections import deque
        # Hàng đợi lưu (vị trí, đường đi tới vị trí đó)
        queue = deque([(_start, [_start])]) 
        visited = set([_start])

        while queue:
            (current_pos, path) = queue.popleft()

            if current_pos == _end:
                return path # Tìm thấy đường đi

            x, y = current_pos
            for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]: # Khám phá lân cận
                nx, ny = x + dx, y + dy
                next_pos = (nx, ny)
                # Kiểm tra hợp lệ: trong bản đồ, không phải vật cản, chưa thăm
                if 0 <= nx < self.size and 0 <= ny < self.size and \
                   self.grid[ny][nx] >= 0 and next_pos not in visited:
                    
                    visited.add(next_pos)
                    new_path = list(path) # Tạo bản sao đường đi
                    new_path.append(next_pos)
                    queue.append((next_pos, new_path))
        
        return None # Không tìm thấy đường đi

    @classmethod
    def generate_random(cls, size: int, num_tolls: int, num_gas: int, num_obstacles: int, max_attempts=50):
        """
        Tạo bản đồ ngẫu nhiên với kích thước và số lượng điểm đặc biệt cho trước.
        Cố gắng tạo tối đa max_attempts lần để đảm bảo có đường đi từ start đến end.

        Args:
            size: Kích thước bản đồ.
            num_tolls: Số lượng trạm thu phí mong muốn.
            num_gas: Số lượng trạm xăng mong muốn.
            num_obstacles: Số lượng vật cản mong muốn.
            max_attempts: Số lần thử tối đa để tạo bản đồ hợp lệ.
            
        Returns:
            Map: Đối tượng bản đồ ngẫu nhiên đã tạo, hoặc None nếu không thành công.
        """
        for _ in range(max_attempts):
            # Gọi phương thức tạo một lần bên trong vòng lặp
            map_obj = cls._generate_random_once(size, num_tolls, num_gas, num_obstacles)
            
            # Kiểm tra xem map có được tạo và có đường đi không
            if map_obj and map_obj.has_path_from_start_to_end():
                return map_obj # Trả về bản đồ hợp lệ
                
        # Nếu vòng lặp kết thúc mà không tìm được bản đồ hợp lệ
        print(f"[Warning] Map.generate_random: Failed to generate a solvable map after {max_attempts} attempts with params: size={size}, tolls={num_tolls}, gas={num_gas}, obstacles={num_obstacles}")
        return None # Trả về None nếu không tạo được map hợp lệ sau max_attempts

    @classmethod
    def _generate_random_once(cls, size: int, num_tolls: int, num_gas: int, num_obstacles: int):
        """Tạo một bản đồ ngẫu nhiên (một lần thử) với số lượng điểm đặc biệt.
        Lưu ý: Không đảm bảo có đường đi từ start đến end trong một lần thử này.
        """
        map_obj = cls(size)
        # Khởi tạo grid với toàn bộ là đường (0), sẽ đặt vật cản sau
        grid = np.zeros((size, size), dtype=int) 

        # --- SỬ DỤNG TRỰC TIẾP SỐ LƯỢNG num_tolls, num_gas, num_obstacles --- 
        # Các giá trị này đã được truyền vào, không cần tính từ ratio nữa.
        
        # Để theo dõi các vị trí đã được sử dụng bởi các phần tử đặc biệt
        occupied_positions = set()
        
        # Bước 1.5: Chọn và đặt Start/End Position vào occupied_positions
        # Cố gắng chọn start/end không trùng nhau và không ở rìa bản đồ nếu có thể
        start_pos = None
        end_pos = None
        for _ in range(size*size): # Giới hạn số lần thử
             # Chọn ngẫu nhiên trên toàn bộ bản đồ, không ưu tiên khu vực giữa
             # pad = max(1, size // 4) # Bỏ phần padding
             start_y = np.random.randint(0, size) # Chọn từ 0 đến size-1
             start_x = np.random.randint(0, size) # Chọn từ 0 đến size-1
             end_y = np.random.randint(0, size)   # Chọn từ 0 đến size-1
             end_x = np.random.randint(0, size)   # Chọn từ 0 đến size-1
             
             _start_pos = (start_x, start_y)
             _end_pos = (end_x, end_y)

             if _start_pos != _end_pos:
                 start_pos = _start_pos
                 end_pos = _end_pos
                 occupied_positions.add(start_pos)
                 occupied_positions.add(end_pos)
                 map_obj.start_pos = start_pos
                 map_obj.end_pos = end_pos
                 break # Đã tìm được vị trí hợp lệ

        if start_pos is None or end_pos is None:
             print("[Warning] _generate_random_once: Could not find distinct start/end positions.")
             # Nếu không tìm được sau nhiều lần thử, chọn ngẫu nhiên bất kỳ
             start_pos = (np.random.randint(0, size), np.random.randint(0, size))
             end_pos = start_pos
             while end_pos == start_pos:
                  end_pos = (np.random.randint(0, size), np.random.randint(0, size))
             occupied_positions.add(start_pos)
             occupied_positions.add(end_pos)
             map_obj.start_pos = start_pos
             map_obj.end_pos = end_pos

        # Bước 2: Đặt trạm xăng (Cố gắng phân bổ đều)
        gas_positions = []
        gas_placed_count = 0
        max_gas_attempts = num_gas * 30 # Tăng nhẹ số lần thử tổng thể
        current_gas_attempts = 0
        mid_x, mid_y = size // 2, size // 2
        # Theo dõi các góc phần tư đã có trạm xăng (0: TL, 1: TR, 2: BL, 3: BR)
        quadrants_filled = {0: False, 1: False, 2: False, 3: False}
        num_quadrants_filled = 0

        while gas_placed_count < num_gas and current_gas_attempts < max_gas_attempts:
            current_gas_attempts += 1
            row, col = np.random.randint(0, size), np.random.randint(0, size)
            pos = (col, row) # Lưu ý: tuple là (x, y) -> (col, row)

            # Chỉ đặt nếu là đường (0) và chưa bị chiếm
            if grid[row, col] == 0 and pos not in occupied_positions:
                # Xác định góc phần tư
                quadrant = -1
                if col < mid_x and row < mid_y: quadrant = 0 # Top-Left
                elif col >= mid_x and row < mid_y: quadrant = 1 # Top-Right
                elif col < mid_x and row >= mid_y: quadrant = 2 # Bottom-Left
                elif col >= mid_x and row >= mid_y: quadrant = 3 # Bottom-Right

                # Ưu tiên đặt vào góc chưa có trạm xăng (nếu còn góc trống)
                place_anyway = (num_quadrants_filled >= 4 or num_quadrants_filled >= num_gas)
                
                if quadrant != -1 and (not quadrants_filled[quadrant] or place_anyway):
                    grid[row, col] = 2  # Đặt trạm xăng
                    gas_positions.append(pos)
                    occupied_positions.add(pos)
                    gas_placed_count += 1
                    if not quadrants_filled[quadrant]:
                        quadrants_filled[quadrant] = True
                        num_quadrants_filled += 1
                elif num_quadrants_filled >= 4: # Nếu tất cả các góc đã đầy, đặt ngẫu nhiên vào ô hợp lệ
                    grid[row, col] = 2
                    gas_positions.append(pos)
                    occupied_positions.add(pos)
                    gas_placed_count += 1
                    
            # Nếu đã thử nhiều lần mà chưa đặt đủ, dừng lại
            # Điều này ít có khả năng xảy ra hơn với logic mới nhưng vẫn là một biện pháp phòng ngừa
            if current_gas_attempts > max_gas_attempts * 0.8 and gas_placed_count < num_gas // 2:
                 print(f"[Warning] Struggling to place gas stations ({gas_placed_count}/{num_gas} after {current_gas_attempts} attempts), may stop early.")
                 # break # Có thể break sớm nếu muốn

        if gas_placed_count < num_gas:
            print(f"[Warning] _generate_random_once: Placed only {gas_placed_count}/{num_gas} gas stations.")

        # Bước 3: Đặt trạm thu phí
        toll_positions = []
        toll_placed_count = 0
        max_toll_attempts = num_tolls * 20 # Tăng số lần thử
        current_toll_attempts = 0
        while toll_placed_count < num_tolls and current_toll_attempts < max_toll_attempts:
            current_toll_attempts += 1
            row, col = np.random.randint(0, size), np.random.randint(0, size)
            pos = (col, row)
            # Chỉ đặt nếu là đường và chưa bị chiếm
            if grid[row, col] == 0 and pos not in occupied_positions:
                 grid[row, col] = 1  # Đặt trạm thu phí
                 toll_positions.append(pos)
                 occupied_positions.add(pos)
                 toll_placed_count += 1
        if toll_placed_count < num_tolls:
            print(f"[Warning] _generate_random_once: Placed only {toll_placed_count}/{num_tolls} toll stations.")
        
        # --- Bước 4: Đặt các vật cản (Chiến lược mới) ---
        
        # 4.1. Tìm đường đi "dễ" ban đầu (trước khi có vật cản)
        initial_path = map_obj.find_path_bfs()
        
        obstacle_positions = []
        obstacle_placed_count = 0
        # Số vật cản đặt chiến lược (gần đường đi dễ)
        num_strategic_obstacles = int(num_obstacles * 0.7) 
        # Số vật cản đặt ngẫu nhiên
        num_random_obstacles = num_obstacles - num_strategic_obstacles 
        
        max_obstacle_attempts = num_obstacles * 15 # Tăng số lần thử tổng thể
        current_obstacle_attempts = 0

        # 4.2. Đặt vật cản chiến lược (gần đường đi dễ)
        if initial_path and len(initial_path) > 2: # Chỉ đặt nếu có đường đi và đủ dài
            # Tạo danh sách các ô mục tiêu: ô trên đường đi (trừ start/end) và các ô lân cận
            strategic_target_cells = set()
            path_cells = set(initial_path[1:-1]) # Các ô trên đường, trừ start/end
            strategic_target_cells.update(path_cells)
            
            for cell in path_cells:
                x, y = cell
                for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1), (-1,-1), (-1,1), (1,-1), (1,1)]: # 8 hướng lân cận
                    nx, ny = x + dx, y + dy
                    adj_pos = (nx, ny)
                    if 0 <= nx < size and 0 <= ny < size and grid[ny][nx] == 0 and adj_pos not in occupied_positions:
                         strategic_target_cells.add(adj_pos)
            
            strategic_target_list = list(strategic_target_cells - set([map_obj.start_pos, map_obj.end_pos]))
            random.shuffle(strategic_target_list) # Xáo trộn để lấy ngẫu nhiên

            placed_strategic = 0
            target_idx = 0
            while placed_strategic < num_strategic_obstacles and target_idx < len(strategic_target_list) and current_obstacle_attempts < max_obstacle_attempts:
                current_obstacle_attempts += 1
                pos = strategic_target_list[target_idx]
                row, col = pos[1], pos[0]
                
                # Kiểm tra lại lần nữa (phòng trường hợp occupied_positions thay đổi)
                if grid[row, col] == 0 and pos not in occupied_positions:
                    grid[row, col] = -1  # Đặt vật cản
                    obstacle_positions.append(pos)
                    occupied_positions.add(pos)
                    obstacle_placed_count += 1
                    placed_strategic += 1
                target_idx += 1
            
            if placed_strategic < num_strategic_obstacles:
                 print(f"[Warning] Placed only {placed_strategic}/{num_strategic_obstacles} strategic obstacles near the initial path.")
                 # Bổ sung số lượng thiếu vào phần đặt ngẫu nhiên
                 num_random_obstacles += (num_strategic_obstacles - placed_strategic)

        else: # Nếu không có đường đi ban đầu hoặc quá ngắn, đặt tất cả ngẫu nhiên
             print("[Info] No initial path found or path too short. Placing all obstacles randomly.")
             num_random_obstacles = num_obstacles

        # 4.3. Đặt vật cản ngẫu nhiên (số còn lại hoặc tất cả nếu không có chiến lược)
        placed_random = 0
        while placed_random < num_random_obstacles and current_obstacle_attempts < max_obstacle_attempts:
             current_obstacle_attempts += 1
             row, col = np.random.randint(0, size), np.random.randint(0, size)
             pos = (col, row)
             # Chỉ đặt nếu là đường và chưa bị chiếm
             if grid[row, col] == 0 and pos not in occupied_positions:
                 grid[row, col] = -1  # Đặt vật cản
                 obstacle_positions.append(pos)
                 occupied_positions.add(pos)
                 obstacle_placed_count += 1
                 placed_random += 1
        
        if obstacle_placed_count < num_obstacles:
            print(f"[Warning] _generate_random_once: Placed only {obstacle_placed_count}/{num_obstacles} total obstacles.")

        # Đảm bảo start_pos và end_pos cuối cùng không phải là vật cản
        if map_obj.start_pos is not None and grid[map_obj.start_pos[1], map_obj.start_pos[0]] == -1:
             grid[map_obj.start_pos[1], map_obj.start_pos[0]] = 0 # Đảm bảo là đường đi
        if map_obj.end_pos is not None and grid[map_obj.end_pos[1], map_obj.end_pos[0]] == -1:
             grid[map_obj.end_pos[1], map_obj.end_pos[0]] = 0 # Đảm bảo là đường đi

        map_obj.grid = grid
        return map_obj
    
    @classmethod
    def create_demo_map(cls, size=8):
        """
        Tạo một bản đồ mẫu với cấu trúc định sẵn
        
        Parameters:
        - size: Kích thước bản đồ (tối thiểu 8x8)
        """
        # Đảm bảo kích thước tối thiểu
        if size < 8:
            size = 8
        elif size > 15:
            size = 15
        map_obj = cls(size)
        
        # Tạo bản đồ với tất cả các ô là vật cản
        for i in range(size):
            for j in range(size):
                map_obj.grid[i][j] = -1
        
        # Tạo đường thẳng ngang và dọc
        mid = size // 2
        
        # Đường ngang giữa
        for j in range(size):
            map_obj.grid[mid][j] = 0
        
        # Đường dọc giữa
        for i in range(size):
            map_obj.grid[i][mid] = 0
        
        # Tạo đường vành đai bên ngoài
        for i in range(1, size-1):
            map_obj.grid[1][i] = 0  # Hàng trên
            map_obj.grid[size-2][i] = 0  # Hàng dưới
            map_obj.grid[i][1] = 0  # Cột trái
            map_obj.grid[i][size-2] = 0  # Cột phải
        
        # Sử dụng set để theo dõi vị trí đã được sử dụng
        occupied_positions = set()
        
        # Đặt trạm thu phí ở bốn hướng chính
        toll_positions = [
            (1, mid),      # Trạm thu phí phía trên
            (size-2, mid), # Trạm thu phí phía dưới
            (mid, 1),      # Trạm thu phí bên trái
            (mid, size-2)  # Trạm thu phí bên phải
        ]
        
        for pos in toll_positions:
            row, col = pos
            map_obj.grid[row][col] = 1  # Đặt trạm thu phí
            occupied_positions.add(pos)
        
        # Đặt trạm xăng ở bốn góc và trung tâm
        gas_positions = [
            (1, 1),            # Trạm xăng ở góc trên trái
            (1, size-2),       # Trạm xăng ở góc trên phải
            (size-2, 1),       # Trạm xăng ở góc dưới trái
            (size-2, size-2),  # Trạm xăng ở góc dưới phải
            (mid, mid)         # Trạm xăng ở trung tâm
        ]
        
        for pos in gas_positions:
            row, col = pos
            # Kiểm tra xem vị trí này đã bị chiếm chưa
            if pos not in occupied_positions:
                map_obj.grid[row][col] = 2  # Đặt trạm xăng
                occupied_positions.add(pos)
        
        # Mở rộng đường đi ở giữa để tạo không gian
        for i in range(mid-1, mid+2):
            for j in range(mid-1, mid+2):
                if i >= 0 and i < size and j >= 0 and j < size:
                    pos = (i, j)
                    if not (i == mid and j == mid) and pos not in occupied_positions:  # Không thay đổi trạm xăng ở trung tâm
                        map_obj.grid[i][j] = 0
        
        # Tạo khu vực đường đi bên trong
        quarter = size // 4
        for i in range(mid-quarter, mid+quarter+1):
            for j in range(mid-quarter, mid+quarter+1):
                if i >= 0 and i < size and j >= 0 and j < size:
                    pos = (i, j)
                    if pos not in occupied_positions:
                        # Làm cho khu vực bên trong thông thoáng với đường đi
                        if random.random() < 0.7:  # 70% là đường đi
                            map_obj.grid[i][j] = 0
        
        # Tạo một vài mê cung đường đi ngẫu nhiên 
        for _ in range(3):
            start_i = random.randint(2, size-3)
            start_j = random.randint(2, size-3)
            
            if map_obj.grid[start_i][start_j] == -1:  # Nếu là vật cản
                # Tạo đường đi ngẫu nhiên
                for length in range(random.randint(3, 6)):
                    directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]  # Phải, xuống, trái, lên
                    di, dj = random.choice(directions)
                    
                    for steps in range(random.randint(2, 4)):
                        next_i, next_j = start_i + di * steps, start_j + dj * steps
                        pos = (next_i, next_j)
                        if (0 <= next_i < size and 0 <= next_j < size and
                            pos not in occupied_positions):
                            map_obj.grid[next_i][next_j] = 0
        
        # Đảm bảo góc trên bên trái và góc dưới bên phải là đường đi để đặt điểm bắt đầu và điểm đích
        map_obj.grid[0][0] = 0  # Góc trên bên trái - Điểm bắt đầu
        map_obj.grid[size-1][size-1] = 0  # Góc dưới bên phải - Điểm đích
        
        # Đặt vị trí bắt đầu tại góc trên bên trái
        map_obj.start_pos = (0, 0)
        occupied_positions.add(map_obj.start_pos)
        
        # Đặt điểm đích tại góc dưới bên phải
        map_obj.end_pos = (size-1, size-1)
        occupied_positions.add(map_obj.end_pos)
        
        # Đảm bảo có đường đi từ điểm bắt đầu đến điểm đích
        # Tạo đường đi từ trên xuống dưới
        for i in range(size):
            pos = (i, 0)
            if pos not in occupied_positions:
                map_obj.grid[i][0] = 0  # Cột đầu tiên
            
            pos = (i, size-1)
            if pos not in occupied_positions:
                map_obj.grid[i][size-1] = 0  # Cột cuối cùng
        
        # Tạo đường đi từ trái sang phải
        for j in range(size):
            pos = (0, j)
            if pos not in occupied_positions:
                map_obj.grid[0][j] = 0  # Hàng đầu tiên
            
            pos = (size-1, j)
            if pos not in occupied_positions:
                map_obj.grid[size-1][j] = 0  # Hàng cuối cùng
        
        return map_obj
    
    def get_statistics(self):
        """Lấy thống kê về số lượng các loại ô trên bản đồ"""
        return {
            'normal_roads': np.sum(self.grid == 0),
            'toll_stations': np.sum(self.grid == 1),
            'gas_stations': np.sum(self.grid == 2),
            'brick_cells': np.sum(self.grid == -1)
        }
    
    def get_cell_type(self, position):
        """
        Lấy loại ô tại vị trí xác định.
        
        Args:
            position (tuple): Tọa độ (x, y) của ô cần kiểm tra
            
        Returns:
            CellType: Loại ô tại vị trí đó (từ enum CellType)
        """
        from .constants import CellType
        
        x, y = position
        if 0 <= x < self.size and 0 <= y < self.size:
            value = self.grid[y, x]
            # Ánh xạ giá trị trong grid với enum CellType
            if value == 0:
                return CellType.ROAD
            elif value == 1:
                return CellType.TOLL
            elif value == 2:
                return CellType.GAS
            elif value == -1:
                return CellType.OBSTACLE
        
        # Ngoài bản đồ hoặc giá trị không xác định - trả về OBSTACLE
        return CellType.OBSTACLE
    
    def save(self, full_filepath: str):
        """
        Lưu bản đồ vào file với đường dẫn đầy đủ được cung cấp.
        
        Parameters:
        - full_filepath: Đường dẫn đầy đủ (bao gồm cả tên file) để lưu.
        
        Returns:
        - Đường dẫn đến file đã lưu (chính là full_filepath)
        """
        # Đảm bảo thư mục cha tồn tại
        parent_dir = os.path.dirname(full_filepath)
        if parent_dir and not os.path.exists(parent_dir):
            os.makedirs(parent_dir, exist_ok=True)
        
        # Đảm bảo full_filepath có đuôi .json (nếu cần)
        # Hiện tại, giả định generate_maps sẽ đảm bảo tên file có .json
        # if not full_filepath.endswith('.json'):
        #     full_filepath += '.json' # Hoặc raise error
        
        data = {
            'size': self.size,
            'grid': self.grid.tolist(),
            'start_pos': self.start_pos,
            'end_pos': self.end_pos,
            'timestamp': datetime.datetime.now().isoformat()
        }
        
        # Lưu vào file chỉ định
        with open(full_filepath, 'w') as f:
            json.dump(data, f)
        
        # Tạm thời comment out việc lưu 'latest_map.json'
        # # Lưu vào latest_map.json
        # # Cần xác định vị trí hợp lý cho latest_map.json nếu vẫn muốn giữ
        # # Ví dụ, lưu vào thư mục gốc của MAPS_DIR
        # global_maps_dir_path = Path(full_filepath).parents[2] # Giả định cấu trúc MAPS_DIR/session/type/file
        # # This assumption about parents[2] is fragile.
        # # A better way would be to pass MAPS_DIR root if this feature is needed.
        # try:
        #     # Attempt to save latest_map.json if a root 'maps' dir can be inferred or is configured
        #     # For now, this is disabled to prevent errors.
        #     # For example, if we knew the root of all maps storage:
        #     # maps_root = get_configured_maps_root() # Placeholder for a function to get this
        #     # if maps_root and (maps_root / 'latest_map.json').parent.exists():
        #     #    with open(maps_root / 'latest_map.json', 'w') as f:
        #     #        json.dump(data, f)
        #     pass
        # except Exception as e:
        #     print(f"DEBUG: Could not save latest_map.json due to: {e}")
            
        return full_filepath
    
    @classmethod
    def load(cls, full_filepath: str):
        """
        Tải bản đồ từ file với đường dẫn đầy đủ.
        
        Parameters:
        - full_filepath: Đường dẫn đầy đủ đến file bản đồ.
        
        Returns:
        - Map object hoặc None nếu không tìm thấy file hoặc lỗi.
        """
        try:
            # from pathlib import Path # Import không cần thiết ở đây nếu chỉ dùng os.path
            
            # print(f"DEBUG: Attempting to load map from: {full_filepath}") # Giảm bớt log thừa
                
            if not os.path.exists(full_filepath):
                 print(f"ERROR: Map file not found at specified path: {full_filepath}")
                 # Check common relative path from CWD if it's not absolute
                 if not os.path.isabs(full_filepath):
                     alt_path_from_cwd = Path(full_filepath).resolve()
                     if alt_path_from_cwd.exists():
                         print(f"INFO: Found map at resolved CWD relative path: {alt_path_from_cwd}")
                         full_filepath = str(alt_path_from_cwd)
                     else:
                         # Try to see if it was meant to be relative to a 'maps' dir from CWD
                         # This is a legacy fallback and ideally should not be needed
                         # if `auto_train_rl.py` correctly constructs full paths.
                         legacy_path = Path('maps') / full_filepath
                         if legacy_path.exists():
                             print(f"WARNING: Map found in legacy 'maps/{full_filepath}'. Consider using full paths.")
                             full_filepath = str(legacy_path.resolve())
                         else:
                             return None # Truly not found
            
            with open(full_filepath, 'r') as f:
                data = json.load(f)
            
            map_obj = cls(data['size'])
            map_obj.grid = np.array(data['grid'])
            
            map_obj.start_pos = data.get('start_pos') 
            map_obj.end_pos = data.get('end_pos')
            map_obj.filename = Path(full_filepath).name # Store the filename
            
            # print(f"DEBUG: Loaded map '{full_filepath}' with size: {map_obj.size}x{map_obj.size}")
            # print(f"DEBUG: Start: {map_obj.start_pos}, End: {map_obj.end_pos}")
            
            return map_obj
        except FileNotFoundError: # Should be caught by os.path.exists now mostly
            print(f"ERROR: FileNotFoundError for map file: {full_filepath}")
            return None
        except json.JSONDecodeError as e:
            print(f"ERROR: Failed to decode JSON from map file: {full_filepath} - {str(e)}")
            return None
        except Exception as e:
            print(f"ERROR: Failed to load map '{full_filepath}': {str(e)}")
            # import traceback # For more detailed debugging if needed
            # print(traceback.format_exc())
            return None

class Node:
    """Class representing a node/cell in the map."""
    pass