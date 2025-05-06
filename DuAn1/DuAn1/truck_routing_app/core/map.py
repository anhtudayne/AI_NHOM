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
    
    @classmethod
    def generate_random(cls, size, toll_ratio, gas_ratio, brick_ratio):
        """
        Tạo bản đồ ngẫu nhiên với tỷ lệ các loại ô cho trước
        
        Parameters:
        - size: Kích thước bản đồ
        - toll_ratio: Tỷ lệ ô trạm thu phí
        - gas_ratio: Tỷ lệ ô đổ xăng
        - brick_ratio: Tỷ lệ ô gạch (không đi được)
        """
        map_obj = cls(size)
        total_cells = size * size
        
        # Tính số lượng ô cho mỗi loại
        num_toll = int(total_cells * toll_ratio)
        num_gas = int(total_cells * gas_ratio)
        num_brick = int(total_cells * brick_ratio)
        
        # Đảm bảo số lượng phù hợp với kích thước bản đồ
        # Giới hạn số lượng trạm xăng dựa trên kích thước bản đồ
        max_gas_stations = max(1, min(5, size // 3))
        num_gas = min(num_gas, max_gas_stations)
        
        # Giới hạn số lượng trạm thu phí dựa trên kích thước bản đồ
        max_toll_stations = max(1, min(8, size // 2))
        num_toll = min(num_toll, max_toll_stations)
        
        # Giảm tỷ lệ vật cản để đảm bảo tính liền mạch
        max_brick_ratio = 0.3  # Giảm tỷ lệ tối đa xuống 30%
        if brick_ratio > max_brick_ratio:
            brick_ratio = max_brick_ratio
            num_brick = int(total_cells * brick_ratio)
        
        total_special_cells = num_toll + num_gas + num_brick
        
        # Đảm bảo không quá nhiều ô đặc biệt
        if total_special_cells > total_cells * 0.5:  # Giảm tỷ lệ tối đa xuống 50%
            # Giảm theo tỷ lệ
            scale_factor = (total_cells * 0.5) / total_special_cells
            num_toll = int(num_toll * scale_factor)
            num_gas = int(num_gas * scale_factor)
            num_brick = int(num_brick * scale_factor)
            total_special_cells = num_toll + num_gas + num_brick
        
        # Tạo grid ban đầu toàn đường thường
        grid = np.zeros((size, size), dtype=int)
        
        # Bước 1: Tạo các đường chính xuyên qua bản đồ
        # Đường ngang chính
        mid_row = size // 2
        for j in range(size):
            grid[mid_row][j] = 0
        
        # Đường dọc chính
        mid_col = size // 2
        for i in range(size):
            grid[i][mid_col] = 0
        
        # Đường chéo chính
        for i in range(size):
            grid[i][i] = 0
        
        # Bước 2: Đặt trạm xăng - đảm bảo phân bố đều
        if num_gas > 0:
            # Chia bản đồ thành các khu vực để đảm bảo phân bố đều trạm xăng
            sections = math.ceil(math.sqrt(num_gas))
            section_size = size / sections
            
            gas_positions = []
            gas_placed = 0
            
            # Đặt trạm xăng ở từng vùng, đảm bảo khoảng cách hợp lý
            for i in range(sections):
                for j in range(sections):
                    if gas_placed < num_gas:
                        # Tính khoảng giữa của vùng
                        center_row = int((i + 0.5) * section_size)
                        center_col = int((j + 0.5) * section_size)
                        
                        # Thêm nhiễu ngẫu nhiên để tránh quá đều
                        offset = int(section_size / 4)
                        row = min(max(0, center_row + np.random.randint(-offset, offset)), size-1)
                        col = min(max(0, center_col + np.random.randint(-offset, offset)), size-1)
                        
                        # Chỉ đặt trạm xăng trên đường thông thường
                        if grid[row][col] == 0:
                            grid[row][col] = 2  # Đặt trạm xăng
                            gas_positions.append((row, col))
                            gas_placed += 1
        
        # Bước 3: Đặt trạm thu phí - tránh đặt quá gần trạm xăng và đảm bảo phân bố đều
        if num_toll > 0:
            min_distance_from_gas = max(2, size // 6)  # Khoảng cách tối thiểu từ trạm xăng
            toll_positions = []
            
            # Đặt trạm thu phí dọc theo các đường chính
            for _ in range(num_toll * 2):
                if len(toll_positions) >= num_toll:
                    break
                    
                # Chọn ngẫu nhiên một vị trí trên đường chính
                if np.random.random() < 0.5:
                    # Chọn trên đường ngang
                    row = mid_row
                    col = np.random.randint(0, size)
                else:
                    # Chọn trên đường dọc
                    row = np.random.randint(0, size)
                    col = mid_col
                
                # Kiểm tra khoảng cách với các trạm xăng
                too_close_to_gas = False
                for gas_row, gas_col in gas_positions:
                    distance = math.sqrt((gas_row - row)**2 + (gas_col - col)**2)
                    if distance < min_distance_from_gas:
                        too_close_to_gas = True
                        break
                
                # Kiểm tra khoảng cách với các trạm thu phí khác
                too_close_to_toll = False
                min_toll_distance = max(2, size // 8)
                for toll_row, toll_col in toll_positions:
                    distance = math.sqrt((toll_row - row)**2 + (toll_col - col)**2)
                    if distance < min_toll_distance:
                        too_close_to_toll = True
                        break
                
                # Nếu vị trí phù hợp, đặt trạm thu phí
                if not too_close_to_gas and not too_close_to_toll and grid[row][col] == 0:
                    grid[row][col] = 1  # Đặt trạm thu phí
                    toll_positions.append((row, col))
        
        # Bước 4: Đặt các vật cản - tạo các nhóm và đường ngăn cách
        if num_brick > 0:
            # Đảm bảo các vật cản không chặn trạm xăng và trạm thu phí
            all_special_positions = gas_positions + toll_positions
            
            # Chiến lược: Tạo các khối vật cản nhỏ và phân tán
            num_clusters = min(2, size // 4)  # Giảm số lượng cụm
            bricks_per_cluster = num_brick // (num_clusters + 1)
            clusters_placed = 0
            
            for _ in range(num_clusters):
                # Chọn tâm cho cụm vật cản
                center_row = np.random.randint(size // 4, size - size // 4)
                center_col = np.random.randint(size // 4, size - size // 4)
                
                # Kiểm tra xem tâm có gần các đối tượng đặc biệt không
                too_close = False
                for sp_row, sp_col in all_special_positions:
                    if abs(sp_row - center_row) < 3 and abs(sp_col - center_col) < 3:
                        too_close = True
                        break
                
                if too_close:
                    continue
                
                # Đặt các vật cản theo hình dạng ngẫu nhiên quanh tâm
                cluster_size = np.random.randint(2, min(3, size // 4))  # Giảm kích thước cụm
                bricks_in_this_cluster = 0
                
                for i in range(max(0, center_row - cluster_size), min(size, center_row + cluster_size + 1)):
                    for j in range(max(0, center_col - cluster_size), min(size, center_col + cluster_size + 1)):
                        # Xác suất đặt vật cản giảm dần theo khoảng cách từ tâm
                        dist_from_center = abs(i - center_row) + abs(j - center_col)
                        if dist_from_center <= cluster_size and grid[i][j] == 0:
                            # Xác suất đặt vật cản giảm theo khoảng cách từ tâm
                            if np.random.random() > dist_from_center / (2 * cluster_size):
                                grid[i][j] = 3  # Đặt vật cản
                                bricks_in_this_cluster += 1
                                
                                if bricks_in_this_cluster >= bricks_per_cluster:
                                    break
                    
                    if bricks_in_this_cluster >= bricks_per_cluster:
                        break
                
                clusters_placed += 1
            
            # Đặt ngẫu nhiên số vật cản còn lại
            num_random_bricks = num_brick - clusters_placed * bricks_per_cluster
            if num_random_bricks > 0:
                for _ in range(num_random_bricks * 2):
                    row = np.random.randint(0, size)
                    col = np.random.randint(0, size)
                    
                    # Không đặt vật cản quá gần trạm xăng hoặc trạm thu phí
                    too_close = False
                    for sp_row, sp_col in all_special_positions:
                        if abs(sp_row - row) < 2 and abs(sp_col - col) < 2:
                            too_close = True
                            break
                    
                    if not too_close and grid[row][col] == 0:
                        grid[row][col] = 3  # Đặt vật cản
                        num_random_bricks -= 1
                        
                        if num_random_bricks <= 0:
                            break
        
        map_obj.grid = grid
        
        # Thiết lập vị trí bắt đầu và kết thúc một cách ngẫu nhiên nhưng hợp lý
        # 1. Tìm các ô đường thường (loại 0) để đặt điểm bắt đầu và điểm đích
        available_positions = []
        edge_positions = []  # Vị trí ở rìa bản đồ
        center_positions = []  # Vị trí ở giữa bản đồ
        
        for i in range(size):
            for j in range(size):
                if grid[i][j] == 0:  # Chỉ xem xét các ô đường thường
                    available_positions.append((i, j))
                    
                    # Phân loại vị trí ở rìa và ở giữa
                    if i < 2 or i >= size - 2 or j < 2 or j >= size - 2:
                        edge_positions.append((i, j))
                    else:
                        center_positions.append((i, j))
        
        # Nếu không có vị trí thích hợp, không thiết lập vị trí bắt đầu/kết thúc
        if not available_positions:
            return map_obj
            
        # 2. Ưu tiên đặt điểm bắt đầu ở rìa bản đồ
        if edge_positions:
            start_positions = edge_positions
        else:
            start_positions = available_positions
            
        if start_positions:
            map_obj.start_pos = start_positions[np.random.randint(0, len(start_positions))]
            # Loại bỏ vị trí đã chọn cho điểm bắt đầu
            available_positions = [pos for pos in available_positions if pos != map_obj.start_pos]
        
        # 3. Đặt điểm đích ở nơi cách xa điểm bắt đầu
        if not available_positions:
            return map_obj  # Không còn vị trí phù hợp
            
        # Sắp xếp các vị trí còn lại theo khoảng cách từ xa đến gần so với điểm bắt đầu
        if map_obj.start_pos:
            start_row, start_col = map_obj.start_pos
            
            # Tính toán khoảng cách Manhattan từ mỗi điểm đến điểm bắt đầu
            positions_with_distance = []
            for pos in available_positions:
                row, col = pos
                distance = abs(row - start_row) + abs(col - start_col)
                positions_with_distance.append((pos, distance))
            
            # Sắp xếp theo khoảng cách giảm dần (xa nhất trước)
            positions_with_distance.sort(key=lambda x: x[1], reverse=True)
            
            # Lấy 25% vị trí xa nhất làm ứng viên cho điểm đích
            top_quarter = max(1, len(positions_with_distance) // 4)
            candidate_positions = [pos for pos, dist in positions_with_distance[:top_quarter]]
            
            if candidate_positions:
                map_obj.end_pos = candidate_positions[np.random.randint(0, len(candidate_positions))]
        
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
        
        # Đảm bảo kích thước tối thiểu
        if size < 8:
            size = 8  # Kích thước tối thiểu để tạo mẫu hợp lý
            map_obj = cls(size)
        
        # Làm đường bao quanh
        for i in range(size):
            for j in range(size):
                # Đặt tất cả các ô là vật cản trước
                map_obj.grid[i][j] = 3
        
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
        
        # Đặt trạm thu phí ở bốn hướng chính
        map_obj.grid[1][mid] = 1  # Trạm thu phí phía trên
        map_obj.grid[size-2][mid] = 1  # Trạm thu phí phía dưới
        map_obj.grid[mid][1] = 1  # Trạm thu phí bên trái
        map_obj.grid[mid][size-2] = 1  # Trạm thu phí bên phải
        
        # Đặt trạm xăng ở năm góc và trung tâm
        map_obj.grid[1][1] = 2  # Trạm xăng ở góc trên trái
        map_obj.grid[1][size-2] = 2  # Trạm xăng ở góc trên phải
        map_obj.grid[size-2][1] = 2  # Trạm xăng ở góc dưới trái
        map_obj.grid[size-2][size-2] = 2  # Trạm xăng ở góc dưới phải
        map_obj.grid[mid][mid] = 2  # Trạm xăng ở trung tâm
        
        # Mở rộng đường đi ở giữa để tạo không gian
        for i in range(mid-1, mid+2):
            for j in range(mid-1, mid+2):
                if i >= 0 and i < size and j >= 0 and j < size:
                    if not (i == mid and j == mid):  # Giữ trạm xăng ở trung tâm
                        map_obj.grid[i][j] = 0
        
        # Tạo khu vực đường đi bên trong
        quarter = size // 4
        for i in range(mid-quarter, mid+quarter+1):
            for j in range(mid-quarter, mid+quarter+1):
                if i >= 0 and i < size and j >= 0 and j < size:
                    # Làm cho khu vực bên trong thông thoáng với đường đi
                    if random.random() < 0.7:  # 70% là đường đi
                        map_obj.grid[i][j] = 0
        
        # Tạo một vài mê cung đường đi ngẫu nhiên 
        for _ in range(3):
            start_i = random.randint(2, size-3)
            start_j = random.randint(2, size-3)
            
            if map_obj.grid[start_i][start_j] == 3:  # Nếu là vật cản
                # Tạo đường đi ngẫu nhiên
                for length in range(random.randint(3, 6)):
                    directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]  # Phải, xuống, trái, lên
                    di, dj = random.choice(directions)
                    
                    for steps in range(random.randint(2, 4)):
                        next_i, next_j = start_i + di * steps, start_j + dj * steps
                        if 0 <= next_i < size and 0 <= next_j < size:
                            map_obj.grid[next_i][next_j] = 0
        
        # Đảm bảo góc trên bên trái và góc dưới bên phải là đường đi để đặt điểm bắt đầu và điểm đích
        map_obj.grid[0][0] = 0  # Góc trên bên trái - Điểm bắt đầu
        map_obj.grid[size-1][size-1] = 0  # Góc dưới bên phải - Điểm đích
        
        # Đặt vị trí bắt đầu tại góc trên bên trái
        map_obj.start_pos = (0, 0)
        
        # Đặt điểm đích tại góc dưới bên phải
        map_obj.end_pos = (size-1, size-1)
        
        # Đảm bảo có đường đi từ điểm bắt đầu đến điểm đích
        # Tạo đường đi từ trên xuống dưới
        for i in range(size):
            map_obj.grid[i][0] = 0  # Cột đầu tiên
            map_obj.grid[i][size-1] = 0  # Cột cuối cùng
        
        # Tạo đường đi từ trái sang phải
        for j in range(size):
            map_obj.grid[0][j] = 0  # Hàng đầu tiên
            map_obj.grid[size-1][j] = 0  # Hàng cuối cùng
        
        return map_obj
    
    def get_statistics(self):
        """Lấy thống kê về số lượng các loại ô trên bản đồ"""
        return {
            'normal_roads': np.sum(self.grid == 0),
            'toll_stations': np.sum(self.grid == 1),
            'gas_stations': np.sum(self.grid == 2),
            'brick_cells': np.sum(self.grid == 3)
        }
    
    def save(self, filename=None):
        """
        Lưu bản đồ vào file
        
        Parameters:
        - filename: Tên file để lưu (nếu không có sẽ tạo tên dựa trên thời gian)
        
        Returns:
        - Đường dẫn đến file đã lưu
        """
        if not os.path.exists('maps'):
            os.makedirs('maps')
        
        # Tạo tên file dựa trên thời gian nếu không có tên file được cung cấp
        if not filename:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"map_{self.size}x{self.size}_{timestamp}.json"
        
        # Đảm bảo filename có đuôi .json
        if not filename.endswith('.json'):
            filename += '.json'
        
        # Lưu cả bản đồ hiện tại vào latest_map.json
        data = {
            'size': self.size,
            'grid': self.grid.tolist(),
            'start_pos': self.start_pos,
            'end_pos': self.end_pos,
            'timestamp': datetime.datetime.now().isoformat()
        }
        
        # Lưu vào file chỉ định
        filepath = os.path.join('maps', filename)
        with open(filepath, 'w') as f:
            json.dump(data, f)
        
        # Lưu vào latest_map.json
        with open('maps/latest_map.json', 'w') as f:
            json.dump(data, f)
        
        return filepath
    
    @classmethod
    def load(cls, filename='latest_map.json'):
        """Tải bản đồ từ file"""
        try:
            with open(f'maps/{filename}', 'r') as f:
                data = json.load(f)
            
            map_obj = cls(data['size'])
            map_obj.grid = np.array(data['grid'])
            map_obj.start_pos = data.get('start_pos')  # Tương thích với file cũ
            map_obj.end_pos = data.get('end_pos')      # Tương thích với file cũ
            return map_obj
        except FileNotFoundError:
            return None

class Node:
    """Class representing a node/cell in the map."""
    pass