"""
Môi trường học tăng cường (RL Environment) cho bài toán định tuyến xe tải.
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import sys
import os

# Thêm thư mục cha vào sys.path để có thể import từ các module khác
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from .map import Map
from .constants import CellType, MovementCosts, StationCosts, PathfindingWeights


class TruckRoutingEnv(gym.Env):
    """
    Môi trường học tăng cường cho bài toán định tuyến xe tải.
    Agent (xe tải) sẽ học cách di chuyển trên bản đồ để đến đích
    trong khi tối ưu hóa nhiên liệu và chi phí.
    """
    
    metadata = {'render.modes': ['human']}
    
    def __init__(self, map_object, initial_fuel=None, initial_money=None, 
                 fuel_per_move=None, gas_station_cost=None, toll_base_cost=None, 
                 max_steps_per_episode=None):
        """
        Khởi tạo môi trường với bản đồ và các tham số.
        
        Args:
            map_object (Map): Đối tượng bản đồ từ lớp Map
            initial_fuel (float): Lượng nhiên liệu ban đầu
            initial_money (float): Số tiền ban đầu
            fuel_per_move (float): Lượng nhiên liệu tiêu thụ mỗi bước
            gas_station_cost (float): Chi phí đổ xăng tại trạm xăng
            toll_base_cost (float): Chi phí cơ bản khi đi qua trạm thu phí
            max_steps_per_episode (int): Số bước tối đa cho mỗi episode
        """
        # Lưu đối tượng bản đồ
        self.map_object = map_object
        
        # Sử dụng giá trị mặc định từ constants.py nếu không được cung cấp
        self.initial_fuel = initial_fuel if initial_fuel is not None else MovementCosts.MAX_FUEL
        self.initial_money = initial_money if initial_money is not None else 2000.0
        self.fuel_per_move = fuel_per_move if fuel_per_move is not None else MovementCosts.FUEL_PER_MOVE
        self.gas_station_cost = gas_station_cost if gas_station_cost is not None else StationCosts.BASE_GAS_COST
        self.toll_base_cost = toll_base_cost if toll_base_cost is not None else StationCosts.BASE_TOLL_COST
        
        # Xác định kích thước bản đồ, điểm bắt đầu và điểm kết thúc
        self.map_size = map_object.size
        self.start_pos = map_object.start_pos
        self.end_pos = map_object.end_pos
        
        # Xác định số bước tối đa cho mỗi episode
        self.max_steps_per_episode = max_steps_per_episode if max_steps_per_episode is not None else 2 * self.map_size * self.map_size
        
        # Khởi tạo trạng thái hiện tại
        self.current_pos = None
        self.current_fuel = None
        self.current_money = None
        self.current_step_in_episode = 0
        
        # Định nghĩa không gian hành động
        # 0: Lên, 1: Xuống, 2: Trái, 3: Phải, 4: Đổ xăng, 5: Bỏ qua trạm xăng
        self.action_space = spaces.Discrete(6)
        
        # Định nghĩa không gian quan sát (observation space)
        self.observation_space = spaces.Dict({
            'agent_pos': spaces.Box(low=0, high=self.map_size-1, shape=(2,), dtype=np.int32),
            'fuel': spaces.Box(low=0.0, high=self.initial_fuel, shape=(1,), dtype=np.float32),
            'money': spaces.Box(low=0.0, high=self.initial_money, shape=(1,), dtype=np.float32),
            'target_pos': spaces.Box(low=0, high=self.map_size-1, shape=(2,), dtype=np.int32),
            'local_map': spaces.Box(low=-2, high=2, shape=(5, 5), dtype=np.int32)  # 5x5 local view
        })
    
    def _get_observation(self):
        """
        Lấy trạng thái quan sát hiện tại của môi trường.
        
        Returns:
            dict: Trạng thái quan sát hiện tại
        """
        # Lấy bản đồ cục bộ 5x5 xung quanh agent
        local_map = self._get_local_map_view(self.current_pos)
        
        return {
            'agent_pos': np.array(self.current_pos, dtype=np.int32),
            'fuel': np.array([self.current_fuel], dtype=np.float32),
            'money': np.array([self.current_money], dtype=np.float32),
            'target_pos': np.array(self.end_pos, dtype=np.int32),
            'local_map': local_map
        }
    
    def _get_local_map_view(self, position):
        """
        Lấy bản đồ cục bộ 5x5 xung quanh vị trí hiện tại.
        
        Args:
            position (tuple): Vị trí hiện tại (x, y)
            
        Returns:
            np.ndarray: Ma trận 5x5 thể hiện bản đồ cục bộ
        """
        x, y = position
        local_map = np.ones((5, 5), dtype=np.int32) * -2  # -2 là giá trị mặc định cho bên ngoài bản đồ
        
        # Lấy vùng 5x5 xung quanh vị trí hiện tại
        for i in range(5):
            for j in range(5):
                map_x = x + (j - 2)  # Dịch để vị trí hiện tại ở giữa (2,2)
                map_y = y + (i - 2)
                
                # Nếu vị trí nằm trong bản đồ, lấy giá trị từ bản đồ
                if 0 <= map_x < self.map_size and 0 <= map_y < self.map_size:
                    local_map[i, j] = self.map_object.grid[map_y, map_x]
        
        return local_map
    
    def reset(self, seed=None, options=None):
        """
        Đặt lại môi trường về trạng thái ban đầu.
        
        Args:
            seed: Seed cho random state (mới trong Gymnasium API)
            options: Tùy chọn bổ sung (mới trong Gymnasium API)
            
        Returns:
            tuple: (observation, info) theo Gymnasium API
        """
        # Khởi tạo seed nếu được cung cấp
        if seed is not None:
            np.random.seed(seed)
            
        # Đặt lại vị trí của agent về điểm bắt đầu
        self.current_pos = self.start_pos
        
        # Đặt lại nhiên liệu và tiền về giá trị ban đầu
        self.current_fuel = self.initial_fuel
        self.current_money = self.initial_money
        
        # Đặt lại số bước trong episode
        self.current_step_in_episode = 0
        
        # Trả về trạng thái quan sát ban đầu và info
        return self._get_observation(), {"reset_info": True}
    
    def step(self, action):
        """
        Thực hiện một bước trong môi trường.
        
        Args:
            action (int): Hành động được thực hiện
                0: Lên (giảm y)
                1: Xuống (tăng y)
                2: Trái (giảm x)
                3: Phải (tăng x)
                4: Đổ xăng (chỉ khi ở trạm xăng)
                5: Bỏ qua trạm xăng (tiếp tục di chuyển)
        
        Returns:
            tuple: (observation, reward, terminated, truncated, info) theo Gymnasium API
                - observation: Trạng thái quan sát kế tiếp
                - reward: Phần thưởng nhận được
                - terminated: Cờ đánh dấu episode đã kết thúc do đạt điều kiện kết thúc
                - truncated: Cờ đánh dấu episode bị cắt do đạt giới hạn số bước
                - info: Thông tin bổ sung
        """
        # Tăng số bước trong episode
        self.current_step_in_episode += 1
        
        # Khởi tạo các giá trị mặc định
        reward = 0.0
        terminated = False
        truncated = False
        info = {}
        
        # Lưu vị trí hiện tại để tính khoảng cách đến đích trước và sau di chuyển
        current_x, current_y = self.current_pos
        previous_distance_to_goal = self._calculate_distance(self.current_pos, self.end_pos)
        
        # Xử lý từng loại hành động
        if action <= 3:  # Các hành động di chuyển
            # Xác định vị trí mới dựa trên hành động
            if action == 0:  # Lên
                new_x, new_y = current_x, current_y - 1
            elif action == 1:  # Xuống
                new_x, new_y = current_x, current_y + 1
            elif action == 2:  # Trái
                new_x, new_y = current_x - 1, current_y
            elif action == 3:  # Phải
                new_x, new_y = current_x + 1, current_y
            
            # Kiểm tra xem vị trí mới có nằm trong bản đồ không
            if 0 <= new_x < self.map_size and 0 <= new_y < self.map_size:
                # Kiểm tra loại ô tại vị trí mới
                cell_type = self.map_object.grid[new_y, new_x]
                
                # Kiểm tra vật cản
                if cell_type == CellType.OBSTACLE:
                    # Va chạm vật cản - phạt nghiêm trọng
                    reward -= 100.0  # P_OBSTACLE
                    terminated = True
                    info["termination_reason"] = "va_cham_vat_can"
                else:
                    # Di chuyển đến vị trí mới
                    self.current_pos = (new_x, new_y)
                    
                    # Tiêu thụ nhiên liệu cho di chuyển
                    self.current_fuel -= self.fuel_per_move
                    
                    # Phạt nhẹ cho mỗi bước di chuyển để khuyến khích tìm đường ngắn nhất
                    reward -= 1.0  # C_MOVE
                    
                    # Xử lý từng loại ô
                    if cell_type == CellType.TOLL:
                        # Trạm thu phí - trừ tiền và thêm phạt
                        toll_cost = self.toll_base_cost
                        self.current_money -= toll_cost
                        reward -= 5.0  # Phạt thêm khi đi qua trạm thu phí
                        info["toll_paid"] = toll_cost
                    
                    elif cell_type == CellType.GAS:
                        # Ở trạm xăng - ghi nhận thông tin
                        info["at_gas_station"] = True
                        # Không tự động đổ xăng, agent phải chọn hành động 4
                    
                    # Phần thưởng khi tiến gần đích hơn (hướng dẫn thêm)
                    current_distance_to_goal = self._calculate_distance(self.current_pos, self.end_pos)
                    distance_improvement = previous_distance_to_goal - current_distance_to_goal
                    reward += distance_improvement * 2.0  # C_PROGRESS
                    
                    # Kiểm tra đến đích
                    if self.current_pos == self.end_pos:
                        reward += 200.0  # R_GOAL - phần thưởng lớn khi đến đích
                        terminated = True
                        info["termination_reason"] = "den_dich"
            else:
                # Ra ngoài biên bản đồ - phạt và không di chuyển
                reward -= 10.0
                info["out_of_bounds"] = True
        
        elif action == 4:  # Đổ xăng
            # Kiểm tra xem có đang ở trạm xăng không
            cell_type = self.map_object.grid[current_y, current_x]
            if cell_type == CellType.GAS:
                # Tính chi phí đổ xăng dựa trên lượng nhiên liệu cần thêm
                fuel_needed = self.initial_fuel - self.current_fuel
                refuel_cost = self.gas_station_cost * fuel_needed
                
                # Kiểm tra xem có đủ tiền không
                if self.current_money >= refuel_cost:
                    # Đổ đầy bình xăng
                    self.current_fuel = self.initial_fuel
                    # Trừ tiền
                    self.current_money -= refuel_cost
                    info["refueled"] = True
                    info["refuel_cost"] = refuel_cost
                    
                    # Phạt nhẹ cho việc dừng đổ xăng
                    reward -= 2.0
                else:
                    # Không đủ tiền đổ xăng
                    info["not_enough_money_for_refuel"] = True
                    reward -= 5.0  # Phạt nhẹ
            else:
                # Không ở trạm xăng nhưng cố gắng đổ xăng
                reward -= 5.0  # Phạt cho hành động không hợp lý
                info["invalid_refuel"] = True
        
        elif action == 5:  # Bỏ qua trạm xăng
            # Không làm gì đặc biệt, chỉ ghi nhận thông tin
            cell_type = self.map_object.grid[current_y, current_x]
            if cell_type == CellType.GAS:
                info["skipped_gas_station"] = True
            else:
                # Hành động không hợp lý nếu không ở trạm xăng
                reward -= 1.0
                info["invalid_skip"] = True
        
        else:
            # Hành động không hợp lệ
            reward -= 5.0
            info["invalid_action"] = True
        
        # Kiểm tra các điều kiện kết thúc bổ sung
        
        # Kiểm tra hết nhiên liệu
        if self.current_fuel <= 0:
            cell_type = self.map_object.grid[self.current_pos[1], self.current_pos[0]]
            if cell_type == CellType.GAS:
                # Nếu hết xăng tại trạm xăng, có thể đổ xăng
                self.current_fuel = 0.0
                reward -= 10.0  # Phạt nhẹ
                info["zero_fuel_at_gas"] = True
            else:
                # Hết nhiên liệu không ở trạm xăng - kết thúc episode
                reward -= 100.0  # P_NO_FUEL
                terminated = True
                info["termination_reason"] = "het_nhien_lieu"
        
        # Kiểm tra hết tiền
        if self.current_money <= 0:
            self.current_money = 0.0
            # Cho phép tiếp tục nếu còn nhiên liệu và không đi qua trạm thu phí
            if self.current_fuel <= 0 or self.map_object.grid[self.current_pos[1], self.current_pos[0]] == CellType.TOLL:
                reward -= 100.0  # P_NO_MONEY
                terminated = True
                info["termination_reason"] = "het_tien"
            else:
                reward -= 10.0  # Phạt nhẹ cho việc hết tiền
                info["zero_money"] = True
        
        # Kiểm tra số bước tối đa - cập nhật truncated thay vì terminated
        if self.current_step_in_episode >= self.max_steps_per_episode:
            truncated = True
            info["termination_reason"] = "vuot_qua_so_buoc_toi_da"
        
        # Phạt cho việc tiêu thụ nhiên liệu
        if action <= 3:  # Chỉ tính khi di chuyển
            reward -= 0.2 * self.fuel_per_move  # C_FUEL_CONSUMPTION_RATE
        
        # Lấy trạng thái quan sát mới
        next_observation = self._get_observation()
        
        return next_observation, reward, terminated, truncated, info
    
    def _calculate_distance(self, pos1, pos2):
        """
        Tính khoảng cách Manhattan giữa hai vị trí
        
        Args:
            pos1: Vị trí thứ nhất (x1, y1)
            pos2: Vị trí thứ hai (x2, y2)
            
        Returns:
            float: Khoảng cách Manhattan giữa hai vị trí
        """
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])
    
    def render(self, mode='human'):
        """
        Hiển thị môi trường.
        
        Args:
            mode (str): Chế độ hiển thị
        """
        if mode == 'human':
            # In ra thông tin cơ bản về trạng thái hiện tại của môi trường
            print(f"Vị trí hiện tại: {self.current_pos}")
            print(f"Nhiên liệu: {self.current_fuel:.2f}")
            print(f"Tiền: {self.current_money:.2f}")
            print(f"Số bước: {self.current_step_in_episode}/{self.max_steps_per_episode}")
            
            # Có thể mở rộng để hiển thị bản đồ dưới dạng text
            print("\nBản đồ:")
            for y in range(self.map_size):
                line = ""
                for x in range(self.map_size):
                    if (x, y) == self.current_pos:
                        line += "A "  # Agent
                    elif (x, y) == self.start_pos:
                        line += "S "  # Start
                    elif (x, y) == self.end_pos:
                        line += "E "  # End
                    else:
                        cell_type = self.map_object.grid[y, x]
                        if cell_type == CellType.OBSTACLE:
                            line += "# "  # Vật cản
                        elif cell_type == CellType.TOLL:
                            line += "T "  # Trạm thu phí
                        elif cell_type == CellType.GAS:
                            line += "G "  # Trạm xăng
                        else:
                            line += ". "  # Đường thường
                print(line)
    
    def close(self):
        """Đóng môi trường."""
        pass 