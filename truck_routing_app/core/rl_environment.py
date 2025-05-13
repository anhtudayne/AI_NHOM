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

# Define global maximums for observation space, based on UI sliders
GLOBAL_MAX_FUEL = 50.0
GLOBAL_MAX_MONEY = 5000.0

# Define default ranges for randomization, based on UI sliders
DEFAULT_MAX_FUEL_RANGE = (10.0, 50.0)
DEFAULT_INITIAL_FUEL_RANGE = (5.0, 50.0) # Will be clamped by current_episode_max_fuel
DEFAULT_INITIAL_MONEY_RANGE = (1000.0, 5000.0)
DEFAULT_FUEL_PER_MOVE_RANGE = (0.1, 1.0)
DEFAULT_GAS_STATION_COST_RANGE = (10.0, 100.0)
DEFAULT_TOLL_BASE_COST_RANGE = (50.0, 300.0)


class TruckRoutingEnv(gym.Env):
    """
    Môi trường học tăng cường cho bài toán định tuyến xe tải.
    Agent (xe tải) sẽ học cách di chuyển trên bản đồ để đến đích
    trong khi tối ưu hóa nhiên liệu và chi phí.
    Các tham số môi trường có thể được ngẫu nhiên hóa mỗi episode.
    """
    
    metadata = {'render.modes': ['human']}
    
    def __init__(self, map_object, 
                 max_fuel_config=None,
                 initial_fuel_config=None, 
                 initial_money_config=None, 
                 fuel_per_move_config=None, 
                 gas_station_cost_config=None, 
                 toll_base_cost_config=None, 
                 max_steps_per_episode=None):
        """
        Khởi tạo môi trường với bản đồ và các cấu hình tham số.

        Các tham số *_config có thể là một giá trị float (cố định) 
        hoặc một tuple (min_val, max_val) cho việc ngẫu nhiên hóa.
        """
        self.map_object = map_object
        
        # Helper to parse config values (float or tuple) into ranges (tuple)
        def parse_config_to_range(config_val, default_val_or_range):
            if config_val is None:
                if isinstance(default_val_or_range, tuple):
                    return default_val_or_range
                else: # Single default value
                    return (default_val_or_range, default_val_or_range)
            if isinstance(config_val, tuple):
                return config_val
            else: # Single float value provided
                return (float(config_val), float(config_val))

        # Parse and store parameter ranges
        self.max_fuel_range = parse_config_to_range(max_fuel_config, DEFAULT_MAX_FUEL_RANGE)
        self.initial_fuel_range = parse_config_to_range(initial_fuel_config, DEFAULT_INITIAL_FUEL_RANGE)
        self.initial_money_range = parse_config_to_range(initial_money_config, DEFAULT_INITIAL_MONEY_RANGE)
        self.fuel_per_move_range = parse_config_to_range(fuel_per_move_config, DEFAULT_FUEL_PER_MOVE_RANGE)
        self.gas_station_cost_range = parse_config_to_range(gas_station_cost_config, DEFAULT_GAS_STATION_COST_RANGE)
        self.toll_base_cost_range = parse_config_to_range(toll_base_cost_config, DEFAULT_TOLL_BASE_COST_RANGE)

        # Parameters for the current episode, to be set in reset()
        self.current_episode_max_fuel = None
        self.current_episode_initial_fuel = None # Value chosen from range before clamping
        self.current_episode_initial_money = None
        self.current_episode_fuel_per_move = None
        self.current_episode_gas_station_cost = None
        self.current_episode_toll_base_cost = None
        
        self.map_size = map_object.size
        self.start_pos = map_object.start_pos
        self.end_pos = map_object.end_pos
        
        self.max_steps_per_episode = max_steps_per_episode if max_steps_per_episode is not None else 2 * self.map_size * self.map_size
        
        self.current_pos = None
        self.current_fuel = None
        self.current_money = None
        self.current_step_in_episode = 0
        
        # Trạng thái cho hình phạt "cố chấp" va chạm
        self._last_collided_pos = None
        self._last_collided_action = None
        self._successful_moves_after_collision = 0
        
        # Cải thiện bộ nhớ ngắn hạn để phát hiện vòng lặp hiệu quả hơn
        self._recent_positions = []
        self._memory_length = 20  # Tăng từ 15 lên 20 vị trí gần nhất
        self._revisit_penalty = 4.0  # Tăng từ 3.0 lên 4.0 để khuyến khích thoát vòng lặp mạnh hơn
        
        # Thêm bộ đếm vòng lặp mới
        self._position_counter = {}  # Đếm số lần xuất hiện của mỗi vị trí
        self._loop_penalty_factor = 0.8  # Tăng hệ số nhân của hình phạt vòng lặp
        self._loop_detected = False  # Cờ đánh dấu đã phát hiện vòng lặp
        self._stuck_threshold = 4  # Giảm ngưỡng phát hiện mắc kẹt (số lần lặp lại vị trí)
        
        # Thêm biến theo dõi khoảng cách đến đích cho shaped reward
        self._last_distance_to_goal = None
        self._potential_scale = 1.0  # Hệ số cho phần thưởng tiềm năng
        
        # Thêm biến theo dõi vị trí đã thăm để tránh quay lại
        self._visited_positions = set()
        self._optimal_path_length = None  # Ước tính độ dài đường đi tối ưu
        
        # Thêm bộ nhớ đường đi đã đi qua
        self._path_taken = []
        
        self.action_space = spaces.Discrete(6)
        
        # Observation space with randomized environment parameters
        self.observation_space = spaces.Dict({
            'agent_pos': spaces.Box(low=0, high=self.map_size-1, shape=(2,), dtype=np.int32),
            'fuel': spaces.Box(low=0.0, high=GLOBAL_MAX_FUEL, shape=(1,), dtype=np.float32),
            'money': spaces.Box(low=0.0, high=GLOBAL_MAX_MONEY, shape=(1,), dtype=np.float32),
            'target_pos': spaces.Box(low=0, high=self.map_size-1, shape=(2,), dtype=np.int32),
            'local_map': spaces.Box(low=-2, high=CellType.GAS.value, shape=(5, 5), dtype=np.int32),
            'env_params': spaces.Box(
                low=np.array([self.max_fuel_range[0], self.fuel_per_move_range[0], self.gas_station_cost_range[0], self.toll_base_cost_range[0]], dtype=np.float32),
                high=np.array([self.max_fuel_range[1], self.fuel_per_move_range[1], self.gas_station_cost_range[1], self.toll_base_cost_range[1]], dtype=np.float32),
                shape=(4,), 
                dtype=np.float32
            ),
            # Thêm thông tin khoảng cách Manhattan đến đích
            'distance_to_goal': spaces.Box(low=0, high=2*self.map_size, shape=(1,), dtype=np.int32),
            # Thêm thông tin số bước đã đi
            'steps_taken': spaces.Box(low=0, high=self.max_steps_per_episode, shape=(1,), dtype=np.int32),
            # Thêm thông tin đã từng thăm vị trí hiện tại chưa
            'visited_current': spaces.Box(low=0, high=1, shape=(1,), dtype=np.int32)
        })
    
    def _get_observation(self):
        """
        Lấy trạng thái quan sát hiện tại của môi trường.
        
        Returns:
            dict: Trạng thái quan sát hiện tại
        """
        # Lấy bản đồ cục bộ 5x5 xung quanh agent
        local_map = self._get_local_map_view(self.current_pos)
        
        # Tính khoảng cách Manhattan đến đích
        distance_to_goal = self._calculate_distance(self.current_pos, self.end_pos)
        
        # Kiểm tra xem vị trí hiện tại đã từng thăm chưa
        visited_current = 1 if self.current_pos in self._visited_positions else 0
        
        return {
            'agent_pos': np.array(self.current_pos, dtype=np.int32),
            'fuel': np.array([self.current_fuel], dtype=np.float32),
            'money': np.array([self.current_money], dtype=np.float32),
            'target_pos': np.array(self.end_pos, dtype=np.int32),
            'local_map': local_map,
            'env_params': np.array([
                self.current_episode_max_fuel,
                self.current_episode_fuel_per_move,
                self.current_episode_gas_station_cost,
                self.current_episode_toll_base_cost
            ], dtype=np.float32),
            'distance_to_goal': np.array([distance_to_goal], dtype=np.int32),
            'steps_taken': np.array([self.current_step_in_episode], dtype=np.int32),
            'visited_current': np.array([visited_current], dtype=np.int32)
        }
    
    def _get_local_map_view(self, position):
        """
        Lấy bản đồ cục bộ 5x5 xung quanh vị trí hiện tại.
        Tầm nhìn nhỏ hơn giúp agent tập trung vào môi trường gần và đơn giản hóa quá trình học.
        
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
    
    def reset(self, seed=None, options=None, evaluation_params=None):
        """
        Đặt lại môi trường về trạng thái ban đầu.
        Ngẫu nhiên hóa các tham số môi trường nếu không ở chế độ đánh giá.
        
        Args:
            seed: Seed cho random state
            options: Tùy chọn bổ sung
            evaluation_params (dict, optional): Nếu được cung cấp, đặt các tham số môi trường
                                               thay vì ngẫu nhiên hóa. Keys dự kiến:
                                               'max_fuel', 'initial_fuel', 'initial_money',
                                               'fuel_per_move', 'gas_station_cost', 'toll_base_cost'.
        Returns:
            tuple: (observation, info)
        """
        if seed is not None:
            np.random.seed(seed) # Ensure reproducibility if seed is passed
            # super().reset(seed=seed) # Call parent reset if gym.Env is directly inherited and handles seed

        if evaluation_params:
            # Evaluation mode: Use provided parameters
            self.current_episode_max_fuel = evaluation_params.get('max_fuel', np.random.uniform(self.max_fuel_range[0], self.max_fuel_range[1]))
            _initial_fuel_eval = evaluation_params.get('initial_fuel', np.random.uniform(self.initial_fuel_range[0], self.initial_fuel_range[1]))
            # Clamp initial fuel by max fuel
            self.current_episode_initial_fuel = min(_initial_fuel_eval, self.current_episode_max_fuel)
            self.current_episode_initial_money = evaluation_params.get('initial_money', np.random.uniform(self.initial_money_range[0], self.initial_money_range[1]))
            self.current_episode_fuel_per_move = evaluation_params.get('fuel_per_move', np.random.uniform(self.fuel_per_move_range[0], self.fuel_per_move_range[1]))
            self.current_episode_gas_station_cost = evaluation_params.get('gas_station_cost', np.random.uniform(self.gas_station_cost_range[0], self.gas_station_cost_range[1]))
            self.current_episode_toll_base_cost = evaluation_params.get('toll_base_cost', np.random.uniform(self.toll_base_cost_range[0], self.toll_base_cost_range[1]))
        else:
            # Training mode: Randomize all parameters
            self.current_episode_max_fuel = np.random.uniform(self.max_fuel_range[0], self.max_fuel_range[1])
            _initial_fuel_random = np.random.uniform(self.initial_fuel_range[0], self.initial_fuel_range[1])
            # Clamp initial fuel by max fuel
            self.current_episode_initial_fuel = min(_initial_fuel_random, self.current_episode_max_fuel)
            self.current_episode_initial_money = np.random.uniform(self.initial_money_range[0], self.initial_money_range[1])
            self.current_episode_fuel_per_move = np.random.uniform(self.fuel_per_move_range[0], self.fuel_per_move_range[1])
            self.current_episode_gas_station_cost = np.random.uniform(self.gas_station_cost_range[0], self.gas_station_cost_range[1])
            self.current_episode_toll_base_cost = np.random.uniform(self.toll_base_cost_range[0], self.toll_base_cost_range[1])

        # Set initial state
        self.current_pos = self.start_pos
        self.current_fuel = self.current_episode_initial_fuel 
        self.current_money = self.current_episode_initial_money
        self.current_step_in_episode = 0
        
        # Reset loop detection
        self._recent_positions = [self.start_pos]  # Start with initial position
        self._position_counter = {self.start_pos: 1}  # Initialize counter
        self._loop_detected = False
        
        # Reset other tracking variables
        self._last_collided_pos = None
        self._last_collided_action = None
        self._successful_moves_after_collision = 0
        
        # Reset visited positions
        self._visited_positions = {self.start_pos}
        
        # Reset path tracking
        self._path_taken = [self.start_pos]
        
        # Calculate initial distance to goal for potential-based reward
        self._last_distance_to_goal = self._calculate_distance(self.current_pos, self.end_pos)
        
        # Estimate optimal path length using Manhattan distance
        self._optimal_path_length = self._calculate_distance(self.start_pos, self.end_pos)
        
        # Get initial observation
        observation = self._get_observation()
        
        # Return observation and info
        info = {
            'episode_params': {
                'max_fuel': self.current_episode_max_fuel,
                'initial_fuel': self.current_episode_initial_fuel,
                'initial_money': self.current_episode_initial_money,
                'fuel_per_move': self.current_episode_fuel_per_move,
                'gas_station_cost': self.current_episode_gas_station_cost,
                'toll_base_cost': self.current_episode_toll_base_cost
            }
        }
        
        return observation, info
    
    def step(self, action):
        """
        Thực hiện một bước trong môi trường với hành động đã chọn.
        
        Args:
            action (int): Hành động của agent (0-5)
                0: Không làm gì (đứng yên)
                1: Di chuyển lên trên
                2: Di chuyển sang phải
                3: Di chuyển xuống dưới
                4: Di chuyển sang trái
                5: Nạp nhiên liệu/trả phí (tùy thuộc vào ô hiện tại)
        
        Returns:
            tuple: (observation, reward, done, truncated, info)
        """
        # Tăng bước hiện tại
        self.current_step_in_episode += 1
        truncated = self.current_step_in_episode >= self.max_steps_per_episode
        
        # Lưu giữ trạng thái trước khi thay đổi để tính phần thưởng
        prev_pos = self.current_pos
        
        # Khởi tạo phần thưởng và trạng thái kết thúc
        reward = 0.0
        done = False
        info = {
            "action_taken": action,
            "step_number": self.current_step_in_episode
        }
        
        # Variable để theo dõi xem đã mua nhiên liệu hay trả phí trong bước này
        did_buy_fuel = False
        did_pay_toll = False
        
        # Xử lý dựa trên hành động
        if action == 0:  # Đứng yên (không làm gì)
            # Phạt nhẹ nếu đứng yên không cần thiết
            reward -= 0.2  # Giảm hình phạt đứng yên từ 0.5 xuống còn 0.2
            # Không trừ nhiên liệu khi đứng yên
            
            # Thêm thông tin
            info["action_result"] = "dung_yen"
            
        elif 1 <= action <= 4:  # Di chuyển
            # Tính toán vị trí mới dựa trên hành động
            new_pos = list(self.current_pos)
            if action == 1:  # Lên trên
                new_pos[1] = max(0, new_pos[1] - 1)
            elif action == 2:  # Sang phải
                new_pos[0] = min(self.map_size - 1, new_pos[0] + 1)
            elif action == 3:  # Xuống dưới
                new_pos[1] = min(self.map_size - 1, new_pos[1] + 1)
            elif action == 4:  # Sang trái
                new_pos[0] = max(0, new_pos[0] - 1)
            
            new_pos = tuple(new_pos)
            
            # Kiểm tra xem ô mới có phải là vật cản không
            cell_type = self.map_object.get_cell_type(new_pos)
                
            if cell_type == CellType.OBSTACLE:
                # Không di chuyển được vào ô vật cản và nhận hình phạt
                reward -= 2.0  # Tăng hình phạt va chạm lên 2.0 để tránh va chạm
                
                # Theo dõi va chạm với cùng vật cản
                if self._last_collided_pos == new_pos and self._last_collided_action == action:
                    # Phạt nặng hơn nếu lặp lại cùng một va chạm
                    reward -= 4.0  # Tăng hình phạt để khuyến khích tránh lặp lại
                    info["action_result"] = "va_cham_vat_can_nhung_khong_ket_thuc"
                else:
                    # Đặt lại bộ đếm va chạm
                    self._last_collided_pos = new_pos
                    self._last_collided_action = action
                    self._successful_moves_after_collision = 0
                    info["action_result"] = "va_cham_vat_can"
            else:
                # Nếu có đủ nhiên liệu, cho phép di chuyển
                if self.current_fuel >= self.current_episode_fuel_per_move:
                    # Di chuyển đến ô mới
                    self.current_pos = new_pos
                    self.current_fuel -= self.current_episode_fuel_per_move
                    
                    # Thêm vị trí mới vào đường đi
                    self._path_taken.append(new_pos)
                    
                    # Ghi nhận vị trí đã thăm
                    self._visited_positions.add(new_pos)
                    
                    # Phần thưởng cơ bản cho di chuyển mới
                    if new_pos not in self._recent_positions:
                        reward += 0.1  # Thưởng cho việc khám phá vị trí mới
                    
                    # Sử dụng phần thưởng tiềm năng dựa trên khoảng cách đến đích
                    current_distance = self._calculate_distance(new_pos, self.end_pos)
                    potential_reward = self._last_distance_to_goal - current_distance
                    reward += potential_reward * self._potential_scale
                    self._last_distance_to_goal = current_distance
                    
                    # Thêm phần thưởng nếu đến được các ô chức năng
                    if cell_type == CellType.GAS:
                        reward += 0.5  # Thưởng cho việc tìm thấy trạm xăng
                    elif cell_type == CellType.TOLL:
                        # Không thưởng cho việc tìm thấy trạm thu phí vì đó không phải lúc nào cũng tốt
                        pass
                    
                    # Ghi nhận di chuyển thành công sau va chạm
                    self._successful_moves_after_collision += 1
                    if self._successful_moves_after_collision >= 3:
                        # Reset trạng thái va chạm nếu đã di chuyển thành công một số bước
                        self._last_collided_pos = None
                        self._last_collided_action = None
                    
                    # Ghi nhận vị trí mới vào bộ nhớ ngắn hạn
                    self._recent_positions.append(new_pos)
                    if len(self._recent_positions) > self._memory_length:
                        self._recent_positions.pop(0)
                    
                    # Cập nhật bộ đếm vị trí
                    self._position_counter[new_pos] = self._position_counter.get(new_pos, 0) + 1
                    
                    # Phát hiện và xử lý vòng lặp
                    if self._position_counter[new_pos] >= self._stuck_threshold:
                        self._loop_detected = True
                        # Phạt để khuyến khích thoát khỏi vòng lặp
                        loop_penalty = self._revisit_penalty * self._loop_penalty_factor * self._position_counter[new_pos]
                        reward -= min(loop_penalty, 10.0)  # Giới hạn hình phạt tối đa
                        info["action_result"] = "phat_hien_vong_lap"
                    else:
                        info["action_result"] = "di_chuyen_thanh_cong"
                    
                    # Kiểm tra xem đã đến đích chưa
                    if new_pos == self.end_pos:
                        done = True
                        
                        # Phần thưởng lớn cho việc đến đích
                        base_reward = 100.0
                        
                        # Thưởng hiệu quả: nhiều nhiên liệu và tiền còn lại
                        efficiency_reward = (self.current_fuel / self.current_episode_max_fuel) * 50.0
                        efficiency_reward += (self.current_money / self.current_episode_initial_money) * 50.0
                        
                        # Thưởng cho việc tìm đường ngắn
                        path_length = len(set(self._path_taken))  # Số ô khác nhau đã đi qua
                        path_efficiency = min(2.0, self._optimal_path_length / max(1, path_length - 1))
                        path_reward = path_efficiency * 50.0
                        
                        # Kết hợp các phần thưởng
                        total_reward = base_reward + efficiency_reward + path_reward
                        reward += total_reward
                        
                        info["termination_reason"] = "den_dich"
                        info["path_length"] = len(self._path_taken) - 1  # Không tính vị trí ban đầu
                        info["unique_cells_visited"] = len(set(self._path_taken))
                        info["remaining_fuel"] = self.current_fuel
                        info["remaining_money"] = self.current_money
                        info["optimal_path_estimate"] = self._optimal_path_length
                else:
                    # Không đủ nhiên liệu để di chuyển
                    reward -= 1.0
                    info["action_result"] = "khong_du_nhien_lieu"
                    
                    # Kết thúc episode nếu không thể di chuyển và không ở trạm xăng
                    if self.map_object.get_cell_type(self.current_pos) != CellType.GAS:
                        done = True
                        reward -= 20.0  # Phạt nặng
                        info["termination_reason"] = "het_nhien_lieu"
                        
        elif action == 5:  # Nạp nhiên liệu hoặc trả phí
            cell_type = self.map_object.get_cell_type(self.current_pos)
            
            if cell_type == CellType.GAS:
                # Ở trạm xăng, có thể nạp nhiên liệu
                if self.current_money >= self.current_episode_gas_station_cost:
                    # Tính toán lượng nhiên liệu cần nạp
                    missing_fuel = self.current_episode_max_fuel - self.current_fuel
                    if missing_fuel > 0:
                        # Nạp nhiên liệu và trừ tiền
                        self.current_fuel = self.current_episode_max_fuel 
                        self.current_money -= self.current_episode_gas_station_cost
                        did_buy_fuel = True
                        
                        # Phần thưởng cho việc nạp nhiên liệu phụ thuộc vào lượng nhiên liệu cần
                        normalized_missing = missing_fuel / self.current_episode_max_fuel
                        fuel_reward = normalized_missing * 2.0  # Phần thưởng dựa vào % nhiên liệu cần nạp
                        reward += max(0.5, fuel_reward)  # Ít nhất 0.5 reward
                        
                        info["action_result"] = "nap_nhien_lieu_thanh_cong"
                        info["fuel_before"] = self.current_fuel - missing_fuel
                        info["fuel_after"] = self.current_fuel
                        info["money_spent"] = self.current_episode_gas_station_cost
                    else:
                        # Không cần nạp nhiên liệu
                        reward -= 0.5  # Phạt nhẹ
                        info["action_result"] = "khong_can_nap_nhien_lieu"
                else:
                    # Không đủ tiền để nạp nhiên liệu
                    reward -= 1.0
                    info["action_result"] = "khong_du_tien_nap_nhien_lieu"
                    
                    # Nếu ở trạm xăng, hết tiền VÀ ít nhiên liệu, kết thúc
                    if self.current_fuel < self.current_episode_fuel_per_move:
                        done = True
                        reward -= 20.0  # Phạt nặng
                        info["termination_reason"] = "het_tien"
            
            elif cell_type == CellType.TOLL:
                # Ở trạm thu phí, phải trả phí để tiếp tục
                toll_cost = self.current_episode_toll_base_cost
                
                if self.current_money >= toll_cost:
                    # Trả phí
                    self.current_money -= toll_cost
                    did_pay_toll = True
                    
                    # Phần thưởng nhỏ cho việc trả phí đúng
                    reward += 0.5
                    
                    info["action_result"] = "tra_phi_thanh_cong"
                    info["money_spent"] = toll_cost
                else:
                    # Không đủ tiền để trả phí
                    reward -= 2.0
                    info["action_result"] = "khong_du_tien_tra_phi"
                    
                    # Không thể tiếp tục nếu không trả phí và không có đủ tiền
                    done = True
                    reward -= 10.0
                    info["termination_reason"] = "het_tien"
            
            else:
                # Không phải ô đặc biệt, hành động không có tác dụng
                reward -= 1.0
                info["action_result"] = "khong_phai_o_dac_biet"
        
        # Kiểm tra trường hợp kết thúc do hết tiền
        if self.current_money <= 0 and not done:
            # Chỉ kết thúc nếu ở trạm thu phí hoặc cần nạp nhiên liệu
            cell_type = self.map_object.get_cell_type(self.current_pos)
            if cell_type == CellType.TOLL or (cell_type == CellType.GAS and self.current_fuel < self.current_episode_fuel_per_move):
                done = True
                reward -= 20.0
                info["termination_reason"] = "het_tien"
        
        # Kiểm tra điều kiện dừng: hết nhiên liệu
        if self.current_fuel <= 0 and not done:
            done = True
            reward -= 20.0
            info["termination_reason"] = "het_nhien_lieu"
        
        # Thêm thông tin cho truncated
        if truncated:
            done = True
            reward -= 10.0  # Phạt nếu vượt quá số bước tối đa
            info["termination_reason"] = "vuot_qua_so_buoc"
        
        # Cập nhật trạng thái và trả về kết quả
        observation = self._get_observation()
        
        # Thêm thông tin chi tiết vào info
        info["current_pos"] = self.current_pos
        info["current_fuel"] = self.current_fuel
        info["current_money"] = self.current_money
        info["visited_positions"] = list(self._visited_positions)
        info["truck_state"] = {
            "position": self.current_pos,
            "fuel": self.current_fuel,
            "money": self.current_money,
            "buy_fuel": did_buy_fuel,
            "pay_toll": did_pay_toll
        }
        
        # Định dạng kết quả theo Gymnasium API
        return observation, reward, done, truncated, info
    
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