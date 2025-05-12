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
        
        self.action_space = spaces.Discrete(6)
        
        # Observation space with randomized environment parameters
        self.observation_space = spaces.Dict({
            'agent_pos': spaces.Box(low=0, high=self.map_size-1, shape=(2,), dtype=np.int32),
            'fuel': spaces.Box(low=0.0, high=GLOBAL_MAX_FUEL, shape=(1,), dtype=np.float32),
            'money': spaces.Box(low=0.0, high=GLOBAL_MAX_MONEY, shape=(1,), dtype=np.float32),
            'target_pos': spaces.Box(low=0, high=self.map_size-1, shape=(2,), dtype=np.int32),
            'local_map': spaces.Box(low=-2, high=CellType.GAS.value, shape=(7, 7), dtype=np.int32),
            'env_params': spaces.Box(
                low=np.array([self.max_fuel_range[0], self.fuel_per_move_range[0], self.gas_station_cost_range[0], self.toll_base_cost_range[0]], dtype=np.float32),
                high=np.array([self.max_fuel_range[1], self.fuel_per_move_range[1], self.gas_station_cost_range[1], self.toll_base_cost_range[1]], dtype=np.float32),
                shape=(4,), 
                dtype=np.float32
            )
        })
    
    def _get_observation(self):
        """
        Lấy trạng thái quan sát hiện tại của môi trường.
        
        Returns:
            dict: Trạng thái quan sát hiện tại
        """
        # Lấy bản đồ cục bộ 7x7 xung quanh agent
        local_map = self._get_local_map_view(self.current_pos)
        
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
            ], dtype=np.float32)
        }
    
    def _get_local_map_view(self, position):
        """
        Lấy bản đồ cục bộ 7x7 xung quanh vị trí hiện tại.
        
        Args:
            position (tuple): Vị trí hiện tại (x, y)
            
        Returns:
            np.ndarray: Ma trận 7x7 thể hiện bản đồ cục bộ
        """
        x, y = position
        local_map = np.ones((7, 7), dtype=np.int32) * -2  # -2 là giá trị mặc định cho bên ngoài bản đồ
        
        # Lấy vùng 7x7 xung quanh vị trí hiện tại
        for i in range(7):
            for j in range(7):
                map_x = x + (j - 3)  # Dịch để vị trí hiện tại ở giữa (3,3)
                map_y = y + (i - 3)
                
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
            self.current_episode_initial_fuel = min(_initial_fuel_eval, self.current_episode_max_fuel)

            self.current_episode_initial_money = evaluation_params.get('initial_money', np.random.uniform(self.initial_money_range[0], self.initial_money_range[1]))
            self.current_episode_fuel_per_move = evaluation_params.get('fuel_per_move', np.random.uniform(self.fuel_per_move_range[0], self.fuel_per_move_range[1]))
            self.current_episode_gas_station_cost = evaluation_params.get('gas_station_cost', np.random.uniform(self.gas_station_cost_range[0], self.gas_station_cost_range[1]))
            self.current_episode_toll_base_cost = evaluation_params.get('toll_base_cost', np.random.uniform(self.toll_base_cost_range[0], self.toll_base_cost_range[1]))
        else:
            # Training mode: Randomize parameters (HOẶC SỬ DỤNG GIÁ TRỊ CỐ ĐỊNH TỪ auto_train_rl.py NẾU ĐANG DEBUG)
            # Logic này sẽ bị ghi đè nếu auto_train_rl.py truyền vào giá trị cố định cho *_config
            self.current_episode_max_fuel = np.random.uniform(self.max_fuel_range[0], self.max_fuel_range[1])
            _sampled_initial_fuel = np.random.uniform(self.initial_fuel_range[0], self.initial_fuel_range[1])
            self.current_episode_initial_fuel = min(_sampled_initial_fuel, self.current_episode_max_fuel)

            self.current_episode_initial_money = np.random.uniform(self.initial_money_range[0], self.initial_money_range[1])
            self.current_episode_fuel_per_move = np.random.uniform(self.fuel_per_move_range[0], self.fuel_per_move_range[1])
            self.current_episode_gas_station_cost = np.random.uniform(self.gas_station_cost_range[0], self.gas_station_cost_range[1])
            self.current_episode_toll_base_cost = np.random.uniform(self.toll_base_cost_range[0], self.toll_base_cost_range[1])

        # Đảm bảo start_pos và end_pos không trùng nhau (thêm kiểm tra này)
        if self.map_object.start_pos == self.map_object.end_pos:
            # Nếu trùng, thử tìm một end_pos mới không phải là start_pos và không phải vật cản
            # Đây là một cách xử lý đơn giản, có thể cần cải thiện nếu map quá nhỏ hoặc nhiều vật cản
            original_end_pos_value = self.map_object.grid[self.map_object.end_pos[1], self.map_object.end_pos[0]]
            found_new_end = False
            for r_idx in range(self.map_size):
                for c_idx in range(self.map_size):
                    if (c_idx, r_idx) != self.map_object.start_pos and self.map_object.grid[r_idx, c_idx] != CellType.OBSTACLE:
                        self.map_object.end_pos = (c_idx, r_idx)
                        # Khôi phục giá trị ô end_pos cũ nếu nó không phải là start_pos mới
                        if self.map_object.start_pos != self.end_pos:
                             self.map_object.grid[self.map_object.start_pos[1], self.map_object.start_pos[0]] = 0 # Đảm bảo start là đường
                             self.map_object.grid[self.map_object.end_pos[1], self.map_object.end_pos[0]] = 0 # Đảm bảo end là đường
                        found_new_end = True
                        break
                if found_new_end:
                    break
            if not found_new_end:
                # Không tìm thấy end_pos mới phù hợp, có thể là vấn đề với map generation
                print("[WARNING] rl_environment.reset(): start_pos and end_pos are the same and could not find a new valid end_pos.")
        
        self.start_pos = self.map_object.start_pos # Cập nhật lại start_pos phòng trường hợp map_object thay đổi
        self.end_pos = self.map_object.end_pos   # Cập nhật lại end_pos

        self.current_pos = self.start_pos
        self.current_fuel = self.current_episode_initial_fuel 
        self.current_money = self.current_episode_initial_money
        self.current_step_in_episode = 0
        
        return self._get_observation(), {"reset_info": True, "randomized_params": not bool(evaluation_params)}
    
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
        self.current_step_in_episode += 1
        reward = 0.05 # Phần thưởng "sống sót" nhỏ cho mỗi bước (Increased from 0.01)
        terminated = False
        truncated = False
        info = {}
        
        current_x, current_y = self.current_pos
        previous_distance_to_goal = self._calculate_distance(self.current_pos, self.end_pos)
        
        if action <= 3:  # Các hành động di chuyển
            if action == 0: new_x, new_y = current_x, current_y - 1
            elif action == 1: new_x, new_y = current_x, current_y + 1
            elif action == 2: new_x, new_y = current_x - 1, current_y
            elif action == 3: new_x, new_y = current_x + 1, current_y
            
            if 0 <= new_x < self.map_size and 0 <= new_y < self.map_size:
                cell_type = self.map_object.grid[new_y, new_x]
                
                if cell_type == CellType.OBSTACLE:
                    reward -= 0.2  # Giảm mạnh hình phạt P_OBSTACLE -> Giảm còn -0.2
                    terminated = True
                    info["termination_reason"] = "va_cham_vat_can"
                else:
                    self.current_pos = (new_x, new_y)
                    self.current_fuel -= self.current_episode_fuel_per_move
                    # reward -= 0.1  # Giảm hình phạt di chuyển C_MOVE (REMOVED)
                    
                    if cell_type == CellType.TOLL:
                        toll_cost = self.current_episode_toll_base_cost
                        self.current_money -= toll_cost
                        # reward -= 0.1  # Tạm thời xóa/giảm mạnh hình phạt đi qua trạm thu phí
                        info["toll_paid"] = toll_cost
                    
                    elif cell_type == CellType.GAS:
                        info["at_gas_station"] = True
                    
                    current_distance_to_goal = self._calculate_distance(self.current_pos, self.end_pos)
                    distance_improvement = previous_distance_to_goal - current_distance_to_goal
                    reward += distance_improvement * 1.5  # Tăng hệ số C_PROGRESS từ 1.0 lên 1.5
                    
                    if self.current_pos == self.end_pos:
                        reward += 300.0  
                        terminated = True
                        info["termination_reason"] = "den_dich"
            else:
                reward -= 5.0 # Tăng hình phạt out_of_bounds và cho kết thúc episode
                terminated = True
                info["termination_reason"] = "out_of_bounds"
        
        elif action == 4:  # Đổ xăng
            cell_type = self.map_object.grid[current_y, current_x]
            if cell_type == CellType.GAS:
                fuel_needed = self.current_episode_max_fuel - self.current_fuel
                if fuel_needed > 1e-5: 
                    cost = fuel_needed * self.current_episode_gas_station_cost
                    if self.current_money >= cost:
                        self.current_money -= cost
                        self.current_fuel = self.current_episode_max_fuel 
                        reward += 20.0  # Tăng R_REFUEL
                        info["refuel_cost"] = cost
                        info["refueled_amount"] = fuel_needed
                    else:
                        reward -= 0.2 # Giảm hình phạt P_NO_MONEY_FOR_FUEL -> Giảm còn -0.2
                        info["refuel_fail_no_money"] = True
                else:
                    reward += 0.0 
                    info["already_full_fuel"] = True
            else:
                reward -= 0.2 # Giảm P_INVALID_ACTION -> Giảm còn -0.2
        
        elif action == 5:  # Bỏ qua trạm xăng
            cell_type = self.map_object.grid[current_y, current_x]
            if cell_type == CellType.GAS:
                info["skipped_gas_station"] = True
            else:
                reward -= 0.1 # Giảm hình phạt invalid_skip
                info["invalid_skip"] = True
        
        else:
            reward -= 0.5 # Giảm hình phạt invalid_action
            info["invalid_action"] = True
        
        # ---- Kiểm tra các điều kiện kết thúc và các trạng thái đặc biệt ----
        if self.current_pos == self.end_pos and not terminated:
            reward += 300.0 
            terminated = True
            info["termination_reason"] = "den_dich"

        if not terminated and self.current_fuel <= 0:
            cell_type_at_pos = self.map_object.grid[self.current_pos[1], self.current_pos[0]]
            if cell_type_at_pos == CellType.GAS:
                fuel_needed_check = self.current_episode_max_fuel - self.current_fuel 
                if fuel_needed_check < 1e-5: fuel_needed_check = 1.0 
                cost_to_refuel_check = fuel_needed_check * self.current_episode_gas_station_cost
                
                if self.current_money < cost_to_refuel_check:
                    reward -= 0.2 # Giảm hình phạt -> Giảm còn -0.2
                    terminated = True
                    info["termination_reason"] = "het_nhien_lieu_va_tien_tai_tram_xang"
                else:
                    self.current_fuel = max(0.0, self.current_fuel) 
                    reward -= 0.1 # Phạt rất nhẹ vì để hết xăng dù ở trạm
                    info["zero_fuel_at_gas_can_refuel"] = True
            else:
                reward -= 0.2  # Giảm hình phạt P_NO_FUEL -> Giảm còn -0.2
                terminated = True
                info["termination_reason"] = "het_nhien_lieu"
        
        if not terminated and self.current_money <= 0:
            reward -= 0.2 # Giảm hình phạt P_NO_MONEY -> Giảm còn -0.2
            terminated = True
            info["termination_reason"] = "het_tien"
            self.current_money = 0.0 

        if not terminated and self.current_step_in_episode >= self.max_steps_per_episode:
            truncated = True
            info["termination_reason"] = "vuot_qua_so_buoc_toi_da"
            reward -= 0.1 # Thêm hình phạt nhẹ khi bị truncated (Reduced from -1.0)
        
        self.current_fuel = max(0.0, self.current_fuel) if terminated else self.current_fuel
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