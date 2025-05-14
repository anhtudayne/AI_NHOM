"""
Môi trường học tăng cường (RL Environment) cho bài toán định tuyến xe tải.
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import sys
import os
from typing import Any

# Thêm thư mục cha vào sys.path để có thể import từ các module khác
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from .map import Map
from .constants import CellType, MovementCosts, StationCosts, PathfindingWeights

# Define global maximums for observation space, based on UI sliders
# GLOBAL_MAX_FUEL = 50.0 # Will be instance variable
# GLOBAL_MAX_MONEY = 5000.0 # Will be instance variable

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
    
    def __init__(self, map_object: Map,
                 initial_fuel: float = MovementCosts.MAX_FUEL,
                 initial_money: float = 1500.0,
                 fuel_per_move: float = MovementCosts.FUEL_PER_MOVE,
                 gas_station_cost: float = StationCosts.BASE_GAS_COST,
                 toll_base_cost: float = StationCosts.BASE_TOLL_COST,
                 max_steps_per_episode: int | None = None, # Will default based on map_size if None
                 obs_max_fuel: float = 70.0, # Default for observation space scaling (updated to match new MAX_FUEL)
                 obs_max_money: float = 5000.0, # Default for observation space scaling
                 moving_obstacles: bool = False):
        """
        Khởi tạo môi trường RL
        
        Args:
            map_object: Đối tượng Map
            initial_fuel: Nhiên liệu ban đầu mặc định
            initial_money: Tiền ban đầu mặc định
            fuel_per_move: Nhiên liệu tiêu thụ mỗi bước mặc định
            gas_station_cost: Chi phí đổ xăng mặc định
            toll_base_cost: Chi phí trạm thu phí mặc định
            max_steps_per_episode: Số bước tối đa mỗi tập
            obs_max_fuel: Giá trị tối đa cho fuel trong observation space
            obs_max_money: Giá trị tối đa cho money trong observation space
            moving_obstacles: Cờ cho chướng ngại vật di chuyển
        """
        super().__init__()
        
        # Lưu tham số
        self.map_object = map_object  
        self.map_size = map_object.size
        
        # Store default parameters
        self.default_initial_fuel = float(initial_fuel)
        self.default_initial_money = float(initial_money)
        self.default_fuel_per_move = float(fuel_per_move)
        self.default_gas_station_cost = float(gas_station_cost)
        self.default_toll_base_cost = float(toll_base_cost)
        self.default_max_steps_per_episode = int(max_steps_per_episode if max_steps_per_episode is not None else self.map_size * 3)

        # Store observation space scaling factors
        self.obs_max_fuel = float(obs_max_fuel)
        self.obs_max_money = float(obs_max_money)

        # Khởi tạo vị trí bắt đầu và kết thúc
        self.start_pos = list(map_object.start_pos)
        self.end_pos = map_object.end_pos
        self.current_pos = list(self.start_pos)
        
        # Đếm số ô chướng ngại vật trên bản đồ
        self._obstacle_count = 0
        for y in range(self.map_size):
            for x in range(self.map_size):
                if self.map_object.is_obstacle(x, y):
                    self._obstacle_count += 1
        
        # Khởi tạo thêm biến trạng thái với các hằng số đúng
        self.current_fuel = self.default_initial_fuel
        self.current_money = self.default_initial_money
        self.current_episode_max_fuel = self.default_initial_fuel # Max fuel for current episode is the initial fuel
        self.current_episode_initial_money = self.default_initial_money
        self.current_episode_fuel_per_move = self.default_fuel_per_move
        self.current_episode_gas_station_cost = self.default_gas_station_cost
        self.current_episode_toll_base_cost = self.default_toll_base_cost
        
        # Định nghĩa cấu trúc mapping hành động
        self.action_mapping = {
            0: 'UP',
            1: 'RIGHT',
            2: 'DOWN',
            3: 'LEFT',
            4: 'STAY'
        }
        
        # Lưu trữ biến theo dõi step hiện tại
        self.current_step_in_episode = 0
        self.max_steps_per_episode = self.default_max_steps_per_episode
        
        # Theo dõi vị trí đã thăm
        self._position_counter = {}
        self._visited_positions = set()  # Positions visited by the agent (set of tuples)
        self._visited_map = np.zeros((self.map_size, self.map_size), dtype=np.float32)  # 2D array to track visits
        
        # Theo dõi lộ trình
        self._path_taken = []
        
        # Theo dõi va chạm với trở ngại gần đây
        self._last_collided_pos = None
        self._last_collided_action = None
        self._successful_moves_after_collision = 0
        
        # Theo dõi tiến độ
        self._progress_tracking = {
            "distance_improvement_count": 0,
            "best_distance_to_goal": float('inf'),
            "no_progress_count": 0
        }
        
        # Định nghĩa không gian observation và action
        self._define_spaces()
        
        # Reset môi trường
        self.reset()
        
    def _define_spaces(self):
        """
        Định nghĩa không gian observation và action
        """
        self.action_space = spaces.Discrete(5)
        
        # Observation space với các tham số có thể thay đổi
        self.observation_space = spaces.Dict({
            'agent_pos': spaces.Box(low=0, high=self.map_size-1, shape=(2,), dtype=np.int32),
            'fuel': spaces.Box(low=0.0, high=self.obs_max_fuel, shape=(1,), dtype=np.float32),
            'money': spaces.Box(low=0.0, high=self.obs_max_money, shape=(1,), dtype=np.float32),
            'target_pos': spaces.Box(low=0, high=self.map_size-1, shape=(2,), dtype=np.int32),
            'local_map': spaces.Box(low=-2, high=CellType.GAS.value, shape=(3, 3), dtype=np.int32),
            'visited_map': spaces.Box(low=0, high=1, shape=(self.map_size, self.map_size), dtype=np.float32),
            'env_params': spaces.Box(
                low=np.array([0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32),
                high=np.array([1.0, 1.0, 2.0, 1.0, 1.0], dtype=np.float32),
                shape=(5,), 
                dtype=np.float32
            )
        }) # Type of observation_space is gym.spaces.Dict
    
    def _get_observation(self) -> dict[str, np.ndarray]:
        """
        Lấy trạng thái quan sát hiện tại của môi trường.
        
        Returns:
            dict: Trạng thái quan sát hiện tại
        """
        # Lấy bản đồ cục bộ 3x3 xung quanh agent
        local_map = self._get_local_map_view(self.current_pos)
        
        # Tính khoảng cách Manhattan đến đích
        distance_to_goal = self._calculate_distance(self.current_pos, self.end_pos)
        
        # Kiểm tra xem vị trí hiện tại đã từng thăm chưa - chuyển thành tuple để có thể làm key
        visited_current = 1 if tuple(self.current_pos) in self._visited_positions else 0
        
        # Lấy thông tin trạng thái xe
        agent_state = np.array([
            self.current_fuel / self.current_episode_max_fuel if self.current_episode_max_fuel > 0 else 0.0,  # Chuẩn hóa nhiên liệu (0-1)
            self.current_money / self.obs_max_money if self.obs_max_money > 0 else 0.0,  # Chuẩn hóa tiền (0-1)
            self.current_episode_fuel_per_move / MovementCosts.FUEL_PER_MOVE if MovementCosts.FUEL_PER_MOVE > 0 else 0.0, # Chuẩn hóa mức tiêu thụ nhiên liệu
            distance_to_goal / (self.map_size * 2) if self.map_size > 0 else 0.0,  # Chuẩn hóa khoảng cách tới đích
            visited_current,  # Đã từng thăm vị trí hiện tại chưa (0/1)
        ], dtype=np.float32)
        
        # Tạo observation dict
        observation = {
            'agent_pos': np.array(self.current_pos, dtype=np.int32),
            'fuel': np.array([self.current_fuel], dtype=np.float32),
            'money': np.array([self.current_money], dtype=np.float32),
            'target_pos': np.array(self.end_pos, dtype=np.int32),
            'local_map': local_map.astype(np.int32),
            'visited_map': self._visited_map,
            'env_params': agent_state,
        }
        
        return observation
    
    def _get_local_map_view(self, position: tuple[int, int]) -> np.ndarray:
        """
        Lấy bản đồ cục bộ 3x3 xung quanh vị trí hiện tại.
        Tầm nhìn nhỏ hơn giúp agent tập trung vào môi trường gần và đơn giản hóa quá trình học.
        
        Args:
            position (tuple): Vị trí hiện tại (x, y)
            
        Returns:
            np.ndarray: Ma trận 3x3 thể hiện bản đồ cục bộ
        """
        x, y = position
        local_map = np.ones((3, 3), dtype=np.int32) * -2  # -2 là giá trị mặc định cho bên ngoài bản đồ
        
        # Lấy vùng 3x3 xung quanh vị trí hiện tại
        for i in range(3):
            for j in range(3):
                map_x = x + (j - 1)  # Dịch để vị trí hiện tại ở giữa (1,1)
                map_y = y + (i - 1)
                
                # Nếu vị trí nằm trong bản đồ, lấy giá trị từ bản đồ
                if 0 <= map_x < self.map_size and 0 <= map_y < self.map_size:
                    local_map[i, j] = self.map_object.grid[map_y, map_x]
        
        return local_map
    
    def reset(self, seed: int | None = None, options: dict | None = None, evaluation_params: dict | None = None) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
        """
        Đặt lại môi trường về trạng thái ban đầu.
        Ngẫu nhiên hóa các tham số môi trường nếu không ở chế độ đánh giá.
        
        Args:
            seed: Seed cho random state
            options: Tùy chọn bổ sung
            evaluation_params (dict, optional): Nếu được cung cấp, đặt các tham số môi trường
                                               theo các giá trị được chỉ định (để đánh giá)
        
        Returns:
            dict: Trạng thái quan sát ban đầu
            dict: Thông tin bổ sung
        """
        # Đặt lại seed (nếu được cung cấp)
        super().reset(seed=seed)
        
        # Đặt lại vị trí hiện tại về điểm bắt đầu
        self.current_pos = list(self.start_pos)
        
        # Đặt lại biến theo dõi thời gian
        self.current_step_in_episode = 0
        
        # Đặt lại biến theo dõi va chạm
        self._last_collided_pos = None
        self._last_collided_action = None
        self._successful_moves_after_collision = 0
        
        # Đặt lại biến theo dõi tiến độ
        self._progress_tracking = {
            "distance_improvement_count": 0,
            "best_distance_to_goal": float('inf'),
            "no_progress_count": 0
        }
        
        # Khởi tạo khoảng cách ban đầu đến đích
        self._last_distance_to_goal = self._calculate_distance(self.current_pos, self.end_pos)
        
        # Đặt lại biến theo dõi vị trí đã thăm
        self._position_counter = {}
        self._visited_positions = set([tuple(self.current_pos)])  # Bắt đầu với vị trí hiện tại
        self._visited_map = np.zeros((self.map_size, self.map_size), dtype=np.float32)
        self._visited_map[self.current_pos[1], self.current_pos[0]] = 1.0  # Đánh dấu vị trí đầu tiên
        
        # Đặt lại lộ trình
        self._path_taken = [tuple(self.current_pos)]
        
        # --- Ngẫu nhiên hóa tham số ---
        if evaluation_params:
            # Chế độ đánh giá: sử dụng các tham số được chỉ định
            self.current_fuel = float(evaluation_params.get('initial_fuel', self.default_initial_fuel))
            self.current_episode_max_fuel = float(evaluation_params.get('initial_fuel', self.default_initial_fuel)) # Max fuel is initial for eval
            self.current_money = float(evaluation_params.get('initial_money', self.default_initial_money))
            self.current_episode_initial_money = float(evaluation_params.get('initial_money', self.default_initial_money))
            self.current_episode_fuel_per_move = float(evaluation_params.get('fuel_per_move', self.default_fuel_per_move))
            self.current_episode_gas_station_cost = float(evaluation_params.get('gas_station_cost', self.default_gas_station_cost))
            self.current_episode_toll_base_cost = float(evaluation_params.get('toll_base_cost', self.default_toll_base_cost))
            self.max_steps_per_episode = int(evaluation_params.get('max_steps_per_episode', self.default_max_steps_per_episode))
        else:
            # Chế độ huấn luyện: sử dụng các giá trị mặc định đã lưu
            self.current_fuel = self.default_initial_fuel
            self.current_episode_max_fuel = self.default_initial_fuel
            self.current_money = self.default_initial_money
            self.current_episode_initial_money = self.default_initial_money
            self.current_episode_fuel_per_move = self.default_fuel_per_move
            self.current_episode_gas_station_cost = self.default_gas_station_cost
            self.current_episode_toll_base_cost = self.default_toll_base_cost
            self.max_steps_per_episode = self.default_max_steps_per_episode
        
        # Thêm thông tin bản đồ để debug
        optimal_path_estimate = self._calculate_distance(self.start_pos, self.end_pos)
        info = {
            'map_info': {
                'size': self.map_size,
                'obstacles': self._obstacle_count,
                'optimal_path_estimate': optimal_path_estimate,
                'start_pos': self.start_pos,
                'end_pos': self.end_pos
            },
            'env_params': {
                'max_fuel': self.current_episode_max_fuel,
                'fuel_per_move': self.current_episode_fuel_per_move,
                'gas_station_cost': self.current_episode_gas_station_cost,
                'toll_base_cost': self.current_episode_toll_base_cost,
                'max_steps': self.max_steps_per_episode
            }
        }
        
        # Lấy observation mới
        return self._get_observation(), info
    
    def step(self, action: int) -> tuple[dict[str, np.ndarray], float, bool, bool, dict[str, Any]]:
        """
        Thực hiện một bước trong môi trường dựa trên action đã chọn.
        
        Args:
            action: ID của hành động (0: Up, 1: Right, 2: Down, 3: Left, 4: Stay)
            
        Returns:
            observation: State mới
            reward: Phần thưởng
            terminated: Episode đã kết thúc chưa
            truncated: Episode đã bị cắt ngắn chưa
            info: Thông tin thêm
        """
        # Tăng số bước đã thực hiện
        self.current_step_in_episode += 1
        
        # Convert action to integer if it's a numpy array or tensor
        if hasattr(action, 'item'):
            action = action.item()  # For numpy arrays and PyTorch tensors
        elif hasattr(action, '__iter__'):
            # For other iterable types like lists
            action = action[0] if len(action) > 0 else 0
        
        # Kiểm tra hành động hợp lệ
        if not (0 <= action <= 4):
            # Hành động không hợp lệ, phạt nặng và kết thúc episode
            observation = self._get_observation()
            info = {
                "termination_reason": "hanh_dong_khong_hop_le",
                "truck_state": {
                    "position": self.current_pos,
                    "fuel": self.current_fuel,
                    "money": self.current_money
                }
            }
            return observation, -50.0, True, False, info  # Đánh dấu terminated=True để kết thúc ngay
        
        # Mặc định reward cho mỗi bước đi là nhỏ âm (để khuyến khích tìm đường đi ngắn nhất)
        reward = -0.01 # Reduced from -0.05
        
        # Lấy hành động dựa vào action ID
        action_name = self.action_mapping[action]

        # Get current observation BEFORE making any move or checking terminal conditions that return it
        observation = self._get_observation()
        
        # Tính toán vị trí mới dựa vào hành động
        new_pos = list(self.current_pos)
        if action_name == 'UP':
            new_pos[1] -= 1
        elif action_name == 'RIGHT':
            new_pos[0] += 1
        elif action_name == 'DOWN':
            new_pos[1] += 1
        elif action_name == 'LEFT':
            new_pos[0] -= 1
        elif action_name == 'STAY':
            pass  # Giữ nguyên vị trí
            
        # Kiểm tra va chạm với biên và vật cản
        if (new_pos[0] < 0 or new_pos[0] >= self.map_object.size or 
            new_pos[1] < 0 or new_pos[1] >= self.map_object.size or
            self.map_object.is_obstacle(new_pos[0], new_pos[1])):
            
            # Va chạm, phạt nhẹ và giữ nguyên vị trí
            collision_penalty = -2.0
            reward += collision_penalty
            
            # Lưu thông tin va chạm gần đây
            self._last_collided_pos = tuple(self.current_pos)
            self._last_collided_action = action
            self._successful_moves_after_collision = 0
            
            # Trả về observation mới
            info = {
                "collision": True,
                "truck_state": {
                    "position": self.current_pos,
                    "fuel": self.current_fuel,
                    "money": self.current_money
                }
            }
            
            # Nếu đã hết số bước tối đa, đánh dấu truncated (cắt ngắn)
            if self.current_step_in_episode >= self.max_steps_per_episode:
                info["termination_reason"] = "het_so_buoc"
                return observation, reward, False, True, info
                
            return observation, reward, False, False, info
            
        # Thực hiện di chuyển nếu không có vật cản
        previous_pos = tuple(self.current_pos)
        self.current_pos = new_pos
        
        # Thêm vị trí mới vào lộ trình
        self._path_taken.append(tuple(self.current_pos))
        
        # Lưu vị trí đã đi qua vào tập các vị trí đã thăm
        current_pos_tuple = tuple(self.current_pos)
        self._visited_positions.add(current_pos_tuple)
        
        # Cập nhật visited map
        self._visited_map[self.current_pos[1], self.current_pos[0]] = 1.0
        
        # Trừ nhiên liệu khi di chuyển
        self.current_fuel -= self.current_episode_fuel_per_move
        
        # Kiểm tra nếu đi qua trạm thu phí
        if self.map_object.grid[self.current_pos[1], self.current_pos[0]] == CellType.TOLL:
            # Tính toán chi phí đi qua trạm thu phí
            toll_cost = self.current_episode_toll_base_cost
            
            # Trừ tiền và ghi lại trạm thu phí đã sử dụng
            if self.current_money >= toll_cost:
                self.current_money -= toll_cost
            else:
                # Nếu không đủ tiền, quay lại vị trí cũ và kết thúc episode
                self.current_pos = list(previous_pos)
                self._path_taken.append(tuple(self.current_pos))
                info = {
                    "termination_reason": "khong_du_tien_qua_tram",
                    "truck_state": {
                        "position": self.current_pos,
                        "fuel": self.current_fuel,
                        "money": self.current_money
                    }
                }
                return observation, -10.0, True, False, info  # Terminated = True khi không đủ tiền qua trạm
        
        # Kiểm tra phát hiện chu trình (agent đi lòng vòng)
        if current_pos_tuple in self._position_counter:
            self._position_counter[current_pos_tuple] += 1
            
            # Tăng mức phạt khi quay lại vị trí đã thăm (từ 0.2 lên 1.0)
            revisit_penalty = min(3.0, 0.5 * (self._position_counter[current_pos_tuple] - 1))
            
            # Giảm ngưỡng kết thúc nếu lặp quá nhiều lần (từ 10 xuống 6)
            if self._position_counter[current_pos_tuple] > 8:
                info = {
                    "termination_reason": "lap_qua_nhieu",
                    "truck_state": {
                        "position": self.current_pos,
                        "fuel": self.current_fuel,
                        "money": self.current_money
                    }
                }
                return observation, -15.0, True, False, info  # Tăng mức phạt từ -10.0 lên -15.0
            
            reward = -revisit_penalty  # Phạt mạnh hơn cho việc quay lại
        else:
            self._position_counter[current_pos_tuple] = 1
            # Khuyến khích khám phá ô mới
            reward = 1.0  # Increased from 0.5 to 1.0
        
        # Cập nhật khoảng cách Manhattan đến đích
        current_dist = self._calculate_distance(self.current_pos, self.end_pos)
        dist_diff = self._last_distance_to_goal - current_dist
        self._last_distance_to_goal = current_dist
        
        # Tăng rewards dựa trên tiến độ về phía đích (tăng từ 0.5 lên 1.5 nếu tiến gần đích)
        if dist_diff > 0:  # Tiến gần đích
            progress_reward = 2.0  # Increased from 1.5 to 2.0
            reward += progress_reward
            # Cập nhật tracking tiến độ
            self._progress_tracking["distance_improvement_count"] += 1
            if current_dist < self._progress_tracking["best_distance_to_goal"]:
                self._progress_tracking["best_distance_to_goal"] = current_dist
                self._progress_tracking["no_progress_count"] = 0  # Reset counter khi có tiến bộ thực sự
            
        elif dist_diff < 0:  # Đi xa đích
            # Phạt mạnh hơn khi đi xa đích
            progress_reward = -1.0  # Kept at -1.0 as per analysis (was already this value)
            reward += progress_reward
            self._progress_tracking["no_progress_count"] += 1
            
            # Nếu không có tiến bộ trong một thời gian dài, tăng mức phạt
            if self._progress_tracking["no_progress_count"] > self.map_size * 3:
                info = {
                    "termination_reason": "khong_tien_trien",
                    "truck_state": {
                        "position": self.current_pos,
                        "fuel": self.current_fuel,
                        "money": self.current_money
                    }
                }
                return observation, -10.0, True, False, info
        
        # LƯU Ý: Kiểm tra đã đến đích sau khi tính toán reward tiến độ
        if tuple(self.current_pos) == self.end_pos:
            # Đã đến đích, thưởng lớn và kết thúc
            distance_reward = self._calculate_path_efficiency_reward()
            fuel_reward = self.current_fuel / self.current_episode_max_fuel  # Thưởng thêm nếu còn nhiều nhiên liệu
            money_reward = self.current_money / self.current_episode_initial_money  # Thưởng thêm nếu còn nhiều tiền
            
            # Tổng hợp phần thưởng - tăng phần thưởng cơ bản từ 20.0 lên 25.0
            total_reward = 25.0 + distance_reward + 5.0 * fuel_reward + 5.0 * money_reward
            
            info = {
                "termination_reason": "den_dich",
                "truck_state": {
                    "position": self.current_pos,
                    "fuel": self.current_fuel,
                    "money": self.current_money
                },
                "path_length": len(self._path_taken),
                "used_gas_stations": len(set(gas_pos for gas_pos in self._path_taken if self.map_object.grid[gas_pos[1], gas_pos[0]] == CellType.GAS)),
                "used_toll_stations": len(set(toll_pos for toll_pos in self._path_taken if self.map_object.grid[toll_pos[1], toll_pos[0]] == CellType.TOLL))
            }
            return observation, total_reward, True, False, info  # Terminated = True khi đến đích
        
        # Kiểm tra hết nhiên liệu
        if self.current_fuel <= 0:
            info = {
                "termination_reason": "het_nhien_lieu",
                "truck_state": {
                    "position": self.current_pos,
                    "fuel": self.current_fuel,
                    "money": self.current_money
                }
            }
            return observation, -10.0, True, False, info  # Terminated = True khi hết nhiên liệu
        
        # Trả về thông tin sau khi di chuyển
        info = {
            "termination_reason": "none",
            "truck_state": {
                "position": self.current_pos,
                "fuel": self.current_fuel,
                "money": self.current_money
            }
        }
        return observation, reward, False, False, info
    
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
    
    def render(self, mode: str = 'human'):
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

    def _get_neighbor_positions(self, position):
        """
        Lấy danh sách các ô lân cận của một vị trí.
        
        Args:
            position (tuple): Vị trí (x, y) cần lấy lân cận
            
        Returns:
            list: Danh sách các ô lân cận hợp lệ (trong bản đồ)
        """
        x, y = position
        neighbors = []
        
        # Thêm 4 ô lân cận nếu chúng nằm trong bản đồ
        for dx, dy in [(0, -1), (1, 0), (0, 1), (-1, 0)]:  # Lên, phải, xuống, trái
            nx, ny = x + dx, y + dy
            # Kiểm tra nằm trong bản đồ
            if 0 <= nx < self.map_size and 0 <= ny < self.map_size:
                neighbors.append((nx, ny))
                
        return neighbors 

    def _calculate_path_efficiency_reward(self) -> float:
        """
        Tính toán phần thưởng dựa trên hiệu quả của đường đi.
        
        Returns:
            float: Phần thưởng cho hiệu quả đường đi (0-10)
        """
        # Đường đi tối ưu là khoảng cách Manhattan
        optimal_path_length = self._calculate_distance(self.start_pos, self.end_pos)
        
        # Đường đi thực tế là số bước đã đi
        actual_path_length = len(self._path_taken)
        
        # Tính hiệu quả (tỷ lệ đường đi tối ưu / đường đi thực tế)
        # Giá trị nhỏ hơn 1 nếu đường đi dài hơn tối ưu
        efficiency_ratio = min(1.0, optimal_path_length / max(1, actual_path_length))
        
        # Thưởng dựa trên hiệu quả đường đi, tối đa 10 điểm
        return 10.0 * efficiency_ratio 

    def _randomize_obstacles(self):
        """
        Phương thức rỗng để tương thích ngược với mã trước đó.
        Vì chúng ta đã tắt tính năng vật cản di chuyển nên phương thức này không làm gì cả.
        """
        pass 