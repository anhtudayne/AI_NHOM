"""
Simulated Annealing algorithm implementation for truck routing.
"""

from typing import List, Tuple, Dict, Set, Optional
import numpy as np
import random
import math
from .base_search import BaseSearch, SearchState

class SimulatedAnnealing(BaseSearch):
    """Simulated Annealing algorithm implementation."""
    
    def __init__(self, grid: np.ndarray, 
                 initial_temperature: float = 100.0, 
                 cooling_rate: float = 0.95, 
                 steps_per_temp: int = 50,
                 initial_money: float = None,
                 max_fuel: float = None, 
                 fuel_per_move: float = None, 
                 gas_station_cost: float = None, 
                 toll_base_cost: float = None,
                 initial_fuel: float = None):
        """Initialize Simulated Annealing with a grid and parameters."""
        super().__init__(grid, initial_money, max_fuel, fuel_per_move, 
                         gas_station_cost, toll_base_cost, initial_fuel)
        self.initial_temperature = initial_temperature
        self.cooling_rate = cooling_rate
        self.steps_per_temp = steps_per_temp
        self.current_state = None
        self.temperature = initial_temperature
        self.start = None
        self.goal = None
        self.best_path = []
        self.best_cost = float('inf')
        self.current_iteration = 0

    def create_random_path(self, start: Tuple[int, int], goal: Tuple[int, int]) -> List[Tuple[int, int]]:
        """Create a random initial path using BFS/DFS to ensure it's at least valid geometrically."""
        # Use BFS to find a valid path from start to goal
        from collections import deque
        
        queue = deque([(start, [start])])
        visited = {start}
        
        while queue:
            (vertex, path) = queue.popleft()
            
            # Get neighbors in random order to introduce randomness
            neighbors = self.get_neighbors(vertex)
            random.shuffle(neighbors)
            
            for next_vertex in neighbors:
                if next_vertex == goal:
                    # Return the complete path to goal
                    return path + [next_vertex]
                    
                if next_vertex not in visited:  # get_neighbors already filters out obstacles
                    visited.add(next_vertex)
                    queue.append((next_vertex, path + [next_vertex]))
        
        # If no path found, return just start and goal (might not be valid)
        return [start, goal]

    def create_neighbor_path(self, current_path: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
        """Create a neighboring path by making a small random change."""
        if len(current_path) <= 2:
            return current_path.copy()  # Can't modify a path with just start and goal
            
        new_path = current_path.copy()
        
        # Choose a random modification strategy
        strategy = random.randint(0, 3)
        
        if strategy == 0 and len(current_path) > 3:
            # Remove a random point (except start and goal)
            idx = random.randint(1, len(new_path) - 2)
            new_path.pop(idx)
            
        elif strategy == 1:
            # Modify a point (except start and goal)
            idx = random.randint(1, len(new_path) - 2)
            current_point = new_path[idx]
            
            # Find valid neighbors
            valid_neighbors = self.get_neighbors(current_point)
            
            if valid_neighbors:
                new_path[idx] = random.choice(valid_neighbors)
                
        elif strategy == 2:
            # Add a new point between two existing points
            idx = random.randint(0, len(new_path) - 2)
            point1 = new_path[idx]
            point2 = new_path[idx + 1]
            
            # Try to find a valid point between point1 and point2
            valid_neighbors = []
            for n1 in self.get_neighbors(point1):
                for n2 in self.get_neighbors(n1):
                    if n2 == point2 and n1 not in new_path:
                        valid_neighbors.append(n1)
                            
            if valid_neighbors:
                new_path.insert(idx + 1, random.choice(valid_neighbors))
                
        elif strategy == 3 and len(current_path) > 4:
            # Swap two non-consecutive points (except start and goal)
            idx1 = random.randint(1, len(new_path) - 3)
            idx2 = random.randint(idx1 + 1, len(new_path) - 2)
            new_path[idx1], new_path[idx2] = new_path[idx2], new_path[idx1]
            
        # Ensure the path is still connected
        self._repair_path(new_path)
        
        return new_path
        
    def _repair_path(self, path: List[Tuple[int, int]]):
        """Ensure the path is connected by adding intermediate points where needed."""
        i = 0
        while i < len(path) - 1:
            current = path[i]
            next_point = path[i + 1]
            
            # Check if points are adjacent
            if next_point not in self.get_neighbors(current):
                # Find a path between non-adjacent points
                mini_path = self._find_mini_path(current, next_point)
                if mini_path:
                    # Insert the mini-path (excluding the first and last points)
                    for j, point in enumerate(mini_path[1:-1]):
                        path.insert(i + 1 + j, point)
                    i += len(mini_path) - 2  # Skip the newly added points
                    
            i += 1
            
    def _find_mini_path(self, start: Tuple[int, int], end: Tuple[int, int]) -> List[Tuple[int, int]]:
        """Find a short path between two points using BFS."""
        from collections import deque
        
        queue = deque([(start, [start])])
        visited = {start}
        
        while queue:
            (vertex, path) = queue.popleft()
            
            for next_vertex in self.get_neighbors(vertex):
                if next_vertex == end:
                    return path + [next_vertex]
                    
                if next_vertex not in visited:
                    visited.add(next_vertex)
                    queue.append((next_vertex, path + [next_vertex]))
                    
        # If no path found, return None
        return None

    def evaluate_path(self, path: List[Tuple[int, int]]) -> Tuple[float, float, bool, str]:
        """Evaluate the cost, fuel, and feasibility of a path."""
        return self.is_path_feasible(path, self.MAX_FUEL)
        
    def search(self, start: Tuple[int, int], goal: Tuple[int, int]) -> List[Tuple[int, int]]:
        """Execute Simulated Annealing to find a path from start to goal."""
        self.start = start
        self.goal = goal
        self.visited.clear()
        self.current_path.clear()
        self.steps = 0
        self.path_length = 0
        self.cost = 0
        self.current_fuel = self.MAX_FUEL
        self.current_total_cost = 0
        self.current_fuel_cost = 0
        self.current_toll_cost = 0
        
        # Add start to visited for visualization
        self.add_visited(start)
        
        # Create initial path using BFS
        current_path = self.create_random_path(start, goal)
        if not current_path or current_path == [start, goal]:
            print("Could not find initial path")
            return []
        
        # Evaluate initial path
        is_feasible, reason, cost, fuel = self.evaluate_path_extended(current_path)
        if not is_feasible:
            print(f"Initial path not feasible: {reason}")
            return []
        
        best_path = current_path.copy()
        best_cost = cost
        
        # Initialize temperature
        temperature = self.initial_temperature
        
        # Add initial path to visualization
        self.current_position = start
        for pos in current_path:
            self.add_visited(pos)
        
        # Main loop
        while temperature > 1.0:
            # For each temperature, take several steps
            for _ in range(self.steps_per_temp):
                self.steps += 1
                
                # Create neighboring solution
                neighbor_path = self.create_neighbor_path(current_path)
                
                # Evaluate neighbor path
                neighbor_feasible, neighbor_reason, neighbor_cost, neighbor_fuel = self.evaluate_path_extended(neighbor_path)
                
                # If neighbor is feasible
                if neighbor_feasible:
                    # Calculate delta (cost difference)
                    cost_delta = neighbor_cost - cost
                    
                    # If neighbor is better or accepted with probability
                    if cost_delta <= 0 or random.random() < math.exp(-cost_delta / temperature):
                        current_path = neighbor_path
                        cost = neighbor_cost
                        
                        # Update best solution if it's better
                        if cost < best_cost:
                            best_path = current_path.copy()
                            best_cost = cost
                        
                        # Add to visited for visualization
                        for pos in neighbor_path:
                            self.add_visited(pos)
            
            # Cool down
            temperature *= self.cooling_rate
            
        # Validate and finalize the best path found
        if not best_path: # If no path was ever considered best (e.g., initial was not feasible)
            print("SIMULATED ANNEALING: No best path found during search.")
            self.current_path = []
            return []

        # First, validate and clean the best path
        validated_path = self.validate_path_no_obstacles(best_path)
        
        if not validated_path or len(validated_path) < 2:
            print(f"SIMULATED ANNEALING: Best path became invalid or too short after validation.")
            self.current_path = []
            return []

        # Second, check overall feasibility of the validated path
        is_still_feasible, reason_after_validation = self.is_path_feasible(validated_path, self.MAX_FUEL)
        if not is_still_feasible:
            print(f"SIMULATED ANNEALING: Best path after validation is not feasible: {reason_after_validation}")
            self.current_path = []
            return []
            
        # If all checks pass, this is our definitive path
        self.current_path = validated_path
        self.path_length = len(self.current_path) -1
        
        # Crucially, recalculate all costs and fuel based on this *final, validated* path
        self.calculate_path_fuel_consumption(self.current_path)
        # self.cost, self.current_fuel, self.current_total_cost, etc., are updated by the above call.
        
        # For compatibility, ensure self.best_cost reflects the final cost if needed elsewhere, though self.cost is primary.
        self.best_cost = self.cost 

        return self.current_path
    
    def evaluate_path_extended(self, path: List[Tuple[int, int]]) -> Tuple[bool, str, float, float]:
        """Extended evaluation that returns all important data about a path."""
        if not path or len(path) < 2:
            return False, "Path is empty or too short", float('inf'), 0
            
        is_feasible, reason = self.is_path_feasible(path, self.MAX_FUEL)
        
        if not is_feasible:
            return False, reason, float('inf'), 0
            
        # Calculate the actual cost of the path
        current_fuel = self.MAX_FUEL
        total_cost = 0
        visited_tolls = set()
        visited_gas = set()
        current_money = self.MAX_MONEY if self.current_money is None else self.current_money
        
        for i in range(len(path) - 1):
            pos1, pos2 = path[i], path[i + 1]
            state = SearchState(
                position=pos1,
                fuel=current_fuel,
                total_cost=total_cost,
                money=current_money,
                path=path[:i+1],
                visited_gas_stations=visited_gas,
                toll_stations_visited=visited_tolls
            )
            
            new_fuel, move_cost, new_money = self.calculate_cost(state, pos2)
            
            # Update state
            current_fuel = new_fuel
            total_cost += move_cost
            current_money = new_money
            
            # Update visited stations
            if self.grid[pos2[1], pos2[0]] == 2:  # Gas station
                visited_gas.add(pos2)
            elif self.grid[pos2[1], pos2[0]] == 1:  # Toll
                visited_tolls.add(pos2)
        
        return True, "Path is feasible", total_cost, current_fuel
    
    def step(self) -> bool:
        """Execute one step of Simulated Annealing."""
        if not self.start or not self.goal:
            return True  # Finished (not initialized)
        
        if self.temperature <= 0.1:
            return True  # Finished (cooled down)
        
        # We'll consider one temperature iteration as one step for visualization
        for _ in range(self.steps_per_temp):
            self.steps += 1
            
            if not hasattr(self, 'current_path') or not self.current_path:
                # Initialize with random path if not already done
                self.current_path = self.create_random_path(self.start, self.goal)
                is_feasible, reason, current_cost, current_fuel = self.evaluate_path_extended(self.current_path)
                self.best_path = self.current_path.copy() if is_feasible else []
                self.best_cost = current_cost if is_feasible else float('inf')
                self.current_position = self.start
                continue
            
            # Generate a neighboring solution
            new_path = self.create_neighbor_path(self.current_path)
            
            # Track visited nodes for visualization
            for pos in new_path:
                self.add_visited(pos)
            
            # Evaluate new path
            new_is_feasible, new_reason, new_cost, new_fuel = self.evaluate_path_extended(new_path)
            
            # Evaluate current path
            is_feasible, reason, current_cost, current_fuel = self.evaluate_path_extended(self.current_path)
            
            # Calculate delta cost
            delta_cost = float('inf')
            if new_is_feasible and is_feasible:
                delta_cost = new_cost - current_cost
            elif new_is_feasible and not is_feasible:
                delta_cost = -float('inf')  # Prefer any feasible path
            
            # Decide whether to accept the new solution
            accept = False
            
            if new_is_feasible:
                if delta_cost <= 0:  # New path is better
                    accept = True
                else:
                    # Accept with probability e^(-delta/T)
                    probability = math.exp(-delta_cost / self.temperature)
                    accept = random.random() < probability
            
            if accept:
                self.current_path = new_path
                
                # Update best path if this is better
                if new_is_feasible and (not self.best_path or new_cost < self.best_cost):
                    self.best_path = new_path.copy()
                    self.best_cost = new_cost
        
        # Cool down temperature
        self.temperature *= self.cooling_rate
        self.current_iteration += 1
        
        # Update current position for visualization
        if self.current_path:
            self.current_position = self.current_path[min(self.current_iteration, len(self.current_path) - 1)]
        
        # Update statistics with the best path found so far
        if self.best_path:
            is_feasible, reason, cost, fuel = self.evaluate_path_extended(self.best_path)
            self.path_length = len(self.best_path) - 1
            self.cost = cost
            self.current_fuel = fuel
            self.current_total_cost = cost
            
            # Estimate fuel and toll costs
            toll_cost = 0
            visited_toll_stations = set()
            for pos in self.best_path:
                if self.grid[pos[1], pos[0]] == 1 and pos not in visited_toll_stations:
                    toll_cost += self.TOLL_BASE_COST + self.TOLL_PENALTY
                    visited_toll_stations.add(pos)
            
            self.current_toll_cost = toll_cost
            self.current_fuel_cost = cost - toll_cost
            
            # Tính toán lượng nhiên liệu tiêu thụ cho đường đi
            self.calculate_path_fuel_consumption(self.best_path)
        
        return self.temperature <= 0.1 

    def get_random_adjacent_vertex(self, vertex: Tuple[int, int], visited: Set[Tuple[int, int]]) -> Optional[Tuple[int, int]]:
        """Lấy một đỉnh kề ngẫu nhiên chưa được thăm."""
        neighbors = []
        
        for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
            next_vertex = (vertex[0] + dx, vertex[1] + dy)
            
            # Kiểm tra trong giới hạn bản đồ và chưa thăm
            if self.is_valid_position(next_vertex) and next_vertex not in visited:
                # THÊM KIỂM TRA CHƯỚNG NGẠI VẬT
                if self.grid[next_vertex[1], next_vertex[0]] != self.OBSTACLE_CELL:
                    neighbors.append(next_vertex)
        
        if neighbors:
            return random.choice(neighbors)
        return None 

    def gen_random_path(self, start: Tuple[int, int], goal: Tuple[int, int], 
                         fuel: float = None, max_attempts: int = 100) -> List[Tuple[int, int]]:
        """Sinh một đường đi ngẫu nhiên từ start đến goal."""
        if fuel is None:
            fuel = self.MAX_FUEL
            
        current = start
        path = [current]
        visited = {current}
        
        attempt = 0
        while current != goal and attempt < max_attempts:
            # Tìm đỉnh kề chưa thăm
            next_vertex = self.get_random_adjacent_vertex(current, visited)
            
            if next_vertex is None:  # Không có đỉnh kề nào, quay lui một bước
                if len(path) > 1:
                    path.pop()
                    current = path[-1]
                else:
                    break  # Không thể tiếp tục
                    
            else:
                current = next_vertex
                path.append(current)
                visited.add(current)
                
                # Giảm nhiên liệu
                fuel -= self.FUEL_PER_MOVE
                
                # Kiểm tra hết nhiên liệu
                if fuel < 0:
                    break
                    
                # Nếu gặp trạm xăng
                if self.grid[current[1], current[0]] == 2:  # Trạm xăng
                    fuel = self.MAX_FUEL  # Nạp đầy nhiên liệu
            
            attempt += 1
        
        # Nếu không đến được đích, trả về đường đi một phần
        return path 