"""
Genetic Algorithm implementation for truck routing.
"""

from typing import List, Tuple, Dict, Set, Optional
import numpy as np
import random
import heapq
from collections import deque
from .base_search import BaseSearch, SearchState

class GeneticAlgorithm(BaseSearch):
    """Genetic Algorithm implementation."""
    
    def __init__(self, grid: np.ndarray, 
                 population_size: int = 50, 
                 crossover_rate: float = 0.8, 
                 mutation_rate: float = 0.2, 
                 generations: int = 100,
                 initial_money: float = None,
                 max_fuel: float = None, 
                 fuel_per_move: float = None, 
                 gas_station_cost: float = None, 
                 toll_base_cost: float = None,
                 initial_fuel: float = None):
        """Initialize Genetic Algorithm with a grid and parameters."""
        super().__init__(grid, initial_money, max_fuel, fuel_per_move, 
                         gas_station_cost, toll_base_cost, initial_fuel)
        self.population_size = population_size
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.generations = generations
        self.population = []
        self.start = None
        self.goal = None
        self.best_individual = None
        self.best_fitness = -float('inf')
        self.current_generation = 0
    
    def create_random_individual(self, start: Tuple[int, int], goal: Tuple[int, int]) -> List[Tuple[int, int]]:
        """Create a random individual (path) that is valid on the grid."""
        # Use BFS with randomization to create a valid path
        from collections import deque
        
        queue = deque([(start, [start])])
        visited = {start}
        
        while queue:
            (vertex, path) = queue.popleft()
            
            # Get neighbors in random order
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
    
    def initialize_population(self, start: Tuple[int, int], goal: Tuple[int, int]) -> List[List[Tuple[int, int]]]:
        """Initialize a population of random paths."""
        population = []
        
        # Create initial population
        for _ in range(self.population_size):
            individual = self.create_random_individual(start, goal)
            population.append(individual)
        
        return population
    
    def calculate_fitness(self, path: List[Tuple[int, int]]) -> float:
        """Calculate the fitness of an individual (path).
        Higher fitness is better."""
        if not path or len(path) < 2:
            return 0.0
            
        is_feasible, reason, cost, fuel = self.evaluate_path_extended(path)
        
        if not is_feasible:
            return 0.0  # Infeasible paths have zero fitness
        
        # For feasible paths, fitness is inverse of cost
        # Adding a constant to avoid division by zero
        return 1.0 / (cost + 1.0)
    
    def selection(self, population: List[List[Tuple[int, int]]], fitness_values: List[float]) -> List[List[Tuple[int, int]]]:
        """Select individuals for breeding using tournament selection."""
        selected = []
        
        # Tournament selection
        tournament_size = max(2, self.population_size // 10)
        
        for _ in range(self.population_size):
            # Select random candidates for tournament
            tournament_indices = random.sample(range(len(population)), tournament_size)
            
            # Find the best candidate
            best_index = tournament_indices[0]
            best_fitness = fitness_values[best_index]
            
            for idx in tournament_indices[1:]:
                if fitness_values[idx] > best_fitness:
                    best_index = idx
                    best_fitness = fitness_values[idx]
            
            selected.append(population[best_index])
        
        return selected
    
    def crossover(self, parent1: List[Tuple[int, int]], parent2: List[Tuple[int, int]]) -> Tuple[List[Tuple[int, int]], List[Tuple[int, int]]]:
        """Perform crossover between two parent paths."""
        if len(parent1) <= 2 or len(parent2) <= 2:
            # Can't crossover paths with just start and goal
            return parent1.copy(), parent2.copy()
        
        # Choose random crossover points
        point1 = random.randint(1, len(parent1) - 2)
        point2 = random.randint(1, len(parent2) - 2)
        
        # Create offspring
        child1 = parent1[:point1] + parent2[point2:]
        child2 = parent2[:point2] + parent1[point1:]
        
        # Repair paths if needed
        self._repair_path(child1)
        self._repair_path(child2)
        
        return child1, child2
    
    def mutate(self, path: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
        """Mutate a path by making small random changes."""
        if len(path) <= 2:
            return path.copy()  # Can't mutate a direct path
            
        mutated_path = path.copy()
        mutation_type = random.choice([0, 1, 2])  # Different mutation types
        
        if mutation_type == 0 and len(path) > 3:
            # Remove a random point (except start and goal)
            idx = random.randint(1, len(mutated_path) - 2)
            mutated_path.pop(idx)
            
        elif mutation_type == 1:
            # Modify a point (except start and goal)
            idx = random.randint(1, len(mutated_path) - 2)
            current_point = mutated_path[idx]
            
            # Find valid neighbors
            valid_neighbors = self.get_neighbors(current_point)
            
            if valid_neighbors:
                mutated_path[idx] = random.choice(valid_neighbors)
                
        elif mutation_type == 2:
            # Add a new point between two existing points
            idx = random.randint(0, len(mutated_path) - 2)
            point1 = mutated_path[idx]
            point2 = mutated_path[idx + 1]
            
            # Try to find a valid point between point1 and point2
            valid_neighbors = []
            for n1 in self.get_neighbors(point1):
                for n2 in self.get_neighbors(n1):
                    if n2 == point2 and n1 not in mutated_path:
                        valid_neighbors.append(n1)
                            
            if valid_neighbors:
                mutated_path.insert(idx + 1, random.choice(valid_neighbors))
        
        # Repair paths if needed
        self._repair_path(mutated_path)
        
        return mutated_path
    
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
    
    def search(self, start: Tuple[int, int], goal: Tuple[int, int]) -> List[Tuple[int, int]]:
        """Execute Genetic Algorithm to find an optimal path."""
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
        
        # Initialize population
        self.population = self.initialize_population(start, goal)
        
        # Add start to visited nodes
        self.add_visited(start)
        
        # Main genetic algorithm loop
        for generation in range(self.generations):
            self.current_generation = generation
            self.steps += 1
            
            # Calculate fitness for each individual
            fitness_values = [self.calculate_fitness(individual) for individual in self.population]
            
            # Track all visited nodes
            for individual in self.population:
                for pos in individual:
                    self.add_visited(pos)
            
            # Find the best individual
            best_idx = fitness_values.index(max(fitness_values))
            current_best = self.population[best_idx]
            current_best_fitness = fitness_values[best_idx]
            
            # Update the best individual found so far
            if current_best_fitness > 0 and (self.best_individual is None or current_best_fitness > self.best_fitness):
                self.best_individual = current_best.copy()
                self.best_fitness = current_best_fitness
            
            # Selection
            parents = self.selection(self.population, fitness_values)
            
            # Create a new population
            new_population = []
            
            # Elitism: Keep the best individual
            if self.best_individual:
                new_population.append(self.best_individual)
            
            # Crossover and mutation
            while len(new_population) < self.population_size:
                # Select two parents
                parent1 = random.choice(parents)
                parent2 = random.choice(parents)
                
                # Crossover
                if random.random() < self.crossover_rate:
                    child1, child2 = self.crossover(parent1, parent2)
                else:
                    child1, child2 = parent1.copy(), parent2.copy()
                
                # Mutation
                if random.random() < self.mutation_rate:
                    child1 = self.mutate(child1)
                if random.random() < self.mutation_rate:
                    child2 = self.mutate(child2)
                
                # Add to new population
                new_population.append(child1)
                if len(new_population) < self.population_size:
                    new_population.append(child2)
            
            # Replace old population
            self.population = new_population
        
        # Update algorithm statistics with the best individual
        if self.best_individual:
            raw_best_path = self.best_individual
            
            # First, validate and clean the best path found
            validated_path = self.validate_path_no_obstacles(raw_best_path)
            
            if not validated_path or len(validated_path) < 2:
                print(f"GENETIC: Best path became invalid or too short after validation.")
                self.current_path = []
                return []

            # Second, check overall feasibility of the validated path
            is_still_feasible, reason_after_validation = self.is_path_feasible(validated_path, self.MAX_FUEL)
            if not is_still_feasible:
                print(f"GENETIC: Best path after validation is not feasible: {reason_after_validation}")
                self.current_path = []
                return []
                
            # If all checks pass, this is our definitive path
            self.current_path = validated_path
            self.path_length = len(self.current_path) - 1
            
            # Crucially, recalculate all costs and fuel based on this *final, validated* path
            # This will set self.cost, self.current_fuel, etc. correctly.
            self.calculate_path_fuel_consumption(self.current_path)
            
            return self.current_path
        
        print("GENETIC: No feasible path found after all generations.")
        self.current_path = []
        return []  # Return empty list if no feasible path is found
    
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
        """Execute one step (generation) of the Genetic Algorithm."""
        if not self.start or not self.goal:
            return True  # Finished (not initialized)
        
        if self.current_generation >= self.generations:
            return True  # Finished all generations
        
        # Initialize population if this is the first step
        if not self.population:
            self.population = self.initialize_population(self.start, self.goal)
            self.current_generation = 0
        
        # Calculate fitness for each individual
        fitness_values = [self.calculate_fitness(individual) for individual in self.population]
        
        # Track all visited nodes for visualization
        for individual in self.population:
            for pos in individual:
                self.add_visited(pos)
        
        # Find the best individual in this generation
        best_idx = fitness_values.index(max(fitness_values))
        current_best = self.population[best_idx]
        current_best_fitness = fitness_values[best_idx]
        
        # Update the best individual found so far
        if current_best_fitness > 0 and (self.best_individual is None or current_best_fitness > self.best_fitness):
            self.best_individual = current_best.copy()
            self.best_fitness = current_best_fitness
            
            # Update statistics for visualization
            is_feasible, reason, cost, fuel = self.evaluate_path_extended(self.best_individual)
            self.current_path = self.best_individual
            self.path_length = len(self.best_individual) - 1
            self.cost = cost
            self.current_fuel = fuel
            self.current_total_cost = cost
            
            # Estimate fuel and toll costs
            toll_cost = 0
            visited_toll_stations = set()
            for pos in self.best_individual:
                if self.grid[pos[1], pos[0]] == 1 and pos not in visited_toll_stations:
                    toll_cost += self.TOLL_BASE_COST + self.TOLL_PENALTY
                    visited_toll_stations.add(pos)
            
            self.current_toll_cost = toll_cost
            self.current_fuel_cost = cost - toll_cost
            
            # Tính toán lượng nhiên liệu tiêu thụ cho đường đi
            self.calculate_path_fuel_consumption(self.best_individual)
        
        # Selection
        parents = self.selection(self.population, fitness_values)
        
        # Create a new population
        new_population = []
        
        # Elitism: Keep the best individual
        if self.best_individual:
            new_population.append(self.best_individual)
        
        # Crossover and mutation
        while len(new_population) < self.population_size:
            # Select two parents
            parent1 = random.choice(parents)
            parent2 = random.choice(parents)
            
            # Crossover
            if random.random() < self.crossover_rate:
                child1, child2 = self.crossover(parent1, parent2)
            else:
                child1, child2 = parent1.copy(), parent2.copy()
            
            # Mutation
            if random.random() < self.mutation_rate:
                child1 = self.mutate(child1)
            if random.random() < self.mutation_rate:
                child2 = self.mutate(child2)
            
            # Add to new population
            new_population.append(child1)
            if len(new_population) < self.population_size:
                new_population.append(child2)
        
        # Replace old population
        self.population = new_population
        
        # Update current position for visualization (use the best individual's position)
        if self.best_individual and len(self.best_individual) > 1:
            current_pos_idx = min(self.current_generation, len(self.best_individual) - 1)
            self.current_position = self.best_individual[current_pos_idx]
        else:
            self.current_position = self.start
        
        # Increment generation counter
        self.current_generation += 1
        self.steps += 1
        
        return self.current_generation >= self.generations 