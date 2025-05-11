from .state_and_actions import State, DriveAction, RepairAction
from .environment import get_action_results
from core.algorithms.base_search import OBSTACLE_CELL # Import OBSTACLE_CELL

class AndOrProblem:
    def __init__(self, map_grid, start_coord, final_dest_coord):
        self.map_grid = map_grid # numpy array
        self.start_coord = start_coord # (x,y) tuple
        self.final_dest_coord = final_dest_coord # (x,y) tuple
        self._initial_state = State(location=self.start_coord, is_broken_down=False)
        self.rows = map_grid.shape[0]
        self.cols = map_grid.shape[1]
        # OBSTACLE_CELL is imported and used directly

    def get_initial_state(self):
        return self._initial_state

    def get_actions(self, state):
        current_x, current_y = state.location
        possible_actions = []
        
        if state.is_broken_down:
            possible_actions.append(RepairAction())
        else:
            if state.location == self.final_dest_coord:
                return [] # Goal reached

            potential_moves = [(0, 1), (0, -1), (1, 0), (-1, 0)] # Down, Up, Right, Left (typical for y,x or row,col indexing if y is first)
                                                                # Or for (x,y): (x, y+1), (x, y-1), (x+1, y), (x-1,y)

            for dx, dy in potential_moves:
                next_x, next_y = current_x + dx, current_y + dy

                if 0 <= next_x < self.cols and 0 <= next_y < self.rows:
                    if self.map_grid[next_y, next_x] != OBSTACLE_CELL:
                        possible_actions.append(DriveAction(destination=(next_x, next_y)))
                        
        return possible_actions

    def get_results(self, state, action):
        return get_action_results(state, action)

    def is_goal(self, state):
        return (state.location == self.final_dest_coord and 
                not state.is_broken_down)