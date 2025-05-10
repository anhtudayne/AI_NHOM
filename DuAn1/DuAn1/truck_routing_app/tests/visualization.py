"""
Visualization utilities for algorithm test results.
"""

import matplotlib.pyplot as plt
import numpy as np
import os

def create_output_dir():
    """Create the output directory for visualizations if it doesn't exist."""
    output_dir = os.path.join(os.path.dirname(__file__), 'output')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    return output_dir

def visualize_map(map_obj, algorithm=None, path=None, visited=None, filename=None):
    """
    Visualize a map with optional algorithm results.
    
    Args:
        map_obj: The Map object to visualize
        algorithm: Optional algorithm object that ran on the map
        path: Optional path found by the algorithm
        visited: Optional list of visited positions
        filename: Optional filename to save the plot
    """
    fig, ax = plt.subplots(figsize=(10, 10))
    
    # Create a color grid for the map
    grid = map_obj.grid
    size = map_obj.size
    color_map = np.zeros((size, size, 3))
    
    # Define colors for different cell types
    road_color = [0.95, 0.95, 0.95]    # Light gray for roads
    toll_color = [0.9, 0.4, 0.4]       # Red for toll stations
    gas_color = [0.4, 0.8, 0.4]        # Green for gas stations
    brick_color = [0.5, 0.5, 0.5]      # Gray for obstacles
    
    # Color the cells based on type
    for i in range(size):
        for j in range(size):
            if grid[j, i] == 0:      # Road
                color_map[j, i] = road_color
            elif grid[j, i] == 1:    # Toll station
                color_map[j, i] = toll_color
            elif grid[j, i] == 2:    # Gas station
                color_map[j, i] = gas_color
            elif grid[j, i] == -1:    # Obstacle
                color_map[j, i] = brick_color
    
    # Draw the grid
    ax.imshow(color_map, origin='upper')
    
    # Draw grid lines
    for i in range(size + 1):
        ax.axhline(i - 0.5, color='black', linewidth=0.5)
        ax.axvline(i - 0.5, color='black', linewidth=0.5)
    
    # Draw visited positions
    if visited:
        x_visited = [pos[0] for pos in visited]
        y_visited = [pos[1] for pos in visited]
        ax.scatter(x_visited, y_visited, c='lightskyblue', s=30, alpha=0.5)
    
    # Draw path
    if path:
        x_path = [pos[0] for pos in path]
        y_path = [pos[1] for pos in path]
        ax.plot(x_path, y_path, c='blue', linewidth=3, alpha=0.7)
    
    # Draw start and goal positions
    if map_obj.start_pos:
        ax.plot(map_obj.start_pos[0], map_obj.start_pos[1], 'go', markersize=15)
    if map_obj.end_pos:
        ax.plot(map_obj.end_pos[0], map_obj.end_pos[1], 'ro', markersize=15)
    
    # Add labels for cell types
    ax.text(0.02, 0.02, 'Road', transform=ax.transAxes, color='black')
    ax.text(0.02, 0.06, 'Toll Station', transform=ax.transAxes, color=toll_color)
    ax.text(0.02, 0.10, 'Gas Station', transform=ax.transAxes, color='green')
    ax.text(0.02, 0.14, 'Obstacle', transform=ax.transAxes, color='dimgray')
    
    # Add algorithm information if provided
    if algorithm:
        title = f"Map {size}x{size}"
        if hasattr(algorithm, '__class__'):
            title += f" - {algorithm.__class__.__name__}"
        if path:
            title += f"\nPath Length: {len(path) - 1}, Cost: {algorithm.cost:.2f}, Steps: {algorithm.steps}"
        plt.title(title)
    else:
        plt.title(f"Map {size}x{size}")
    
    # Save the plot if filename is provided
    if filename:
        output_dir = create_output_dir()
        plt.savefig(os.path.join(output_dir, filename), dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.tight_layout()
        plt.show()

def plot_performance_comparison(results, metric='execution_time', title='Algorithm Performance', filename=None):
    """
    Create a bar chart comparing algorithm performance.
    
    Args:
        results: List of result dictionaries from algorithm tests
        metric: The metric to compare ('execution_time', 'path_length', 'cost', or 'steps')
        title: Title for the plot
        filename: Optional filename to save the plot
    """
    algorithms = [r['algorithm'] for r in results]
    values = [r[metric] for r in results]
    
    plt.figure(figsize=(10, 6))
    bars = plt.bar(algorithms, values)
    
    # Add values above bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.01 * max(values),
                f'{height:.2f}', ha='center', va='bottom')
    
    plt.title(title)
    plt.xlabel('Algorithm')
    
    if metric == 'execution_time':
        plt.ylabel('Time (seconds)')
    elif metric == 'path_length':
        plt.ylabel('Path Length')
    elif metric == 'cost':
        plt.ylabel('Path Cost')
    elif metric == 'steps':
        plt.ylabel('Algorithm Steps')
    
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    # Save the plot if filename is provided
    if filename:
        output_dir = create_output_dir()
        plt.savefig(os.path.join(output_dir, filename), dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

def plot_trend_analysis(results, x_key, y_key, title, filename=None):
    """
    Create a line plot showing trends in the results.
    
    Args:
        results: List of result dictionaries
        x_key: The key to use for the x-axis
        y_key: The key to use for the y-axis
        title: Title for the plot
        filename: Optional filename to save the plot
    """
    # Aggregate data by x_key
    x_values = sorted(set(r[x_key] for r in results))
    y_values = []
    
    for x in x_values:
        matching_results = [r[y_key] for r in results if r[x_key] == x]
        y_values.append(sum(matching_results) / len(matching_results) if matching_results else 0)
    
    plt.figure(figsize=(10, 6))
    plt.plot(x_values, y_values, 'o-', linewidth=2)
    
    # Add data points
    for i, (x, y) in enumerate(zip(x_values, y_values)):
        plt.text(x, y + 0.01 * max(y_values), f'{y:.2f}', ha='center')
    
    plt.title(title)
    plt.xlabel(x_key.replace('_', ' ').title())
    plt.ylabel(y_key.replace('_', ' ').title())
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    # Save the plot if filename is provided
    if filename:
        output_dir = create_output_dir()
        plt.savefig(os.path.join(output_dir, filename), dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show() 