"""
Module for detailed evaluation of RL agents and comparison with other pathfinding algorithms.
"""

import os
import time
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import seaborn as sns
from tqdm import tqdm

from core.map import Map
from core.rl_environment import TruckRoutingEnv
from core.algorithms.rl_DQNAgent import DQNAgentTrainer

# Import available pathfinding algorithms for comparison
from core.algorithms.astar import AStar
from core.algorithms.greedy import Greedy
from core.algorithms.genetic_algorithm import GeneticAlgorithm
from core.algorithms.simulated_annealing import SimulatedAnnealing
from core.algorithms.local_beam import LocalBeamSearch

class RLEvaluator:
    """
    Class for evaluating RL agents and comparing with traditional algorithms.
    """
    def __init__(self, maps_dir="maps"):
        """
        Initialize the evaluator.
        
        Args:
            maps_dir: Directory containing test maps
        """
        self.maps_dir = maps_dir
        self.results_dir = "evaluation_results"
        os.makedirs(self.results_dir, exist_ok=True)
        
        # Load all maps
        self.maps = self._load_maps()
    
    def _load_maps(self):
        """Load all maps from the maps directory."""
        maps = []
        for map_file in os.listdir(self.maps_dir):
            if map_file.endswith(".json"):
                map_path = os.path.join(self.maps_dir, map_file)
                try:
                    map_obj = Map.load(map_path)
                    if map_obj is not None:
                        maps.append((map_file, map_obj))
                except Exception as e:
                    print(f"Error loading map {map_file}: {e}")
        return maps
    
    def evaluate_rl_agent(self, model_path, n_episodes=10, initial_fuel=5.0, 
                          initial_money=1000.0, fuel_per_move=0.3, map_filter=None):
        """
        Evaluate a trained RL agent on all maps.
        
        Args:
            model_path: Path to the trained model
            n_episodes: Number of episodes per map
            initial_fuel: Initial fuel amount
            initial_money: Initial money amount
            fuel_per_move: Fuel consumed per move
            map_filter: Optional string to filter maps by name
            
        Returns:
            pd.DataFrame: Evaluation results for the RL agent
        """
        results = []
        
        # Lọc bản đồ theo pattern nếu được chỉ định
        filtered_maps = self.maps
        if map_filter:
            filtered_maps = [(name, map_obj) for name, map_obj in self.maps if map_filter in name]
            
        # Nếu không có bản đồ nào thỏa điều kiện lọc, sử dụng tất cả bản đồ
        if not filtered_maps and map_filter:
            print(f"Warning: No maps found with filter '{map_filter}', using all maps.")
            filtered_maps = self.maps
            
        for map_name, map_obj in tqdm(filtered_maps, desc="Evaluating RL agent on maps"):
            # Create environment
            env = TruckRoutingEnv(
                map_object=map_obj,
                initial_fuel=initial_fuel,
                initial_money=initial_money,
                fuel_per_move=fuel_per_move,
                max_steps_per_episode=2 * map_obj.size * map_obj.size
            )
            
            # Create agent and load model
            agent = DQNAgentTrainer(env)
            agent.load_model(model_path)
            
            # Evaluate agent on multiple episodes
            map_results = self._evaluate_agent_on_map(agent, env, map_name, n_episodes)
            results.extend(map_results)
        
        # Convert to DataFrame
        df = pd.DataFrame(results)
        
        # Calculate additional metrics
        df['fuel_efficiency'] = df['remaining_fuel'] / df['path_length']
        df['money_efficiency'] = df['remaining_money'] / df['path_length']
        df['success_rate'] = df['success'].astype(float)
        
        # Group by map and calculate statistics
        map_stats = df.groupby('map_name').agg({
            'success': ['mean', 'std'],
            'path_length': ['mean', 'std', 'min', 'max'],
            'remaining_fuel': ['mean', 'std', 'min', 'max'],
            'remaining_money': ['mean', 'std', 'min', 'max'],
            'fuel_efficiency': ['mean', 'std'],
            'money_efficiency': ['mean', 'std']
        }).round(2)
        
        # Save detailed results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_path = os.path.join(self.results_dir, f"rl_evaluation_{timestamp}.csv")
        df.to_csv(results_path, index=False)
        
        # Save map statistics
        stats_path = os.path.join(self.results_dir, f"rl_evaluation_stats_{timestamp}.csv")
        map_stats.to_csv(stats_path)
        
        # Generate evaluation report
        self._generate_evaluation_report(df, map_stats, timestamp)
        
        return df
    
    def _evaluate_agent_on_map(self, agent, env, map_name, n_episodes):
        """Evaluate agent on a single map for multiple episodes."""
        results = []
        
        for episode in range(n_episodes):
            obs, _ = env.reset()
            done = False
            truncated = False
            episode_reward = 0
            path_length = 0
            actions_taken = []
            
            while not (done or truncated):
                action = agent.predict_action(obs)
                obs, reward, done, truncated, info = env.step(action)
                episode_reward += reward
                path_length += 1
                actions_taken.append(action)
            
            # Record episode results
            results.append({
                "map_name": map_name,
                "episode": episode + 1,
                "success": info.get("success", False),
                "path_length": path_length,
                "remaining_fuel": float(obs["fuel"][0]),
                "remaining_money": float(obs["money"][0]),
                "reward": episode_reward,
                "termination_reason": info.get("termination_reason", "unknown"),
                "actions": actions_taken
            })
        
        return results
    
    def _generate_evaluation_report(self, df, map_stats, timestamp):
        """Generate a detailed evaluation report with visualizations."""
        report_dir = os.path.join(self.results_dir, f"evaluation_report_{timestamp}")
        os.makedirs(report_dir, exist_ok=True)
        
        # 1. Overall Success Rate
        plt.figure(figsize=(10, 6))
        success_rate = df['success'].mean() * 100
        plt.bar(['Success Rate'], [success_rate], color='green')
        plt.title('Overall Success Rate')
        plt.ylabel('Percentage')
        plt.ylim(0, 100)
        plt.savefig(os.path.join(report_dir, 'success_rate.png'))
        plt.close()
        
        # 2. Path Length Distribution
        plt.figure(figsize=(10, 6))
        sns.histplot(data=df[df['success']], x='path_length', bins=20)
        plt.title('Path Length Distribution (Successful Episodes)')
        plt.xlabel('Path Length')
        plt.ylabel('Count')
        plt.savefig(os.path.join(report_dir, 'path_length_dist.png'))
        plt.close()
        
        # 3. Resource Usage
        plt.figure(figsize=(12, 6))
        resources = pd.DataFrame({
            'Fuel': df['remaining_fuel'],
            'Money': df['remaining_money']
        })
        sns.boxplot(data=resources)
        plt.title('Resource Usage Distribution')
        plt.ylabel('Amount')
        plt.savefig(os.path.join(report_dir, 'resource_usage.png'))
        plt.close()
        
        # 4. Map-wise Performance
        plt.figure(figsize=(12, 6))
        map_success = df.groupby('map_name')['success'].mean() * 100
        map_success.plot(kind='bar')
        plt.title('Success Rate by Map')
        plt.xlabel('Map')
        plt.ylabel('Success Rate (%)')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(report_dir, 'map_performance.png'))
        plt.close()
        
        # 5. Efficiency Metrics
        plt.figure(figsize=(12, 6))
        efficiency = pd.DataFrame({
            'Fuel Efficiency': df['fuel_efficiency'],
            'Money Efficiency': df['money_efficiency']
        })
        sns.boxplot(data=efficiency)
        plt.title('Resource Efficiency Distribution')
        plt.ylabel('Efficiency Score')
        plt.savefig(os.path.join(report_dir, 'efficiency_metrics.png'))
        plt.close()
        
        # Save summary statistics
        summary = {
            "timestamp": timestamp,
            "total_episodes": len(df),
            "overall_success_rate": success_rate,
            "avg_path_length": df['path_length'].mean(),
            "avg_remaining_fuel": df['remaining_fuel'].mean(),
            "avg_remaining_money": df['remaining_money'].mean(),
            "fuel_efficiency": df['fuel_efficiency'].mean(),
            "money_efficiency": df['money_efficiency'].mean()
        }
        
        with open(os.path.join(report_dir, 'summary.json'), 'w') as f:
            json.dump(summary, f, indent=2)
    
    def evaluate_traditional_algorithm(self, algorithm_name, n_runs=5, initial_fuel=5.0, 
                                      initial_money=1000.0, fuel_per_move=0.3):
        """
        Evaluate a traditional pathfinding algorithm on all maps.
        
        Args:
            algorithm_name: Name of the algorithm to evaluate
            n_runs: Number of runs per map (for stochastic algorithms)
            initial_fuel: Initial fuel amount
            initial_money: Initial money amount
            fuel_per_move: Fuel consumed per move
            
        Returns:
            pd.DataFrame: Evaluation results for the algorithm
        """
        if algorithm_name not in self.algorithms:
            raise ValueError(f"Algorithm {algorithm_name} not supported")
        
        results = []
        
        for map_name, map_obj in tqdm(self.maps, desc=f"Evaluating {algorithm_name} on maps"):
            # Get algorithm class
            AlgorithmClass = self.algorithms[algorithm_name]
            
            # Create algorithm instance
            algorithm = AlgorithmClass(map_obj)
            
            for run in range(n_runs):
                # Start time
                start_time = time.time()
                
                # Run algorithm
                if algorithm_name in ["Genetic Algorithm", "Simulated Annealing", "Local Beam Search"]:
                    # Stochastic algorithms
                    path, metrics = algorithm.search(
                        start_pos=map_obj.start_pos,
                        goal_pos=map_obj.end_pos,
                        initial_fuel=initial_fuel,
                        initial_money=initial_money,
                        fuel_consumption=fuel_per_move
                    )
                else:
                    # Deterministic algorithms
                    path, metrics = algorithm.search(
                        start_pos=map_obj.start_pos,
                        goal_pos=map_obj.end_pos,
                        initial_fuel=initial_fuel,
                        initial_money=initial_money,
                        fuel_per_move=fuel_per_move
                    )
                
                # End time
                end_time = time.time()
                execution_time = end_time - start_time
                
                # Check success
                success = path is not None and len(path) > 0 and path[-1] == map_obj.end_pos
                
                # Compute path length
                path_length = len(path) - 1 if path else 0  # Exclude starting position
                
                # Store results
                results.append({
                    "algorithm": algorithm_name,
                    "map_name": map_name,
                    "run": run,
                    "success": success,
                    "path_length": path_length,
                    "execution_time": execution_time,
                    **metrics  # Include all metrics from the algorithm
                })
        
        # Convert to DataFrame
        df = pd.DataFrame(results)
        
        # Save results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_path = os.path.join(self.results_dir, f"{algorithm_name.lower().replace(' ', '_')}_evaluation_{timestamp}.csv")
        df.to_csv(results_path, index=False)
        
        return df
    
    def compare_algorithms(self, rl_model_path, algorithms_to_compare=None, n_episodes=10, n_runs=5,
                          initial_fuel=5.0, initial_money=1000.0, fuel_per_move=0.3):
        """
        Compare RL agent with traditional algorithms.
        
        Args:
            rl_model_path: Path to the trained RL model
            algorithms_to_compare: List of algorithm names to compare with
            n_episodes: Number of episodes for RL agent
            n_runs: Number of runs for traditional algorithms
            initial_fuel: Initial fuel amount
            initial_money: Initial money amount
            fuel_per_move: Fuel consumed per move
            
        Returns:
            pd.DataFrame: Combined evaluation results
        """
        if algorithms_to_compare is None:
            algorithms_to_compare = list(self.algorithms.keys())
        
        # Evaluate RL agent
        rl_results = self.evaluate_rl_agent(
            model_path=rl_model_path,
            n_episodes=n_episodes,
            initial_fuel=initial_fuel,
            initial_money=initial_money,
            fuel_per_move=fuel_per_move
        )
        
        # Evaluate traditional algorithms
        all_results = [rl_results]
        
        for algorithm_name in algorithms_to_compare:
            try:
                algorithm_results = self.evaluate_traditional_algorithm(
                    algorithm_name=algorithm_name,
                    n_runs=n_runs,
                    initial_fuel=initial_fuel,
                    initial_money=initial_money,
                    fuel_per_move=fuel_per_move
                )
                all_results.append(algorithm_results)
            except Exception as e:
                print(f"Error evaluating {algorithm_name}: {e}")
        
        # Combine results
        combined_results = pd.concat(all_results, ignore_index=True)
        
        # Save combined results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_path = os.path.join(self.results_dir, f"algorithm_comparison_{timestamp}.csv")
        combined_results.to_csv(results_path, index=False)
        
        # Generate comparison report
        self.generate_comparison_report(combined_results)
        
        return combined_results
    
    def generate_comparison_report(self, results_df):
        """
        Generate a comparison report with visualizations.
        
        Args:
            results_df: DataFrame with comparison results
        """
        # Create a results directory for the report
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_dir = os.path.join(self.results_dir, f"comparison_report_{timestamp}")
        os.makedirs(report_dir, exist_ok=True)
        
        # Save the full results
        results_df.to_csv(os.path.join(report_dir, "full_results.csv"), index=False)
        
        # Compute aggregated metrics by algorithm
        agg_metrics = results_df.groupby("algorithm").agg({
            "success": "mean",
            "path_length": ["mean", "std"],
            "execution_time": ["mean", "std"],
            "fuel_consumed": ["mean", "std"] if "fuel_consumed" in results_df.columns else None,
            "money_spent": ["mean", "std"] if "money_spent" in results_df.columns else None,
        }).reset_index()
        
        # Save aggregated metrics
        agg_metrics.to_csv(os.path.join(report_dir, "aggregated_metrics.csv"))
        
        # Create visualizations
        
        # 1. Success Rate Comparison
        plt.figure(figsize=(10, 6))
        success_rates = results_df.groupby("algorithm")["success"].mean().sort_values(ascending=False)
        success_rates.plot(kind="bar", color="green")
        plt.title("Success Rate by Algorithm")
        plt.ylabel("Success Rate")
        plt.xlabel("Algorithm")
        plt.tight_layout()
        plt.savefig(os.path.join(report_dir, "success_rates.png"))
        plt.close()
        
        # 2. Path Length Comparison
        plt.figure(figsize=(10, 6))
        sns.boxplot(x="algorithm", y="path_length", data=results_df[results_df["success"] == True])
        plt.title("Path Length Comparison (Successful Runs Only)")
        plt.ylabel("Path Length")
        plt.xlabel("Algorithm")
        plt.tight_layout()
        plt.savefig(os.path.join(report_dir, "path_length_comparison.png"))
        plt.close()
        
        # 3. Execution Time Comparison
        plt.figure(figsize=(10, 6))
        sns.boxplot(x="algorithm", y="execution_time", data=results_df)
        plt.title("Execution Time Comparison")
        plt.ylabel("Execution Time (seconds)")
        plt.xlabel("Algorithm")
        plt.tight_layout()
        plt.savefig(os.path.join(report_dir, "execution_time_comparison.png"))
        plt.close()
        
        # 4. Radar Chart for Multiple Metrics
        # Normalize metrics for radar chart
        metrics_to_plot = ["success", "path_length", "execution_time"]
        if "fuel_consumed" in results_df.columns:
            metrics_to_plot.extend(["fuel_consumed", "money_spent"])
        
        # Prepare data for radar chart
        radar_data = {}
        for metric in metrics_to_plot:
            if metric == "success":
                # Higher is better for success rate
                values = results_df.groupby("algorithm")[metric].mean()
                normalized = values / values.max()
            elif metric in ["path_length", "execution_time", "fuel_consumed", "money_spent"]:
                # Lower is better for these metrics
                values = results_df.groupby("algorithm")[metric].mean()
                normalized = 1 - (values / values.max())
            
            radar_data[metric] = normalized
        
        # Create radar chart
        self._create_radar_chart(radar_data, report_dir)
        
        # Create HTML report
        self._create_html_report(report_dir, agg_metrics, radar_data)
        
        print(f"Comparison report generated at {report_dir}")
    
    def _create_radar_chart(self, radar_data, report_dir):
        """
        Create a radar chart comparing algorithms on multiple metrics.
        
        Args:
            radar_data: Dictionary with normalized metric values by algorithm
            report_dir: Directory to save the chart
        """
        # Get algorithms and metrics
        algorithms = radar_data[list(radar_data.keys())[0]].index.tolist()
        metrics = list(radar_data.keys())
        
        # Create radar chart
        angles = np.linspace(0, 2*np.pi, len(metrics), endpoint=False).tolist()
        angles += angles[:1]  # Close the loop
        
        fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))
        
        for algorithm in algorithms:
            values = [radar_data[metric][algorithm] for metric in metrics]
            values += values[:1]  # Close the loop
            
            ax.plot(angles, values, linewidth=2, label=algorithm)
            ax.fill(angles, values, alpha=0.1)
        
        # Set labels and ticks
        ax.set_thetagrids(np.degrees(angles[:-1]), metrics)
        ax.set_ylim(0, 1)
        plt.legend(loc="upper right", bbox_to_anchor=(0.1, 0.1))
        plt.title("Algorithm Performance Comparison")
        
        # Save the chart
        plt.tight_layout()
        plt.savefig(os.path.join(report_dir, "radar_chart.png"))
        plt.close()
    
    def _create_html_report(self, report_dir, agg_metrics, radar_data):
        """
        Create an HTML report with results and visualizations.
        
        Args:
            report_dir: Directory to save the report
            agg_metrics: Aggregated metrics DataFrame
            radar_data: Data used for radar chart
        """
        html_content = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>Algorithm Comparison Report</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 20px; }
                h1, h2 { color: #333366; }
                table { border-collapse: collapse; width: 100%; margin-bottom: 20px; }
                th, td { padding: 8px; text-align: left; border-bottom: 1px solid #ddd; }
                th { background-color: #333366; color: white; }
                tr:hover { background-color: #f5f5f5; }
                .chart-container { margin: 20px 0; text-align: center; }
                .chart-container img { max-width: 100%; height: auto; }
            </style>
        </head>
        <body>
            <h1>Algorithm Comparison Report</h1>
            <p>Generated on: {timestamp}</p>
            
            <h2>Aggregated Metrics</h2>
            {agg_metrics_table}
            
            <h2>Visualizations</h2>
            
            <div class="chart-container">
                <h3>Success Rate Comparison</h3>
                <img src="success_rates.png" alt="Success Rate Comparison">
            </div>
            
            <div class="chart-container">
                <h3>Path Length Comparison</h3>
                <img src="path_length_comparison.png" alt="Path Length Comparison">
            </div>
            
            <div class="chart-container">
                <h3>Execution Time Comparison</h3>
                <img src="execution_time_comparison.png" alt="Execution Time Comparison">
            </div>
            
            <div class="chart-container">
                <h3>Radar Chart (Normalized Performance)</h3>
                <img src="radar_chart.png" alt="Radar Chart">
            </div>
        </body>
        </html>
        """.format(
            timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            agg_metrics_table=agg_metrics.to_html()
        )
        
        # Save the HTML report
        with open(os.path.join(report_dir, "report.html"), "w") as f:
            f.write(html_content)
    
    def evaluate_single_episode(self, model_path, map_obj, initial_fuel=5.0, 
                               initial_money=1000.0, fuel_per_move=0.3, render=False):
        """
        Evaluate a single episode for visualization in the UI.
        
        Args:
            model_path: Path to the trained model
            map_obj: Map object
            initial_fuel: Initial fuel amount
            initial_money: Initial money amount
            fuel_per_move: Fuel consumed per move
            render: Whether to render each step
            
        Returns:
            tuple: (path, metrics, step_by_step_info)
        """
        # Create environment
        env = TruckRoutingEnv(
            map_object=map_obj,
            initial_fuel=initial_fuel,
            initial_money=initial_money,
            fuel_per_move=fuel_per_move,
            max_steps_per_episode=2 * map_obj.size * map_obj.size
        )
        
        # Create agent and load model
        agent = DQNAgentTrainer(env)
        agent.load_model(model_path)
        
        # Reset environment
        observation, _ = env.reset()
        
        # Initialize metrics
        total_reward = 0
        total_steps = 0
        fuel_consumed = 0
        money_spent = 0
        toll_visits = 0
        gas_station_visits = 0
        refuels = 0
        path = [env.current_pos]
        step_by_step_info = []
        terminated = False
        truncated = False
        
        # Episode loop
        while not (terminated or truncated):
            # Select action
            action = agent.predict_action(observation)
            
            # Current state
            current_pos = env.current_pos
            current_fuel = env.current_fuel
            current_money = env.current_money
            
            # Take action
            next_observation, reward, terminated, truncated, info = env.step(action)
            
            # Update metrics
            total_reward += reward
            total_steps += 1
            
            # Step info
            step_info = {
                "step": total_steps,
                "action": action,
                "reward": reward,
                "position": current_pos,
                "new_position": env.current_pos,
                "fuel_before": current_fuel,
                "fuel_after": env.current_fuel,
                "money_before": current_money,
                "money_after": env.current_money,
                "info": info
            }
            step_by_step_info.append(step_info)
            
            if "toll_paid" in info:
                money_spent += info["toll_paid"]
                toll_visits += 1
            
            if "refueled" in info and info["refueled"]:
                refuels += 1
                money_spent += info["refuel_cost"]
            
            if "at_gas_station" in info and info["at_gas_station"]:
                gas_station_visits += 1
            
            # Update fuel consumed (only for move actions)
            if action <= 3:  # Move actions
                fuel_consumed += env.fuel_per_move
            
            # Update path
            if env.current_pos not in path:
                path.append(env.current_pos)
            
            # Update observation
            observation = next_observation
            
            # Render if needed
            if render:
                env.render()
        
        # Final metrics
        success = info.get("termination_reason") == "den_dich"
        remaining_fuel = float(observation["fuel"][0]) if "fuel" in observation else 0
        remaining_money = float(observation["money"][0]) if "money" in observation else 0
        
        metrics = {
            "success": success,
            "total_reward": total_reward,
            "path_length": len(path) - 1,  # Exclude starting position
            "total_steps": total_steps,
            "fuel_consumed": fuel_consumed,
            "money_spent": money_spent,
            "toll_visits": toll_visits,
            "gas_station_visits": gas_station_visits,
            "refuels": refuels,
            "remaining_fuel": remaining_fuel,
            "remaining_money": remaining_money,
            "termination_reason": info.get("termination_reason", "unknown")
        }
        
        return path, metrics, step_by_step_info
    
    def compare_with_algorithm(self, model_path, algorithm_name, map_obj, initial_fuel=5.0,
                              initial_money=1000.0, fuel_per_move=0.3):
        """
        Compare RL agent with a traditional algorithm on a single map.
        
        Args:
            model_path: Path to the trained RL model
            algorithm_name: Name of the algorithm to compare with
            map_obj: Map object
            initial_fuel: Initial fuel amount
            initial_money: Initial money amount
            fuel_per_move: Fuel consumed per move
            
        Returns:
            tuple: (rl_path, rl_metrics, algorithm_path, algorithm_metrics)
        """
        # Evaluate RL agent
        rl_path, rl_metrics, _ = self.evaluate_single_episode(
            model_path=model_path,
            map_obj=map_obj,
            initial_fuel=initial_fuel,
            initial_money=initial_money,
            fuel_per_move=fuel_per_move
        )
        
        # Evaluate traditional algorithm
        if algorithm_name not in self.algorithms:
            raise ValueError(f"Algorithm {algorithm_name} not supported")
        
        # Get algorithm class
        AlgorithmClass = self.algorithms[algorithm_name]
        
        # Create algorithm instance
        algorithm = AlgorithmClass(map_obj)
        
        # Run algorithm
        if algorithm_name in ["Genetic Algorithm", "Simulated Annealing", "Local Beam Search"]:
            # Stochastic algorithms
            algorithm_path, algorithm_metrics = algorithm.search(
                start_pos=map_obj.start_pos,
                goal_pos=map_obj.end_pos,
                initial_fuel=initial_fuel,
                initial_money=initial_money,
                fuel_consumption=fuel_per_move
            )
        else:
            # Deterministic algorithms
            algorithm_path, algorithm_metrics = algorithm.search(
                start_pos=map_obj.start_pos,
                goal_pos=map_obj.end_pos,
                initial_fuel=initial_fuel,
                initial_money=initial_money,
                fuel_per_move=fuel_per_move
            )
        
        return rl_path, rl_metrics, algorithm_path, algorithm_metrics


# Example usage
if __name__ == "__main__":
    # Define directories
    test_maps_dir = "./maps/test"
    results_dir = "./evaluation_results"
    
    # Make sure directories exist
    os.makedirs(test_maps_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)
    
    # If no maps exist, create some test maps
    if len(os.listdir(test_maps_dir)) == 0:
        print("No test maps found. Creating some...")
        for i in range(5):
            map_obj = Map.generate_random(size=10, toll_ratio=0.05, gas_ratio=0.05, brick_ratio=0.2)
            map_obj.save(os.path.join(test_maps_dir, f"test_map_{i}.json"))
    
    # Placeholder for demonstration
    # In practice, you would specify a real model path
    model_path = "./saved_models/best_dqn_agent"
    
    # Create evaluator
    evaluator = RLEvaluator(maps_dir=test_maps_dir, results_dir=results_dir)
    
    # Example usage (commented out to avoid accidental execution)
    # To evaluate just the RL agent:
    # rl_results = evaluator.evaluate_rl_agent(model_path=model_path)
    
    # To compare with all algorithms:
    # results = evaluator.compare_algorithms(rl_model_path=model_path)
    
    # To compare with specific algorithms:
    # results = evaluator.compare_algorithms(
    #     rl_model_path=model_path,
    #     algorithms_to_compare=["A*", "Greedy"]
    # )
    
    print("RL evaluation module created. Use this module to evaluate RL agents and compare with other algorithms.") 