from core.and_or_search_logic.problem_definition import AndOrProblem
from core.and_or_search_logic.search_algorithm import solve_and_or_problem, FAILURE, NO_PLAN

# --- Helper function to print the plan ---
def format_plan(plan, indent=""):
    if plan == FAILURE:
        return f"{indent}Failed to find a plan."
    if plan == NO_PLAN:
        return f"{indent}Goal reached (no further actions needed)."

    plan_str = ""
    if plan.get("type") == "OR_PLAN_STEP":
        action = plan['action']
        sub_plan = plan['sub_plan']
        plan_str += f"{indent}IF current state allows, DO: {action}\n"
        plan_str += format_plan(sub_plan, indent + "  THEN ")
    
    elif plan.get("type") == "AND_PLAN_CONDITIONAL":
        plan_str += f"{indent}EXPECT one of the following outcomes:\n"
        for desc, contingent_plan in plan['contingencies'].items():
            plan_str += f"{indent}  - IF ({desc}):\n"
            plan_str += format_plan(contingent_plan, indent + "    - THEN ")
            plan_str += "\n"
            
    elif plan.get("type") == "AND_PLAN_SINGLE_OUTCOME":
        # This case handles the result of a deterministic action like Repair
        desc = plan['description']
        actual_plan = plan['plan']
        plan_str += f"{indent}EXPECTED OUTCOME ({desc}):\n"
        plan_str += format_plan(actual_plan, indent + "  - THEN ")

    elif isinstance(plan, str): # Should be caught by FAILURE/NO_PLAN but as a fallback
        plan_str += f"{indent}{plan}\n"
        
    return plan_str.strip("\n ")

# --- Main execution ---
def main():
    start_node = "S"
    destination_node = "D"

    print(f"Starting AND-OR search from '{start_node}' to '{destination_node}'...")

    problem = AndOrProblem(start_location_name=start_node, 
                           final_destination_name=destination_node)

    solution = solve_and_or_problem(problem)

    print("\n--- Search Complete ---")
    if solution == FAILURE:
        print("Result: No guaranteed plan found.")
    else:
        print("Result: Guaranteed plan found!")
        print("\n--- Plan Details ---")
        print(format_plan(solution))

if __name__ == "__main__":
    main() 