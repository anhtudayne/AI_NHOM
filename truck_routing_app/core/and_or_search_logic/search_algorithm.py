FAILURE = "failure"
NO_PLAN = "no_plan" # Represents an empty plan, e.g., when already at goal

# Dictionary to store results of solved states for or_search
solved_or_states = {}

def or_search(state, problem, path):
    """
    Searches for a plan from an OR node (a state where the agent chooses an action).
    Returns a plan (e.g., [action, sub_plan]) or FAILURE.
    path: A list of states on the current path from the root to 'state' to detect cycles.
    """
    if problem.is_goal(state):
        return NO_PLAN # Successfully reached goal

    if state in path:
        return FAILURE # Cycle detected

    # Check memoized results for or_search
    if state in solved_or_states:
        # Important: Need to handle cycles if a memoized state would lead to a cycle with the current path.
        # For simplicity here, we assume that if a state was solved, it was solved optimally without creating a cycle 
        # from the point it was first encountered. This might need more sophisticated cycle handling if a memoized plan
        # itself could lead back into the current path from a *different* entry point.
        # However, standard AND-OR search with cycle detection on path should mostly prevent this.
        # If solved_or_states[state] itself can be FAILURE due to a cycle found when it was originally solved, that's fine.
        return solved_or_states[state]

    # Try each action
    for action in problem.get_actions(state):
        # For each action, we need to ensure all its outcomes can be handled.
        # The outcomes are obtained from problem.get_results, which now returns a list of (state, prob, desc)
        # We only need the states for and_search's first argument.
        outcomes_with_details = problem.get_results(state, action)
        # Filter out cases where outcomes_with_details might be empty (e.g. repair on non-broken vehicle)
        if not outcomes_with_details:
            continue
        
        outcome_states = [res[0] for res in outcomes_with_details] # Extract just the states
        
        # Pass the full outcomes_with_details to and_search so it can build the conditional plan
        plan_for_action_outcomes = and_search(outcomes_with_details, problem, [state] + path)

        if plan_for_action_outcomes != FAILURE:
            # Found a successful plan for this action's outcomes
            # The plan from OR node is: [this_action, plan_for_its_outcomes]
            # The 'plan_for_action_outcomes' already describes the conditional handling of AND branches
            action_plan_result = {"type": "OR_PLAN_STEP", "action": action, "sub_plan": plan_for_action_outcomes}
            solved_or_states[state] = action_plan_result # Store successful plan
            return action_plan_result
            
    solved_or_states[state] = FAILURE # Store failure if no action leads to solution
    return FAILURE

def and_search(outcomes_with_details, problem, path):
    """
    Searches for a plan from an AND node (a set of outcomes for a chosen action).
    'outcomes_with_details' is a list of (state, probability, description) tuples.
    Returns a conditional plan structure or FAILURE.
    """
    # This function needs to build a conditional plan based on the outcomes.
    # For each outcome, we recursively call or_search.
    # If all recursive calls succeed, we construct a plan.
    
    # The plan structure for an AND node can be a dictionary mapping outcome descriptions
    # to their respective sub-plans found by or_search.
    conditional_sub_plans = {}
    all_outcomes_handled = True

    for outcome_state, _, outcome_description in outcomes_with_details:
        sub_plan = or_search(outcome_state, problem, path) # Path is already extended by or_search caller
        if sub_plan == FAILURE:
            all_outcomes_handled = False
            break # If one outcome cannot be handled, this AND path fails
        conditional_sub_plans[outcome_description] = sub_plan
    
    if all_outcomes_handled:
        # If there's only one outcome (e.g. after RepairAction), 
        # the plan is just the sub_plan for that outcome.
        if len(outcomes_with_details) == 1:
            # The key of the single item in conditional_sub_plans
            single_outcome_desc = list(conditional_sub_plans.keys())[0]
            return {"type": "AND_PLAN_SINGLE_OUTCOME", "description": single_outcome_desc, "plan": conditional_sub_plans[single_outcome_desc]}

        return {"type": "AND_PLAN_CONDITIONAL", "contingencies": conditional_sub_plans}
    else:
        return FAILURE

def solve_and_or_problem(problem):
    """Main entry point to solve the AND-OR problem."""
    global solved_or_states
    solved_or_states = {} # Clear memoization table for each new problem
    initial_state = problem.get_initial_state()
    solution_plan = or_search(initial_state, problem, [])
    return solution_plan 