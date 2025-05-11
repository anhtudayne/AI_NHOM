from .state_and_actions import State

PROB_OK = 0.9
PROB_BROKEN = 0.1

def get_action_results(current_state, action):
    """
    Simulates the non-deterministic results of an action.
    Location/destination in State/Action are now (x,y) tuples.
    Returns a list of tuples, where each tuple is (resulting_state, probability, description).
    """
    results = []
    if action.type == "DRIVE":
        target_location = action.destination # This is now an (x,y) tuple

        # Outcome 1: Arrives OK
        state_ok = State(location=target_location, is_broken_down=False)
        results.append((state_ok, PROB_OK, f"Arrived at {target_location} OK."))

        # Outcome 2: Arrives but is Broken
        state_broken_at_dest = State(location=target_location, is_broken_down=True)
        results.append((state_broken_at_dest, PROB_BROKEN, f"Arrived at {target_location} but BROKEN."))

    elif action.type == "REPAIR":
        # Assumption: RepairAction is only possible if currently broken.
        # Repair is always successful and happens at the current location.
        if current_state.is_broken_down:
            state_repaired = State(location=current_state.location, is_broken_down=False)
            results.append((state_repaired, 1.0, f"Repaired at {current_state.location}."))
        else:
            # Should not happen if action generation is correct, but as a safeguard:
            results.append((current_state.clone(), 1.0, "Repair attempted on non-broken vehicle."))
            
    return results 