import math

def distance(node1, node2, nodes):
    """Calculate Euclidean distance between two nodes."""
    x1, y1 = nodes[node1]
    x2, y2 = nodes[node2]
    return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

def calculate_energy_to_reach(from_node, to_node, cost_matrix):
    """Get energy cost between nodes, returning infinity if not reachable."""
    return cost_matrix.get((from_node, to_node), float('inf'))

def find_nearest_charging_station(current_node, charging_stations, cost_matrix, battery):
    """Finds the closest reachable charging station from the current node."""
    best_station = None
    min_cost = float('inf')

    for cs in charging_stations:
        if (current_node, cs) in cost_matrix:
            travel_cost = cost_matrix[(current_node, cs)]
            if travel_cost <= battery and travel_cost < min_cost:
                min_cost = travel_cost
                best_station = cs

    if best_station:
        print(f"[DEBUG] Nearest CS to {current_node}: {best_station} (Cost: {min_cost})")
    else:
        print(f"[DEBUG] No CS reachable from {current_node}, Battery Left: {battery}")

    return best_station

def find_charging_chain(from_node, to_node, charging_stations, cost_matrix, E_max, battery_left, max_depth=2):
    """
    Attempts to find a short chain of charging stations between from_node and to_node.
    Allows one or two intermediate charging stations (depth-limited DFS).
    """
    for cs1 in charging_stations:
        if cost_matrix.get((from_node, cs1), float('inf')) <= battery_left:
            # Try direct CS1 → to_node
            if cost_matrix.get((cs1, to_node), float('inf')) <= E_max:
                return [cs1]

            # Try CS1 → CS2 → to_node
            for cs2 in charging_stations:
                if cs2 == cs1:
                    continue
                if (cost_matrix.get((cs1, cs2), float('inf')) <= E_max and
                    cost_matrix.get((cs2, to_node), float('inf')) <= E_max):
                    return [cs1, cs2]
    return None  # No valid chain found

def make_routes_battery_feasible(routes, cost_matrix, E_max, charging_stations, depot):
    """
    Converts a list of routes into battery-feasible routes by inserting charging stations
    or splitting the route when necessary.
    """
    battery_feasible_routes = []

    for idx, route in enumerate(routes):
        print(f"\n🔋 Repairing Route {idx + 1}: {route}")
        current_subroute = [depot]
        battery = E_max

        i = 1
        while i < len(route):
            from_node = current_subroute[-1]
            to_node = route[i]
            energy_cost = cost_matrix.get((from_node, to_node), float('inf'))

            if energy_cost <= battery:
                current_subroute.append(to_node)
                battery -= energy_cost

                # Recharge if at depot or charging station
                if to_node == depot or to_node in charging_stations:
                    battery = E_max
            else:
                print(f"⚠️ Battery insufficient for {from_node} → {to_node} | Remaining: {battery}, Cost: {energy_cost}")

                # Try inserting CS chain
                chain = find_charging_chain(from_node, to_node, charging_stations, cost_matrix, E_max, battery)

                if chain:
                    for cs in chain:
                        if current_subroute[-1] == cs:
                            print(f"⛔ Skipping duplicate CS: {cs}")
                            continue
                        print(f"🔌 Inserting intermediate CS: {cs}")
                        current_subroute.append(cs)
                        battery = E_max - cost_matrix.get((from_node, cs), float('inf'))
                        from_node = cs

                    current_subroute.append(to_node)
                    battery -= cost_matrix[(from_node, to_node)]
                else:
                    # Split and restart from depot
                    current_subroute.append(depot)
                    battery_feasible_routes.append(current_subroute)
                    current_subroute = [depot, to_node]
                    battery = E_max - cost_matrix.get((depot, to_node), float('inf'))

            i += 1

        # Finalize route if not already closed
        if current_subroute[-1] != depot:
            current_subroute.append(depot)
        if current_subroute != [depot, depot]:
            battery_feasible_routes.append(current_subroute)

        print(f"✅ Final Repaired Subroutes so far: {battery_feasible_routes}")

    return battery_feasible_routes
