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

def make_routes_battery_feasible(routes, cost_matrix, E_max, charging_stations, depot, safety_margin=10):
    """
    Enhanced version that repairs infeasible EVRP routes by inserting charging stations when needed.
    Uses a greedy-nearest-CS strategy with a safety margin, and ensures customers are not skipped.
    """
    feasible_routes = []

    for route in routes:
        new_route = [route[0]]  # Start with depot
        remaining_battery = E_max
        infeasible = False
        i = 1  # start from second node

        while i < len(route):
            prev_node = new_route[-1]
            curr_node = route[i]

            if prev_node == curr_node:
                i += 1
                continue  # Skip duplicate

            cost = cost_matrix.get((prev_node, curr_node), float('inf'))
            if cost == float('inf'):
                print(f"[WARNING] No path from {prev_node} to {curr_node}")
                infeasible = True
                break

            # Check if we can reach next node safely
            if remaining_battery >= cost + safety_margin:
                new_route.append(curr_node)


                remaining_battery -= cost
                i += 1  # Only advance if customer is added
                continue

            # Try to find a valid CS
            cs_candidates = []
            for cs in charging_stations:
                to_cs = cost_matrix.get((prev_node, cs), float('inf'))
                cs_to_next = cost_matrix.get((cs, curr_node), float('inf'))

                if to_cs <= remaining_battery and cs_to_next <= E_max:
                    total_cost = to_cs + cs_to_next
                    cs_candidates.append((total_cost, cs, cs_to_next))

            if cs_candidates:
                cs_candidates.sort()
                total_cost, chosen_cs, cs_to_next = cs_candidates[0]
                print(f"[INFO] Inserting CS {chosen_cs} between {prev_node} and {curr_node} (detour cost {total_cost})")
                new_route.append(chosen_cs)
                remaining_battery = E_max - cs_to_next
                new_route.append(curr_node)
                i += 1
                # i stays the same to re-check curr_node
            else:
                print(f"[ERROR] No feasible CS found between {prev_node} and {curr_node}")
                infeasible = True
                break

        if infeasible:
            print(f"[SKIPPED] Route marked infeasible and skipped: {route}")
            continue

        # === Final leg: ensure return to depot ===
        if new_route[-1] != depot:
            cost_to_depot = cost_matrix.get((new_route[-1], depot), float('inf'))
            if remaining_battery < cost_to_depot:
                cs_candidates = []
                for cs in charging_stations:
                    to_cs = cost_matrix.get((new_route[-1], cs), float('inf'))
                    cs_to_depot = cost_matrix.get((cs, depot), float('inf'))

                    if to_cs <= remaining_battery and cs_to_depot <= E_max:
                        total_cost = to_cs + cs_to_depot
                        cs_candidates.append((total_cost, cs, cs_to_depot))

                if cs_candidates:
                    cs_candidates.sort()
                    _, chosen_cs, cs_to_depot = cs_candidates[0]
                    print(f"[INFO] Inserting CS {chosen_cs} before returning to depot from {new_route[-1]}")
                    new_route.append(chosen_cs)
                    remaining_battery = E_max - cs_to_depot
                else:
                    print(f"[ERROR] Could not reach depot from {new_route[-1]}")
                    infeasible = True
                    continue

            new_route.append(depot)

        feasible_routes.append(new_route)

    return feasible_routes

