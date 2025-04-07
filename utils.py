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
    """Finds the closest charging station that is reachable from the current node given remaining battery."""
    best_station = None
    min_cost = float('inf')

    for cs in charging_stations:
        if (current_node, cs) in cost_matrix:
            travel_cost = cost_matrix[(current_node, cs)]
            #  Ensure the station is within battery range
            if travel_cost <= battery and travel_cost < min_cost:
                min_cost = travel_cost
                best_station = cs

    if best_station:
        print(f" [DEBUG] Nearest Charging Station to {current_node}: {best_station} (Cost: {min_cost})")
    else:
        print(f" [DEBUG] No reachable charging station from {current_node}, Battery Remaining: {battery}")

    return best_station

def make_routes_battery_feasible(routes, cost_matrix, E_max, charging_stations, depot):
    """
    Repairs each route by inserting charging stations to maintain battery feasibility.
    If a segment is not repairable, it skips that segment and continues with remaining nodes.
    """
    feasible_routes = []

    for route_idx, route in enumerate(routes):
        print(f"\n🔋 Repairing Route {route_idx + 1}: {route}")
        battery = E_max
        repaired = [route[0]]  # start at depot
        i = 1

        while i < len(route):
            from_node = repaired[-1]
            to_node = route[i]
            cost = cost_matrix.get((from_node, to_node), float('inf'))

            if cost <= battery:
                repaired.append(to_node)
                battery -= cost
                print(f"✅ From {from_node} → {to_node} | Cost: {cost}, Battery Left: {battery}")
                if to_node == depot or to_node in charging_stations:
                    battery = E_max  # recharge
                    print(f"🔌 Recharged at {to_node}, Battery Reset to {E_max}")
                i += 1
            else:
                # Try inserting a CS between from_node and to_node
                insertable_cs = [
                    cs for cs in charging_stations
                    if cost_matrix.get((from_node, cs), float('inf')) <= battery and
                       cost_matrix.get((cs, to_node), float('inf')) <= E_max
                ]
                if insertable_cs:
                    best_cs = min(insertable_cs, key=lambda cs: cost_matrix[(from_node, cs)] + cost_matrix[(cs, to_node)])
                    print(f"⚡ Inserting CS {best_cs} between {from_node} → {to_node}")
                    repaired.append(best_cs)
                    battery = E_max - cost_matrix[(best_cs, to_node)]  # simulate move to to_node next
                    repaired.append(to_node)
                    if to_node == depot or to_node in charging_stations:
                        battery = E_max
                    i += 1
                else:
                    print(f"❌ [ERROR] Cannot reach {to_node} from {from_node}, skipping.")
                    i += 1  # Skip to_node, continue with next

        print(f"✅ Final Repaired Route: {repaired}")
        feasible_routes.append(repaired)

    return feasible_routes



