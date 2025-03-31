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

