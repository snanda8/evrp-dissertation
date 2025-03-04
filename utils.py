import math

def distance(node1, node2, nodes):
    """Calculate Euclidean distance between two nodes."""
    x1, y1 = nodes[node1]
    x2, y2 = nodes[node2]
    return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

def calculate_energy_to_reach(from_node, to_node, cost_matrix):
    """Get energy cost between nodes, returning infinity if not reachable."""
    return cost_matrix.get((from_node, to_node), float('inf'))

def find_nearest_charging_station(current_node, charging_stations, nodes):
    """Find the nearest charging station to the current node using Euclidean distance."""
    if not charging_stations:
        return None
    nearest = None
    min_dist = float('inf')
    for cs in charging_stations:
        dist = distance(current_node, cs, nodes)
        if dist < min_dist:
            min_dist = dist
            nearest = cs
    return nearest
