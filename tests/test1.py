import math
import random
import xml.etree.ElementTree as ET
import random
import math
import numpy as np
from sklearn.cluster import KMeans


def distance(node1, node2, nodes):
    """Calculate Euclidean distance between two nodes."""
    x1, y1 = nodes[node1]
    x2, y2 = nodes[node2]
    return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)


def calculate_energy_to_reach(from_node, to_node, cost_matrix):
    """Get energy cost between nodes, returning infinity if not reachable."""
    return cost_matrix.get((from_node, to_node), float('inf'))


def find_nearest_charging_station(current_node, charging_stations, nodes):
    """Find the nearest charging station to the current node."""
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


"""Clustering and Nearest Neighbor Route Functions"""


def cluster_customers(customers, nodes, num_clusters):
    """Cluster customers using K-means."""
    coords = np.array([nodes[c] for c in customers])
    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    cluster_labels = kmeans.fit_predict(coords)
    clusters = [[] for _ in range(num_clusters)]
    for i, customer in enumerate(customers):
        cluster_id = cluster_labels[i]
        clusters[cluster_id].append(customer)
    return clusters


def nearest_neighbor_route(start_node, customer_set, nodes, cost_matrix, E_max,
                           charging_stations, requests, vehicle_capacity):
    """
    Construct a route using a nearest neighbor heuristic with battery constraints.
    """
    current_node = start_node
    route = [start_node]
    remaining_customers = set(customer_set)
    remaining_battery = E_max
    remaining_capacity = vehicle_capacity

    while remaining_customers:
        best_customer = None
        best_distance = float('inf')
        for customer in remaining_customers:
            if customer in requests and requests[customer]['quantity'] > remaining_capacity:
                continue
            energy_needed = calculate_energy_to_reach(current_node, customer, cost_matrix)
            if energy_needed == float('inf') or energy_needed > remaining_battery:
                continue
            dist = distance(current_node, customer, nodes)
            if dist < best_distance:
                best_distance = dist
                best_customer = customer
        if best_customer is None:
            # No feasible customer found; try inserting a charging station.
            nearest_cs = find_nearest_charging_station(current_node, charging_stations, nodes)
            if nearest_cs is None:
                break
            energy_to_cs = calculate_energy_to_reach(current_node, nearest_cs, cost_matrix)
            if energy_to_cs == float('inf') or energy_to_cs > remaining_battery:
                break
            route.append(nearest_cs)
            current_node = nearest_cs
            remaining_battery = E_max  # Recharge
        else:
            route.append(best_customer)
            remaining_customers.remove(best_customer)
            remaining_battery -= calculate_energy_to_reach(current_node, best_customer, cost_matrix)
            if best_customer in requests:
                remaining_capacity -= requests[best_customer]['quantity']
            current_node = best_customer

    # Ensure return to depot is feasible
    energy_to_depot = calculate_energy_to_reach(current_node, start_node, cost_matrix)
    if energy_to_depot > remaining_battery:
        nearest_cs = find_nearest_charging_station(current_node, charging_stations, nodes)
        if nearest_cs is not None:
            energy_to_cs = calculate_energy_to_reach(current_node, nearest_cs, cost_matrix)
            if energy_to_cs <= remaining_battery:
                route.append(nearest_cs)
                current_node = nearest_cs
                remaining_battery = E_max
    route.append(start_node)
    return route

"""Route repair fucntions"""


def insert_charging_stations_strategically(route, cost_matrix, E_max, charging_stations, depot):
    """Insert charging stations at strategic points to ensure battery feasibility."""
    if not route or len(route) <= 2:
        return route

    result_route = [route[0]]  # Start with depot
    remaining_battery = E_max

    for i in range(1, len(route)):
        next_node = route[i]
        current_node = result_route[-1]
        energy_needed = calculate_energy_to_reach(current_node, next_node, cost_matrix)
        if energy_needed <= remaining_battery:
            result_route.append(next_node)
            remaining_battery -= energy_needed
            if next_node in charging_stations or next_node == depot:
                remaining_battery = E_max
        else:
            # Try to insert a charging station
            candidates = []
            for cs in charging_stations:
                if cs == current_node:
                    continue
                energy_to_cs = calculate_energy_to_reach(current_node, cs, cost_matrix)
                energy_from_cs = calculate_energy_to_reach(cs, next_node, cost_matrix)
                if energy_to_cs <= remaining_battery and energy_from_cs <= E_max:
                    detour_cost = energy_to_cs + energy_from_cs - energy_needed
                    candidates.append((cs, detour_cost))
            if candidates:
                candidates.sort(key=lambda x: x[1])
                best_cs = candidates[0][0]
                result_route.append(best_cs)
                remaining_battery = E_max - calculate_energy_to_reach(best_cs, next_node, cost_matrix)
                result_route.append(next_node)
            else:
                return route  # Optimization failed; return original route
    return result_route


def remove_unnecessary_charging_stations(route, cost_matrix, E_max, charging_stations, depot):
    """Remove charging stations that are not necessary for battery feasibility."""
    if not route or len(route) <= 2:
        return route

    cs_indices = [i for i, node in enumerate(route) if node in charging_stations and node != depot]
    for idx in sorted(cs_indices, reverse=True):
        test_route = route.copy()
        test_route.pop(idx)
        battery = E_max
        feasible = True
        for i in range(len(test_route) - 1):
            energy = calculate_energy_to_reach(test_route[i], test_route[i + 1], cost_matrix)
            if energy == float('inf'):
                feasible = False
                break
            battery -= energy
            if battery < 0:
                feasible = False
                break
            if test_route[i + 1] in charging_stations or test_route[i + 1] == depot:
                battery = E_max
        if feasible:
            route = test_route
    return route


"""Heurstic population initlisation"""


def heuristic_initial_solution(nodes, cost_matrix, travel_time_matrix, depot,
                               E_max, recharge_amount, charging_stations,
                               vehicle_capacity, max_travel_time, requests, num_vehicles):
    # Start by clustering customers if more than one vehicle is used.
    if num_vehicles > 1:
        clusters = cluster_customers(list(requests.keys()), nodes, num_vehicles)
    else:
        clusters = [list(requests.keys())]

    solution = []
    for cluster in clusters:
        route = nearest_neighbor_route(depot, cluster, nodes, cost_matrix, E_max, charging_stations, requests,
                                       vehicle_capacity)
        route = insert_charging_stations_strategically(route, cost_matrix, E_max, charging_stations, depot)
        route = remove_unnecessary_charging_stations(route, cost_matrix, E_max, charging_stations, depot)
        solution.append(route)

    # If there are any unvisited customers, assign them to the route with lowest load.
    visited = set()
    for route in solution:
        visited.update(set(route))
    unvisited = set(requests.keys()) - visited
    if unvisited:
        for cust in list(unvisited):
            best_route = min(solution, key=lambda r: sum(
                requests.get(n, {'quantity': 0})['quantity'] for n in r if n in requests))
            best_route.insert(-1, cust)
            unvisited.remove(cust)
    return solution


def heuristic_population_initialization(population_size, nodes, cost_matrix, travel_time_matrix, depot,
                                        E_max, recharge_amount, charging_stations,
                                        vehicle_capacity, max_travel_time, requests, num_vehicles):
    population = []
    for _ in range(population_size):
        sol = heuristic_initial_solution(nodes, cost_matrix, travel_time_matrix, depot,
                                         E_max, recharge_amount, charging_stations,
                                         vehicle_capacity, max_travel_time, requests, num_vehicles)
        population.append(sol)
    # Optionally, add a few random solutions for diversity.
    num_random = max(1, population_size // 5)
    for _ in range(num_random):
        customers_list = list(requests.keys())
        random.shuffle(customers_list)
        random_routes = []
        for i in range(num_vehicles):
            route = [depot]
            assigned_customers = customers_list[i::num_vehicles]
            route.extend(assigned_customers)
            route.append(depot)
            random_routes.append(route)
        population.append(random_routes)
    return population


# --- 1. XML Parsing Function ---
def parse_instance(file_path):
    tree = ET.parse(file_path)
    root = tree.getroot()

    nodes = {}
    charging_stations = set()
    customers = set()
    depot = None

    # Parse nodes and classify them by type:
    network = root.find('network')
    nodes_xml = network.find('nodes')
    for node in nodes_xml:
        node_id = int(node.attrib['id'])
        cx = float(node.find('cx').text)
        cy = float(node.find('cy').text)
        nodes[node_id] = (cx, cy)
        node_type = node.attrib['type']
        if node_type == "0":
            depot = node_id
        elif node_type == "1":
            customers.add(node_id)
        elif node_type == "2":
            charging_stations.add(node_id)

    # Build cost and travel time matrices.
    cost_matrix = {}
    travel_time_matrix = {}
    links_xml = network.find('links')
    for link in links_xml:
        head = int(link.attrib['head'])
        tail = int(link.attrib['tail'])
        energy_consumption = float(link.find('energy_consumption').text)
        travel_time = float(link.find('travel_time').text)
        cost_matrix[(head, tail)] = energy_consumption
        cost_matrix[(tail, head)] = energy_consumption  # symmetric
        travel_time_matrix[(head, tail)] = travel_time
        travel_time_matrix[(tail, head)] = travel_time

    # Get fleet parameters.
    fleet = root.find('fleet')
    vehicle_profile = fleet.find('vehicle_profile')
    battery_capacity = float(vehicle_profile.find('custom').find('battery_capacity').text)
    num_vehicles = int(vehicle_profile.attrib['number'])
    vehicle_capacity = float(vehicle_profile.find('capacity').text)
    max_travel_time = float(vehicle_profile.find('max_travel_time').text)

    # Parse requests (for demand/service time).
    requests = {}
    requests_xml = root.find('requests')
    for req in requests_xml:
        req_node = int(req.attrib['node'])
        quantity = float(req.find('quantity').text)
        service_time = float(req.find('service_time').text)
        requests[req_node] = {'quantity': quantity, 'service_time': service_time}

    return (nodes, charging_stations, depot, customers, cost_matrix, travel_time_matrix,
            battery_capacity, num_vehicles, vehicle_capacity, max_travel_time, requests)


# --- 2. Global Setup: Parse Instance and Define Global Variables ---
instance_file = "C101-10.xml"  # Change this to the correct path if necessary
(nodes, charging_stations, depot, customers, cost_matrix, travel_time_matrix,
 battery_capacity, num_vehicles_instance, vehicle_capacity, max_travel_time, requests) = parse_instance(instance_file)

# Define global variables based on the instance
DEPOT = depot
num_vehicles = num_vehicles_instance
E_max = battery_capacity
recharge_amount = 30  # You may adjust this value if needed

penalty_weights = {
    'missing_customers': 1e6,
    'battery_depletion': 1e5,
    'unreachable_node': 1e5,
    'unnecessary_recharges': 1000,
    'low_battery': 5000,
    'capacity_overload': 1e5,
    'max_travel_time_exceeded': 1e5,
    'invalid_route': 1e5,
}


# --- 3. Function Definitions ---

def generate_random_routes(nodes, num_vehicles, depot):
    """
    Generates routes for multiple vehicles.
    """
    customers_list = list(set(nodes.keys()) - {depot})
    random.shuffle(customers_list)
    routes = []
    for i in range(num_vehicles):
        route = [depot]
        assigned_customers = customers_list[i::num_vehicles]
        route += assigned_customers
        route.append(depot)
        routes.append(route)
    return routes


def update_battery(route, cost_matrix, E_max, charging_stations, recharge_amount, depot):
    battery = E_max
    recharged = 0
    unnecessary_recharges = 0
    low_battery_penalty = 0
    for i in range(len(route) - 1):
        from_node = route[i]
        to_node = route[i + 1]
        if isinstance(from_node, list):
            print(f"⚠️ Warning: `from_node` is a list {from_node}, taking first element.")
            from_node = from_node[0]
        if isinstance(to_node, list):
            print(f"⚠️ Warning: `to_node` is a list {to_node}, taking first element.")
            to_node = to_node[0]
        if from_node == depot:
            battery = E_max
        energy_cost = cost_matrix.get((from_node, to_node), float('inf'))
        if energy_cost == float('inf'):
            print(f"Unreachable node pair: ({from_node}, {to_node}). Adding penalty.")
            return battery, False, recharged, unnecessary_recharges, low_battery_penalty
        battery -= energy_cost
        print(f"From {from_node} to {to_node}: Cost={energy_cost}, Battery={battery}")
        if battery < 0.25 * E_max and to_node not in charging_stations:
            print("Low battery warning! No charging station ahead.")
            low_battery_penalty += 5000
        if to_node in charging_stations:
            battery = min(battery + recharge_amount, E_max)
            recharged += 1
        if battery < 0:
            print(f"Battery depleted between {from_node} and {to_node}.")
            return battery, False, recharged, unnecessary_recharges, low_battery_penalty
        if to_node in charging_stations:
            if battery > 0.75 * E_max:
                unnecessary_recharges += 1
            battery = min(battery + recharge_amount, E_max)
            recharged += 1
    return battery, True, recharged, unnecessary_recharges, low_battery_penalty


def fitness_function(solution, cost_matrix, travel_time_matrix, E_max, charging_stations,
                     recharge_amount, penalty_weights, depot, nodes, vehicle_capacity,
                     max_travel_time, requests):
    total_distance = 0
    total_penalty = 0
    visited_customers = set()
    print("\n=== Multi-Vehicle Fitness Debug ===")
    for idx, route in enumerate(solution):
        print(f"\nProcessing Vehicle {idx + 1} Route: {route}")
        if not route or route[0] != depot or route[-1] != depot:
            print("  Route must start and end at depot.")
            total_penalty += penalty_weights.get('invalid_route', 1e5)
        route_distance = 0
        route_travel_time = 0
        for i in range(len(route) - 1):
            from_node = route[i]
            to_node = route[i + 1]
            distance = cost_matrix.get((from_node, to_node), float('inf'))
            travel_time = travel_time_matrix.get((from_node, to_node), float('inf'))
            route_distance += distance
            route_travel_time += travel_time
        print(f"  Route Distance: {route_distance}")
        print(f"  Route Travel Time: {route_travel_time}")
        total_distance += route_distance

        battery, valid, recharged, unnecessary_recharges, low_battery_penalty = update_battery(
            route, cost_matrix, E_max, charging_stations, recharge_amount, depot
        )
        print(f"  Battery after route: {battery}, Valid: {valid}")
        if not valid:
            total_penalty += penalty_weights['battery_depletion']
        if unnecessary_recharges > 0:
            penalty = penalty_weights['unnecessary_recharges'] * unnecessary_recharges
            total_penalty += penalty
            print(f"  Unnecessary recharge penalty: {penalty}")
        if low_battery_penalty > 0:
            total_penalty += low_battery_penalty
            print(f"  Low battery penalty: {low_battery_penalty}")

        route_demand = 0
        for node in route:
            if node in requests and node not in charging_stations and node != depot:
                route_demand += requests[node]['quantity']
        print(f"  Route Demand: {route_demand}")
        if route_demand > vehicle_capacity:
            overload = route_demand - vehicle_capacity
            penalty = penalty_weights.get('capacity_overload', 1e5) * overload
            print(f"  Capacity overload: {overload}, Penalty: {penalty}")
            total_penalty += penalty

        if route_travel_time > max_travel_time:
            excess_time = route_travel_time - max_travel_time
            penalty = penalty_weights.get('max_travel_time_exceeded', 1e5) * excess_time
            print(f"  Travel time exceeded by {excess_time}, Penalty: {penalty}")
            total_penalty += penalty

        visited_customers.update(set(route) - {depot})
    expected_customers = set(nodes.keys()) - {depot}
    if visited_customers != expected_customers:
        missing = expected_customers - visited_customers
        missing_penalty = penalty_weights['missing_customers'] * len(missing)
        total_penalty += missing_penalty
        print(f"  Missing customers {missing} -> penalty: {missing_penalty}")
    total_fitness = total_distance + total_penalty
    print(f"\nTotal Distance: {total_distance}, Total Penalty: {total_penalty}, Overall Fitness: {total_fitness}")
    return total_fitness, (total_penalty == 0)


def order_crossover_evrp(parent1, parent2, cost_matrix, E_max, charging_stations, recharge_amount, depot):
    child = []
    for r1, r2 in zip(parent1, parent2):
        chosen_route = r1.copy() if random.random() < 0.5 else r2.copy()
        child.append(chosen_route)
    for route in child:
        if route[0] != depot:
            route.insert(0, depot)
        if route[-1] != depot:
            route.append(depot)
    return child


def mutate_route(solution, mutation_rate=0.2):
    mutated_solution = []
    for route in solution:
        if random.random() < mutation_rate and len(route) > 3:
            indices = list(range(1, len(route) - 1))
            idx1, idx2 = random.sample(indices, 2)
            route[idx1], route[idx2] = route[idx2], route[idx1]
        mutated_solution.append(route)
    return mutated_solution


# --- 4. Generate Initial Population ---
population_size = 10  # Number of solutions per generation
"""population = [generate_random_routes(nodes, num_vehicles, DEPOT) for _ in range(population_size)]"""
population = heuristic_population_initialization(
    population_size, nodes, cost_matrix, travel_time_matrix, DEPOT,
    E_max, recharge_amount, charging_stations,
    vehicle_capacity, max_travel_time, requests, num_vehicles
)

print("Generated Routes:", population)

# --- 5. Main GA Loop ---
num_generations = 100  # Number of generations

for generation in range(num_generations):
    print(f"\n=== Generation {generation + 1} ===")
    evaluated_population = []
    for individual in population:
        total_fitness, valid = fitness_function(
            individual, cost_matrix, travel_time_matrix, E_max, charging_stations,
            recharge_amount, penalty_weights, DEPOT, nodes, vehicle_capacity,
            max_travel_time, requests
        )
        evaluated_population.append((individual, total_fitness, valid))

    # Select valid individuals and sort by fitness (lower is better).
    valid_population = [ind for ind in evaluated_population if ind[2]]
    valid_population.sort(key=lambda x: x[1])
    selected_parents = [ind[0] for ind in valid_population[:max(1, population_size // 2)]]

    if len(selected_parents) < 2:
        print("Not enough valid individuals; falling back to full evaluated population.")
        selected_parents = [ind[0] for ind in evaluated_population]
    if len(selected_parents) < 2:
        print("Still fewer than 2 parents; duplicating available parent.")
        selected_parents = selected_parents * 2

    children = []
    while len(children) < population_size - len(selected_parents):
        p1, p2 = random.sample(selected_parents, 2)
        child = order_crossover_evrp(p1, p2, cost_matrix, E_max, charging_stations, recharge_amount, DEPOT)
        child = mutate_route(child, mutation_rate=0.4)
        children.append(child)

    population = selected_parents + children
    print(f"Population size at end of generation: {len(population)}")
