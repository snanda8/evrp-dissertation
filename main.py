import math
import random
import xml.etree.ElementTree as ET

def parse_instance(file_path):
    import xml.etree.ElementTree as ET
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

    # Parse requests for demand and service time.
    requests = {}
    requests_xml = root.find('requests')
    for req in requests_xml:
        req_node = int(req.attrib['node'])
        quantity = float(req.find('quantity').text)
        service_time = float(req.find('service_time').text)
        requests[req_node] = {'quantity': quantity, 'service_time': service_time}

    return (nodes, charging_stations, depot, customers, cost_matrix,
            travel_time_matrix, battery_capacity, num_vehicles, vehicle_capacity,
            max_travel_time, requests)

# Parse instance and set global variables
instance_file = "C101-10.xml"
nodes, charging_stations, depot, customers, cost_matrix, battery_capacity, num_vehicles_instance, requests = parse_instance(
    instance_file)
DEPOT = depot  # Global depot from instance
num_vehicles = num_vehicles_instance
E_max = battery_capacity


def generate_random_routes(nodes, num_vehicles, depot):
    """
    Generates routes for multiple vehicles, ensuring an even distribution of customers.
    Uses the provided depot.
    """
    customers_list = list(set(nodes.keys()) - {depot})
    random.shuffle(customers_list)
    routes = []
    for i in range(num_vehicles):
        route = [depot]  # Start at depot
        assigned_customers = customers_list[i::num_vehicles]
        route += assigned_customers
        route.append(depot)  # End at depot
        routes.append(route)
    return routes

def validate_no_duplicates(routes, nodes, depot):
    custs = set(nodes.keys()) - {depot}
    visited_customers = set()
    for route in routes:
        route_customers = set(route) - {depot}
        if visited_customers.intersection(route_customers):
            print("Duplicate customers detected across routes.")
            return False
        visited_customers.update(route_customers)
    if visited_customers != custs:
        print("Not all customers are visited across routes.")
        return False
    return True



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
            print("")
            return battery, False, recharged, unnecessary_recharges, low_battery_penalty
        if to_node in charging_stations:
            if battery > 0.75 * E_max:
                unnecessary_recharges += 1
            battery = min(battery + recharge_amount, E_max)
            recharged += 1
    return battery, True, recharged, unnecessary_recharges, low_battery_penalty


def is_fully_connected(nodes, cost_matrix, depot):
    visited = set()
    queue = [depot]
    while queue:
        current = queue.pop()
        if current not in visited:
            visited.add(current)
            queue.extend([j for j in nodes if (current, j) in cost_matrix])
    return visited == set(nodes.keys())


def fitness_function(solution, cost_matrix, travel_time_matrix, E_max, charging_stations,
                     recharge_amount, penalty_weights, depot, nodes, vehicle_capacity,
                     max_travel_time, requests):
    total_distance = 0
    total_penalty = 0
    visited_customers = set()
    print("\n=== Multi-Vehicle Fitness Debug ===")
    for idx, route in enumerate(solution):
        print(f"\nProcessing Vehicle {idx + 1} Route: {route}")
        # Check that the route starts and ends at the depot.
        if not route or route[0] != depot or route[-1] != depot:
            print("  Route must start and end at depot.")
            total_penalty += penalty_weights.get('invalid_route', 1e5)
        # Calculate route distance and travel time.
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

        # Check battery constraints.
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

        # Check vehicle capacity: sum the demand of customers on the route.
        route_demand = 0
        for node in route:
            # Only count demand for nodes that are customers (exclude depot and charging stations).
            if node in requests and node not in charging_stations and node != depot:
                route_demand += requests[node]['quantity']
        print(f"  Route Demand: {route_demand}")
        if route_demand > vehicle_capacity:
            overload = route_demand - vehicle_capacity
            penalty = penalty_weights.get('capacity_overload', 1e5) * overload
            print(f"  Capacity overload: {overload}, Penalty: {penalty}")
            total_penalty += penalty

        # Check maximum travel time constraint.
        if route_travel_time > max_travel_time:
            excess_time = route_travel_time - max_travel_time
            penalty = penalty_weights.get('max_travel_time_exceeded', 1e5) * excess_time
            print(f"  Travel time exceeded by {excess_time}, Penalty: {penalty}")
            total_penalty += penalty

        # Collect visited customers.
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




def tournament_selection(routes, route_fitness_scores, num_parents, tournament_size=3):
    selected_routes = []
    for _ in range(num_parents):
        tournament_contestants = random.sample(list(zip(routes, route_fitness_scores)), tournament_size)
        best_route = min(tournament_contestants, key=lambda x: (x[1], -len(x[0])))
        selected_routes.append(best_route[0])
    return selected_routes



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


def enforce_battery_constraints(route, cost_matrix, E_max, charging_stations, recharge_amount, depot):
    battery = E_max
    valid_route = []
    for i in range(len(route) - 1):
        from_node = route[i]
        to_node = route[i + 1]
        energy_cost = cost_matrix.get((from_node, to_node), float('inf'))
        if energy_cost == float('inf'):
            print(f"Warning: No direct path between {from_node} and {to_node}.")
            return None
        battery -= energy_cost
        valid_route.append(from_node)
        if battery < 0:
            print(f"Battery depleted between {from_node} and {to_node}. Adding nearest charging station.")
            nearest_station = find_nearest_charging_station(from_node, charging_stations, cost_matrix)
            if nearest_station:
                valid_route.append(nearest_station)
                battery = min(battery + recharge_amount, E_max)
        if to_node in charging_stations:
            battery = min(battery + recharge_amount, E_max)
    valid_route.append(route[-1])
    return valid_route


def find_nearest_charging_station(current_node, charging_stations, cost_matrix):
    nearest_station = None
    min_cost = float('inf')
    for station in charging_stations:
        cost = cost_matrix.get((current_node, station), float('inf'))
        if cost < min_cost:
            min_cost = cost
            nearest_station = station
    return nearest_station



def select_next_generation(population, fitness_scores, elitism_rate=0.1, cost_matrix=None, E_max=None, charging_stations=None, recharge_amount=None, depot=None):
    sorted_population = [x for _, x in sorted(zip(fitness_scores, population))]
    elite_size = int(len(population) * elitism_rate)
    next_generation = sorted_population[:elite_size]
    while len(next_generation) < len(population):
        parent1, parent2 = random.choices(sorted_population[:elite_size * 2], k=2)
        child = order_crossover_evrp(parent1, parent2, cost_matrix, E_max, charging_stations, recharge_amount, depot)
        next_generation.append(child)
    return next_generation



def mutate_route(solution, mutation_rate=0.2):
    mutated_solution = []
    for route in solution:
        if random.random() < mutation_rate and len(route) > 3:
            indices = list(range(1, len(route) - 1))
            idx1, idx2 = random.sample(indices, 2)
            route[idx1], route[idx2] = route[idx2], route[idx1]
        mutated_solution.append(route)
    return mutated_solution


# GA Parameters
num_generations = 100  # Number of generations
population_size = 10  # Number of solutions per generation
tournament_size = 3  # Tournament selection parameter
recharge_amount = 30  # Charge gained at stations

penalty_weights = {
    'missing_customers': 1e6,           # Penalty for customers not visited
    'battery_depletion': 1e5,           # Penalty for battery running out
    'unreachable_node': 1e5,            # Penalty for unreachable nodes
    'unnecessary_recharges': 1000,      # Penalty for recharging too soon
    'low_battery': 5000,                # Penalty for low battery warnings
    'capacity_overload': 1e5,           # Penalty per unit of overload
    'max_travel_time_exceeded': 1e5,    # Penalty per unit of excess travel time
    'invalid_route': 1e5,               # Penalty for a route not starting/ending at depot
}




# Generate initial population using the parsed instance values
population = [generate_random_routes(nodes, num_vehicles, DEPOT) for _ in range(population_size)]
print("Generated Routes:", population)

# GA Loop

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
    # ... rest of your GA loop remains unchanged ...


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










