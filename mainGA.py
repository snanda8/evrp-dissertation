import math
import random


def generate_nodes(num_nodes, charging_station_ratio=0.2):
    """
    Dynamically generates nodes, ensuring a depot at (0,0) and randomly assigned customers and charging stations.

    :param num_nodes: Total number of nodes to generate (including depot).
    :param charging_station_ratio: Percentage of nodes that should be charging stations.
    :return: Tuple containing nodes dictionary and charging station set.
    """
    if num_nodes < 5:
        raise ValueError("Number of nodes must be at least 5 for a valid EVRP instance.")

    nodes = {0: (0, 0)}  # The depot is always at (0,0)

    # Generate random (x, y) coordinates for other nodes in a bounded space (avoid placing too close to depot)
    for i in range(1, num_nodes):
        x, y = random.uniform(5, 100), random.uniform(5, 100)
        nodes[i] = (x, y)


    num_charging_stations = max(1, int(num_nodes * charging_station_ratio))  # Ensure at least 1 charging station
    charging_stations = set(random.sample(list(nodes.keys())[1:], num_charging_stations))  # Exclude depot

    return nodes, charging_stations


# Example usage
num_nodes = 15
nodes, charging_stations = generate_nodes(num_nodes)
print("Generated Nodes:", nodes)
print("")
print("Charging Stations:", charging_stations)
print("")


def calculate_cost_matrix(nodes):
    cost_matrix = {}
    for i, (x1, y1) in nodes.items():
        for j, (x2, y2) in nodes.items():
            if i != j:
                cost_matrix[(i, j)] = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
    return cost_matrix

cost_matrix = calculate_cost_matrix(nodes)
print("Cost Matrix:", cost_matrix)
print("")


def generate_random_routes(nodes, num_vehicles):
    """
    Generates routes for multiple vehicles, ensuring an even distribution of customers.

    :param nodes: Dictionary of nodes.
    :param num_vehicles: Number of vehicles.
    :return: List of routes.
    """
    customers = list(set(nodes.keys()) - {0})  # Exclude depot
    random.shuffle(customers)

    # Distribute customers evenly across vehicles
    min_customers_per_vehicle = max(1, len(customers) // num_vehicles)

    routes = []
    for i in range(num_vehicles):
        route = [0]  # Start at depot
        assigned_customers = customers[i::num_vehicles]  # Distribute customers evenly
        route += assigned_customers
        route.append(0)  # End at depot
        routes.append(route)

    return routes

# Example usage
num_vehicles = 5
routes = generate_random_routes(nodes, num_vehicles)
print("Generated Routes:", routes)








def validate_no_duplicates(routes, nodes):
    customers = set(nodes) - {0}
    visited_customers = set()

    for route in routes:
        route_customers = set(route) - {0}
        if visited_customers.intersection(route_customers):
            print("Duplicate customers detected across routes.")
            return False
        visited_customers.update(route_customers)

    if visited_customers != customers:
        print("Not all customers are visited across all routes.")
        return False

    return True


def update_battery(route, cost_matrix, E_max, charging_stations, recharge_amount):
    battery = E_max
    recharged = 0
    unnecessary_recharges = 0
    low_battery_penalty = 0

    for i in range(len(route) - 1):
        from_node = route[i]
        to_node = route[i + 1]

        if isinstance(from_node, list):
            print(f"️ Warning: `from_node` is a list {from_node}, taking first element.")
            from_node = from_node[0]
        if isinstance(to_node, list):
            print(f"️ Warning: `to_node` is a list {to_node}, taking first element.")
            to_node = to_node[0]

        if from_node == 0:
            battery = E_max


        energy_cost = cost_matrix.get((from_node, to_node), float('inf'))
        if energy_cost == float('inf'):
            print(f"Unreachable node pair: ({from_node}, {to_node}). Adding penalty.")
            return battery, False, recharged, unnecessary_recharges, low_battery_penalty

        battery -= energy_cost
        print(f"From {from_node} to {to_node}: Cost={energy_cost}, Battery={battery}")

        if battery < 0.25 * E_max and to_node not in charging_stations:
            print(f" Low battery warning! No charging station ahead.")
            low_battery_penalty += 5000  # Customizable

        if to_node in charging_stations:
            battery = min(battery + recharge_amount, E_max)
            recharged += 1

        if battery < 0:
            print(f"Battery depleted between {from_node} and {to_node}.")
            print("")
            return battery, False, recharged, unnecessary_recharges, low_battery_penalty

        if to_node in charging_stations:
            if battery > 0.75 * E_max:  #
                unnecessary_recharges += 1
            battery = min(battery + recharge_amount, E_max)
            recharged += 1


    return battery, True, recharged, unnecessary_recharges, low_battery_penalty


def validate_route(routes, nodes):

    for route in routes:
        if not route or route[0] != 0 or route[-1] != 0:
            print(f"Invalid segment: {route}. Must start and end at depot.")
            return False


    customers = set(nodes) - {0}
    visited_customers = set()
    for route in routes:
        segment_customers = set(route) - {0}
        visited_customers.update(segment_customers)


    if visited_customers != customers:
        missing = customers - visited_customers
        extra = visited_customers - customers
        if missing:
            print(f"Missing customers: {missing}")
        if extra:
            print(f"Extra customers: {extra}")
        return False


def is_fully_connected(nodes, cost_matrix):
    """
    Ensures that all nodes are reachable from the depot.
    Uses BFS/DFS to check connectivity.
    """
    visited = set()
    queue = [0]

    while queue:
        current = queue.pop()
        if current not in visited:
            visited.add(current)
            queue.extend([j for j in nodes if (current, j) in cost_matrix])

    return visited == set(nodes.keys())


# Before running the GA, check connectivity
if not is_fully_connected(nodes, cost_matrix):
    print("Warning: Some nodes are not reachable from the depot. Adjust locations.")


def fitness_function(solution, cost_matrix, E_max, charging_stations, recharge_amount, penalty_weights):
    total_distance = 0
    total_penalty = 0
    visited_customers = set()
    overall_valid = True

    print("\n=== Multi-Vehicle Fitness Debug ===")
    for idx, route in enumerate(solution):
        print(f"\nProcessing Vehicle {idx + 1} Route: {route}")

        # Check that route starts and ends at the depot.
        if not route or route[0] != 0 or route[-1] != 0:
            print("  Route must start and end at depot.")
            total_penalty += penalty_weights.get('invalid_route', 1e5)
            overall_valid = False

        # Calculate the route's distance.
        route_distance = 0
        for i in range(len(route) - 1):
            from_node = route[i]
            to_node = route[i + 1]
            distance = cost_matrix.get((from_node, to_node), float('inf'))
            route_distance += distance
        print(f"  Route Distance: {route_distance}")
        total_distance += route_distance

        # Check battery constraints for this route.
        battery, valid, recharged, unnecessary_recharges, low_battery_penalty = update_battery(
            route, cost_matrix, E_max, charging_stations, recharge_amount
        )
        print(f"  Battery after route: {battery}, Valid: {valid}")
        if not valid:
            total_penalty += penalty_weights['battery_depletion']
            overall_valid = False
        if unnecessary_recharges > 0:
            penalty = penalty_weights['unnecessary_recharges'] * unnecessary_recharges
            total_penalty += penalty
            print(f"  Unnecessary recharge penalty: {penalty}")
        if low_battery_penalty > 0:
            total_penalty += low_battery_penalty
            print(f"  Low battery penalty: {low_battery_penalty}")

        # Collect the customers visited in this route.
        visited_customers.update(set(route) - {0})

    # Ensure all customers are visited.
    expected_customers = set(nodes.keys()) - {0}
    if visited_customers != expected_customers:
        missing = expected_customers - visited_customers
        missing_penalty = penalty_weights['missing_customers'] * len(missing)
        total_penalty += missing_penalty
        print(f"  Missing customers {missing} -> penalty: {missing_penalty}")

    total_fitness = total_distance + total_penalty
    print(f"\nTotal Distance: {total_distance}, Total Penalty: {total_penalty}, Overall Fitness: {total_fitness}")
    return total_fitness, overall_valid



def tournament_selection(routes, route_fitness_scores, num_parents, tournament_size=3):
    """
    Selects parents using Tournament Selection.
    A subset of routes is randomly chosen, and the best one is selected.
    """
    selected_routes = []
    for _ in range(num_parents):
        # Select random routes for the tournament
        tournament_contestants = random.sample(list(zip(routes, route_fitness_scores)), tournament_size)

        # Select the route with the best (lowest) fitness score
        best_route = min(tournament_contestants, key=lambda x: (x[1], -len(x[0])))
        selected_routes.append(best_route[0])

    return selected_routes


def order_crossover_evrp(parent1, parent2, cost_matrix, E_max, charging_stations, recharge_amount):
    """
    Performs a simple route-level crossover for multi-vehicle solutions.
    For each vehicle, it randomly selects the route from one of the two parents.
    """
    child = []
    for r1, r2 in zip(parent1, parent2):
        chosen_route = r1.copy() if random.random() < 0.5 else r2.copy()
        # (Optional) You can add minor intra-route reordering here if desired.
        child.append(chosen_route)

    # Ensure that each route starts and ends with depot.
    for route in child:
        if route[0] != 0:
            route.insert(0, 0)
        if route[-1] != 0:
            route.append(0)
    return child


def enforce_battery_constraints(route, cost_matrix, E_max, charging_stations, recharge_amount):
    """
    Ensures battery constraints are met by adding necessary charging stations.

    :param route: List of nodes in the route.
    :param cost_matrix: Dictionary containing travel costs (energy consumption).
    :param E_max: Maximum battery capacity.
    :param charging_stations: Set of charging station nodes.
    :param recharge_amount: Amount of charge gained at a charging station.
    :return: A modified route that respects battery constraints.
    """
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
                battery = min(battery + recharge_amount, E_max)  # Recharge battery

        if to_node in charging_stations:
            battery = min(battery + recharge_amount, E_max)  # Recharge at the charging station

    valid_route.append(route[-1])  # Ensure depot is included at the end
    return valid_route


def find_nearest_charging_station(current_node, charging_stations, cost_matrix):
    """
    Finds the nearest charging station from the current location.

    :param current_node: The current node where the vehicle is located.
    :param charging_stations: Set of charging station nodes.
    :param cost_matrix: Dictionary containing travel costs (energy consumption).
    :return: The closest charging station node.
    """
    nearest_station = None
    min_cost = float('inf')

    for station in charging_stations:
        cost = cost_matrix.get((current_node, station), float('inf'))
        if cost < min_cost:
            min_cost = cost
            nearest_station = station

    return nearest_station


def select_next_generation(population, fitness_scores, elitism_rate=0.1):
    """
    Selects the next generation while preserving elite individuals.
    """
    sorted_population = [x for _, x in sorted(zip(fitness_scores, population))]
    elite_size = int(len(population) * elitism_rate)

    # Ensure best solutions persist
    next_generation = sorted_population[:elite_size]  # Keep top X% solutions

    while len(next_generation) < len(population):
        parent1, parent2 = random.choices(sorted_population[:elite_size * 2], k=2)
        child = order_crossover_evrp(parent1, parent2)
        next_generation.append(child)

    return next_generation


def mutate_route(solution, mutation_rate=0.2):
    """
    Mutates a multi-vehicle solution by applying a swap mutation on each route with a given probability.
    """
    mutated_solution = []
    for route in solution:
        # With probability 'mutation_rate', swap two customers in the route (ignoring depots).
        if random.random() < mutation_rate and len(route) > 3:
            indices = list(range(1, len(route) - 1))
            idx1, idx2 = random.sample(indices, 2)
            route[idx1], route[idx2] = route[idx2], route[idx1]
        mutated_solution.append(route)
    return mutated_solution



#GA Parameters
num_generations = 100  # Number of generations
population_size = 10  # Number of routes per generation
num_vehicles = 5  # Number of vehicles
tournament_size = 3  # Tournament selection parameter
E_max = 175  # Max battery capacity
recharge_amount = 30  # Charge gained at stations

# Penalty Weights
penalty_weights = {
    'missing_customers': 1e6,  # Penalty for customers not visited
    'battery_depletion': 1e5,  # Penalty for battery running out
    'unreachable_node': 1e5,  # Penalty for unreachable nodes
    'unnecessary_recharges': 1000,  # Penalty for recharging too soon
    'low_battery': 5000,  # Penalty for low battery warning
}

# Generate Initial Population
population = [generate_random_routes(nodes, num_vehicles) for _ in range(population_size)]

# GA loop
for generation in range(num_generations):
    print(f"\n=== Generation {generation + 1} ===")

    # Evaluate fitness for each individual.
    evaluated_population = []
    for individual in population:
        total_fitness, valid = fitness_function(
            individual, cost_matrix, E_max, charging_stations, recharge_amount, penalty_weights
        )
        evaluated_population.append((individual, total_fitness, valid))


    valid_population = [ind for ind in evaluated_population if ind[2]]
    valid_population.sort(key=lambda x: x[1])

    # Select parents
    selected_parents = [ind[0] for ind in valid_population[:max(1, population_size // 2)]]

    # If there are fewer than 2 valid parents, fall back to using the full evaluated population.
    if len(selected_parents) < 2:
        print("Not enough valid individuals; falling back to full evaluated population.")
        selected_parents = [ind[0] for ind in evaluated_population]

    # If still fewer than 2, duplicate the available parent.
    if len(selected_parents) < 2:
        print("Still fewer than 2 parents; duplicating available parent.")
        selected_parents = selected_parents * 2  # duplicate the single parent if necessary

    # Generate children using crossover and mutation functions.
    children = []
    while len(children) < population_size - len(selected_parents):
        p1, p2 = random.sample(selected_parents, 2)
        child = order_crossover_evrp(p1, p2, cost_matrix, E_max, charging_stations, recharge_amount)
        child = mutate_route(child, mutation_rate=0.4)
        children.append(child)

    # Form the new population.
    population = selected_parents + children
    print(f"Population size at end of generation: {len(population)}")









