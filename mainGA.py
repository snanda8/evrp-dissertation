import math
import random


nodes = {
    0: (0, 0),
    1: (10, 20),
    2: (15, 10),
    3: (20, 5),
    4: (5, 25),
    5: (10, 15),
    6: (18, 23),
    7: (20, 10)
}


def calculate_cost_matrix(nodes):
    cost_matrix = {}
    for i, (x1, y1) in nodes.items():
        for j, (x2, y2) in nodes.items():
            if i != j:
                cost_matrix[(i, j)] = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
    return cost_matrix

cost_matrix = calculate_cost_matrix(nodes)
print("Cost Matrix:", cost_matrix)


def generate_random_routes(nodes, num_vehicles):
    customers = list(set(nodes.keys()) - {0})
    random.shuffle(customers)


    routes = []
    for i in range(num_vehicles):
        route = [0]
        route += customers[i::num_vehicles]
        route.append(0)
        routes.append(route)
    return routes



routes = generate_random_routes(nodes, num_vehicles=5)
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
    for i in range(len(route) - 1):
        from_node = route[i]
        to_node = route[i + 1]


        if from_node == 0:
            battery = E_max


        energy_cost = cost_matrix.get((from_node, to_node), float('inf'))
        if energy_cost == float('inf'):
            print(f"Unreachable node pair: ({from_node}, {to_node}). Adding penalty.")
            return battery, False, recharged, unnecessary_recharges
        battery -= energy_cost

        print(f"From {from_node} to {to_node}: Cost={energy_cost}, Battery={battery}")

        if to_node in charging_stations:
            battery = min(battery + recharge_amount, E_max)
            recharged += 1


        if battery < 0:
            print(f"Battery depleted between {from_node} and {to_node}.")
            return battery, False, recharged, unnecessary_recharges

        if to_node in charging_stations:
            if battery > 0.75 * E_max:  #
                unnecessary_recharges += 1
            battery = min(battery + recharge_amount, E_max)
            recharged += 1


    return battery, True, recharged, unnecessary_recharges


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

    return True


def fitness_function(routes, cost_matrix, E_max, charging_stations, recharge_amount, penalty_weights):
    total_distance = 0
    total_penalty = 0
    route_fitness_scores = []  # Store individual route scores

    customers = set(nodes) - {0}  # Exclude depot
    visited_customers = set()

    print("\n=== FITNESS FUNCTION DEBUG OUTPUT ===")
    for idx, route in enumerate(routes):
        print(f"Processing Route {idx + 1}: {route}")

        # Calculate route distance
        route_distance = 0
        route_penalty = 0  # Track penalty for this route
        for i in range(len(route) - 1):
            from_node = route[i]
            to_node = route[i + 1]
            distance = cost_matrix.get((from_node, to_node), float('inf'))
            if distance == float('inf'):
                route_penalty += penalty_weights['unreachable_node']
                print(f"  Penalty for unreachable node pair ({from_node}, {to_node}): {penalty_weights['unreachable_node']}")
            else:
                route_distance += distance

        print(f"  Route Distance: {route_distance}")
        total_distance += route_distance

        # Update battery and check constraints
        battery, valid, recharged, unnecessary_recharges = update_battery(
            route, cost_matrix, E_max, charging_stations, recharge_amount
        )

        # Penalty for battery depletion
        if not valid:
            route_penalty += penalty_weights['battery_depletion']
            print(f"  Penalty for battery depletion on route {route}: {penalty_weights['battery_depletion']}")

        # Penalty for unnecessary recharges
        if unnecessary_recharges > 0:
            penalty = penalty_weights['unnecessary_recharges'] * unnecessary_recharges
            route_penalty += penalty
            print(f"  Penalty for unnecessary recharges ({unnecessary_recharges}): {penalty}")

        print(f"  Charging Stations Visited: {recharged}")

        # Update visited customers
        segment_customers = set(route) - {0}
        visited_customers.update(segment_customers)

        # Compute **individual route fitness score**
        route_fitness_score = route_distance + route_penalty
        route_fitness_scores.append(route_fitness_score)
        print(f"  Fitness Score for Route {idx + 1}: {route_fitness_score}")

        print()

    # Penalty for missing customers
    if visited_customers != customers:
        missing = customers - visited_customers
        penalty = penalty_weights['missing_customers'] * len(missing)
        total_penalty += penalty
        print(f"\nPenalty for missing customers {missing}: {penalty}")

    # Final overall fitness score
    total_fitness_score = total_distance + total_penalty
    print(f"\nTotal Distance: {total_distance}")
    print(f"Total Penalty: {total_penalty}")
    print(f"Overall Fitness Score: {total_fitness_score}")

    return route_fitness_scores, total_fitness_score



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
        best_route = min(tournament_contestants, key=lambda x: x[1])[0]
        selected_routes.append(best_route)

    return selected_routes


# Main Program Logic
charging_stations = [4]
recharge_amount = 30
all_valid = True
E_max = 50


penalty_weights = {
    'missing_customers': 1e6,
    'battery_depletion': 1e5,
    'unreachable_node': 1e5,
    'unnecessary_recharges': 1000,
}
if validate_route(routes, nodes):
    route_fitness_scores, total_fitness_score = fitness_function(
        routes=routes,
        cost_matrix=cost_matrix,
        E_max=E_max,
        charging_stations=charging_stations,
        recharge_amount=recharge_amount,
        penalty_weights=penalty_weights
    )
    print(f"\nTotal Population Fitness Score: {total_fitness_score}")

    num_parents = len(routes) // 2

    selected_parents = tournament_selection(routes, route_fitness_scores, num_parents, tournament_size=3)

    print("\nSelected Routes using Tournament Selection:")
    for idx, parent in enumerate(selected_parents, 1):
        print(f"Parent {idx}: {parent}")
else:
    # If routes are invalid, print detailed validation results
    print("Some routes are invalid.")
    for route in routes:
        # Optionally, process individual routes for debugging purposes
        battery, valid, recharged = update_battery(route, cost_matrix, E_max, charging_stations, recharge_amount)
        if valid:
            print(f"Route {route} is valid with remaining battery: {battery}")
            print(f"Charging stations visited: {recharged}")
        else:
            print(f"Route {route} is invalid due to battery depletion.")






