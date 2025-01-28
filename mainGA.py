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


    customers = set(nodes) - {0}
    visited_customers = set()


    for route in routes:
        for i in range(len(route) -1):
            from_node = route[i]
            to_node = route[i + 1]
            distance = cost_matrix.get((from_node, to_node), float('inf'))
            if distance == float('inf'):
                total_penalty += penalty_weights['unreachable_node']
            total_distance += distance

        battery, valid, recharged, unnecessary_recharges = update_battery(
            route, cost_matrix, E_max, charging_stations, recharge_amount
        )

        if not valid:
            total_penalty += penalty_weights['battery_depletion']

        total_penalty += penalty_weights['unnecessary_recharges'] * unnecessary_recharges


        segment_customers = set(route) - {0}
        visited_customers.update(segment_customers)


    if visited_customers != customers:
        missing = customers - visited_customers
        total_penalty += penalty_weights['missing_customers'] * len(missing)


    fitness_score = total_distance + total_penalty
    return fitness_score


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
    fitness_score = fitness_function(
        routes=routes,
        cost_matrix=cost_matrix,
        E_max=E_max,
        charging_stations=charging_stations,
        recharge_amount=recharge_amount,
        penalty_weights=penalty_weights
    )
    print(f"Fitness Score: {fitness_score}")
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






