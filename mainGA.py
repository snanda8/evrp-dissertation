# Depot = 0, Customers = 1, 2, 3, Charging Station = 4
nodes = [0,1,2,3,4]


# Define a Cost Matrix
cost_matrix = {
    (0, 1): 10,
    (0, 2): 15,
    (0, 3): 20,
    (0, 4): 25,
    (1, 2): 12,
    (1, 3): 8,
    (2, 3): 5,
    (1, 4): 18,
    (2, 4): 10,
    (3, 4): 10,
}

for (i, j), cost in list(cost_matrix.items()):
    cost_matrix[(j, i)] = cost  # Adding reverse order pairs


# Define Chromosomes and Routes
chromosome = [0, 1, 2, 0, 3, 4, 0]  # Two vehicles
routes = [[0, 1, 2,4,3, 0], [0, 1, 3, 4, 2, 0]]


# Define Battery
E_max = 50


# Update Battery Level Along Route
def update_battery(route, cost_matrix, E_max, charging_stations, recharge_amount):
    battery = E_max
    recharged = 0
    for i in range(len(route) - 1):
        from_node = route[i]
        to_node = route[i + 1]

        # Reset battery at the depot
        if from_node == 0:
            battery = E_max

        # Deduct energy for travel
        energy_cost = cost_matrix.get((from_node, to_node), None)
        if energy_cost is None:
            raise ValueError(f"Missing cost for ({from_node}, {to_node})")
        battery -= energy_cost

        # Recharge at charging stations
        if to_node in charging_stations:
            battery = min(battery + recharge_amount, E_max)  # Recharge battery
            recharged += 1

        # Check for battery depletion
        if battery < 0:
            print(f"Battery depleted between {from_node} and {to_node}.")
            return battery, False, recharged

    return battery, True, recharged

# Validate Route
def validate_route(chromosome, nodes, routes):
    # Ensure the main route starts and ends at the depot
    if not chromosome or chromosome[0] != 0 or chromosome[-1] != 0:
        return False

    # Ensure all customers are visited exactly once
    customers = set(nodes) - {0}  # Exclude depot
    visited_customers = set(chromosome) - {0}
    if customers != visited_customers:
        print("All customers NOT visited")
        return False

    # Validate each route (segment) for individual vehicles
    for route in routes:
        # Each route must start and end at the depot
        if not route or route[0] != 0 or route[-1] != 0:
            print(f"Invalid segment: {route}")
            return False

        # Ensure no duplicate customers within this route
        segment_customers = set(route) - {0}
        if not segment_customers.issubset(customers):
            print(f"Invalid customers in segment: {route}")
            return False

    return True  # All validations passed



# Main Program Logic
charging_stations = [4]  # Define charging stations
recharge_amount = 30
all_valid = True  # Track overall route validity

for route in routes:
    if validate_route(route, nodes, [route]):  # Validate individual route
        battery, valid, recharged = update_battery(route, cost_matrix, E_max, charging_stations, recharge_amount)
        if valid:
            print(f"Route {route} is valid with remaining battery: {battery}")
            print(f"Charging stations visited: {recharged}")
        else:
            print(f"Route {route} is invalid due to battery depletion.")
            all_valid = False
    else:
        print(f"Route {route} is invalid.")
        all_valid = False

if all_valid:
    print("All routes are valid.")
else:
    print("Some routes are invalid.")




