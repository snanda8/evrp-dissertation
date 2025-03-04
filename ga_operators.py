def update_battery(route, cost_matrix, E_max, charging_stations, recharge_amount, depot):
    battery = E_max
    recharged = 0
    low_battery_penalty = 0
    for i in range(len(route) - 1):
        from_node = route[i]
        to_node = route[i + 1]
        if isinstance(from_node, list):
            from_node = from_node[0]
        if isinstance(to_node, list):
            to_node = to_node[0]
        if from_node == depot:
            battery = E_max
        energy_cost = cost_matrix.get((from_node, to_node), float('inf'))
        if energy_cost == float('inf'):
            print(f"Unreachable node pair: ({from_node}, {to_node}).")
            return battery, False, recharged, low_battery_penalty
        battery -= energy_cost
        print(f"From {from_node} to {to_node}: Cost={energy_cost}, Battery={battery}")
        if battery < 0.25 * E_max and to_node not in charging_stations:
            print("Low battery warning! No charging station ahead.")
            low_battery_penalty += 5000
        if battery < 0:
            print(f"Battery depleted between {from_node} and {to_node}.")
            return battery, False, recharged, low_battery_penalty
        if to_node in charging_stations:
            battery = min(battery + recharge_amount, E_max)
            recharged += 1
    return battery, True, recharged, low_battery_penalty

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
            distance_val = cost_matrix.get((from_node, to_node), float('inf'))
            travel_time = travel_time_matrix.get((from_node, to_node), float('inf'))
            route_distance += distance_val
            route_travel_time += travel_time
        print(f"  Route Distance: {route_distance}")
        print(f"  Route Travel Time: {route_travel_time}")
        total_distance += route_distance

        battery, valid, recharged, low_battery_penalty = update_battery(
            route, cost_matrix, E_max, charging_stations, recharge_amount, depot
        )
        print(f"  Battery after route: {battery}, Valid: {valid}")
        if not valid:
            total_penalty += penalty_weights['battery_depletion']
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
