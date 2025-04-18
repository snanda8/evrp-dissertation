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
    try:
        print("🚀 Entered fitness_function")

        total_distance = 0
        total_penalty = 0
        visited_customers = set()

        expected_customers = (set(nodes.keys()) - charging_stations) - {depot}

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
                    visited_customers.add(node)
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

        if visited_customers != expected_customers:
            missing = expected_customers - visited_customers
            missing_penalty = penalty_weights['missing_customers'] * len(missing)
            total_penalty += missing_penalty
            print(f"  Missing customers {missing} -> penalty: {missing_penalty}")

        vehicle_penalty = len(solution) * penalty_weights.get('vehicle_count', 1e4)
        print(f"  Vehicle count penalty: {vehicle_penalty}")
        total_penalty += vehicle_penalty

        unnecessary_cs_count = 0
        for route in solution:
            unnecessary_cs_count += sum(1 for node in route if node in charging_stations)

        print(f"  Total unnecessary CS visits: {unnecessary_cs_count}")
        total_penalty += penalty_weights['unnecessary_recharges'] * unnecessary_cs_count

        total_fitness = total_distance + total_penalty
        print(f"\n✅ Returning from fitness_function with fitness: {total_fitness}, Valid: {total_penalty == 0}")
        return total_fitness, (total_penalty == 0)

    except Exception as e:
        print(f"\n💥 Exception in fitness_function: {e}")
        import traceback
        traceback.print_exc()
        return float('inf'), False
