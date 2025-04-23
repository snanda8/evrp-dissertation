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
                     max_travel_time, requests, customers):
    try:
        print("🚀 Entered fitness_function")

        total_distance = 0
        total_penalty = 0
        visited_customers = set()
        battery_depletion_flag = False

        expected_customers = set(customers)

        print(f"Expected customers: {expected_customers}")

        for idx, route in enumerate(solution):
            print(f"\nProcessing Vehicle {idx + 1} Route: {route}")
            if not route or route[0] != depot or route[-1] != depot:
                print("  ⚠️ Route must start and end at depot.")
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

            total_distance += route_distance
            print(f"  Route Distance: {route_distance}")
            print(f"  Route Travel Time: {route_travel_time}")

            # Track customers served
            # Track customers served
            route_demand = 0
            for node in route:
                if node in requests and node not in charging_stations and node != depot:
                    visited_customers.add(node)
                    route_demand += requests[node]['quantity']

            # Detect duplicate customer visits and hard-invalidate
            customer_counts = {}
            for node in route:
                if node not in [depot] and node not in charging_stations and node in customers:
                    customer_counts[node] = customer_counts.get(node, 0) + 1

            duplicates = {k: v for k, v in customer_counts.items() if v > 1}
            if duplicates:
                print(f"❌ HARD INVALIDATION: duplicate customers in route {idx + 1}: {duplicates}")
                return float('inf'), False

            print(f"  Route Demand: {route_demand}")
            if route_demand > vehicle_capacity:
                overload = route_demand - vehicle_capacity
                print(f"  ⚠️ Capacity overload: {overload}")
                # Optional: apply reduced or zero penalty here for debugging
                # total_penalty += penalty_weights.get('capacity_overload', 1e5) * overload

            if route_travel_time > max_travel_time:
                excess_time = route_travel_time - max_travel_time
                penalty = penalty_weights.get('max_travel_time_exceeded', 1e5) * excess_time
                print(f"  ⌛ Travel time exceeded by {excess_time}, Penalty: {penalty}")
                total_penalty += penalty

            # Battery checks
            battery, valid, recharged, low_battery_penalty = update_battery(
                route, cost_matrix, E_max, charging_stations, recharge_amount, depot
            )
            print(f"  🔋 Battery after route: {battery}, Valid: {valid}")
            if not valid:
                total_penalty += penalty_weights['battery_depletion']
                battery_depletion_flag = True

            if low_battery_penalty > 0 and not battery_depletion_flag:
                # Apply reduced penalty or ignore if route survives
                reduced_penalty = low_battery_penalty * 0.2
                print(f"  ⚠️ Low battery warning: {low_battery_penalty} → applying reduced penalty: {reduced_penalty}")
                total_penalty += reduced_penalty

        # === Global Customer Check ===
        if visited_customers != expected_customers:
            missing = expected_customers - visited_customers
            missing_penalty = penalty_weights['missing_customers'] * len(missing)
            print(f"❌ Missing customers: {missing} → Penalty: {missing_penalty}")
            total_penalty += missing_penalty

        # === Vehicle Count Penalty (but not invalidating) ===
        vehicle_penalty = len(solution) * penalty_weights.get('vehicle_count', 1e4)
        print(f"🚚 Vehicle count penalty: {vehicle_penalty}")
        total_penalty += vehicle_penalty

        # === Unnecessary Recharges ===
        cs_visits = sum(1 for route in solution for node in route if node in charging_stations)
        print(f"🔌 Charging station visits: {cs_visits}")
        total_penalty += penalty_weights['unnecessary_recharges'] * cs_visits * 0.25  # reduced impact

        total_fitness = total_distance + total_penalty

        # === Validity flag: relaxed check ===
        valid = (
            visited_customers == expected_customers
            and not battery_depletion_flag
        )

        print(f"\n✅ Final Fitness: {total_fitness} | Battery Feasible: {'Yes' if valid else 'No'}")
        return total_fitness, valid

    except Exception as e:
        print(f"💥 Exception in fitness_function: {e}")
        import traceback
        traceback.print_exc()
        return float('inf'), False

