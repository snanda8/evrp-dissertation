import random
import numpy as np
from sklearn.cluster import KMeans
from utils import distance, calculate_energy_to_reach, find_nearest_charging_station

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
                           charging_stations, requests, vehicle_capacity, battery):
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
            # Optional: Check that battery remaining after going to customer allows returning to depot.
            dist = distance(current_node, customer, nodes)
            if dist < best_distance:
                best_distance = dist
                best_customer = customer
        if best_customer is None:
            # No feasible customer found; try inserting a charging station.
            nearest_cs = find_nearest_charging_station(current_node, charging_stations, nodes, battery)
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

    # Ensure return to depot is feasible.
    energy_to_depot = calculate_energy_to_reach(current_node, start_node, cost_matrix)
    if energy_to_depot > remaining_battery:
        nearest_cs = find_nearest_charging_station(current_node, charging_stations, nodes, battery)
        if nearest_cs is not None:
            energy_to_cs = calculate_energy_to_reach(current_node, nearest_cs, cost_matrix)
            if energy_to_cs <= remaining_battery:
                route.append(nearest_cs)
                current_node = nearest_cs
                remaining_battery = E_max
    route.append(start_node)
    return route

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
                return route  # If optimization fails, return original route.


    return result_route
"""
def remove_unnecessary_charging_stations(route, cost_matrix, E_max, charging_stations, depot):
    Remove charging stations that are not necessary for battery feasibility.
    if not route or len(route) <= 2:
        return route
    cs_indices = [i for i, node in enumerate(route) if node in charging_stations and node != depot]
    for idx in sorted(cs_indices, reverse=True):
        test_route = route.copy()
        test_route.pop(idx)
        battery = E_max
        feasible = True
        for i in range(len(test_route) - 1):
            energy = calculate_energy_to_reach(test_route[i], test_route[i+1], cost_matrix)
            if energy == float('inf'):
                feasible = False
                break
            battery -= energy
            if battery < 0:
                feasible = False
                break
            if test_route[i+1] in charging_stations or test_route[i+1] == depot:
                battery = E_max
        if feasible:
            route = test_route
    return route
"""
def heuristic_initial_solution(nodes, cost_matrix, travel_time_matrix, depot,
                               E_max, recharge_amount, charging_stations,
                               vehicle_capacity, max_travel_time, requests, num_vehicles):
    # Cluster customers if multiple vehicles are used.
    if num_vehicles > 1:
        clusters = cluster_customers(list(requests.keys()), nodes, num_vehicles)
    else:
        clusters = [list(requests.keys())]
    solution = []
    for cluster in clusters:
        route = nearest_neighbor_route(depot, cluster, nodes, cost_matrix, E_max, charging_stations, requests, vehicle_capacity, battery=E_max)
        route = insert_charging_stations_strategically(route, cost_matrix, E_max, charging_stations, depot)
        #route = remove_unnecessary_charging_stations(route, cost_matrix, E_max, charging_stations, depot)
        solution.append(route)
    # Assign any unvisited customers to the route with the lowest load.
    visited = set()
    for route in solution:
        visited.update(set(route))
    unvisited = set(requests.keys()) - visited
    if unvisited:
        for cust in list(unvisited):
            best_route = min(solution, key=lambda r: sum(requests.get(n, {'quantity': 0})['quantity'] for n in r if n in requests))
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

def generate_giant_tour(depot, customers, nodes, cost_matrix):
    """
    Generate a giant tour (ordering) of all customers using a simple nearest neighbor heuristic.
    Returns a list of customer nodes in the order they are visited.
    """
    unvisited = set(customers)
    tour = []
    current = depot
    while unvisited:
        next_customer = min(unvisited, key=lambda c: cost_matrix.get((current, c), float('inf')))
        tour.append(next_customer)
        unvisited.remove(next_customer)
        current = next_customer
    return tour


def split_giant_tour(giant_tour, depot, cost_matrix, travel_time_matrix, requests,
                     vehicle_capacity, max_travel_time, E_max, battery_threshold=0.8):
    """
    Splits a giant tour (list of customer nodes) into routes.
    It forces a split when adding a new customer would cause the cumulative energy cost (plus cost to depot),
    total demand, or travel time to exceed the respective limits.

    battery_threshold is a fraction of E_max to trigger splitting early.
    """
    routes = []
    current_route = [depot]
    current_demand = 0
    current_travel_time = 0
    current_energy = 0  # cumulative energy cost from depot along this route
    current = depot

    for cust in giant_tour:
        demand = requests[cust]['quantity']
        travel_time = travel_time_matrix.get((current, cust), float('inf'))
        energy_cost = cost_matrix.get((current, cust), float('inf'))
        # Estimated cost from customer back to depot
        energy_to_depot = cost_matrix.get((cust, depot), float('inf'))
        travel_time_to_depot = travel_time_matrix.get((cust, depot), float('inf'))

        new_demand = current_demand + demand
        new_travel_time = current_travel_time + travel_time + travel_time_to_depot
        new_energy = current_energy + energy_cost + energy_to_depot

        # If adding this customer would exceed any constraint, close the current route.
        if (new_demand > vehicle_capacity or
                new_travel_time > max_travel_time or
                new_energy > E_max * battery_threshold):
            current_route.append(depot)
            routes.append(current_route)
            # Start a new route from the depot with this customer.
            current_route = [depot, cust]
            current_demand = demand
            current_travel_time = travel_time_matrix.get((depot, cust), 0)
            current_energy = cost_matrix.get((depot, cust), 0)
            current = cust
        else:
            current_route.append(cust)
            current_demand = new_demand
            current_travel_time += travel_time
            current_energy += energy_cost
            current = cust

    # Close the final route.
    current_route.append(depot)
    routes.append(current_route)
    return routes


def generate_giant_tour_and_split(depot, customers, nodes, cost_matrix, travel_time_matrix, requests, vehicle_capacity, max_travel_time, E_max):
    """
    Generates a giant tour over all customers and splits it into routes using the updated splitting function.
    """
    giant_tour = generate_giant_tour(depot, customers, nodes, cost_matrix)

    routes = split_giant_tour(giant_tour, depot, cost_matrix, travel_time_matrix, requests, vehicle_capacity, max_travel_time, E_max)


    return routes


from collections import defaultdict

def repair_route_battery_feasibility(route, cost_matrix, E_max, recharge_amount, charging_stations, depot, nodes):
    """
    Repairs a route by inserting charging stations when battery is insufficient or a trap is predicted.
    Ensures the route starts and ends at the depot and avoids CS looping.
    """
    print(f"\n🔧 [DEBUG] REPAIRING ROUTE: {route}")
    repaired_route = [route[0]]
    battery = E_max
    i = 1

    visited_transitions = set()
    visited_cs_pairs = set()
    cs_visit_counts = defaultdict(int)
    last_charging_stations = []
    MAX_INSERTIONS = 15
    insertions = 0

    while i < len(route):
        from_node = repaired_route[-1]
        to_node = route[i]
        energy_needed = cost_matrix.get((from_node, to_node), float('inf'))

        if energy_needed == float('inf'):
            print(f"❌ [DEBUG] Unreachable segment: {from_node} → {to_node}")
            return repaired_route

        # Prevent retry loops
        transition = (from_node, to_node)
        if transition in visited_transitions:
            print(f"🛑 [DEBUG] Loop detected at {transition}. Aborting.")
            return repaired_route
        visited_transitions.add(transition)

        # Predict trap ahead
        battery_after_move = battery - energy_needed
        trap_predicted = battery_after_move <= 0 or all(
            cost_matrix.get((to_node, cs), float('inf')) > battery_after_move for cs in charging_stations
        )

        # If move is infeasible or leads to a trap
        if energy_needed > battery or trap_predicted:
            print(f"⚠️ [DEBUG] Battery too low or trap predicted from {from_node} to {to_node}.")
            cs_candidates = charging_stations - set(last_charging_stations[-2:])
            cs = find_nearest_charging_station(from_node, cs_candidates, cost_matrix, battery)

            if cs is None:
                print(f"❌ [DEBUG] No reachable CS from {from_node}. Ending repair.")
                return repaired_route

            if cs == from_node or cs == to_node:
                print(f"🚫 [DEBUG] Skipping invalid CS {cs} between {from_node} → {to_node}")
                return repaired_route

            if (from_node, cs) in visited_cs_pairs:
                print(f"🛑 [DEBUG] CS pair {from_node} → {cs} already used. Aborting loop.")
                return repaired_route

            visited_cs_pairs.add((from_node, cs))
            cs_visit_counts[cs] += 1
            if cs_visit_counts[cs] > 3:
                print(f"🛑 [DEBUG] CS {cs} visited too often. Ending repair.")
                return repaired_route

            repaired_route.append(cs)
            battery = E_max
            last_charging_stations.append(cs)
            insertions += 1
            if insertions >= MAX_INSERTIONS:
                print("🛑 [DEBUG] Max CS insertions reached. Aborting.")
                return repaired_route
            continue  # Retry to_node after recharge

        # Make the move
        battery -= energy_needed
        repaired_route.append(to_node)

        if to_node in charging_stations or to_node == depot:
            battery = E_max
            last_charging_stations.append(to_node)
            print(f"🔋 [DEBUG] Recharged at {'Depot' if to_node == depot else 'CS'} {to_node}. Battery: {battery}")

        print(f"🔋 [DEBUG] Battery after move: {battery}")
        i += 1

    # Ensure route ends at depot
    if repaired_route[-1] != depot:
        cost_to_depot = cost_matrix.get((repaired_route[-1], depot), float('inf'))
        if cost_to_depot <= battery:
            repaired_route.append(depot)
            print(f"✅ [DEBUG] Appended depot at end.")
        else:
            cs = find_nearest_charging_station(repaired_route[-1], charging_stations, cost_matrix, battery)
            if cs and cost_matrix.get((cs, depot), float('inf')) <= E_max:
                repaired_route.extend([cs, depot])
                print(f"✅ [DEBUG] Added CS {cs} and depot to finish.")
            else:
                print(f"❌ [DEBUG] No path to depot. Route truncated.")

    print(f"\n✅ [DEBUG] Final repaired route: {repaired_route}")
    return repaired_route



def repair_giant_solution(routes, cost_matrix, E_max, recharge_amount, charging_stations, depot, nodes):
    """
    Applies the repair_route_battery_feasibility function to each route in the solution.
    Returns the repaired set of routes.
    """
    repaired_routes = []
    for route in routes:
        repaired = repair_route_battery_feasibility(route, cost_matrix, E_max, recharge_amount, charging_stations, depot, nodes)
        repaired_routes.append(repaired)
    return repaired_routes
