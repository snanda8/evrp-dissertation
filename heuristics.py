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
            # Optional: Check that battery remaining after going to customer allows returning to depot.
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

    # Ensure return to depot is feasible.
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
        route = nearest_neighbor_route(depot, cluster, nodes, cost_matrix, E_max, charging_stations, requests, vehicle_capacity)
        route = insert_charging_stations_strategically(route, cost_matrix, E_max, charging_stations, depot)
        route = remove_unnecessary_charging_stations(route, cost_matrix, E_max, charging_stations, depot)
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
