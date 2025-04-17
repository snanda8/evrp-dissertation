import os
import time
import csv

from instance_parser import parse_instance
from constructive_solver import construct_initial_solution, post_merge_routes
from local_search import apply_local_search
from ga_operators import fitness_function
from utils import make_routes_battery_feasible


def filter_overloaded_routes(routes, vehicle_capacity, requests, depot, charging_stations):
    filtered = []
    for route in routes:
        demand = sum(requests[n]['quantity'] for n in route if n not in charging_stations and n != depot)
        if demand <= vehicle_capacity:
            filtered.append(route)
        else:
            print(f"[⚠️] Overloaded route dropped (Demand: {demand}): {route}")
    return filtered


def sanitize_routes(routes, depot, charging_stations):
    cleaned = []

    for route in routes:
        route = [n for i, n in enumerate(route) if i == 0 or n != route[i - 1]]  # Remove adjacent duplicates

        while len(route) >= 2 and route[-1] == depot and route[-2] == depot:
            route.pop()

        if route[0] != depot:
            route = [depot] + route
        if route[-1] != depot:
            route.append(depot)

        customer_nodes = [n for n in route if n != depot and n not in charging_stations]
        if customer_nodes:
            cleaned.append(route)

    return cleaned


# === CONFIGURATION ===
INSTANCE_DIR = "instance_files"
OUTPUT_CSV = "evaluation_results.csv"

penalty_weights = {
    'missing_customers': 1e6,
    'battery_depletion': 1e4,
    'unreachable_node': 1e5,
    'unnecessary_recharges': 1000,
    'low_battery': 5000,
    'capacity_overload': 1e5,
    'max_travel_time_exceeded': 1e5,
    'invalid_route': 1e5,
    'vehicle_count': 1e4,
}

# === EVALUATION LOOP ===
results = []
instance_files = [f for f in os.listdir(INSTANCE_DIR) if f.endswith(".xml")]

for filename in sorted(instance_files):
    instance_path = os.path.join(INSTANCE_DIR, filename)
    instance_id = filename.replace(".xml", "")
    print(f"\n🔍 Evaluating instance: {instance_id}")

    # Parse instance
    (nodes, cs, depot, customers, cost_matrix, travel_time_matrix,
     E_max, _, vehicle_capacity, max_travel_time, requests) = parse_instance(instance_path)

    print(f"[INFO] Parsed {len(customers)} customers, {len(cs)} charging stations, depot: {depot}")
    recharge_amount = E_max

    start = time.time()

    # === Constructive Phase ===
    initial_routes = construct_initial_solution(
        nodes=nodes,
        depot=depot,
        customers=customers,
        cost_matrix=cost_matrix,
        vehicle_capacity=vehicle_capacity,
        E_max=E_max,
        requests=requests,
        charging_stations=cs
    )
    print(f"[INFO] Initial routes: {initial_routes}")
    for i, route in enumerate(initial_routes):
        print(f"[DEBUG] Initial Route {i + 1}: {route}")

    battery_routes = make_routes_battery_feasible(initial_routes, cost_matrix, E_max, cs, depot)
    battery_routes = post_merge_routes(battery_routes, cost_matrix, vehicle_capacity, E_max, cs, depot, requests)
    battery_routes = make_routes_battery_feasible(battery_routes, cost_matrix, E_max, cs, depot)
    battery_routes = sanitize_routes(battery_routes, depot, cs)

    print(f"[INFO] Battery-feasible, cleaned routes: {battery_routes}")

    # === Local Search ===
    optimized_routes = apply_local_search(
        battery_routes,
        cost_matrix=cost_matrix,
        travel_time_matrix=travel_time_matrix,
        E_max=E_max,
        charging_stations=cs,
        recharge_amount=recharge_amount,
        penalty_weights=penalty_weights,
        depot=depot,
        nodes=nodes,
        vehicle_capacity=vehicle_capacity,
        max_travel_time=max_travel_time,
        requests=requests
    )

    optimized_routes = filter_overloaded_routes(optimized_routes, vehicle_capacity, requests, depot, cs)

    served_customers = set(n for r in optimized_routes for n in r if n in customers)
    missing_customers = set(customers) - served_customers
    print(f"[DEBUG] Served customers: {served_customers}")
    if missing_customers:
        print(f"[WARNING] Missing customers: {missing_customers}")

    # === Evaluation ===
    total_fitness, battery_valid = fitness_function(
        optimized_routes,
        cost_matrix,
        travel_time_matrix,
        E_max,
        cs,
        recharge_amount,
        penalty_weights,
        depot,
        nodes,
        vehicle_capacity,
        max_travel_time,
        requests
    )

    end = time.time()
    runtime_sec = round(end - start, 2)

    total_distance = sum(
        sum(cost_matrix[(route[i], route[i + 1])] for i in range(len(route) - 1))
        for route in optimized_routes if len(route) > 1
    )
    num_cs_visits = sum(1 for route in optimized_routes for node in route if node in cs)

    results.append({
        "instance_id": instance_id,
        "num_customers": len(customers),
        "num_routes": len(optimized_routes),
        "total_distance": round(total_distance, 2),
        "fitness_score": round(total_fitness, 2),
        "is_feasible": battery_valid,
        "num_CS_visits": num_cs_visits,
        "runtime_sec": runtime_sec
    })

# === WRITE TO CSV ===
with open(OUTPUT_CSV, "w", newline="") as csvfile:
    fieldnames = list(results[0].keys())
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(results)

print("\n✅ Evaluation complete. Results saved to:", OUTPUT_CSV)
