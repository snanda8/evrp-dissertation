import os
import time
import csv

from instance_parser import parse_instance
from constructive_solver import construct_initial_solution, post_merge_routes
from local_search import apply_local_search
from ga_operators import fitness_function
from utils import make_routes_battery_feasible

# === CONFIGURATION ===
INSTANCE_DIR = "instance_files"
OUTPUT_CSV = "evaluation_results.csv"

# Penalty settings
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

    recharge_amount = E_max

    start = time.time()

    # Construct solution
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

    battery_routes = make_routes_battery_feasible(initial_routes, cost_matrix, E_max, cs, depot)
    battery_routes = post_merge_routes(battery_routes, cost_matrix, vehicle_capacity, E_max, cs, depot, requests)
    battery_routes = make_routes_battery_feasible(battery_routes, cost_matrix, E_max, cs, depot)

    # Local search + evaluation
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
        sum(cost_matrix[(route[i], route[i+1])] for i in range(len(route)-1))
        for route in optimized_routes if len(route) > 1
    )

    num_cs_visits = sum(1 for route in optimized_routes for node in route if node in cs)
    num_customers = len(customers)

    results.append({
        "instance_id": instance_id,
        "num_customers": num_customers,
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
