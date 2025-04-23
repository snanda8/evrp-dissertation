
import os
import time
import csv

from instance_parser import parse_instance
from pipeline import run_pipeline, run_ga_pipeline

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

ga_config = {
    "num_generations": 50,
    "population_size": 30,
    "mutation_rate": 0.2,
    "crossover_rate": 0.8,
    "elite_fraction": 0.1,
    "verbose": False,
    "num_vehicles": 3
}

# === EVALUATION LOOP ===
results = []

# Target specific instances for debugging/plot verification
target_instances = ["C101-10.xml", "C101-5.xml", "C103-5.xml"]


# Get all .xml instance files in directory
all_instance_files = [f for f in os.listdir(INSTANCE_DIR) if f.endswith(".xml")]

# Filter only those we're targeting
instance_files = [f for f in all_instance_files if f in target_instances]

print(f"[INFO] Found {len(all_instance_files)} total instance files.")
print(f"[INFO] Evaluating {len(instance_files)} filtered instances: {instance_files}")

# Main evaluation loop
for filename in sorted(instance_files):
    print(f"\n🔍 Attempting to evaluate instance: {filename}")
    instance_path = os.path.join(INSTANCE_DIR, filename)
    instance_id = filename.replace(".xml", "")
    print(f"🔍 Evaluating instance: {instance_id}")

    # === Parse the instance ===
    (nodes, cs, depot, customers, cost_matrix, travel_time_matrix,
     E_max, _, vehicle_capacity, max_travel_time, requests) = parse_instance(instance_path)
    print(f"[DEBUG] Parsed instance '{instance_id}' → Depot: {depot}, Customers: {len(customers)}, CS: {len(cs)}")

    print(f"[DEBUG] Instance: {instance_id}")
    print(f"  Depot: {depot}")
    print(f"  Customers: {sorted(customers)}")
    print(f"  Charging Stations: {sorted(cs)}")
    print(f"  All nodes: {sorted(nodes.keys())}")

    instance_data = (
        nodes, cs, depot, customers, cost_matrix, travel_time_matrix,
        E_max, _, vehicle_capacity, max_travel_time, requests
    )

    # === CWS Evaluation ===
    try:
        print(f"\n🔧 Running CWS for: {instance_id}")
        routes, stats = run_pipeline(
            instance_data,
            penalty_weights,
            method="CWS",
            visualize=False # handle plotting explicitly
        )
        print(f"[SUCCESS] CWS completed for {instance_id}. Generated {len(routes)} routes.")

        battery_feasible = stats.get("battery_feasible", False)

        if not routes or all(len(r) <= 2 for r in routes):  # Possibly filtered to nothing
            print(f"[WARNING] No feasible or useful routes returned for {instance_id}, using fallback for plotting...")
            # Fallback to reparsed route for plotting baseline
            from constructive_solver import construct_initial_solution

            (nodes, cs, depot, customers, cost_matrix, travel_time_matrix,
             E_max, _, vehicle_capacity, max_travel_time, requests) = instance_data

            fallback_routes = construct_initial_solution(
                nodes, depot, customers, cost_matrix, vehicle_capacity, E_max, requests, cs
            )
            routes = fallback_routes

        from local_search import plot_routes

        plot_routes(routes, instance_data[0], instance_data[2], instance_id=instance_id, method="CWS")

        stats['instance_id'] = instance_id
        stats['method'] = "CWS"
        stats['num_customers'] = len(customers)
        results.append(stats)

    except Exception as e:
        print(f"[ERROR] CWS failed on {instance_id}: {e}")

    # === GA Evaluation ===

try:
    ga_routes, ga_stats = run_ga_pipeline(
        instance_data,
        penalty_weights,
        ga_config,
        visualize=True,
        instance_id=instance_id
    )
    ga_stats['instance_id'] = instance_id
    ga_stats['method'] = "GA"
    ga_stats['num_customers'] = len(customers)
    results.append(ga_stats)
except Exception as e:
    print(f"[ERROR] GA failed on {instance_id}: {e}")

# === WRITE TO CSV ===
if results:
    with open(OUTPUT_CSV, "w", newline="") as csvfile:
        fieldnames = list(results[0].keys())
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    print("\n✅ Evaluation complete. Results saved to:", OUTPUT_CSV)
else:
    print("\n️ No evaluation results to save.")
