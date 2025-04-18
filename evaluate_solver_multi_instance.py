
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
instance_files = [f for f in os.listdir(INSTANCE_DIR) if f.endswith(".xml")]

for filename in sorted(instance_files):
    instance_path = os.path.join(INSTANCE_DIR, filename)
    instance_id = filename.replace(".xml", "")
    print(f"\n🔍 Evaluating instance: {instance_id}")

    # Parse instance
    (nodes, cs, depot, customers, cost_matrix, travel_time_matrix,
     E_max, _, vehicle_capacity, max_travel_time, requests) = parse_instance(instance_path)

    instance_data = (nodes, cs, depot, customers, cost_matrix, travel_time_matrix,
                     E_max, _, vehicle_capacity, max_travel_time, requests)

    # === CWS Evaluation ===
    try:
        routes, stats = run_pipeline(
            instance_data,
            penalty_weights,
            method="CWS",
            visualize=True,
            instance_id=instance_id
        )
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
    print("\n⚠️ No evaluation results to save.")
