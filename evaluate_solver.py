import os
import time
import csv

from instance_parser import parse_instance
from constructive_solver import construct_initial_solution, post_merge_routes
from local_search import apply_local_search
from ga_operators import fitness_function
from utils import make_routes_battery_feasible
from pipeline import run_pipeline, run_ga_pipeline
from evrp_utils import sanitize_routes,filter_overloaded_routes



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

    instance_data = (nodes, cs, depot, customers, cost_matrix, travel_time_matrix,
                     E_max, _, vehicle_capacity, max_travel_time, requests)

    # === CWS EVALUATION ===
    routes, stats = run_pipeline(instance_data, penalty_weights, method="CWS", visualize=True, instance_id=instance_id)
    stats['instance_id'] = instance_id
    stats['method'] = "CWS"
    stats['num_customers'] = len(customers)
    results.append(stats)

    # === GA EVALUATION ===
    ga_config = {
        "num_generations": 50,
        "population_size": 30,
        "mutation_rate": 0.2,
        "crossover_rate": 0.8,
        "elite_fraction": 0.1,
        "verbose": False
    }
    ga_routes, ga_stats = run_ga_pipeline(instance_data, penalty_weights, ga_config, visualize=True, instance_id=instance_id)
    ga_stats['instance_id'] = instance_id
    ga_stats['method'] = "GA"
    ga_stats['num_customers'] = len(customers)
    results.append(ga_stats)



# === WRITE TO CSV ===
with open(OUTPUT_CSV, "w", newline="") as csvfile:
    fieldnames = list(results[0].keys())
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(results)

print("\n✅ Evaluation complete. Results saved to:", OUTPUT_CSV)
