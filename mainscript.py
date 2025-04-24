import os
import csv
import random
from instance_parser import parse_instance
from heuristics import (
    heuristic_population_initialization,
    generate_giant_tour_and_split,
    repair_route_battery_feasibility
)
from ga_operators import order_crossover_evrp, mutate_route
from validation import (
    validate_solution,
    validate_no_duplicates_route,
    ensure_all_customers_present,
    validate_and_finalize_routes
)
from local_search import apply_local_search, plot_routes, route_cost
from utils import make_routes_battery_feasible
from route_utils import sanitize_routes
from fitness import fitness_function
import matplotlib.pyplot as plt

# === CONFIG ===
INSTANCE_DIR = "instance_files"
RESULTS_FILE = "evaluation_results.csv"
TARGET_INSTANCES = [
    "C101-10.xml", "C101-5.xml", "C104-10.xml",
    "R102-10.xml", "RC102-10.xml", "C103-15.xml"
]

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

def save_result_to_csv(instance_name, method, fitness, battery_feasible, route_count, vehicle_count, comment=""):
    file_exists = os.path.isfile(RESULTS_FILE)
    with open(RESULTS_FILE, mode='a', newline='') as file:
        writer = csv.writer(file)
        if not file_exists:
            writer.writerow(["Instance", "Method", "Fitness", "Battery Feasible", "Route Count", "Vehicle Count", "Comments"])
        writer.writerow([
            instance_name,
            method,
            f"{fitness:.2f}",
            "YES" if battery_feasible else "NO",
            route_count,
            vehicle_count,
            comment
        ])

# === RUN FOR EACH INSTANCE ===
for filename in TARGET_INSTANCES:
    print(f"\nProcessing: {filename}")
    filepath = os.path.join(INSTANCE_DIR, filename)

    # === PARSE INSTANCE ===
    (nodes, charging_stations, depot, customers, cost_matrix, travel_time_matrix,
     battery_capacity, num_vehicles_instance, vehicle_capacity, max_travel_time, requests) = parse_instance(filepath)

    DEPOT = depot
    E_max = battery_capacity
    recharge_amount = E_max
    num_generations = 30
    population_size = 10

    # === INITIAL POPULATION ===
    print("\nInitializing GA Population...")
    base_solution = generate_giant_tour_and_split(
        DEPOT, list(customers), nodes, cost_matrix, travel_time_matrix,
        requests, vehicle_capacity, max_travel_time, E_max
    )

    repaired_base = [repair_route_battery_feasibility(route, cost_matrix, E_max, recharge_amount,
                                                      charging_stations, DEPOT, nodes)
                     for route in base_solution]

    population = [repaired_base]
    additional_population = heuristic_population_initialization(
        population_size - 1, nodes, cost_matrix, travel_time_matrix, DEPOT,
        E_max, recharge_amount, charging_stations, vehicle_capacity,
        max_travel_time, requests, num_vehicles_instance
    )
    population.extend(additional_population)

    # === GA LOOP ===
    for generation in range(num_generations):
        print(f"\nGeneration {generation+1}...")
        new_population = []
        for i in range(population_size // 2):
            parent1, parent2 = random.sample(population, 2)
            child = order_crossover_evrp(parent1, parent2, cost_matrix, E_max, charging_stations, recharge_amount, DEPOT)
            child = mutate_route(child)
            repaired_child = make_routes_battery_feasible(child, cost_matrix, E_max, charging_stations, DEPOT)
            repaired_child = sanitize_routes(repaired_child, DEPOT, charging_stations)
            new_population.append(repaired_child)

        population = sorted(new_population + population, key=lambda sol: fitness_function(
            sol, cost_matrix, travel_time_matrix, E_max, charging_stations, recharge_amount, penalty_weights,
            DEPOT, nodes, vehicle_capacity, max_travel_time, requests, customers
        )[0])[:population_size]

    # === BEST SOLUTION ===
    best_solution = population[0]
    final_fitness, battery_ok = fitness_function(
        best_solution, cost_matrix, travel_time_matrix, E_max, charging_stations, recharge_amount,
        penalty_weights, DEPOT, nodes, vehicle_capacity, max_travel_time, requests, customers
    )

    print(f"\nFinal Evaluation for {filename}:")
    print(f"  Total Routes: {len(best_solution)}")
    print(f"  Fitness Score: {final_fitness:.2f}")
    print(f"  Battery Feasible: {'YES' if battery_ok else 'NO'}")

    for i, route in enumerate(best_solution):
        print(f"    Route {i+1}: {route} (Cost: {route_cost(route, cost_matrix)})")

    instance_id = filename.replace(".xml", "")
    plot_routes(
        best_solution,
        nodes=nodes,
        depot=depot,
        charging_stations=charging_stations,
        cost_matrix=cost_matrix,
        E_max=E_max,
        save_plot=False,
        method="GA",
        instance_id=instance_id
    )
    plt.show(block=True)

    save_result_to_csv(
        instance_name=filename,
        method="GA",
        fitness=final_fitness,
        battery_feasible=battery_ok,
        route_count=len(best_solution),
        vehicle_count=len(best_solution),
        comment="GA result from mainscript.py"
    )
