import os
import csv
from instance_parser import parse_instance
from constructive_solver import construct_initial_solution, post_merge_routes
from utils import make_routes_battery_feasible
from local_search import apply_local_search, plot_routes, route_cost
from fitness import fitness_function
import matplotlib.pyplot as plt

# === CONFIG ===
INSTANCE_DIR = "instance_files"
RESULTS_FILE = "evaluation_results.csv"
TARGET_INSTANCES = ["C103-5.xml", "C101-10.xml", "C101-5.xml", "C104-10.xml", "R102-10.xml", "RC102-10.xml", "C103-15.xml"]
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

    # === Parse ===
    (nodes, charging_stations, depot, customers,
     cost_matrix, travel_time_matrix, E_max, _,
     vehicle_capacity, max_travel_time, requests) = parse_instance(filepath)

    # Patch self-loops
    for n in nodes:
        cost_matrix[(n, n)] = 0
        travel_time_matrix[(n, n)] = 0

    # === Initial Construction ===
    initial_routes = construct_initial_solution(
        nodes, depot, customers, cost_matrix, vehicle_capacity,
        E_max, requests, charging_stations
    )

    battery_routes = make_routes_battery_feasible(
        initial_routes, cost_matrix, E_max, charging_stations, depot
    )

    battery_routes = post_merge_routes(
        battery_routes, cost_matrix, vehicle_capacity, E_max,
        charging_stations, depot, requests
    )

    # Re-fix battery feasibility
    battery_routes = make_routes_battery_feasible(
        battery_routes, cost_matrix, E_max, charging_stations, depot
    )

    # === Local Search ===
    optimized_routes = apply_local_search(
        battery_routes,
        cost_matrix=cost_matrix,
        travel_time_matrix=travel_time_matrix,
        E_max=E_max,
        charging_stations=charging_stations,
        recharge_amount=E_max,
        penalty_weights=penalty_weights,
        depot=depot,
        nodes=nodes,
        vehicle_capacity=vehicle_capacity,
        max_travel_time=max_travel_time,
        requests=requests,
        customers=customers
    )

    from evrp_utils import sanitize_routes

    # Deduplicate and clean up
    optimized_routes = sanitize_routes(
        optimized_routes,
        depot=depot,
        charging_stations=charging_stations
    )

    # === Evaluate ===
    fitness, battery_ok = fitness_function(
        optimized_routes, cost_matrix, travel_time_matrix,
        E_max, charging_stations, E_max, penalty_weights,
        depot, nodes, vehicle_capacity, max_travel_time,
        requests, customers
    )

    print(f"\nFinal Evaluation for {filename}:")
    print(f"  Total Routes: {len(optimized_routes)}")
    print(f"  Fitness Score: {fitness}")
    print(f"  Battery Feasible: {'YES' if battery_ok else 'NO'}")

    for i, route in enumerate(optimized_routes):
        print(f"    Route {i+1}: {route} (Cost: {route_cost(route, cost_matrix)})")

    # Use filename as instance_id
    instance_id = filename.replace(".xml", "")

    plot_routes(
        optimized_routes,
        nodes=nodes,
        depot=depot,
        charging_stations=charging_stations,
        cost_matrix=cost_matrix,
        E_max=E_max,
        save_plot=True,
        method="CWS",
        instance_id=instance_id
    )

    # Ensure plot stays open until manually closed
    plt.show(block=True)

    save_result_to_csv(
        instance_name=filename,
        method="CWS",
        fitness=fitness,
        battery_feasible=battery_ok,
        route_count=len(optimized_routes),
        vehicle_count=len(optimized_routes),
        comment="CWS result from mainscript_constructive.py"
    )
