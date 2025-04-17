import time
from constructive_solver import construct_initial_solution, post_merge_routes
from utils import make_routes_battery_feasible
from local_search import apply_local_search, route_cost
from ga_operators import fitness_function
from evaluate_solver import filter_overloaded_routes  # If it's still in evaluate_solver.py, you can move it here
from mainscript_constructive import sanitize_routes

def run_pipeline(instance_data, penalty_weights, method="CWS", visualize=False, save_plot=False):
    """
    Runs the EVRP pipeline (construct → repair → local search → evaluate).
    Supports method='CWS' or 'GA'. Returns final routes and performance stats.
    """

    # === Unpack instance data ===
    (nodes, charging_stations, depot, customers, cost_matrix, travel_time_matrix,
     E_max, _, vehicle_capacity, max_travel_time, requests) = instance_data

    DEPOT = depot
    recharge_amount = E_max

    start = time.time()

    # === 1. Construct initial solution ===
    if method == "CWS":
        initial_routes = construct_initial_solution(
            nodes, depot, customers, cost_matrix, vehicle_capacity, E_max, requests, charging_stations
        )
    elif method == "GA":
        raise NotImplementedError("GA pipeline not yet connected. Will be added in next step.")
    else:
        raise ValueError(f"Unsupported method: {method}")

    # === 2. Battery repair and merge ===
    battery_routes = make_routes_battery_feasible(initial_routes, cost_matrix, E_max, charging_stations, depot)
    battery_routes = post_merge_routes(battery_routes, cost_matrix, vehicle_capacity, E_max, charging_stations, depot, requests)
    battery_routes = make_routes_battery_feasible(battery_routes, cost_matrix, E_max, charging_stations, depot)
    battery_routes = sanitize_routes(battery_routes, depot, charging_stations)

    # Remove routes that don’t serve customers
    battery_routes = [r for r in battery_routes if any(n in customers for n in r)]

    # Final cleanup (remove extra depots)
    for i in range(len(battery_routes)):
        while battery_routes[i].count(depot) > 2:
            battery_routes[i].remove(depot)

    # === 3. Local search ===
    optimized_routes = apply_local_search(
        battery_routes,
        cost_matrix=cost_matrix,
        travel_time_matrix=travel_time_matrix,
        E_max=E_max,
        charging_stations=charging_stations,
        recharge_amount=recharge_amount,
        penalty_weights=penalty_weights,
        depot=depot,
        nodes=nodes,
        vehicle_capacity=vehicle_capacity,
        max_travel_time=max_travel_time,
        requests=requests
    )

    # === 4. Filter and Evaluate ===
    optimized_routes = filter_overloaded_routes(optimized_routes, vehicle_capacity, requests, depot, charging_stations)

    # Fitness evaluation
    fitness_score, battery_valid = fitness_function(
        optimized_routes,
        cost_matrix,
        travel_time_matrix,
        E_max,
        charging_stations,
        recharge_amount,
        penalty_weights,
        depot,
        nodes,
        vehicle_capacity,
        max_travel_time,
        requests
    )

    # Optional visualization
    if visualize:
        from local_search import plot_routes
        plot_routes(optimized_routes, nodes, depot, save_plot=save_plot, instance_name=None)

    runtime = round(time.time() - start, 2)
    total_distance = sum(route_cost(r, cost_matrix) for r in optimized_routes if len(r) > 1)
    cs_count = sum(1 for r in optimized_routes for n in r if n in charging_stations)

    return optimized_routes, {
        'fitness_score': round(fitness_score, 2),
        'is_feasible': battery_valid,
        'num_routes': len(optimized_routes),
        'total_distance': round(total_distance, 2),
        'num_CS_visits': cs_count,
        'runtime_sec': runtime
    }
