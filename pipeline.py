from constructive_solver import construct_initial_solution, post_merge_routes
#from mainscript_constructive import battery_routes
from utils import make_routes_battery_feasible
from local_search import apply_local_search, route_cost
from route_utils import sanitize_routes
from evrp_utils import sanitize_routes, filter_overloaded_routes
import time
from heuristics import heuristic_population_initialization
from validation import validate_and_finalize_routes
from ga_operators import genetic_algorithm
from fitness import fitness_function
from local_search import route_cost
from local_search import plot_routes
from ga_operators import remove_trivial_routes
from constructive_solver import construct_initial_solution
from utils import make_routes_battery_feasible
from local_search import apply_local_search, route_cost, plot_routes
from heuristics import heuristic_population_initialization
from validation import validate_and_finalize_routes
import time

def run_pipeline(instance_data, penalty_weights, method="CWS", visualize=False):
    (
        nodes, charging_stations, depot, customers, cost_matrix, travel_time_matrix,
        E_max, _, vehicle_capacity, max_travel_time, requests
    ) = instance_data

    print(f"\n🛠️  Running {method} pipeline...")

    # === Step 1: Construct Initial Routes ===
    initial_routes = construct_initial_solution(
        nodes=nodes,
        depot=depot,
        customers=customers,
        cost_matrix=cost_matrix,
        vehicle_capacity=vehicle_capacity,
        E_max=E_max,
        requests=requests,
        charging_stations=charging_stations
    )
    print(f"[INFO] Initial constructed routes: {initial_routes}")

    # === Step 2: Make Battery Feasible ===
    battery_routes = make_routes_battery_feasible(
        initial_routes,
        cost_matrix,
        E_max,
        charging_stations,
        depot
    )
    print(f"[INFO] After battery repair: {battery_routes}")

    # === Step 3: Local Search Optimization ===
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
    print(f"[INFO] After local search: {optimized_routes}")

    # === Step 4: Final Battery Repair ===
    battery_routes = make_routes_battery_feasible(
        optimized_routes,
        cost_matrix,
        E_max,
        charging_stations,
        depot
    )
    print(f"[INFO] Final battery-feasible routes: {battery_routes}")

    # === Step 5: Evaluation ===
    fitness, battery_feasible = fitness_function(
        battery_routes,
        cost_matrix,
        travel_time_matrix,
        E_max,
        charging_stations,
        E_max,
        penalty_weights,
        depot,
        nodes,
        vehicle_capacity,
        max_travel_time,
        requests,
        customers
    )

    print(f"\n📊 Final Fitness: {fitness:.2f}")
    print(f"🔋 Battery Feasible: {'✅ Yes' if battery_feasible else '❌ No'}")

    return battery_routes, {
        "fitness": fitness,
        "battery_feasible": battery_feasible
    }


def run_ga_pipeline(instance_data, penalty_weights, ga_config, visualize=False, instance_id=""):


    """
    Wrapper to run the GA-based EVRP solver and return comparable output for evaluation.
    """

    # === Unpack instance ===
    (nodes, charging_stations, depot, customers, cost_matrix, travel_time_matrix,
     E_max, _, vehicle_capacity, max_travel_time, requests) = instance_data

    try:
        warm_routes, _ = run_pipeline(
            instance_data,
            penalty_weights,
            method="CWS",
            visualize=False,

        )
    except Exception as e:
        print(f"[WARNING] Failed to generate warm-start CWS route: {e}")
        warm_routes = []

    DEPOT = depot
    recharge_amount = E_max

    start_time = time.time()

    # === 1. Build initial population ===
    population = heuristic_population_initialization(
        nodes=nodes,
        depot=depot,
        vehicle_capacity=vehicle_capacity,
        cost_matrix=cost_matrix,
        travel_time_matrix=travel_time_matrix,
        E_max=E_max,
        charging_stations=charging_stations,
        recharge_amount=recharge_amount,
        requests=requests,
        num_vehicles=ga_config.get("num_vehicles", 3),
        population_size=ga_config.get("population_size", 30),
        max_travel_time=max_travel_time,
        initial_routes=warm_routes
    )

    # === 2. Run Genetic Algorithm ===
    best_solution = genetic_algorithm(
        population=population,
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
        requests=requests,
        customers=customers,
        num_generations=ga_config.get("num_generations", 50),
        population_size=ga_config.get("population_size", 30),
        mutation_rate=ga_config.get("mutation_rate", 0.2),
        crossover_rate=ga_config.get("crossover_rate", 0.8),
        elite_fraction=ga_config.get("elite_fraction", 0.1),
        verbose=ga_config.get("verbose", False)
    )

    # === 3. Final cleanup of best solution ===
    best_routes = validate_and_finalize_routes(
        best_solution,
        cost_matrix,
        E_max,
        recharge_amount,
        charging_stations,
        depot,
        nodes
    )
    best_solution = remove_trivial_routes(best_solution, depot, charging_stations)

    # === 4. Fitness Evaluation ===
    fitness_score, battery_valid = fitness_function(
        battery_routes,
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

    total_distance = sum(route_cost(route, cost_matrix) for route in best_routes if len(route) > 1)
    cs_visits = sum(1 for route in best_routes for n in route if n in charging_stations)
    runtime = round(time.time() - start_time, 2)


    if best_routes and any(len(r) > 1 for r in best_routes):
        plot_routes(
            routes=best_routes,
            nodes=nodes,
            depot=depot,
            cost_matrix=cost_matrix,
            E_max=E_max,
            save_plot=True,
            method="GA",
            instance_id=instance_id,
            charging_stations=charging_stations
        )

    else:
        print("[WARNING] No valid GA routes to plot.")


    # Optional: visualize best solution

    if visualize:
        print(f"[DEBUG] Plotting GA final solution for {instance_id}")
        plot_routes(
            best_routes,
            nodes=nodes,
            depot=depot,
            charging_stations=charging_stations,
            method="GA",
            save_plot=True,
            instance_id=instance_id,
            E_max=E_max,
            cost_matrix=cost_matrix
        )

    return best_routes, {
        'fitness_score': round(fitness_score, 2),
        'is_feasible': battery_valid,
        'num_routes': len(best_routes),
        'total_distance': round(total_distance, 2),
        'num_CS_visits': cs_visits,
        'runtime_sec': runtime
    }

