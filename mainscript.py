import random
from instance_parser import parse_instance
from heuristics import (
    heuristic_population_initialization,
    generate_giant_tour_and_split,
    repair_route_battery_feasibility
)
from ga_operators import fitness_function, order_crossover_evrp, mutate_route
from validation import (
    validate_solution,
    validate_no_duplicates_route,
    ensure_all_customers_present,
    validate_and_finalize_routes
)
from constructive_solver import construct_initial_solution
from local_search import apply_local_search, plot_routes, route_cost
from utils import make_routes_battery_feasible


# === CONFIGURATION ===
instance_file = "instance_files/C101-10.xml"

# === PARSE INSTANCE ===
(nodes, charging_stations, depot, customers, cost_matrix, travel_time_matrix,
 battery_capacity, num_vehicles_instance, vehicle_capacity, max_travel_time, requests) = parse_instance(instance_file)

DEPOT = depot
num_vehicles = num_vehicles_instance
E_max = battery_capacity
recharge_amount = E_max

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

# === CONSTRUCTIVE SOLVER TEST ===
print("\n\U0001F527 Testing Constructive Initial Solution...\n")
initial_routes = construct_initial_solution(
    nodes=nodes,
    depot=DEPOT,
    customers=customers,
    cost_matrix=cost_matrix,
    vehicle_capacity=vehicle_capacity,
    E_max=E_max,
    requests=requests,
    charging_stations=charging_stations
)

for i, route in enumerate(initial_routes):
    print(f"🚚 Vehicle {i+1}: {route}")

battery_aware_routes = make_routes_battery_feasible(
    initial_routes, cost_matrix, E_max, charging_stations, DEPOT
)

# Print updated routes
for i, route in enumerate(battery_aware_routes):
    print(f"🔋 Battery-Aware Route {i+1}: {route}")


# === LOCAL SEARCH IMPROVEMENT ===
try:
    optimized_routes = apply_local_search(
        battery_aware_routes,
        cost_matrix=cost_matrix,
        travel_time_matrix=travel_time_matrix,
        E_max=E_max,
        charging_stations=charging_stations,
        recharge_amount=recharge_amount,
        penalty_weights=penalty_weights,
        depot=DEPOT,
        nodes=nodes,
        vehicle_capacity=vehicle_capacity,
        max_travel_time=max_travel_time,
        requests=requests
    )
except Exception as e:
    print(f"[ERROR] Local search failed: {e}")
    optimized_routes = battery_aware_routes

plot_routes(optimized_routes, nodes, DEPOT)

for i, route in enumerate(optimized_routes):
    print(f"Route {i+1}: {route} — Cost: {route_cost(route, cost_matrix)}")

# === FITNESS EVALUATION ===
# === FITNESS EVALUATION OF BATTERY-AWARE ROUTES ===
print("\n🧪 Fitness of Battery-Aware Routes:")

total_fitness, is_battery_valid = fitness_function(
    battery_aware_routes, cost_matrix, travel_time_matrix, E_max, charging_stations,
    recharge_amount, penalty_weights, DEPOT, nodes, vehicle_capacity,
    max_travel_time, requests
)

print(f"⚙️  Fitness Score: {total_fitness:.2f}")
print(f"🔋 Battery Feasible: {'✅ YES' if is_battery_valid else '❌ NO'}")
print(f"🚚 Vehicles Used: {len(battery_aware_routes)}")

total_distance = sum(route_cost(route, cost_matrix) for route in battery_aware_routes)
print(f"📏 Total Distance: {total_distance:.2f}")


# === GA COMPONENTS  ===
giant_solution = generate_giant_tour_and_split(
    DEPOT, list(customers), nodes, cost_matrix, travel_time_matrix,
    requests, vehicle_capacity, max_travel_time, E_max
)

repaired_solution = [repair_route_battery_feasibility(route, cost_matrix, E_max, recharge_amount,
                                                      charging_stations, DEPOT, nodes) for route in giant_solution]

population_size = 10
population = [repaired_solution]
additional_population = heuristic_population_initialization(
    population_size, nodes, cost_matrix, travel_time_matrix, DEPOT,
    E_max, recharge_amount, charging_stations, vehicle_capacity,
    max_travel_time, requests, num_vehicles
)
population.extend(additional_population)

# === DEBUG VALIDATION ===
assigned_customers = {node for route in giant_solution for node in route if node in customers}
missing_customers = customers - assigned_customers
print(f"Assigned Customers: {assigned_customers}")
if missing_customers:
    print(f"ERROR: Missing Customers in Initial Routes: {missing_customers}")

overlap = customers.intersection(charging_stations)
if overlap:
    print(f" [DEBUG] ERROR: Charging stations are incorrectly in the customers list: {overlap}")


