from instance_parser import parse_instance
from constructive_solver import construct_initial_solution
from local_search import apply_local_search, plot_routes, route_cost
from ga_operators import fitness_function
from utils import make_routes_battery_feasible  # If it’s in utils.py
from constructive_solver import post_merge_routes

# === PARSE INSTANCE ===
instance_file = "instance_files/C101-10.xml"
(nodes, charging_stations, depot, customers, cost_matrix, travel_time_matrix,
 battery_capacity, num_vehicles, vehicle_capacity, max_travel_time, requests) = parse_instance(instance_file)

DEPOT = depot
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

# === CONSTRUCTIVE SOLVER ===
print("\n🔧 Constructing Initial Solution using CWS...\n")
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

# === BATTERY-AWARE CONVERSION ===
print("🔍 Before Battery-Aware Conversion")
battery_routes = make_routes_battery_feasible(
    initial_routes, cost_matrix, E_max, charging_stations, DEPOT
)
print("🔍 After Battery-Aware Conversion")



battery_routes = post_merge_routes(
    battery_routes, cost_matrix, vehicle_capacity, E_max,
    charging_stations, DEPOT, requests
)

# Re-apply battery fixing to merged routes
battery_routes = make_routes_battery_feasible(
    battery_routes, cost_matrix, E_max, charging_stations, DEPOT
)

# clean-up to remove [15, 15] or duplicate depot nodes
battery_routes = [r for r in battery_routes if len(set(r)) > 2 and r != [DEPOT, DEPOT]]
for route in battery_routes:
    while route.count(DEPOT) > 2:
        route.remove(DEPOT)


for i, route in enumerate(battery_routes):
    print(f"🔁 Post-Merged Battery Route {i+1}: {route}")

def sanitize_routes(routes, depot, charging_stations):
    cleaned = []

    for route in routes:
        # Remove immediate consecutive duplicates (e.g. [15, 15, ...])
        route = [n for i, n in enumerate(route) if i == 0 or n != route[i - 1]]

        # Remove trailing duplicate depot (e.g. [..., 15, 15])
        while len(route) >= 2 and route[-1] == depot and route[-2] == depot:
            route.pop()

        # Ensure route starts and ends with depot
        if route[0] != depot:
            route = [depot] + route
        if route[-1] != depot:
            route.append(depot)

        # Route must contain at least one customer (not CS or depot)
        customer_nodes = [n for n in route if n != depot and n not in charging_stations]
        if customer_nodes:
            cleaned.append(route)

    return cleaned

battery_routes = sanitize_routes(battery_routes, DEPOT, charging_stations)


# === LOCAL SEARCH ===
optimized_routes = apply_local_search(battery_routes, cost_matrix)

# === VISUALIZE ===
plot_routes(optimized_routes, nodes, DEPOT)

# === FITNESS EVALUATION ===
print("🔍 Before Fitness Evaluation")
total_fitness, is_battery_valid = fitness_function(
    optimized_routes, cost_matrix, travel_time_matrix, E_max, charging_stations,
    recharge_amount, penalty_weights, DEPOT, nodes, vehicle_capacity,
    max_travel_time, requests
)
print("🔍 After Fitness Evaluation")

print("\n📊 Final Evaluation of Optimized Routes:")
print(f"Total Routes: {len(optimized_routes)}")
print(f"Fitness Score: {total_fitness}")
print(f"Battery Feasible: {'✅ Yes' if is_battery_valid else '❌ No'}")

for i, route in enumerate(optimized_routes):
    print(f"✅ Route {i+1} | Cost: {route_cost(route, cost_matrix)} | Nodes: {route}")
