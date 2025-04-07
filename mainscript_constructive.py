import random
from instance_parser import parse_instance
from constructive_solver import construct_initial_solution
from local_search import apply_local_search, plot_routes, route_cost
from ga_operators import fitness_function
from utils import make_routes_battery_feasible  # If it’s in utils.py

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

for i, route in enumerate(battery_routes):
    print(f"🔋 Battery-Aware Route {i+1}: {route}")

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
