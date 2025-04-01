import random
from instance_parser import parse_instance
from utils import *
from heuristics import heuristic_population_initialization, generate_giant_tour_and_split, repair_route_battery_feasibility
from ga_operators import fitness_function, order_crossover_evrp, mutate_route
from validation import validate_solution, validate_no_duplicates_route, ensure_all_customers_present
from merging import merge_routes

# --- Global Setup: Parse Instance ---
instance_file = "instance_files/C101-10.xml"  # Adjust path as needed
(nodes, charging_stations, depot, customers, cost_matrix, travel_time_matrix,
 battery_capacity, num_vehicles_instance, vehicle_capacity, max_travel_time, requests) = parse_instance(instance_file)

print(f"cost_matrix:{cost_matrix}")

# Define global variables based on the instance
DEPOT = depot
num_vehicles = num_vehicles_instance
E_max = battery_capacity
recharge_amount = E_max  # Adjust as needed

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

# --- New Giant Tour Splitting Initialization ---
# Generate a giant tour splitting initial solution using the new functions.
giant_solution = generate_giant_tour_and_split(
    DEPOT,
    list(customers),
    nodes,
    cost_matrix,
    travel_time_matrix,
    requests,
    vehicle_capacity,
    max_travel_time,
    E_max
)
print("Giant Tour Splitting Initial Solution:")
print(giant_solution)

# Repair the giant solution to improve battery feasibility.
repaired_solution = []
for route in giant_solution:
    repaired = repair_route_battery_feasibility(route, cost_matrix, E_max, recharge_amount, charging_stations, DEPOT, nodes)
    repaired_solution.append(repaired)
print("Repaired Giant Tour Solution:")
print(repaired_solution)

# Initialize population with the repaired giant solution and additional heuristic solutions.
population_size = 10  # Number of solutions per generation
population = [repaired_solution]

additional_population = heuristic_population_initialization(
    population_size, nodes, cost_matrix, travel_time_matrix, DEPOT,
    E_max, recharge_amount, charging_stations, vehicle_capacity,
    max_travel_time, requests, num_vehicles)
population.extend(additional_population)
print("Combined Initial Population:")
for sol in population:
    print(sol)

assigned_customers = set()
for route in giant_solution:
    for node in route:
        if node in customers:
            assigned_customers.add(node)

missing_customers = customers - assigned_customers
print(f"Assigned Customers: {assigned_customers}")
if missing_customers:
    print(f"ERROR: Missing Customers in Initial Routes: {missing_customers}")

print(f"\n[DEBUG] Full Customers Set: {customers}")
print(f"[DEBUG] Charging Stations Set: {charging_stations}")

# Check if any charging stations were mistakenly included in customers
overlap = customers.intersection(charging_stations)
if overlap:
    print(f" [DEBUG] ERROR: Charging stations are incorrectly in the customers list: {overlap}")



# --- GA Loop ---
num_generations = 10  # Number of generations

for generation in range(num_generations):
    print(f"\n=== Generation {generation + 1} ===")

    evaluated_population = []

    # 🔍 Evaluate current population
    for individual in population:
        # 🔧 Repair every route in the individual before validation/fitness
        repaired_individual = []
        for route in individual:
            repaired = repair_route_battery_feasibility(
                route, cost_matrix, E_max, recharge_amount, charging_stations, DEPOT, nodes
            )
            repaired_individual.append(repaired)

        # ✅ Use repaired version for validation and fitness
        valid_solution = validate_solution(
            solution=repaired_individual,
            depot=DEPOT,
            requests=requests,
            expected_customers=customers,
            charging_stations=charging_stations
        )

        if not valid_solution:
            print("Solution validation failed (duplicate or missing customers).")

        total_fitness, is_battery_valid = fitness_function(
            repaired_individual, cost_matrix, travel_time_matrix, E_max, charging_stations,
            recharge_amount, penalty_weights, DEPOT, nodes, vehicle_capacity,
            max_travel_time, requests
        )

        overall_valid = valid_solution and is_battery_valid
        evaluated_population.append((individual, total_fitness, overall_valid))

    # 📉 Select valid individuals
    valid_population = [ind for ind in evaluated_population if ind[2]]
    valid_population.sort(key=lambda x: x[1])

    selected_parents = [ind[0] for ind in valid_population[:max(1, population_size // 2)]]

    if len(selected_parents) < 2:
        print("Not enough valid individuals; falling back to full evaluated population.")
        selected_parents = [ind[0] for ind in evaluated_population]

    if len(selected_parents) < 2:
        print("Still fewer than 2 parents; duplicating available parent.")
        selected_parents = selected_parents * 2

    # 👶 Generate children
    children = []
    while len(children) < population_size - len(selected_parents):
        p1, p2 = random.sample(selected_parents, 2)
        child = order_crossover_evrp(
            p1, p2, cost_matrix, E_max, charging_stations, recharge_amount, DEPOT
        )
        child = mutate_route(child, mutation_rate=0.4)

        # 🚑 Repair child routes for battery feasibility
        repaired_child = []
        for route in child:
            repaired = repair_route_battery_feasibility(
                route, cost_matrix, E_max, recharge_amount, charging_stations, DEPOT, nodes
            )
            repaired_child.append(repaired)

        # 🧩 Ensure all customers are present
        repaired_child = ensure_all_customers_present(
            repaired_child, customers, DEPOT, cost_matrix, nodes, charging_stations, E_max
        )

        children.append(repaired_child)

    # 👥 Update population
    population = selected_parents + children
    print(f"Population size at end of generation: {len(population)}")
