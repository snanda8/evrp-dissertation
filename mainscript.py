import random
from instance_parser import parse_instance
from utils import *
from heuristics import heuristic_population_initialization
from ga_operators import fitness_function, order_crossover_evrp, mutate_route

# --- Global Setup: Parse Instance ---
instance_file = "instance_files/C101-10.xml"  # Adjust path as needed
(nodes, charging_stations, depot, customers, cost_matrix, travel_time_matrix,
 battery_capacity, num_vehicles_instance, vehicle_capacity, max_travel_time, requests) = parse_instance(instance_file)

# Define global variables based on the instance
DEPOT = depot
num_vehicles = num_vehicles_instance
E_max = battery_capacity
recharge_amount = 30  # Adjust as needed

penalty_weights = {
    'missing_customers': 1e6,
    'battery_depletion': 1e5,
    'unreachable_node': 1e5,
    'unnecessary_recharges': 1000,
    'low_battery': 5000,
    'capacity_overload': 1e5,
    'max_travel_time_exceeded': 1e5,
    'invalid_route': 1e5,
}

# --- Initialize Population ---
population_size = 10  # Number of solutions per generation
population = heuristic_population_initialization(
    population_size, nodes, cost_matrix, travel_time_matrix, DEPOT,
    E_max, recharge_amount, charging_stations, vehicle_capacity,
    max_travel_time, requests, num_vehicles
)
print("Heuristic-based Initial Population:")
for sol in population:
    print(sol)

# --- GA Loop ---
num_generations = 100  # Number of generations
for generation in range(num_generations):
    print(f"\n=== Generation {generation + 1} ===")
    evaluated_population = []
    for individual in population:
        total_fitness, valid = fitness_function(
            individual, cost_matrix, travel_time_matrix, E_max, charging_stations,
            recharge_amount, penalty_weights, DEPOT, nodes, vehicle_capacity,
            max_travel_time, requests
        )
        evaluated_population.append((individual, total_fitness, valid))

    valid_population = [ind for ind in evaluated_population if ind[2]]
    valid_population.sort(key=lambda x: x[1])
    selected_parents = [ind[0] for ind in valid_population[:max(1, population_size // 2)]]

    if len(selected_parents) < 2:
        print("Not enough valid individuals; falling back to full evaluated population.")
        selected_parents = [ind[0] for ind in evaluated_population]
    if len(selected_parents) < 2:
        print("Still fewer than 2 parents; duplicating available parent.")
        selected_parents = selected_parents * 2

    children = []
    while len(children) < population_size - len(selected_parents):
        p1, p2 = random.sample(selected_parents, 2)
        child = order_crossover_evrp(p1, p2, cost_matrix, E_max, charging_stations, recharge_amount, DEPOT)
        child = mutate_route(child, mutation_rate=0.4)
        children.append(child)

    population = selected_parents + children
    print(f"Population size at end of generation: {len(population)}")
