import random
from local_search import apply_local_search
from validation import validate_and_finalize_routes, ensure_all_customers_present, validate_solution
from fitness import fitness_function
print("📦 ga_operators.py loaded successfully")


def remove_trivial_routes(routes, depot, charging_stations):
    """
    Removes routes with only one customer (ignoring depots and CS).
    """
    cleaned = []
    for route in routes:
        customer_count = sum(1 for n in route if n not in charging_stations and n != depot)
        if customer_count > 1:
            cleaned.append(route)
        else:
            print(f"[FILTER] Removing trivial route: {route}")
    return cleaned




def order_crossover_evrp(parent1, parent2, cost_matrix, E_max, charging_stations, recharge_amount, depot):
    child = []
    for r1, r2 in zip(parent1, parent2):
        chosen_route = r1.copy() if random.random() < 0.5 else r2.copy()
        child.append(chosen_route)
    for route in child:
        if route[0] != depot:
            route.insert(0, depot)
        if route[-1] != depot:
            route.append(depot)
    return child

def mutate_route(solution, mutation_rate=0.2):
    mutated_solution = []
    for route in solution:
        if random.random() < mutation_rate and len(route) > 3:
            indices = list(range(1, len(route) - 1))
            idx1, idx2 = random.sample(indices, 2)
            route[idx1], route[idx2] = route[idx2], route[idx1]
        mutated_solution.append(route)
    return mutated_solution

def genetic_algorithm(
    population,
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
    requests,
    customers,
    num_generations=10,
    population_size=30,
    mutation_rate=0.2,
    crossover_rate=0.8,
    elite_fraction=0.1,
    verbose=False
):
    best_solution = None
    best_fitness = float('inf')

    for generation in range(num_generations):
        print(f"\n=== Generation {generation + 1} ===")

        evaluated_population = []

        for individual in population:
            repaired = validate_and_finalize_routes(
                individual, cost_matrix, E_max, recharge_amount, charging_stations, depot, nodes
            )
            repaired = ensure_all_customers_present(
                repaired, customers, depot, cost_matrix, nodes, charging_stations, E_max
            )
            # Filter trivial routes AFTER re-adding missing customers
            repaired = [r for r in repaired if sum(n not in charging_stations and n != depot for n in r) > 1]
            # Prevent routes starting with unreachable edges
            # Filter out routes with unreachable start from depot
            filtered_repaired = []
            for route in repaired:
                if len(route) > 1 and cost_matrix.get((depot, route[1]), float('inf')) != float('inf'):
                    filtered_repaired.append(route)
                else:
                    print(f"[SKIP] Unreachable or trivial route: {route}")
            repaired = filtered_repaired

            valid = validate_solution(repaired, depot, requests, customers, charging_stations)

            fitness, battery_ok = fitness_function(
                repaired, cost_matrix, travel_time_matrix, E_max, charging_stations,
                recharge_amount, penalty_weights, depot, nodes, vehicle_capacity,
                max_travel_time, requests
            )

            evaluated_population.append((repaired, fitness,True))

        valid_population = [ind for ind in evaluated_population if ind[2]]
        valid_population.sort(key=lambda x: x[1])
        selected_parents = [ind[0] for ind in valid_population[:max(2, population_size // 2)]]

        if len(selected_parents) < 2:
            print("⚠️ Not enough valid individuals. Using full population.")
            selected_parents = [ind[0] for ind in evaluated_population]

        if len(selected_parents) < 2:
            print("⚠️ Duplicating available parent.")
            selected_parents *= 2

        children = []
        while len(children) < population_size - len(selected_parents):
            p1, p2 = random.sample(selected_parents, 2)
            child = order_crossover_evrp(p1, p2, cost_matrix, E_max, charging_stations, recharge_amount, depot)
            child = mutate_route(child, mutation_rate=mutation_rate)

            repaired = validate_and_finalize_routes(
                child, cost_matrix, E_max, recharge_amount, charging_stations, depot, nodes
            )
            repaired = ensure_all_customers_present(
                repaired, customers, depot, cost_matrix, nodes, charging_stations, E_max
            )
            repaired = [r for r in repaired if len(r) > 2 and any(n in customers for n in r)]

            valid = validate_solution(repaired, depot, requests, customers, charging_stations)
            fitness, battery_ok = fitness_function(
                repaired, cost_matrix, travel_time_matrix, E_max, charging_stations,
                recharge_amount, penalty_weights, depot, nodes, vehicle_capacity,
                max_travel_time, requests
            )
            evaluated_population.append((repaired, fitness, valid and battery_ok))
            children.append(repaired)

        population = selected_parents + children

        # Update best solution
        best_candidate = min(evaluated_population, key=lambda x: x[1])
        if best_candidate[1] < best_fitness:
            best_solution = best_candidate[0]
            best_fitness = best_candidate[1]

        print(f"✅ Population size at end of generation: {len(population)}")

    if best_solution is None:
        print("❌ No valid GA solution found.")
        return [], float('inf')

    # Final revalidation of best solution
    best_solution = validate_and_finalize_routes(
        best_solution, cost_matrix, E_max, recharge_amount, charging_stations, depot, nodes
    )
    best_solution = ensure_all_customers_present(
        best_solution, customers, depot, cost_matrix, nodes, charging_stations, E_max
    )

    # Remove invalid routes again
    best_solution = [
        r for r in best_solution
        if len(r) > 2 and any(n in customers for n in r)
    ]

    if not best_solution:
        print("⚠️ Final GA repair yielded no valid solution. Using fallback.")
        fallback = min(evaluated_population, key=lambda x: x[1])[0]
        best_solution = validate_and_finalize_routes(
            fallback, cost_matrix, E_max, recharge_amount, charging_stations, depot, nodes
        )
        best_solution = ensure_all_customers_present(
            best_solution, customers, depot, cost_matrix, nodes, charging_stations, E_max
        )


    # Final cleanup using local search to improve structure and reduce trivial routes
    best_solution = apply_local_search(
        best_solution,
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

    return best_solution

