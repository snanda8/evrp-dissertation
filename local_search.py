import matplotlib.pyplot as plt
from copy import deepcopy
from ga_operators import fitness_function

def calculate_savings(depot, customers, cost_matrix):
    savings = []
    for i in customers:
        for j in customers:
            if i != j:
                saving = cost_matrix[(depot, i)] + cost_matrix[(depot, j)] - cost_matrix[(i, j)]
                savings.append((saving, i, j))
    savings.sort(reverse=True)
    return savings

def construct_initial_solution(nodes, depot, customers, cost_matrix, vehicle_capacity, E_max, requests):
    routes = {cust: [depot, cust, depot] for cust in customers}
    loads = {cust: requests[cust]['quantity'] for cust in customers}
    savings = calculate_savings(depot, customers, cost_matrix)

    for _, i, j in savings:
        ri = routes.get(i)
        rj = routes.get(j)
        if not ri or not rj or ri == rj:
            continue
        if ri[-2] == i and rj[1] == j:
            merged = ri[:-1] + rj[1:]
            total_demand = sum(requests[c]['quantity'] for c in merged if c != depot)
            if total_demand <= vehicle_capacity:
                # Check energy feasibility (optional here)
                for node in merged:
                    routes[node] = merged
    unique_routes = []
    seen = set()
    for r in routes.values():
        rt = tuple(r)
        if rt not in seen:
            unique_routes.append(r)
            seen.add(rt)
    return unique_routes



def route_cost(route, cost_matrix):
    total = 0
    for i in range(len(route) - 1):
        total += cost_matrix.get((route[i], route[i + 1]), float('inf'))
    return total


# === INTRA-ROUTE OPTIMIZATION ===
def fitness_based_two_opt(route, cost_matrix, travel_time_matrix, full_solution, **fitness_kwargs):
    best = route
    best_solution = deepcopy(full_solution)
    best_solution[full_solution.index(route)] = best
    best_fitness, _ = fitness_function(best_solution, cost_matrix, travel_time_matrix, **fitness_kwargs)

    improved = True
    while improved:
        improved = False
        for i in range(1, len(best) - 2):
            for j in range(i + 1, len(best) - 1):
                new_route = best[:i] + best[i:j + 1][::-1] + best[j + 1:]
                new_solution = deepcopy(full_solution)
                new_solution[full_solution.index(route)] = new_route
                new_fitness, _ = fitness_function(new_solution, cost_matrix, travel_time_matrix, **fitness_kwargs)
                if new_fitness < best_fitness:
                    best = new_route
                    best_fitness = new_fitness
                    improved = True
    return best


# === INTER-ROUTE OPERATORS ===
def try_relocate(solution, cost_matrix, travel_time_matrix, **fitness_kwargs):
    improved = False
    for i in range(len(solution)):
        for j in range(len(solution)):
            if i == j:
                continue
            from_route = solution[i]
            to_route = solution[j]

            for k in range(1, len(from_route) - 1):
                customer = from_route[k]
                for m in range(1, len(to_route)):
                    new_from = from_route[:k] + from_route[k + 1:]
                    new_to = to_route[:m] + [customer] + to_route[m:]

                    new_solution = deepcopy(solution)
                    new_solution[i] = new_from
                    new_solution[j] = new_to

                    old_fitness, _ = fitness_function(solution, cost_matrix, travel_time_matrix, **fitness_kwargs)
                    new_fitness, _ = fitness_function(new_solution, cost_matrix, travel_time_matrix, **fitness_kwargs)

                    if new_fitness < old_fitness:
                        solution[i] = new_from
                        solution[j] = new_to
                        return True  # early exit for restart
    return improved


def try_swap(solution, cost_matrix, travel_time_matrix, **fitness_kwargs):
    improved = False
    for i in range(len(solution)):
        for j in range(i + 1, len(solution)):
            route_a = solution[i]
            route_b = solution[j]

            for a_idx in range(1, len(route_a) - 1):
                for b_idx in range(1, len(route_b) - 1):
                    new_a = route_a[:]
                    new_b = route_b[:]
                    new_a[a_idx], new_b[b_idx] = new_b[b_idx], new_a[a_idx]

                    new_solution = deepcopy(solution)
                    new_solution[i] = new_a
                    new_solution[j] = new_b

                    old_fitness, _ = fitness_function(solution, cost_matrix, travel_time_matrix, **fitness_kwargs)
                    new_fitness, _ = fitness_function(new_solution, cost_matrix, travel_time_matrix, **fitness_kwargs)

                    if new_fitness < old_fitness:
                        solution[i] = new_a
                        solution[j] = new_b
                        return True
    return improved


# === FULL LOCAL SEARCH ===
def apply_local_search(solution, cost_matrix, travel_time_matrix, **fitness_kwargs):
    print("🔍 Running local search with 2-opt, relocate, and swap...")
    routes = deepcopy(solution)

    # Intra-route: 2-opt
    for idx, route in enumerate(routes):
        routes[idx] = fitness_based_two_opt(route, cost_matrix, travel_time_matrix, routes, **fitness_kwargs)

    # Inter-route: Relocate and Swap until no improvement
    improving = True
    while improving:
        improving = (
            try_relocate(routes, cost_matrix, travel_time_matrix, **fitness_kwargs)
            or try_swap(routes, cost_matrix, travel_time_matrix, **fitness_kwargs)
        )

    return routes


# === VISUALIZATION ===
def plot_routes(routes, nodes, depot):
    colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
    for i, route in enumerate(routes):
        x, y = zip(*[nodes[n][:2] for n in route])
        plt.plot(x, y, marker='o', label=f'Route {i + 1}', color=colors[i % len(colors)])
    dx, dy = nodes[depot][:2]
    plt.plot(dx, dy, marker='s', markersize=10, color='black', label='Depot')
    plt.title("EVRP Routes")
    plt.legend()
    plt.show()
