import matplotlib.pyplot as plt
from copy import deepcopy
from ga_operators import fitness_function
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import os

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
def plot_routes(routes, nodes, depot, charging_stations=None, method="CWS", save_plot=False, instance_name=None, E_max=None, cost_matrix=None):
    """
    Enhanced route plotter with color-coded routes, depot/CS markers, and optional battery annotation.
    """
    plt.figure(figsize=(10, 7))
    cmap = cm.get_cmap('tab20', len(routes))

    for i, route in enumerate(routes):
        x = [nodes[n][0] for n in route]
        y = [nodes[n][1] for n in route]
        plt.plot(x, y, label=f"Vehicle {i+1}", color=cmap(i), marker='o')

        # Optional battery annotation
        if E_max and cost_matrix:
            battery = E_max
            for j in range(1, len(route)):
                from_n, to_n = route[j-1], route[j]
                cost = cost_matrix.get((from_n, to_n), 0)
                battery -= cost
                battery = max(0, battery)
                plt.text(nodes[to_n]['x'], nodes[to_n]['y'], f"{int(battery)}", fontsize=7, color="red")

    # Mark depot
    plt.scatter(nodes[depot][0], nodes[depot][1], c='black', marker='s', s=150, label='Depot')
    plt.text(nodes[depot][0] + 0.5, nodes[depot][1] + 0.5, f"Depot ({depot})", fontsize=9)

    # Mark charging stations
    if charging_stations:
        for cs in charging_stations:
            plt.scatter(nodes[cs]['x'], nodes[cs]['y'], c='green', marker='*', s=150)
            plt.text(nodes[cs]['x'] + 0.5, nodes[cs]['y'] + 0.5, f"CS {cs}", fontsize=8)

    # Mark other nodes
    for n in nodes:
        if n != depot and (charging_stations is None or n not in charging_stations):
            plt.text(nodes[n][0] + 0.3, nodes[n][1] + 0.3, str(n), fontsize=8)

    plt.title(f"EVRP Route Visualization – {method}")
    plt.legend()
    plt.grid(True)

    if save_plot and instance_name:
        os.makedirs("plots", exist_ok=True)
        filepath = f"plots/{instance_name}_{method}_routes.png"
        plt.savefig(filepath)
        print(f"[INFO] Plot saved to {filepath}")
        plt.close()
    else:
        plt.show()
