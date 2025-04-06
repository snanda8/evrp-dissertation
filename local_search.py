import matplotlib.pyplot as plt

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
    return sum(cost_matrix[(route[i], route[i+1])] for i in range(len(route)-1))

def two_opt(route, cost_matrix):
    best = route
    improved = True
    while improved:
        improved = False
        for i in range(1, len(route) - 2):
            for j in range(i+1, len(route) - 1):
                new_route = route[:i] + route[i:j+1][::-1] + route[j+1:]
                if route_cost(new_route, cost_matrix) < route_cost(best, cost_matrix):
                    best = new_route
                    improved = True
        route = best
    return best

def apply_local_search(solution, cost_matrix):
    return [two_opt(route, cost_matrix) for route in solution]

def plot_routes(routes, nodes, depot):
    colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
    for i, route in enumerate(routes):
        x, y = zip(*[nodes[n][:2] for n in route])
        plt.plot(x, y, marker='o', label=f'Route {i+1}', color=colors[i % len(colors)])
    dx, dy = nodes[depot][:2]
    plt.plot(dx, dy, marker='s', markersize=10, color='black', label='Depot')
    plt.title("EVRP Routes")
    plt.legend()
    plt.show()
