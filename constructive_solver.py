import math
from copy import deepcopy


def calculate_savings(depot, customers, cost_matrix):
    """
    Computes savings for each pair of customers using Clarke & Wright's method.
    """
    savings = []
    for i in customers:
        for j in customers:
            if i != j:
                saving = cost_matrix[(depot, i)] + cost_matrix[(depot, j)] - cost_matrix[(i, j)]
                savings.append((saving, i, j))
    savings.sort(reverse=True)
    return savings


def construct_initial_solution(nodes, depot, customers, cost_matrix, vehicle_capacity, E_max, requests, charging_stations):

    """
    Builds an initial solution using Clarke & Wright Savings heuristic adapted for EVRP.
    """
    # 1. Start with each customer on their own route

    #print(f"[DEBUG] Sample request: {next(iter(requests.items()))}")

    routes = {cust_id: [depot, cust_id, depot] for cust_id in customers}
    loads = {cust_id: requests[cust_id]['quantity'] for cust_id in customers}
    batteries = {cust_id: E_max - (cost_matrix[(depot, cust_id)] + cost_matrix[(cust_id, depot)]) for cust_id in customers}

    savings = calculate_savings(depot, customers, cost_matrix)

    for s, i, j in savings:
        route_i = routes.get(i)
        route_j = routes.get(j)

        if not route_i or not route_j or route_i == route_j:
            continue

        if route_i[-2] == i and route_j[1] == j:
            new_route = route_i[:-1] + route_j[1:]
            new_load = loads[i] + loads[j]

            if new_load <= vehicle_capacity:
                # 🚫 Do not check battery feasibility here — defer to repair phase
                for node in route_j[1:-1]:
                    routes[node] = new_route
                for node in route_i[1:-1]:
                    routes[node] = new_route
                routes[i] = new_route

                # ✅ Recalculate loads
                for node in new_route:
                    if node != depot:
                        loads[node] = requests[node]['quantity']

    # Extract unique routes
    seen = set()
    unique_routes = []
    for r in routes.values():
        r_tuple = tuple(r)
        if r_tuple not in seen:
            seen.add(r_tuple)
            unique_routes.append(r)

    return unique_routes

def is_battery_feasible(route, cost_matrix, charging_stations, E_max, depot):
    """
    Checks and repairs a route by inserting charging stations if needed.
    Returns a battery-feasible version of the route.
    """
    battery = E_max
    new_route = [route[0]]  # Start from depot

    for i in range(1, len(route)):
        from_node = new_route[-1]
        to_node = route[i]
        cost = cost_matrix.get((from_node, to_node), float('inf'))

        if cost > battery:
            # Try inserting a CS
            reachable_cs = [
                cs for cs in charging_stations
                if cost_matrix.get((from_node, cs), float('inf')) <= battery and
                   cost_matrix.get((cs, to_node), float('inf')) <= E_max
            ]

            if reachable_cs:
                best_cs = min(reachable_cs, key=lambda cs: cost_matrix[(from_node, cs)])
                print(f"[DEBUG] Inserting CS {best_cs} between {from_node} and {to_node}")
                new_route.append(best_cs)
                battery = E_max - cost_matrix[(best_cs, to_node)]  # recharge then consume
                new_route.append(to_node)
            else:
                print(f"[ERROR] No CS can be inserted between {from_node} and {to_node}")
                return None  # Route is infeasible
        else:
            battery -= cost
            new_route.append(to_node)

        if to_node == depot or to_node in charging_stations:
            battery = E_max

    return new_route
