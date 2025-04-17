def filter_overloaded_routes(routes, vehicle_capacity, requests, depot, charging_stations):
    """
    Remove any routes that exceed vehicle capacity based on summed customer demands.
    Charging stations and depot are ignored in the demand calculation.
    """
    filtered = []
    for route in routes:
        demand = sum(
            requests[n]['quantity']
            for n in route
            if n not in charging_stations and n != depot
        )
        if demand <= vehicle_capacity:
            filtered.append(route)
        else:
            print(f"[⚠️] Overloaded route dropped (Demand: {demand}): {route}")
    return filtered


def sanitize_routes(routes, depot, charging_stations):
    """
    Clean routes by:
    - Removing adjacent duplicates
    - Ensuring depot at start and end
    - Dropping routes that don’t visit any customers
    """
    cleaned = []

    for route in routes:
        if not route:
            continue

        # Remove adjacent duplicates
        route = [n for i, n in enumerate(route) if i == 0 or n != route[i - 1]]

        # Clean trailing duplicate depots
        while len(route) >= 2 and route[-1] == depot and route[-2] == depot:
            route.pop()

        # Ensure start and end at depot
        if route[0] != depot:
            route = [depot] + route
        if route[-1] != depot:
            route.append(depot)

        # Keep route only if it visits at least one customer
        customer_nodes = [n for n in route if n != depot and n not in charging_stations]
        if customer_nodes:
            cleaned.append(route)

    return cleaned
