def sanitize_routes(routes, depot, charging_stations):
    cleaned = []

    for route in routes:
        route = [n for i, n in enumerate(route) if i == 0 or n != route[i - 1]]

        while len(route) >= 2 and route[-1] == depot and route[-2] == depot:
            route.pop()

        if route[0] != depot:
            route = [depot] + route
        if route[-1] != depot:
            route.append(depot)

        customer_nodes = [n for n in route if n != depot and n not in charging_stations]
        if customer_nodes:
            cleaned.append(route)

    return cleaned
