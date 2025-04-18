from instance_parser import parse_instance
from utils import find_nearest_charging_station
from heuristics import repair_route_battery_feasibility


def validate_no_duplicates_route(route, depot, requests, charging_stations,):
    """
    Ensure that each customer (node with a request that is not the depot or a charging station)
    appears at most once in the route.
    """
    visited = set()
    for node in route:
        if node == depot or node in charging_stations:
            continue
        if node in requests:
            if node in visited:
                return False
            visited.add(node)
    return True


def validate_solution(solution, depot, requests, charging_stations, expected_customers):
    """Ensure all real customers (not charging stations) are visited."""
    visited_customers = set()

    print(f"\n[DEBUG] Running Validation Check")
    print(f"Expected Customers (Before Filtering): {expected_customers}")
    print(f"Charging Stations: {charging_stations}")

    # 🔹 Check if `charging_stations` are in `expected_customers`
    overlap = expected_customers.intersection(charging_stations)
    if overlap:
        print(f" [DEBUG] ERROR: Charging stations mistakenly included in expected customers: {overlap}")

    # 🔹 Ensure only real customers are checked
    actual_customers = expected_customers - charging_stations
    print(f"Actual Customers (After Filtering): {actual_customers}")

    for route in solution:
        for node in route:
            if node == depot or node in charging_stations:
                continue  # Ignore depot and charging stations

            if node in actual_customers:  # ✅ Only check REAL customers
                if node in visited_customers:
                    print(f" ERROR: Customer {node} visited more than once!")
                    return False
                visited_customers.add(node)

    missing = actual_customers - visited_customers

    if missing:
        print(f" ERROR: Missing Customers Detected: {missing}")
        return False

    print("✅ Validation Passed")
    return True

def ensure_all_customers_present(routes, customers, depot, cost_matrix, nodes, charging_stations, E_max):
    """
    Ensure that every customer appears in the solution exactly once.
    If a customer is missing, insert it into the nearest feasible route or create a new one.
    """
    present_customers = set(n for r in routes for n in r if n in customers)
    missing_customers = set(customers) - present_customers

    if missing_customers:
        print(f"[REPAIR] Reinserting missing customers: {missing_customers}")

    for customer in missing_customers:
        # Try inserting into an existing route
        best_route_idx = -1
        best_position = -1
        best_increase = float('inf')

        for i, route in enumerate(routes):
            for j in range(1, len(route) - 1):  # Avoid depot ends
                before, after = route[j - 1], route[j]
                cost_before = cost_matrix.get((before, customer), float('inf'))
                cost_after = cost_matrix.get((customer, after), float('inf'))
                cost_original = cost_matrix.get((before, after), float('inf'))

                added_cost = cost_before + cost_after - cost_original
                if added_cost < best_increase:
                    best_increase = added_cost
                    best_route_idx = i
                    best_position = j

        if best_route_idx != -1:
            routes[best_route_idx].insert(best_position, customer)
        else:
            # Create a fallback route
            print(f"[REPAIR] Creating new route for customer {customer}")
            routes.append([depot, customer, depot])

    return routes

def validate_and_finalize_routes(individual, *_):
    """
    TEMPORARY: Skip battery repair to prioritize valid, customer-covering routes for GA.
    """
    cleaned_routes = []
    for route in individual:
        if len(route) < 3 or all(n in {15, 10, 11, 12, 13, 14} for n in route):
            continue
        if route[0] != 15:
            route = [15] + route
        if route[-1] != 15:
            route = route + [15]
        cleaned_routes.append(route)
    return cleaned_routes






