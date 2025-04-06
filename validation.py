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

def ensure_all_customers_present(solution, expected_customers, depot, cost_matrix, nodes, charging_stations, E_max):
    """
    Adds any missing customers to a new route to ensure feasibility.
    """
    assigned = set()
    for route in solution:
        assigned.update([n for n in route if n in expected_customers])
    missing = expected_customers - assigned

    if not missing:
        return solution  # All good

    print(f"🧩 [DEBUG] Adding missing customers: {missing}")
    new_route = [depot]
    battery = E_max
    for customer in missing:
        cost = cost_matrix.get((new_route[-1], customer), float('inf'))
        if cost > battery:
            # Need CS
            cs = find_nearest_charging_station(new_route[-1], charging_stations, cost_matrix, battery)
            if cs:
                new_route.append(cs)
                battery = E_max
        battery -= cost
        new_route.append(customer)
    new_route.append(depot)
    solution.append(new_route)
    return solution

def validate_and_finalize_routes(individual, cost_matrix, E_max, recharge_amount, charging_stations, depot, nodes):
    """
    Validates and repairs all routes in an individual. Ensures each route ends at the depot.
    """
    repaired = []
    for route in individual:
        # Ensure starts and ends at depot
        if route[0] != depot:
            route = [depot] + route
        if route[-1] != depot:
            route.append(depot)

        repaired_route = repair_route_battery_feasibility(
            route, cost_matrix, E_max, recharge_amount, charging_stations, depot, nodes
        )
        repaired.append(repaired_route)
    return repaired




