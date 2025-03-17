from instance_parser import parse_instance

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


