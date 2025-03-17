from validation import validate_solution

def merge_routes(solution, cost_matrix, E_max, charging_stations, recharge_amount, depot, vehicle_capacity, requests):
    merged = True
    while merged:
        merged = False
        for i in range(len(solution)):
            for j in range(i + 1, len(solution)):
                # Create a candidate merge by removing depot from end of route i and start of route j
                candidate = solution[i][:-1] + solution[j][1:]
                # Validate candidate using overall validation function
                if validate_solution([candidate], depot, requests, charging_stations):
                    solution[i] = candidate
                    del solution[j]
                    merged = True
                    break
            if merged:
                break
    return solution
