import pulp

# Define the Problem
prob = pulp.LpProblem("EVRP_Minimize_Distance", pulp.LpMinimize)

# Define the Nodes and Costs
nodes = [0, 1, 2]  # Depot and customers
costs = {
    (0, 1): 10,
    (0, 2): 15,
    (1, 2): 5,
    (1, 0): 10,
    (2, 0): 15,
    (2, 1): 5
}

# Battery Parameters
E_max = 100  # Maximum battery capacity
consumption = {k: v for k, v in costs.items()}  # Energy consumption same as cost
M = E_max  # Big-M constant for linearization

# Define Decision Variables
x = pulp.LpVariable.dicts("x", costs, cat=pulp.LpBinary)  # Route selection
E = pulp.LpVariable.dicts("E", nodes, lowBound=0, upBound=E_max, cat=pulp.LpContinuous)  # Battery level
r = pulp.LpVariable.dicts("r", nodes, lowBound=0, upBound=E_max, cat=pulp.LpContinuous)  # Recharge amount
s = pulp.LpVariable.dicts("s", nodes, cat=pulp.LpBinary)  # Charging station indicator

# Objective Function (Minimize Total Distance)
prob += pulp.lpSum([costs[i, j] * x[i, j] for i, j in costs]), "Total_Distance"

# Flow Conservation Constraint (Visit Each Node Once)
for node in nodes:
    if node != 0:
        prob += pulp.lpSum([x[i, node] for i in nodes if (i, node) in costs]) == 1
        prob += pulp.lpSum([x[node, j] for j in nodes if (node, j) in costs]) == 1

# Battery Capacity Constraint (Big-M Linearization)
for (i, j) in costs:
    prob += E[j] >= E[i] - consumption[i, j] * x[i, j] - M * (1 - s[j]) + r[j]
    prob += E[j] <= E[i] - consumption[i, j] * x[i, j] + r[j]  # Prevent overcharging beyond capacity

# Prevent battery from going below zero
for node in nodes:
    prob += E[node] >= 0
    prob += E[node] <= E_max

# Charging Constraints (only charge if charging station is present)
for node in nodes:
    prob += r[node] <= E_max * s[node]

# Solve the Problem
prob.solve()

# Output the Results
print(f"Status: {pulp.LpStatus[prob.status]}")
for i, j in costs:
    if x[i, j].varValue == 1:
        print(f"Route taken: {i} -> {j} with cost: {costs[i, j]}")

for node in nodes:
    print(f"Battery Level at Node {node}: {E[node].varValue}")
    print(f"Recharge Amount at Node {node}: {r[node].varValue}")
    print(f"Charging Station Active at Node {node}: {s[node].varValue}")

# Objective Value (Minimum Distance)
print(f"Total Distance: {pulp.value(prob.objective)}")
