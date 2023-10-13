import numpy as np
from scipy.optimize import linprog

# Define the transportation problem
cost_matrix = np.array([
    [6, 8, 10],
    [9, 12, 13],
    [15, 9, 10],
])

supply = np.array([20, 30, 25])
demand = np.array([10, 15, 30])

# Flatten the cost matrix to a 1D array
c = cost_matrix.flatten()

# Define the inequality constraints Ax <= b
A_eq = []
for i in range(len(supply)):
    row = np.zeros_like(c)
    row[i * len(demand):(i + 1) * len(demand)] = 1
    A_eq.append(row)

b_eq = supply

# Solve the linear programming problem
result = linprog(c, A_eq=A_eq, b_eq=b_eq, bounds=(0, None))

if result.success:
    # Reshape the solution to the original matrix shape
    solution = result.x.reshape(cost_matrix.shape)
    print("Optimal Solution Found:")
    print(solution)
    print("Total Cost =", result.fun)
else:
    print("No optimal solution found.")
