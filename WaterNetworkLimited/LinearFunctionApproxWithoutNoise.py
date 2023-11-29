import numpy as np
import matplotlib.pyplot as plt

# Generate some random data for value function approximation
np.random.seed(42)
state_values = np.linspace(0, 5, 100)
true_values = 2 * state_values + 1  # Removed noise from true values

# Linear approximation function
def linear_approximation(state, weight):
    return weight * state

# Set up plot
plt.figure(figsize=(10, 6))

# Initial weight
weight_initial = 0.0

# Plot true values and initial approximation
plt.plot(state_values, true_values, label='True Values (Linear)', linestyle='solid', linewidth=2)
plt.xlabel('State')
plt.ylabel('Value')
plt.title('True Values and Linear Value Function Approximation without Noise')

# Plot initial approximation
plt.plot(state_values, linear_approximation(state_values, weight_initial),
         label=f'Initial Approximation (w={weight_initial})', linestyle='solid', alpha=0.7)

# Perform linear value function approximation
learning_rate = 0.01
num_updates = 30  # Change the number of updates to 30
min_cost = float('inf')
min_cost_weight = None

for i in range(num_updates):
    # Update weight directly for linear approximation
    weight_initial += learning_rate * np.mean(true_values - linear_approximation(state_values, weight_initial))
    
    # Calculate cost
    cost = np.mean((true_values - linear_approximation(state_values, weight_initial))**2)
    
    # Update minimum cost and weight
    if cost < min_cost:
        min_cost = cost
        min_cost_weight = weight_initial

# Plot final approximation without noise
approx_values_final = linear_approximation(state_values, weight_initial)
plt.plot(state_values, approx_values_final, label=f'Final Approximation (w={weight_initial:.2f})', linestyle='solid', alpha=0.7, color='purple')

# Add annotations for linear value function approximation without noise
plt.scatter(0, 0, color='blue', marker='o', label=f'Initial Weight (w={weight_initial:.2f})', alpha=0.7)
plt.text(0.1, min_cost + 0.2, f'Min Cost: {min_cost:.2f}\n(w={min_cost_weight:.2f})', color='green', fontsize=8, ha='left')

# Move the point to the top
plt.scatter(min_cost_weight, min_cost, color='green', marker='o', label=f'Minimum Cost (w={min_cost_weight:.2f})', alpha=0.7, zorder=5)

# Move the legend to the top middle
plt.legend(loc='upper center')

# Show the plot
plt.show()