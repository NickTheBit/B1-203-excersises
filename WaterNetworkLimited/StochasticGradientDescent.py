import matplotlib.pyplot as plt
import numpy as np

# Generate random data for illustration
np.random.seed(42)
s_values = np.linspace(0, 4, 50)  # Adjusted range of state values
true_values = np.exp(s_values) - 1  # Exponential true values starting from the origin

# Linear approximation function
def linear_approximation(s, w):
    return w * s

# Stochastic Gradient Descent update function
def sgd_update(w, learning_rate, error, s):
    return w + learning_rate * error * s  # Use element-wise multiplication for the error term

# Set up plot
plt.figure(figsize=(8, 6))
plt.plot(s_values, true_values, label='True Values', linestyle='solid', linewidth=2)
plt.xlabel('State (s)')
plt.ylabel('Value')
plt.title('Stochastic Gradient Descent Update in Linear Approximation')

# Initial weight
w_initial = 0.0

# Plot initial approximation
approx_values_initial = linear_approximation(s_values, w_initial)
plt.plot(s_values, approx_values_initial, label=f'Initial Approximation (w={w_initial})', linestyle='solid', alpha=0.7)

# Perform multiple SGD updates
learning_rate = 0.001  # Adjusted learning rate
num_updates = 100

for _ in range(num_updates):
    error = true_values - linear_approximation(s_values, w_initial)
    w_initial = sgd_update(w_initial, learning_rate, error, s_values)

# Plot final approximation
approx_values_final = linear_approximation(s_values, w_initial)
plt.plot(s_values, approx_values_final, label=f'Final Approximation (w={w_initial[0]:.2f})', linestyle='solid', alpha=0.7)

# Set x-axis ticks to integers with 1 in between
plt.xticks(np.arange(0, 5, 1))

plt.legend()
plt.show()