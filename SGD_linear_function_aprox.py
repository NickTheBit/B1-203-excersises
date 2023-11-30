import matplotlib.pyplot as plt
import numpy as np
import random

# Generate random data for illustration
np.random.seed(42)
N_points=50
s_values = np.linspace(0, 4, N_points)  # Adjusted range of state value
m=7
true_values = s_values*(m+ np.random.normal(0, 1, size=N_points))  # Exponential true values starting from the origin

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

# Perform multiple SGD updates (It takes one true value per update)

num_updates = 100
w_history=[]
for _ in range(1,num_updates):
    learning_rate = 1/(_)  # Adjusted learning rate
    s=random.randint(0, N_points-1) # This is not necessary, its more of a proof of concept
    error = true_values[s] - linear_approximation(s_values[s], w_initial)
    w_initial = sgd_update(w_initial, learning_rate, error, s_values[s])
    w_history.append(w_initial)

# Plot final approximation
approx_values_final=[]
for i in range(0,N_points):
    approx_values_final.append( linear_approximation(s_values[i], w_initial))

print(f"w estimation relative error: {abs((w_initial-m)/m)*100}%")
plt.style.use('dark_background')
plt.figure(1)
plt.plot(s_values, approx_values_final)
#plt.plot(s_values, approx_values_final, label=f'Final Approximation (w={w_initial[0]:.2f})', linestyle='solid', alpha=0.7)

plt.legend()
plt.figure(2)
plt.title("w updates")
plt.plot(w_history)

plt.show()