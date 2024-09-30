import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Convert the data to numpy arrays
dollar_rate = np.array([30.3394, 31.2159, 32.3436, 32.3971, 32.2454, 32.9854, 33.0102, 33.9844])
policy_interest_rate = np.array([45, 45, 50, 50, 50, 50, 50, 50])
oil_price = np.array([82.18, 82.05, 87.26, 87.25, 82.24, 84.85, 79.12, 78.94])
gasoline_price = np.array([39.79, 42.67, 44.98, 45.28, 43.11, 44.15, 46.55, 44.36])

# Normalize the data
dollar_rate_normalized = (dollar_rate - np.mean(dollar_rate)) / np.std(dollar_rate)
policy_interest_rate_normalized = (policy_interest_rate - np.mean(policy_interest_rate)) / np.std(policy_interest_rate)
oil_price_normalized = (oil_price - np.mean(oil_price)) / np.std(oil_price)

# Create input data
X = np.vstack((dollar_rate_normalized, policy_interest_rate_normalized, oil_price_normalized)).T
y = gasoline_price  # All gasoline prices (including the final value)

# Convert to tensors
X_train = torch.tensor(X, dtype=torch.float32)
y_train = torch.tensor(y, dtype=torch.float32).view(-1, 1)  # Output should be 2D (N x 1)

# Simple linear regression model
class LinearRegressionModel(nn.Module):
    def __init__(self):
        super(LinearRegressionModel, self).__init__()
        self.linear = nn.Linear(3, 1)  # 3 features -> 1 output

    def forward(self, x):
        return self.linear(x)

# Initialize the model
model = LinearRegressionModel()

# Use MSE as the loss function
criterion = nn.MSELoss()

# Use PyTorch's built-in optimizer (SGD)
optimizer = optim.SGD(model.parameters(), lr=0.01)

# Training loop
num_epochs = 1000
for epoch in range(num_epochs):
    outputs = model(X_train)
    loss = criterion(outputs, y_train)
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    if (epoch+1) % 100 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

# Predict gasoline price for the 9th month
dollar_rate_9 = (34.1514 - np.mean(dollar_rate)) / np.std(dollar_rate)
policy_interest_rate_9 = (50 - np.mean(policy_interest_rate)) / np.std(policy_interest_rate)
oil_price_9 = (72 - np.mean(oil_price)) / np.std(oil_price)

X_9 = torch.tensor([[dollar_rate_9, policy_interest_rate_9, oil_price_9]], dtype=torch.float32)

# Make predictions using the model
model.eval()  
with torch.no_grad():
    predicted_train = model(X_train).numpy()
    predicted_9 = model(X_9).numpy()

# Extend the gasoline price array for the prediction
gasoline_price_extended = np.append(gasoline_price, [np.nan])
predicted_extended = np.append(predicted_train.flatten(), predicted_9.flatten())

# Plot settings (2 graphs side by side)
fig = plt.figure(figsize=(14, 6))

# 1st plot: 3D
ax1 = fig.add_subplot(121, projection='3d')
ax1.scatter(dollar_rate_normalized, policy_interest_rate_normalized, gasoline_price, color='red', label='Actual Gasoline Prices')
ax1.scatter(dollar_rate_normalized, policy_interest_rate_normalized, predicted_train.flatten(), color='blue', label='Predicted Gasoline Prices')

# 9th month prediction (green dot)
ax1.scatter(dollar_rate_9, policy_interest_rate_9, predicted_9[0][0], color='green', s=100, label='9th Month Prediction')

# Create a meshgrid for surface plotting
dollar_range = np.linspace(min(dollar_rate_normalized), max(dollar_rate_normalized), 20)
interest_range = np.linspace(min(policy_interest_rate_normalized), max(policy_interest_rate_normalized), 20)
dollar_grid, interest_grid = np.meshgrid(dollar_range, interest_range)
oil_mean = np.mean(oil_price_normalized)
X_grid = np.c_[dollar_grid.ravel(), interest_grid.ravel(), np.full_like(dollar_grid.ravel(), oil_mean)]
X_grid_tensor = torch.tensor(X_grid, dtype=torch.float32)

with torch.no_grad():
    Z_grid = model(X_grid_tensor).numpy().reshape(dollar_grid.shape)

# Plot the surface
ax1.plot_surface(dollar_grid, interest_grid, Z_grid, color='orange', alpha=0.5)

ax1.set_xlabel('Dollar Rate (Normalized)')
ax1.set_ylabel('Policy Interest Rate (Normalized)')
ax1.set_zlabel('Gasoline Price (TL)')
ax1.set_title('3D Actual vs Predicted Gasoline Prices')
ax1.legend()

# 2nd plot: 2D (Time vs Gasoline Price)
ax2 = fig.add_subplot(122)
time = np.arange(1, len(gasoline_price_extended) + 1)

# Actual prices
ax2.plot(time[:-1], gasoline_price, 'ro-', label='Actual Gasoline Prices')

# Predicted prices
ax2.plot(time, predicted_extended, 'bx-', label='Predicted Gasoline Prices')

# 9th month prediction (green dot)
ax2.plot(len(time), predicted_9[0][0], 'go', markersize=10, label='9th Month Prediction')

# Create a smooth linear model line similar to the 3D surface
# Generate synthetic range for `dollar_rate` and `policy_interest_rate`, keep oil price fixed
dollar_range_2d = np.linspace(min(dollar_rate_normalized), max(dollar_rate_normalized), len(time))
interest_mean = np.mean(policy_interest_rate_normalized)  # Fixed interest rate
oil_mean_2d = np.mean(oil_price_normalized)  # Fixed oil price
X_smooth = np.vstack((dollar_range_2d, np.full_like(dollar_range_2d, interest_mean), np.full_like(dollar_range_2d, oil_mean_2d))).T
X_smooth_tensor = torch.tensor(X_smooth, dtype=torch.float32)

# Predict gasoline prices using this smoothed grid
with torch.no_grad():
    y_smooth = model(X_smooth_tensor).numpy()

# Plot the smooth model line
ax2.plot(time, y_smooth.flatten(), 'k--', label='Model Trend Line')

# Axis labels and title
ax2.set_xlabel('Time (Months)')
ax2.set_ylabel('Gasoline Price (TL/Liter)')
ax2.set_title('2D Actual vs Predicted Gasoline Prices')
ax2.legend()

# Show the plot
plt.tight_layout()
plt.show()
