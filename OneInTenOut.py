import numpy as np
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import OneHotEncoder
import matplotlib.pyplot as plt

# Set random seed for reproducibility
np.random.seed(42)

# Generate training data
n_samples = 1000
X = np.random.randint(0, 10, size=(n_samples, 1))  # Input: random numbers 0-9
encoder = OneHotEncoder(sparse_output=False)
y = encoder.fit_transform(X)  # Output: one-hot encoded vectors

# Create and train the model
model = MLPRegressor(
    hidden_layer_sizes=(1,),  # Single hidden layer with 1 unit
    activation='relu',
    solver='adam',
    max_iter=1000,
    batch_size=32,
    verbose=True
)

# Train the model and capture the loss history
history = model.fit(X, y)

# Print first 10 input/output pairs
print("\nFirst 10 input/output pairs:")
for i in range(10):
    print(f"Input: {X[i][0]}, Output: {y[i]}")

# Print training progress
print("\nTraining Progress:")
print(f"Final MSE = {history.loss_curve_[-1]:.6f}")

# Test the model
test_input = np.array([[0]])  # Test with input 0
predicted_output = model.predict(test_input)
print(f"\nTest input: {test_input[0][0]}")
print(f"Predicted output: {np.round(predicted_output[0]).astype(int)}")
print(f"Final Mean squared error: {history.loss_curve_[-1]}")

# Plot the training loss
plt.figure(figsize=(10, 6))
plt.plot(history.loss_curve_)
plt.title('Training Loss Over Time')
plt.xlabel('Iteration')
plt.ylabel('Mean Squared Error')
plt.grid(True)
plt.show() 