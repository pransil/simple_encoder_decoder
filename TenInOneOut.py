import numpy as np
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Generate training data where each input has exactly one 1 and nine 0s
np.random.seed(42)
X = np.zeros((1000, 10))  # Initialize with all zeros
y = np.zeros(1000)        # Initialize output array as 1D
for i in range(1000):
    pos = np.random.randint(0, 10)  # Random position for the 1
    X[i, pos] = 1  # Set that position to 1
    y[i] = pos     # Store the position as the target

# Display first 10 and last 5 inputs
print("First 10 inputs and their target indices:")
for i in range(10):
    print(f"Input {i+1}:", X[i], "Target index:", int(y[i]))
print("\nLast 5 inputs and their target indices:")
for i in range(995, 1000):
    print(f"Input {i+1}:", X[i], "Target index:", int(y[i]))

# Scale the data
scaler_X = StandardScaler()
scaler_y = StandardScaler()
X_scaled = scaler_X.fit_transform(X)
y_scaled = scaler_y.fit_transform(y.reshape(-1, 1)).ravel()

# Create the neural network with no hidden layers
model = MLPRegressor(
    hidden_layer_sizes=(),  # No hidden layers
    activation='identity',  # Linear activation
    solver='adam',
    max_iter=10,  # Set to 10 per epoch
    warm_start=True,  # Allows incremental fitting
    random_state=42
)

# Track MSE during training
mse_history = []
num_epochs = 100
print("\nTraining Progress:")
for epoch in range(num_epochs):
    model.fit(X_scaled, y_scaled)
    mse_history.append(model.loss_)
    if (epoch + 1) % 10 == 0 or epoch == 0:
        print(f"Epoch {epoch+1}: MSE = {model.loss_:.6f}")
print(f"Final Mean squared error: {model.loss_:.6f}")

# Plot the training loss
plt.figure(figsize=(10, 6))
plt.plot(range(1, num_epochs + 1), mse_history, marker='o')
plt.title('MSE, 10 in, 1 out')
plt.xlabel('Epoch')
plt.ylabel('Mean Squared Error')
plt.grid(True)
plt.tight_layout()
plt.savefig('mse_training_curve.png')
plt.show()

# Test the model with a few new samples
print("\nTesting with new samples:")
for _ in range(5):
    test_input = np.zeros((1, 10))
    test_pos = np.random.randint(0, 10)
    test_input[0, test_pos] = 1
    test_input_scaled = scaler_X.transform(test_input)
    prediction_scaled = model.predict(test_input_scaled)
    prediction = scaler_y.inverse_transform(prediction_scaled.reshape(-1, 1)).ravel()
    
    print(f"Input: {test_input[0]}")
    print(f"True index: {test_pos}")
    print(f"Predicted index: {round(prediction[0])}")
    print() 