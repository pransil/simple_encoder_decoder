import numpy as np
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

# Generate training data
np.random.seed(42)
X = np.random.randint(0, 10, size=(1000, 1))  # Input: random number between 0 and 9
# Output: 10-digit array with 1 in the position from input
Y = np.zeros((1000, 10))
for i in range(1000):
    Y[i, X[i, 0]] = 1

# Show first 10 input/output pairs
print("First 10 input/output pairs:")
for i in range(10):
    print(f"Input: {X[i,0]}, Output: {Y[i]}")

# Scale the data
x_scaler = StandardScaler()
y_scaler = StandardScaler()
X_scaled = x_scaler.fit_transform(X)
Y_scaled = y_scaler.fit_transform(Y)

# Create and train the neural network
model = MLPRegressor(
    hidden_layer_sizes=(10,),
    activation='relu',
    solver='adam',
    max_iter=10000,
    random_state=42
)

print("\nTraining Progress:")
model.fit(X_scaled, Y_scaled)
print(f"Final MSE = {model.loss_:.6f}")

# Test the model with a new sample
test_input = np.array([[np.random.randint(0, 10)]])
test_input_scaled = x_scaler.transform(test_input)
prediction_scaled = model.predict(test_input_scaled)
prediction = y_scaler.inverse_transform(prediction_scaled)

print("\nTest input:", test_input[0,0])
print("Predicted output:", np.round(prediction[0]).astype(int))
print("Final Mean squared error:", model.loss_) 