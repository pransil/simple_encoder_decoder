import numpy as np
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

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
    hidden_layer_sizes=(1,),  # Single hidden unit
    activation='relu',
    solver='adam',
    max_iter=1,  # We'll fit one epoch at a time
    warm_start=True,  # Allows incremental fitting
    random_state=42
)

# Track MSE during training
mse_history = []
num_epochs = 100
print("\nTraining Progress:")
for epoch in range(num_epochs):
    model.fit(X_scaled, Y_scaled)
    mse_history.append(model.loss_)
    if (epoch + 1) % 10 == 0 or epoch == 0:
        print(f"Epoch {epoch+1}: MSE = {model.loss_:.6f}")

print(f"\nFinal MSE = {model.loss_:.6f}")

# Plot the training loss
plt.figure(figsize=(10, 6))
plt.plot(range(1, num_epochs + 1), mse_history, marker='o')
plt.title('MSE, single hidden unit')
plt.xlabel('Epoch')
plt.ylabel('Mean Squared Error')
plt.grid(True)
plt.tight_layout()
plt.savefig('mse_training_curve_simple_ml.png')
plt.show()

# Test the model with a new sample
test_input = np.array([[np.random.randint(0, 10)]])
test_input_scaled = x_scaler.transform(test_input)
prediction_scaled = model.predict(test_input_scaled)
prediction = y_scaler.inverse_transform(prediction_scaled)

print("\nTest input:", test_input[0,0])
print("Predicted output:", np.round(prediction[0]).astype(int))
print("Final Mean squared error:", model.loss_) 