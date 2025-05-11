# Simple Neural Network: 10-in-1-out Mapping

This project implements a simple neural network using scikit-learn that learns to map a 10-dimensional one-hot encoded input vector to a single output value representing the index of the '1' in the input.

## Project Structure

- `TenInOneOut.py`: Main script implementing the neural network
- `requirements.txt`: Python package dependencies
- `mse_training_curve.png`: Plot of the training loss over epochs

## Requirements

- Python 3.x
- numpy
- scikit-learn
- matplotlib

## Installation

1. Clone the repository:
```bash
git clone [repository-url]
cd simple_encode_decode
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

Run the main script:
```bash
python TenInOneOut.py
```

The script will:
1. Generate training data (1000 samples of one-hot encoded vectors)
2. Train a neural network with no hidden layers
3. Display training progress and MSE values
4. Test the model with new samples
5. Generate and save a plot of the training loss

## Model Details

- Input: 10-dimensional one-hot encoded vector (one '1' and nine '0's)
- Output: Single value representing the index of the '1' in the input
- Architecture: Single layer neural network (no hidden layers)
- Activation: Linear (identity)
- Optimizer: Adam
- Training: 100 epochs with 10 iterations per epoch

## Example

Input: `[0, 0, 0, 1, 0, 0, 0, 0, 0, 0]`
Output: `3` (index of the '1' in the input vector) 