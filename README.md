# A simple encoder - decoder
simple_ml.py is a simple encoder, 10 inputs, a single hidden unit, reproducing the input at the output. The input is an array of 10 digits, all zero except for a single 1. I'm just proving to myself that this will work, as a building block for later projects.

I broke this into 3 parts:
1. TenInOneOut.py - Two layers, 10 units input, 1 output that gives the index of the 1 in the input array.
2. OneInTenOut.py - Two layers, 1 input unit (int from 0 to 9), 10 outputs, an array of zeros with a 1 whose index is the input int.
3. simple_ml.py - The real goal, a 3 layer encoder - decoder. 10 in, 1 hidden unit, 10 out. Output reproduces the input, which is 10 ints, all 0 except a 1 in a single location. The hidden layer just learns the index of the 1.

Using scikit-learn. Files contain a plot of the MSE during training for all 3 networks.

## Requirements

- Python 3.x
- numpy
- scikit-learn
- matplotlib

## Installation

1. Clone the repository:
```bash
git clone [https://github.com/pransil/simple_encoder_decoder]
cd simple_encoder_decoder
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

Run the main script:
```bash
python simple_ml.py
```

The script will:
1. Generate training data (1000 samples of one-hot encoded vectors)
2. Train a neural network
3. Display training progress and MSE values
4. Test the model with new samples
5. Generate and save a plot of the training loss

## Model Details

- Input: 10-dimensional one-hot encoded vector (one '1' and nine '0's)
- Output: Same as input
- Architecture: 10 unit input layer, 1 unit hidden, 10 out
- Activation: Linear (identity)
- Optimizer: Adam
- Training: 100 epochs with 10 iterations per epoch

## Example


Input: `[0, 0, 0, 1, 0, 0, 0, 0, 0, 0]`
Output: `[0, 0, 0, 1, 0, 0, 0, 0, 0, 0]`
The hidden layer learns to encode the index of the 1.