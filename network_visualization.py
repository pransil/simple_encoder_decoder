import matplotlib.pyplot as plt
import numpy as np

def draw_neural_network():
    # Create figure and axis
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Set up the plot
    ax.set_xlim(-1, 2.5)
    ax.set_ylim(-1, 1)
    ax.axis('off')
    
    # Draw input layer (10 units)
    input_x = 0
    input_y = np.linspace(-0.8, 0.8, 10)
    for y in input_y:
        circle = plt.Circle((input_x, y), 0.09, color='lightblue', ec='black')
        ax.add_patch(circle)
    
    # Draw hidden layer (1 unit)
    hidden_x = 1
    hidden_y = 0
    circle = plt.Circle((hidden_x, hidden_y), 0.09, color='lightgreen', ec='black')
    ax.add_patch(circle)
    
    # Draw output layer (10 units)
    output_x = 2
    output_y = np.linspace(-0.8, 0.8, 10)
    for y in output_y:
        circle = plt.Circle((output_x, y), 0.09, color='lightcoral', ec='black')
        ax.add_patch(circle)
    
    # Draw connections from input to hidden
    for y in input_y:
        ax.plot([input_x + 0.09, hidden_x - 0.09], [y, hidden_y], 'gray', alpha=0.3)
    
    # Draw connections from hidden to output
    for y in output_y:
        ax.plot([hidden_x + 0.09, output_x - 0.09], [hidden_y, y], 'gray', alpha=0.3)
    
    # Add labels
    ax.text(input_x, 0.9, 'Input Layer\n(10 units)', ha='center', va='bottom')
    ax.text(hidden_x, 0.9, 'Hidden Layer\n(1 unit)', ha='center', va='bottom')
    ax.text(output_x, 0.9, 'Output Layer\n(10 units)', ha='center', va='bottom')
    
    # Add title
    plt.title('Neural Network Architecture\n10 → 1 → 10', pad=20)
    
    # Save and show
    plt.tight_layout()
    plt.savefig('network_architecture.png', dpi=300, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    draw_neural_network() 