# Create the README.md file content
readme_content = """# NeuralNet

**NeuralNet** is a PHP implementation of a simple feedforward neural network. It supports forward propagation, gradient calculation via backpropagation, parameter saving/loading, and basic gradient-based updates. The implementation is fully object-oriented and uses a custom `Neuron` class (included separately).

## Features

- Configurable multi-layer architecture
- Forward propagation (`calc`)
- Backpropagation for gradient computation (`calcGradient`)
- Weight and bias update using gradients (`applyGradient`)
- Network serialization and deserialization (`save`, `read`)
- Adjustable learning rate

## Getting Started

### Requirements

- PHP 7.0 or higher
- `neuron.php` must be available in the same directory

### Example Usage

```php
require "NeuralNet.php";

// Create a neural network with 2 inputs, one hidden layer of 3 neurons, and an output layer with 1 neuron. With learnig rate equal 0.01
$nn = new NeuralNet([2, 3, 1], 0.01);

// Forward pass
$result = $nn->calc([1.0, 0.5]);

// Compute gradient and apply update
$gradient = $nn->calcGradient([1.0, 0.5], [0.8]); // Target output is 0.8
$nn->applyGradient($gradient);

// Save to file
$nn->save("model.txt");

// Load from file
$nn->read("model.txt");
