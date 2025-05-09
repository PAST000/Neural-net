# 🤖 NeuralNet

**NeuralNet** is a PHP implementation of a simple feedforward neural network. It supports forward propagation, gradient calculation via backpropagation, parameter saving/loading, and basic gradient-based updates. The implementation is fully object-oriented and uses a custom `Neuron` class (included separately).

## ✨ Features

- 🧠 Configurable multi-layer architecture
- 🔄 Forward propagation (`calc`)
- 🔙 Backpropagation for gradient computation (`calcGradient`)
- 🧮 Weight and bias update using gradients (`applyGradient`)
- 💾 Network serialization and deserialization (`save`, `read`)
- ⚙️ Adjustable learning rate

## 🚀 Getting Started

### 📦 Requirements

- PHP 7.0 or higher
- `neuron.php` must be available in the same directory

### 🧪 Example Usage

```php
require "neuron.php";
require "NeuralNet.php";

// Create a neural network with 2 inputs, one hidden layer of 3 neurons, and an output layer with 1 neuron
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
```

## 📄 File Format

Saved network files use the `.txt` extension and a custom text format consisting of:

- 🧱 Layer sizes and neuron weights/biases
- 📌 Separators:
  - Layers: `' '` (space)
  - Neurons: `'/'`
  - Weights: `';'`

## 🔧 Key Constants

```php
NeuralNet::FILE_EXTENSION = ".txt";
NeuralNet::MAX_RAND_WEIGHT = 10;
NeuralNet::MAX_RAND_BIAS = 10;
NeuralNet::MAX_RATE = 100;
NeuralNet::VALUE_PRECISION = 5;
```

## 📚 Methods Summary

| Method | Description |
|--------|-------------|
| `__construct(array $layers, float $rate)` | 🛠️ Initialize the network |
| `calc(array $inputs)` | ➡️ Forward propagation |
| `calcGradient(array $inputs, array $expected)` | 🧮 Compute negative gradient vector |
| `applyGradient(array $gradient, float $rateMultiplier = 1)` | 🧰 Apply gradient update |
| `save(string $filename)` | 💾 Save model to file |
| `read(string $filename)` | 📂 Load model from file |
| `setRate(float $newRate)` | 🔧 Update learning rate |
| `setExpected(array $expected)` | 🎯 Set expected output (used internally) |
| `getResult()` | 📊 Get current output from the last forward pass |
| `getGradient()` | 📈 Get last computed gradient |
| `getGradientRate()` | ⚙️ Get learning rate |
| `getNeurons()` | 🧬 Get network structure |

## 📝 Notes

- This implementation is primarily for educational or experimental use.
- The `Neuron` class must provide methods such as `calc()`, `getValue()`, `getPreSigmoid()`, `getWeightsCount()`, `getWeight()`, `getBias()`, `incrementWeight()`, and `incrementBias()`.
- `sigmoidDerivative()` is expected to be a standalone function.

## 📜 License

This project is licensed under the MIT License. See the `LICENSE` file for details.
