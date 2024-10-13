# Learning XOR with PyTorch Lightning: An Educational Journey

![](unnamed-ml.jpg)

## Introduction

In this post, we'll explore the implementation of a simple neural network to solve the XOR (exclusive OR) problem using PyTorch Lightning. The XOR problem is a classic example that showcases the limitations of linear classifiers and highlights the strengths of neural networks. This experience will help you avoid common pitfalls and understand the significance of the XOR function in machine learning.

## Understanding the XOR Problem

The XOR function is a binary operation that takes two inputs and returns `1` if the inputs are different and `0` if they are the same. The truth table for the XOR function is as follows:

| Input A | Input B | Output |
|---------|---------|--------|
| 0       | 0       | 0      |
| 0       | 1       | 1      |
| 1       | 0       | 1      |
| 1       | 1       | 0      |

### Why XOR?

The XOR problem is particularly interesting because it cannot be solved by a linear classifier; there is no straight line that can separate the outputs correctly. This non-linearity makes XOR an ideal benchmark for testing the capabilities of neural networks.

#### XOR vs. Decision Trees

While decision trees can model the XOR function by recursively partitioning the input space, they may struggle with more complex, high-dimensional data. Here’s how neural networks differ from decision trees:

1. **Complexity and Non-Linearity**: Neural networks can learn complex decision boundaries through non-linear transformations, making them well-suited for intricate problems.

2. **Modeling Capability**: Neural networks can capture hierarchical representations of data, allowing them to generalize better across diverse datasets and tasks.

3. **Robustness**: With appropriate regularization techniques, neural networks can mitigate overfitting, providing more reliable predictions compared to decision trees, especially with smaller datasets.

### Real-World Applications of XOR and Neural Networks

The ability to solve the XOR problem showcases the potential of neural networks in various applications, including:

- **Binary Classification**: Many tasks, such as sentiment analysis and fraud detection, can be modeled similarly to XOR, where the output is binary.
- **Feature Interactions**: Neural networks excel at capturing complex interactions between features, making them invaluable in fields like finance and healthcare for predictive modeling.

## Implementing the XOR Model in PyTorch Lightning

Now, let's dive into the code implementation of the XOR model using PyTorch Lightning. Below is the complete code:

## Code Breakdown

### Step 1: Import Necessary Libraries

```python
import torch
from torch import nn, optim
from torch.autograd import Variable
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.utils.data import DataLoader
```

**Explanation:**

- `torch`: The main library for building neural networks in Python.
- `nn`: A sub-library within `torch` that contains building blocks for neural networks.
- `optim`: A module in `torch` used for optimizing the model.
- `Variable`: A wrapper for tensors that allows for automatic differentiation (used in older versions of PyTorch, now tensors themselves support this).
- `pytorch_lightning`: A wrapper around PyTorch to simplify the training process.
- `ModelCheckpoint`: A utility that saves the model at certain checkpoints during training.
- `DataLoader`: A class that helps manage and load data in batches for training.

### Step 2: Print Library Versions

```python
print("PyTorch version:", torch.__version__)
print("Torch Lightning version:", pl.__version__)
```

**Explanation:**

This code simply prints out the versions of PyTorch and PyTorch Lightning you are using. It’s a good practice to know your library versions, especially when debugging.

### Step 3: Define XOR Inputs and Targets

```python
xor_inputs = [
    Variable(torch.Tensor([0, 0])),
    Variable(torch.Tensor([0, 1])),
    Variable(torch.Tensor([1, 0])),
    Variable(torch.Tensor([1, 1])),
]

xor_targets = [
    Variable(torch.Tensor([0])),
    Variable(torch.Tensor([1])),
    Variable(torch.Tensor([1])),
    Variable(torch.Tensor([0])),
]
```

**Explanation:**

Here, we define the inputs and expected outputs for the XOR function:

- **Inputs**: Each input is a pair of values (0 or 1).
- **Targets**: Each target is the expected output of the XOR function for the corresponding input.

### Step 4: Create a Dataset and DataLoader

```python
xor_data = list(zip(xor_inputs, xor_targets))
train_loader = DataLoader(xor_data, batch_size=1, shuffle=True)
```

**Explanation:**

- `zip`: Combines inputs and targets into pairs. For example, it combines `([0, 0], [0])`, `([0, 1], [1])`, etc.
- `DataLoader`: Prepares the data for training by loading it in batches. Here, we set the batch size to 1, meaning the model will process one input-output pair at a time, and we shuffle the data to improve training.

### Step 5: Define the XOR Model

```python
class XORModel(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.input_layer = nn.Linear(2, 4)  # Input layer
        self.output_layer = nn.Linear(4, 1)  # Output layer
        self.sigmoid = nn.Sigmoid()  # Sigmoid activation function
        self.loss = nn.MSELoss()  # Mean squared error loss
```

**Explanation:**

- **XORModel**: This is our custom model class that inherits from `pl.LightningModule`, which provides helpful functions for training and validation.
- `__init__`: This function initializes the model layers:
  - `nn.Linear(2, 4)`: This creates a layer that takes 2 inputs (the XOR inputs) and outputs 4 values (hidden layer).
  - `nn.Linear(4, 1)`: This layer takes the 4 hidden values and outputs a single value (the XOR output).
  - `nn.Sigmoid()`: This is an activation function that converts the output to a probability between 0 and 1.
  - `nn.MSELoss()`: This is the loss function that calculates how far off the model's predictions are from the actual targets.

### Step 6: Define the Forward Pass

```python
    def forward(self, input):
        x = self.input_layer(input)
        x = self.sigmoid(x)
        x = self.output_layer(x)
        return x
```

**Explanation:**

- The `forward` method defines how data passes through the model.
- It takes `input`, passes it through the input layer, applies the sigmoid function, and finally passes it through the output layer to produce a prediction.

### Step 7: Configure the Optimizer

```python
    def configure_optimizers(self):
        optimizer = optim.Adam(params=self.parameters(), lr=0.01)
        return optimizer
```

**Explanation:**

- This method sets up the optimizer used for training the model. In this case, we use the **Adam** optimizer, which is popular for training neural networks. 
- `lr=0.01` sets the learning rate, which determines how quickly the model learns.

### Step 8: Define the Training Step

```python
    def training_step(self, batch, batch_idx):
        xor_input, xor_target = batch
        outputs = self(xor_input)
        loss = self.loss(outputs, xor_target)
        return loss
```

**Explanation:**

- The `training_step` method is called for each batch of data during training.
- `xor_input, xor_target = batch`: This extracts the input and target values from the current batch.
- `outputs = self(xor_input)`: This line makes a prediction using the model.
- `loss = self.loss(outputs, xor_target)`: This calculates the loss by comparing the model’s output to the actual target.

### Step 9: Train the Model

```python
checkpoint_callback = ModelCheckpoint()
model = XORModel()
trainer = pl.Trainer(max_epochs=300, callbacks=[checkpoint_callback])
trainer.fit(model, train_loader)
```

**Explanation:**

- `ModelCheckpoint()`: This will save the model at different stages during training, allowing you to keep the best version.
- `XORModel()`: We create an instance of our model.
- `pl.Trainer()`: This initializes the training process. `max_epochs=300` means the model will train for up to 300 epochs (passes over the entire dataset).
- `trainer.fit()`: This starts the training process using our model and training data.

### Step 10: Test the Model

```python
# Load the best model
trained_model = XORModel.load_from_checkpoint(checkpoint_callback.best_model_path)

# Test the model
for val in xor_inputs:
    print("Input:", val)
    output = trained_model(val)
    print("Output:", output)
```

**Explanation:**

- `load_from_checkpoint()`: This loads the best-performing model from the training process.
- The `for` loop goes through each input in `xor_inputs`, prints the input, and then gets the output from the trained model.


## **First Pitfall: Training and Device Issues**

When I first trained the model, I encountered an issue with the device setup—my inputs were on the CPU, while my model was using an MPS (Apple Silicon) device. This led to runtime errors.

### **Solution: Check Device Compatibility**

To solve this, I moved both the model and the data to the correct device using:

```python
device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
xor_inputs = [x.to(device) for x in xor_inputs]
xor_targets = [y.to(device) for y in xor_targets]
```

---

## **Second Pitfall: Unstable Output Values**

After fixing the device issue, I ran the training process, but the outputs were strange and did not reflect the XOR truth table.

```
Input: tensor([0., 0.], device='mps:0')
Output: tensor([0.1465], device='mps:0', grad_fn=<LinearBackward0>)
Input: tensor([0., 1.], device='mps:0')
Output: tensor([0.7000], device='mps:0', grad_fn=<LinearBackward0>)
Input: tensor([1., 0.], device='mps:0')
Output: tensor([0.5763], device='mps:0', grad_fn=<LinearBackward0>)
Input: tensor([1., 1.], device='mps:0')
Output: tensor([0.5607], device='mps:0', grad_fn=<LinearBackward0>)
```

This was a clear sign that the network was failing to learn the XOR function.

### **Solution: Change the Activation Function**

It turned out that using `Sigmoid` in the hidden layers caused saturation. This meant the gradients weren't flowing properly during training. I switched to the `ReLU` activation function for the hidden layers:

```python
self.relu = nn.ReLU()
```

---

### **Third Pitfall: Choosing the Right Loss Function**

I initially used Mean Squared Error (MSE) as my loss function. While this might seem intuitive for a regression-style problem, it turned out to be a poor choice for XOR, where we're dealing with binary classification (0 or 1).


### Expected XOR Outputs
The truth table for XOR is:

| Input A | Input B | Output (Expected) |
|---------|---------|-------------------|
| 0       | 0       | 0                 |
| 0       | 1       | 1                 |
| 1       | 0       | 1                 |
| 1       | 1       | 0                 |

### Model Outputs

1. **Input: tensor([0., 0.])**
   - Output: **0.0018** (Expected: **0**)
   
2. **Input: tensor([0., 1.])**
   - Output: **0.9996** (Expected: **1**)
   
3. **Input: tensor([1., 0.])**
   - Output: **0.9995** (Expected: **1**)
   
4. **Input: tensor([1., 1.])**
   - Output: **1.4948e-05** (Expected: **0**)

### Analysis of the Outputs
- The model is correctly identifying the outputs for the cases where the inputs differ (0, 1) and (1, 0), as both output values are close to **1**. 
- However, the outputs for (0, 0) and (1, 1) are very close to **0**, but they are not exactly zero. The model is giving non-zero outputs for these cases, which indicates it is struggling to fully learn the XOR function.
- It seems to be leaning towards **1** for inputs that should return **0**, which indicates a learning issue.

### Potential Reasons and Solutions
1. **Learning Rate**: Your learning rate might be too high or too low. Try adjusting it to see if the model performs better.

2. **Model Architecture**: Ensure the architecture is appropriate for learning XOR. A more complex network with more hidden neurons may be necessary. You can try increasing the number of neurons in the hidden layers.

3. **Activation Functions**: If you’re using only the output layer's linear activation without a sigmoid or other activation function, try using `torch.sigmoid()` when getting outputs in the testing phase.

4. **Loss Function**: The choice of loss function is crucial in training a neural network. For binary classification problems like XOR, Binary Cross Entropy (BCE) is often a good choice. If you're currently using a different loss function, try switching to BCE. If you're already using BCE and still encountering issues, consider using `BCEWithLogitsLoss`, which combines a sigmoid activation function and BCE loss in one class, making it more numerically stable.

5. **Training Duration**: If the model has not been trained long enough, consider increasing the number of epochs to give it more time to learn.


#### **Solution: Use `BCEWithLogitsLoss`**

`BCEWithLogitsLoss` combines both the sigmoid activation and the binary cross-entropy loss into one function, making it numerically more stable. I updated my loss function:

```python
self.loss = nn.BCEWithLogitsLoss()
```

---

## **Fourth Pitfall: Insufficient Training Time**

Even after switching the activation function and loss function, my model wasn’t fully converging. After some experimentation, I realized I needed more epochs to allow the model to converge fully.

### **Solution: Increase Epochs and Tune Learning Rate**

I increased the number of epochs and fine-tuned the learning rate:

```python
trainer = pl.Trainer(max_epochs=1000)
```

---

## **Success: The Model Finally Learns XOR**

After applying all these changes, the model started producing the expected results:

```text
Input: [0, 0] → Output: ~0
Input: [0, 1] → Output: ~1
Input: [1, 0] → Output: ~1
Input: [1, 1] → Output: ~0
```

This was the correct behavior for XOR!

---

## **Key Learnings**

Here are the main takeaways from this experience:
- **Activation Functions Matter**: Avoid using `Sigmoid` in hidden layers for non-linear problems—use `ReLU` or `tanh` instead.
- **Right Loss for the Right Task**: Use `BCEWithLogitsLoss` for binary classification tasks instead of MSE.
- **Check Devices**: Make sure that all data and model parameters are moved to the appropriate device (CPU, GPU, or MPS).
- **Allow Time for Convergence**: If the model isn't converging, increasing the number of epochs and fine-tuning the learning rate can help.

## Final Code (After improvements to get the model to 'learn' XOR)

```python
import torch
from torch import nn, optim
from torch.autograd import Variable
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.utils.data import DataLoader

# Check if MPS is available and set the device accordingly
device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
print("Using device:", device)

# XOR inputs and targets
xor_inputs = [
    Variable(torch.Tensor([0, 0]).to(device)),
    Variable(torch.Tensor([0, 1]).to(device)),
    Variable(torch.Tensor([1, 0]).to(device)),
    Variable(torch.Tensor([1, 1]).to(device)),
]
xor_targets = [
    Variable(torch.Tensor([0]).to(device)),
    Variable(torch.Tensor([1]).to(device)),
    Variable(torch.Tensor([1]).to(device)),
    Variable(torch.Tensor([0]).to(device)),
]

xor_data = list(zip(xor_inputs, xor_targets))
train_loader = DataLoader(xor_data, batch_size=1, shuffle=True)

class XORModel(pl.LightningModule):
    def __init__(self):
        super().__init__()
        # Input layer: 2 input features, 8 neurons in hidden layer
        self.input_layer = nn.Linear(2, 8)
        # Hidden layer: 8 neurons in, 4 neurons out
        self.hidden_layer = nn.Linear(8, 4)
        # Output layer: 4 neurons in, 1 output
        self.output_layer = nn.Linear(4, 1)
        # ReLU for hidden layers
        self.relu = nn.ReLU()
        # Loss function: Binary Cross Entropy with Logits
        self.loss = nn.BCEWithLogitsLoss()

        # Initialize weights explicitly
        torch.nn.init.xavier_uniform_(self.input_layer.weight)
        torch.nn.init.xavier_uniform_(self.hidden_layer.weight)
        torch.nn.init.xavier_uniform_(self.output_layer.weight)

    def forward(self, input):
        x = self.input_layer(input)
        x = self.relu(x)  # Use ReLU for better gradient flow
        x = self.hidden_layer(x)
        x = self.relu(x)
        x = self.output_layer(x)
        return x  # No sigmoid here because BCEWithLogitsLoss handles it

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=0.01)  # Increased LR
        return optimizer

    def training_step(self, batch, batch_idx):
        xor_input, xor_target = batch
        outputs = self(xor_input)
        loss = self.loss(outputs, xor_target)
        self.log('train_loss', loss)
        return loss

# Initialize the model and move to the same device
model = XORModel().to(device)

# Train the model
checkpoint_callback = ModelCheckpoint()
trainer = pl.Trainer(max_epochs=1000, callbacks=[checkpoint_callback])  # Increased epochs
trainer.fit(model, train_loader)

print("Training Complete")
print("Best model path:", checkpoint_callback.best_model_path)

# Load the best model checkpoint
trained_model = XORModel.load_from_checkpoint(checkpoint_callback.best_model_path).to(device)

# Test the model with inputs
for val in xor_inputs:
    val = val.to(device)
    output = torch.sigmoid(trained_model(val))  # Apply sigmoid at test time
    print("Input:", val)
    print("Output:", output)

```

## Conclusion

This blog post has walked you through the implementation of a neural network to solve the XOR problem using PyTorch Lightning. We discussed the significance of XOR in neural network training, compared it to decision trees, and highlighted its real-world applications. By understanding the nuances of the XOR problem, you can better appreciate the capabilities of neural networks and their potential applications across various domains.

### Key Takeaways

- The XOR problem is a classic example of non-linear learning, making it a valuable benchmark for neural networks.
- Neural networks provide greater flexibility and robustness compared to traditional models like decision trees.
- The ability to model complex interactions in data allows neural networks to excel in various real-world applications.

With this knowledge, you’re better equipped to tackle similar problems and implement neural networks for a wide range of tasks!
