# PyTorch Core Functions and Functionalities

This document provides an overview of major PyTorch functions and functionalities, with references to their source files in the repository, short code snippets, and comments.

## ASCII Tree of Key PyTorch Directories

```
torch/
├── _tensor.py
├── autograd/
│   └── __init__.py
└── nn/
    ├── functional.py
    ├── __init__.py
    └── modules/
        ├── __init__.py
        ├── module.py
        ├── linear.py
        ├── conv.py
        ├── rnn.py
        ├── dropout.py
        └── batchnorm.py
```

## Core Components

### `torch.Tensor`

A `torch.Tensor` is a multi-dimensional matrix containing elements of a single data type. It's the fundamental data structure in PyTorch, similar to NumPy arrays, but with the added capability of running on GPUs to accelerate computation and tracking gradients for automatic differentiation.

**File Reference:** [torch/_tensor.py](torch/_tensor.py)

**Code Snippet:**
```python
import torch

# Create a tensor from a list
data = [[1, 2], [3, 4]]
x_data = torch.tensor(data)
print(f"Tensor from list:\\n{x_data}")

# Create a tensor of random numbers
x_rand = torch.rand(2, 3) # Creates a 2x3 tensor with random values between 0 and 1
print(f"Random tensor:\\n{x_rand}")

# Create a tensor of zeros
x_zeros = torch.zeros(2, 2, dtype=torch.float16)
print(f"Zeros tensor (float16):\\n{x_zeros}")

# Get tensor properties
print(f"Shape of tensor: {x_data.shape}")
print(f"Datatype of tensor: {x_data.dtype}")
print(f"Device tensor is stored on: {x_data.device}")
```

### `torch.autograd`

The `torch.autograd` package provides classes and functions implementing automatic differentiation of arbitrary scalar valued functions. It is the core of PyTorch's ability to train neural networks. When a tensor has `requires_grad=True`, PyTorch tracks all operations on it. When you finish your computation you can call `.backward()` and have all the gradients computed automatically. The gradient for a tensor will be accumulated into its `.grad` attribute.

**File Reference:** [torch/autograd/__init__.py](torch/autograd/__init__.py)

**Code Snippet:**
```python
import torch

# Create a tensor and set requires_grad=True to track computation
x = torch.ones(2, 2, requires_grad=True)
print(f"Tensor x:\\n{x}")

# Perform some tensor operations
y = x + 2
print(f"Tensor y (x + 2):\\n{y}")
z = y * y * 3
out = z.mean()
print(f"Tensor z (y*y*3):\\n{z}")
print(f"Tensor out (z.mean()):\\n{out}")

# Compute gradients
out.backward() # out is a scalar, so no gradient argument is needed

# Print gradients d(out)/dx
print(f"Gradients of out with respect to x (x.grad):\\n{x.grad}")
# x.grad will be a tensor of shape (2,2)
# For a single element x_i, out = (1/4) * sum(3 * (x_i+2)^2)
# d(out)/dx_i = (1/4) * 3 * 2 * (x_i+2) = (3/2) * (x_i+2)
# For x_i = 1, d(out)/dx_i = (3/2) * 3 = 4.5
```

### `torch.nn.functional`

The `torch.nn.functional` module (usually imported as `F`) contains functions that are used to build neural networks. These are typically stateless versions of the layers found in `torch.nn.modules`. For example, `F.relu` is the functional version of the `nn.ReLU` module. Using functional interfaces can be more flexible for certain network architectures, especially when you don't need to store parameters.

**File Reference:** [torch/nn/functional.py](torch/nn/functional.py)

#### Convolution Functions

These functions apply convolution operations.

*   **`F.conv1d(input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1)`**
    *   Applies a 1D convolution.
    *   **Snippet:**
        ```python
        import torch
        import torch.nn.functional as F

        inputs = torch.randn(32, 20, 50)  # (batch_size, in_channels, input_length)
        weights = torch.randn(10, 20, 5)   # (out_channels, in_channels/groups, kernel_size)
        output = F.conv1d(inputs, weights, padding=2)
        print(f"Conv1d output shape: {output.shape}") # Expected: (32, 10, 50)
        ```

*   **`F.conv2d(input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1)`**
    *   Applies a 2D convolution.
    *   **Snippet:**
        ```python
        inputs = torch.randn(32, 3, 28, 28)  # (batch, in_channels, height, width)
        weights = torch.randn(16, 3, 3, 3)    # (out_channels, in_channels/groups, kernel_H, kernel_W)
        output = F.conv2d(inputs, weights, padding=1)
        print(f"Conv2d output shape: {output.shape}") # Expected: (32, 16, 28, 28)
        ```

*   **`F.conv3d(input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1)`**
    *   Applies a 3D convolution.
    *   **Snippet:**
        ```python
        inputs = torch.randn(32, 3, 16, 28, 28)  # (batch, in_channels, depth, height, width)
        weights = torch.randn(16, 3, 3, 3, 3)     # (out_channels, in_channels/groups, kernel_D, kernel_H, kernel_W)
        output = F.conv3d(inputs, weights, padding=1)
        print(f"Conv3d output shape: {output.shape}") # Expected: (32, 16, 16, 28, 28)
        ```

#### Pooling Functions

These functions apply pooling operations.

*   **`F.avg_pool1d(input, kernel_size, stride=None, padding=0, ...)`**
    *   Applies 1D average pooling.
    *   **Snippet:**
        ```python
        input_tensor = torch.tensor([[[1., 2., 3., 4., 5.]]]) # (batch, channels, length)
        output = F.avg_pool1d(input_tensor, kernel_size=2, stride=2)
        print(f"AvgPool1d output: {output}") # Expected: [[[1.5, 3.5]]]
        ```

*   **`F.avg_pool2d(input, kernel_size, stride=None, padding=0, ...)`**
    *   Applies 2D average pooling.
    *   **Snippet:**
        ```python
        input_tensor = torch.randn(1, 3, 4, 4) # (batch, channels, height, width)
        output = F.avg_pool2d(input_tensor, kernel_size=2, stride=2)
        print(f"AvgPool2d output shape: {output.shape}") # Expected: (1, 3, 2, 2)
        ```

*   **`F.avg_pool3d(input, kernel_size, stride=None, padding=0, ...)`**
    *   Applies 3D average pooling.
    *   **Snippet:**
        ```python
        input_tensor = torch.randn(1, 3, 4, 4, 4) # (batch, channels, depth, height, width)
        output = F.avg_pool3d(input_tensor, kernel_size=2, stride=2)
        print(f"AvgPool3d output shape: {output.shape}") # Expected: (1, 3, 2, 2, 2)
        ```

*   **`F.max_pool1d(input, kernel_size, stride=None, padding=0, ...)`**
    *   Applies 1D max pooling.
    *   **Snippet:**
        ```python
        input_tensor = torch.tensor([[[1., 2., 3., 4., 5.]]])
        output = F.max_pool1d(input_tensor, kernel_size=2, stride=2)
        print(f"MaxPool1d output: {output}") # Expected: [[[2., 4.]]]
        ```

*   **`F.max_pool2d(input, kernel_size, stride=None, padding=0, ...)`**
    *   Applies 2D max pooling.
    *   **Snippet:**
        ```python
        input_tensor = torch.randn(1, 3, 4, 4)
        output = F.max_pool2d(input_tensor, kernel_size=2, stride=2)
        print(f"MaxPool2d output shape: {output.shape}") # Expected: (1, 3, 2, 2)
        ```

*   **`F.max_pool3d(input, kernel_size, stride=None, padding=0, ...)`**
    *   Applies 3D max pooling.
    *   **Snippet:**
        ```python
        input_tensor = torch.randn(1, 3, 4, 4, 4)
        output = F.max_pool3d(input_tensor, kernel_size=2, stride=2)
        print(f"MaxPool3d output shape: {output.shape}") # Expected: (1, 3, 2, 2, 2)
        ```

#### Activation Functions

These functions apply non-linear activations.

*   **`F.relu(input, inplace=False)`**
    *   Applies the Rectified Linear Unit function element-wise.
    *   **Snippet:**
        ```python
        input_tensor = torch.randn(2, 2)
        output = F.relu(input_tensor)
        print(f"ReLU input:\\n{input_tensor}\\nReLU output:\\n{output}")
        ```

*   **`F.sigmoid(input)`**
    *   Applies the sigmoid function element-wise.
    *   **Snippet:**
        ```python
        input_tensor = torch.randn(2, 2)
        output = F.sigmoid(input_tensor)
        print(f"Sigmoid input:\\n{input_tensor}\\nSigmoid output:\\n{output}")
        ```

*   **`F.tanh(input)`**
    *   Applies the hyperbolic tangent function element-wise.
    *   **Snippet:**
        ```python
        input_tensor = torch.randn(2, 2)
        output = F.tanh(input_tensor)
        print(f"Tanh input:\\n{input_tensor}\\nTanh output:\\n{output}")
        ```

#### Linear Function

*   **`F.linear(input, weight, bias=None)`**
    *   Applies a linear transformation: `y = xA^T + b`.
    *   **Snippet:**
        ```python
        input_tensor = torch.randn(128, 20)  # (batch_size, in_features)
        weights = torch.randn(30, 20)     # (out_features, in_features)
        bias_tensor = torch.randn(30)        # (out_features)
        output = F.linear(input_tensor, weights, bias_tensor)
        print(f"Linear output shape: {output.shape}") # Expected: (128, 30)
        ```

#### Dropout Function

*   **`F.dropout(input, p=0.5, training=True, inplace=False)`**
    *   During training, randomly zeroes some elements of the input tensor with probability `p`.
    *   **Snippet:**
        ```python
        input_tensor = torch.randn(1, 10)
        # training=True is important, otherwise dropout does nothing
        output_train = F.dropout(input_tensor, p=0.5, training=True)
        output_eval = F.dropout(input_tensor, p=0.5, training=False)
        print(f"Dropout input:\\n{input_tensor}")
        print(f"Dropout output (training=True):\\n{output_train}") # Some elements will be zero
        print(f"Dropout output (training=False):\\n{output_eval}") # Should be same as input
        ```

#### Normalization Functions

*   **`F.batch_norm(input, running_mean, running_var, weight=None, bias=None, training=False, momentum=0.1, eps=1e-05)`**
    *   Applies Batch Normalization.
    *   **Snippet:**
        ```python
        input_tensor = torch.randn(20, 100) # (batch_size, features)
        # For training, running_mean and running_var are updated.
        # For evaluation, they are used for normalization.
        running_mean = torch.zeros(100)
        running_var = torch.ones(100)
        weights = torch.rand(100)
        biases = torch.rand(100)
        output = F.batch_norm(input_tensor, running_mean, running_var, weight=weights, bias=biases, training=True)
        print(f"BatchNorm output shape: {output.shape}") # Expected: (20, 100)
        print(f"Updated running_mean (first 5): {running_mean[:5]}") # Will be updated
        ```

#### Loss Functions

*   **`F.cross_entropy(input, target, weight=None, reduction='mean', ignore_index=-100, label_smoothing=0.0)`**
    *   Computes the cross entropy loss between input logits and target.
    *   **Snippet:**
        ```python
        input_logits = torch.randn(3, 5) # (batch_size, num_classes)
        target_indices = torch.tensor([1, 0, 4]) # (batch_size)
        loss = F.cross_entropy(input_logits, target_indices)
        print(f"CrossEntropyLoss: {loss.item()}")
        ```

*   **`F.mse_loss(input, target, reduction='mean')`**
    *   Computes the mean squared error (squared L2 norm) between each element in the input `x` and target `y`.
    *   **Snippet:**
        ```python
        input_preds = torch.randn(3, 5)
        target_values = torch.randn(3, 5)
        loss = F.mse_loss(input_preds, target_values)
        print(f"MSELoss: {loss.item()}")
        ```

### `torch.nn.Module`

`torch.nn.Module` is the base class for all neural network modules in PyTorch. Your models should also subclass this class. Modules can also contain other Modules, allowing to nest them in a tree structure. You can assign submodules as regular attributes:

**File Reference:** [torch/nn/modules/module.py](torch/nn/modules/module.py)

**Code Snippet:**
```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        # Define a linear layer: 10 input features, 20 output features
        self.fc1 = nn.Linear(10, 20)
        # Define another linear layer: 20 input features, 5 output features
        self.fc2 = nn.Linear(20, 5)

    def forward(self, x):
        # Apply ReLU activation after the first linear layer
        x = F.relu(self.fc1(x))
        # Apply the second linear layer
        x = self.fc2(x)
        # Apply log_softmax to the output
        return F.log_softmax(x, dim=1)

# Create an instance of the model
model = SimpleModel()
print("Model structure:")
print(model)

# Create a dummy input tensor (batch_size=4, in_features=10)
dummy_input = torch.randn(4, 10)
output = model(dummy_input)
print(f"Output shape: {output.shape}") # Expected: (4, 5)

# Print model parameters
print("\\nModel parameters:")
for name, param in model.named_parameters():
    if param.requires_grad:
        print(name, param.data.shape)
```

### Core `torch.nn.modules`

These are some of the most commonly used layers when building neural networks. They are all subclasses of `nn.Module`.

#### `nn.Linear`
*   **Description:** Applies a linear transformation to the incoming data: `y = xA^T + b`.
*   **File Reference:** [torch/nn/modules/linear.py](torch/nn/modules/linear.py)
*   **Snippet:**
    ```python
    import torch
    import torch.nn as nn

    # Define a linear layer: 10 input features, 5 output features
    linear_layer = nn.Linear(in_features=10, out_features=5)
    print(f"Linear layer: {linear_layer}")

    # Create a dummy input (batch_size=4, in_features=10)
    input_tensor = torch.randn(4, 10)
    output = linear_layer(input_tensor)
    print(f"Output shape from Linear layer: {output.shape}") # Expected: (4, 5)
    print(f"Linear layer weight shape: {linear_layer.weight.shape}") # Expected: (5, 10)
    print(f"Linear layer bias shape: {linear_layer.bias.shape}") # Expected: (5,)
    ```

#### Convolution Layers (`nn.Conv1d`, `nn.Conv2d`, `nn.Conv3d`)
*   **Description:** Apply 1D, 2D, or 3D convolution operations. These layers are fundamental for processing spatial or temporal data, like images or sequences.
*   **File Reference:** [torch/nn/modules/conv.py](torch/nn/modules/conv.py)
*   **Snippets:**
    *   `nn.Conv1d`:
        ```python
        # 1D Convolutional Layer
        # Expects input of shape (N, C_in, L_in) or (C_in, L_in)
        # Outputs shape (N, C_out, L_out) or (C_out, L_out)
        conv1d_layer = nn.Conv1d(in_channels=16, out_channels=33, kernel_size=3, stride=2, padding=1)
        input_1d = torch.randn(20, 16, 50) # (batch_size, in_channels, length)
        output_1d = conv1d_layer(input_1d)
        print(f"Conv1d output shape: {output_1d.shape}") # Expected: (20, 33, 25)
        ```
    *   `nn.Conv2d`:
        ```python
        # 2D Convolutional Layer
        # Expects input of shape (N, C_in, H_in, W_in) or (C_in, H_in, W_in)
        # Outputs shape (N, C_out, H_out, W_out) or (C_out, H_out, W_out)
        conv2d_layer = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1)
        input_2d = torch.randn(10, 3, 28, 28) # (batch_size, in_channels, height, width)
        output_2d = conv2d_layer(input_2d)
        print(f"Conv2d output shape: {output_2d.shape}") # Expected: (10, 32, 28, 28)
        ```
    *   `nn.Conv3d`:
        ```python
        # 3D Convolutional Layer
        # Expects input of shape (N, C_in, D_in, H_in, W_in) or (C_in, D_in, H_in, W_in)
        # Outputs shape (N, C_out, D_out, H_out, W_out) or (C_out, D_out, H_out, W_out)
        conv3d_layer = nn.Conv3d(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1)
        input_3d = torch.randn(5, 3, 10, 28, 28) # (batch_size, in_channels, depth, height, width)
        output_3d = conv3d_layer(input_3d)
        print(f"Conv3d output shape: {output_3d.shape}") # Expected: (5, 16, 10, 28, 28)
        ```

#### Recurrent Layers (`nn.RNN`, `nn.LSTM`, `nn.GRU`)
*   **Description:** Process sequential data. `RNN` is the basic recurrent layer. `LSTM` (Long Short-Term Memory) and `GRU` (Gated Recurrent Unit) are more advanced versions designed to handle long-range dependencies and vanishing/exploding gradient problems.
*   **File Reference:** [torch/nn/modules/rnn.py](torch/nn/modules/rnn.py)
*   **Snippets:**
    *   `nn.RNN`:
        ```python
        # RNN Layer
        rnn_layer = nn.RNN(input_size=10, hidden_size=20, num_layers=2, batch_first=True)
        # Input shape: (batch, seq_len, input_size) if batch_first=True
        input_seq = torch.randn(5, 3, 10) # (batch_size, sequence_length, features)
        h0 = torch.randn(2*1, 5, 20) # (num_layers * num_directions, batch_size, hidden_size)
        output_seq, hn = rnn_layer(input_seq, h0)
        print(f"RNN output shape: {output_seq.shape}") # Expected: (5, 3, 20)
        print(f"RNN hidden state shape: {hn.shape}")    # Expected: (2, 5, 20)
        ```
    *   `nn.LSTM`:
        ```python
        # LSTM Layer
        lstm_layer = nn.LSTM(input_size=10, hidden_size=20, num_layers=2, batch_first=True)
        input_seq = torch.randn(5, 3, 10)
        h0 = torch.randn(2*1, 5, 20) # (num_layers * num_directions, batch, hidden_size)
        c0 = torch.randn(2*1, 5, 20) # (num_layers * num_directions, batch, cell_state_size)
        output_seq, (hn, cn) = lstm_layer(input_seq, (h0, c0))
        print(f"LSTM output shape: {output_seq.shape}") # Expected: (5, 3, 20)
        print(f"LSTM hidden state shape: {hn.shape}")    # Expected: (2, 5, 20)
        print(f"LSTM cell state shape: {cn.shape}")      # Expected: (2, 5, 20)
        ```
    *   `nn.GRU`:
        ```python
        # GRU Layer
        gru_layer = nn.GRU(input_size=10, hidden_size=20, num_layers=2, batch_first=True)
        input_seq = torch.randn(5, 3, 10)
        h0 = torch.randn(2*1, 5, 20)
        output_seq, hn = gru_layer(input_seq, h0)
        print(f"GRU output shape: {output_seq.shape}") # Expected: (5, 3, 20)
        print(f"GRU hidden state shape: {hn.shape}")    # Expected: (2, 5, 20)
        ```

#### `nn.Dropout`
*   **Description:** During training, randomly zeroes some of the elements of the input tensor with probability `p`. This helps prevent overfitting.
*   **File Reference:** [torch/nn/modules/dropout.py](torch/nn/modules/dropout.py)
*   **Snippet:**
    ```python
    # Dropout Layer
    dropout_layer = nn.Dropout(p=0.5)
    input_tensor = torch.randn(1, 10)
    # Apply dropout (implicitly uses training=True if model is in train mode)
    model_dropout = nn.Sequential(dropout_layer)
    model_dropout.train() # Set to training mode
    output_train = model_dropout(input_tensor)
    model_dropout.eval()  # Set to evaluation mode
    output_eval = model_dropout(input_tensor)
    print(f"Dropout input:\\n{input_tensor}")
    print(f"Dropout output (training):\\n{output_train}") # Some elements likely zeroed
    print(f"Dropout output (evaluation):\\n{output_eval}") # Should be same as input
    ```

#### Batch Normalization Layers (`nn.BatchNorm1d`, `nn.BatchNorm2d`, `nn.BatchNorm3d`)
*   **Description:** Applies Batch Normalization over a 1D, 2D, or 3D input. Batch Normalization helps stabilize and accelerate training by normalizing the inputs to a layer for each mini-batch.
*   **File Reference:** [torch/nn/modules/batchnorm.py](torch/nn/modules/batchnorm.py)
*   **Snippets:**
    *   `nn.BatchNorm1d`:
        ```python
        # 1D Batch Normalization
        # Expects input of shape (N, C) or (N, C, L)
        bn1d_layer = nn.BatchNorm1d(num_features=100) # num_features is C
        input_1d_bn = torch.randn(20, 100) # (batch_size, features)
        output_1d_bn = bn1d_layer(input_1d_bn)
        print(f"BatchNorm1d output shape: {output_1d_bn.shape}") # Expected: (20, 100)
        # Check running_mean and running_var (updated during training)
        print(f"BatchNorm1d running mean (first 5): {bn1d_layer.running_mean[:5]}")
        ```
    *   `nn.BatchNorm2d`:
        ```python
        # 2D Batch Normalization
        # Expects input of shape (N, C, H, W)
        bn2d_layer = nn.BatchNorm2d(num_features=3) # num_features is C
        input_2d_bn = torch.randn(10, 3, 28, 28) # (batch_size, channels, height, width)
        output_2d_bn = bn2d_layer(input_2d_bn)
        print(f"BatchNorm2d output shape: {output_2d_bn.shape}") # Expected: (10, 3, 28, 28)
        ```
    *   `nn.BatchNorm3d`:
        ```python
        # 3D Batch Normalization
        # Expects input of shape (N, C, D, H, W)
        bn3d_layer = nn.BatchNorm3d(num_features=3) # num_features is C
        input_3d_bn = torch.randn(5, 3, 10, 28, 28) # (batch_size, channels, depth, height, width)
        output_3d_bn = bn3d_layer(input_3d_bn)
        print(f"BatchNorm3d output shape: {output_3d_bn.shape}") # Expected: (5, 3, 10, 28, 28)
        ```
