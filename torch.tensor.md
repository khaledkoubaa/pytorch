# torch.tensor

Constructs a tensor with no autograd history (also known as a “leaf tensor”) by copying data.

## Syntax

```python
torch.tensor(data, *, dtype=None, device=None, requires_grad=False, pin_memory=False) -> Tensor
```

### Parameters

*   **data** (`array_like`): Initial data for the tensor. Can be a list, tuple, NumPy ndarray, scalar, and other types.
*   **dtype** (`torch.dtype`, optional): The desired data type of the returned tensor. Default: if `None`, infers data type from `data`.
*   **device** (`torch.device`, optional): The device of the constructed tensor. If `None` and `data` is a tensor, the device of `data` is used. If `None` and `data` is not a tensor, the result tensor is constructed on the current device.
*   **requires_grad** (`bool`, optional): If autograd should record operations on the returned tensor. Default: `False`.
*   **pin_memory** (`bool`, optional): If set, the returned tensor would be allocated in the pinned memory. Works only for CPU tensors. Default: `False`.

## In-depth explanation of Parameters

### `data`

The `data` parameter is the most crucial argument for `torch.tensor()`. It accepts various array-like objects and scalars to initialize the tensor. PyTorch is highly flexible in the types of data it can handle.

#### Accepted Data Types:

1.  **Lists or Tuples:** You can create tensors from Python lists or tuples. Nested lists or tuples will result in multi-dimensional tensors.

    ```python
    import torch

    # From a list
    list_data = [[1, 2], [3, 4]]
    tensor_from_list = torch.tensor(list_data)
    print(tensor_from_list)
    # tensor([[1, 2],
    #         [3, 4]])

    # From a tuple
    tuple_data = ((1.0, 2.0), (3.0, 4.0))
    tensor_from_tuple = torch.tensor(tuple_data)
    print(tensor_from_tuple)
    # tensor([[1., 2.],
    #         [3., 4.]])
    ```

2.  **NumPy Arrays:** `torch.tensor()` can create a tensor from a NumPy `ndarray`. It's important to note that this function *copies* the data. If you want to create a tensor that shares the underlying memory with the NumPy array (to avoid a copy), use `torch.from_numpy()`.

    ```python
    import numpy as np

    numpy_array = np.array([[1, 2], [3, 4]])
    tensor_from_numpy = torch.tensor(numpy_array)
    print(tensor_from_numpy)
    # tensor([[1, 2],
    #         [3, 4]], dtype=torch.int64)
    ```

3.  **Scalars:** You can create a zero-dimensional tensor (a scalar) by passing a single number.

    ```python
    scalar_tensor = torch.tensor(3.14)
    print(scalar_tensor)
    # tensor(3.1400)
    ```

4.  **Other Tensors:** If you pass an existing tensor to `torch.tensor()`, it will create a *copy* of the tensor. This is equivalent to using `existing_tensor.clone().detach()`.

    ```python
    existing_tensor = torch.tensor([1, 2, 3])
    new_tensor = torch.tensor(existing_tensor)
    new_tensor[0] = 100
    print(existing_tensor) # The original tensor is unchanged
    # tensor([1, 2, 3])
    print(new_tensor)
    # tensor([100, 2, 3])
    ```

### `dtype`

The `dtype` parameter allows you to specify the data type of the resulting tensor. If you don't specify it, PyTorch will infer the type from the input `data`. For example, a list of integers will result in a `torch.int64` tensor, while a list of floating-point numbers will create a `torch.float32` tensor.

#### Common Data Types:

*   `torch.float32` or `torch.float`: 32-bit floating-point.
*   `torch.float64` or `torch.double`: 64-bit floating-point.
*   `torch.int8`: Signed 8-bit integer.
*   `torch.uint8`: Unsigned 8-bit integer.
*   `torch.int16` or `torch.short`: Signed 16-bit integer.
*   `torch.int32` or `torch.int`: Signed 32-bit integer.
*   `torch.int64` or `torch.long`: Signed 64-bit integer.
*   `torch.bool`: Boolean.

#### Example of specifying `dtype`:

```python
# Create a tensor with a specific dtype
float_tensor = torch.tensor([1, 2, 3], dtype=torch.float32)
print(float_tensor)
# tensor([1., 2., 3.])

# Another example with a different dtype
long_tensor = torch.tensor([1.0, 2.0, 3.0], dtype=torch.long)
print(long_tensor)
# tensor([1, 2, 3])
```

### `device`

The `device` parameter determines the memory where the tensor will be stored. This is critical for performance, especially when using GPUs.

#### Available Devices:

*   **CPU:** `'cpu'`
*   **CUDA GPU:** `'cuda'` or `'cuda:0'` (for the first GPU), `'cuda:1'`, etc.

You can create a `torch.device` object to specify the device.

#### Example of specifying `device`:

```python
# Create a tensor on the CPU (default)
cpu_tensor = torch.tensor([1, 2, 3], device='cpu')
print(cpu_tensor.device)
# cpu

# Create a tensor on a CUDA-enabled GPU (if available)
if torch.cuda.is_available():
    cuda_tensor = torch.tensor([1, 2, 3], device='cuda')
    print(cuda_tensor.device)
    # cuda:0
else:
    print("CUDA is not available.")
```

### `requires_grad`

The `requires_grad` parameter is essential for automatic differentiation in PyTorch. If you set `requires_grad=True`, PyTorch will track all operations on the tensor, building a computational graph. This allows you to compute gradients with respect to this tensor by calling `.backward()` on a resulting scalar tensor.

This is primarily used for training neural networks, where you need to compute gradients of the loss function with respect to the model's parameters.

#### Example of `requires_grad`:

```python
# Create a tensor that requires gradients
x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
print(x.requires_grad)
# True

# Perform an operation
y = x * 2
z = y.mean()

# Compute gradients
z.backward()

# The gradients are stored in the .grad attribute of the tensor
print(x.grad)
# tensor([0.6667, 0.6667, 0.6667])
```

### `pin_memory`

Setting `pin_memory=True` allocates the tensor in "pinned" memory on the CPU. Pinned memory is a special type of memory that the host (CPU) can't page out. This is useful because it allows for much faster data transfers from the CPU to a CUDA-enabled GPU.

When you're working with GPUs and need to move a lot of data from the CPU to the GPU, using pinned memory can significantly speed up your code. You'll often see this used in data loading pipelines.

#### Example of `pin_memory`:

```python
# Create a tensor in pinned memory
pinned_tensor = torch.tensor([1, 2, 3], pin_memory=True)

# Check if the tensor is in pinned memory
print(pinned_tensor.is_pinned())
# True
```

**Note:** This option only has an effect on CPU tensors. It is typically used in conjunction with moving the tensor to a GPU.

## Comprehensive Examples

Here are some more examples that combine the different parameters of `torch.tensor`.

### 1. Creating a 2D tensor of floats on the GPU

```python
# Create a 2x3 tensor of floating-point numbers on the first CUDA device
if torch.cuda.is_available():
    gpu_tensor = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
                              dtype=torch.float32,
                              device='cuda:0')
    print(gpu_tensor)
    # tensor([[1., 2., 3.],
    #         [4., 5., 6.]], device='cuda:0')
```

### 2. Creating a tensor from a NumPy array and enabling gradients

```python
import numpy as np

# Create a NumPy array
numpy_data = np.array([10, 20, 30], dtype=np.float32)

# Create a tensor from the NumPy array that requires gradients
grad_tensor = torch.tensor(numpy_data, requires_grad=True)
print(grad_tensor)
# tensor([10., 20., 30.], requires_grad=True)
```

### 3. Creating a boolean tensor

```python
# Create a boolean tensor from a list of booleans
bool_tensor = torch.tensor([True, False, True, False])
print(bool_tensor)
# tensor([ True, False,  True, False])
```

### 4. Creating an empty tensor

You can create an empty tensor, which will have a size of `(0,)`.

```python
empty_tensor = torch.tensor([])
print(empty_tensor)
# tensor([])
print(empty_tensor.shape)
# torch.Size([0])
```

## See Also

*   [**`torch.as_tensor()`**](https://pytorch.org/docs/stable/generated/torch.as_tensor.html): Creates a tensor that shares data with the original `array_like` object whenever possible, avoiding a data copy. This is more memory-efficient if you don't need to modify the underlying data independently.

*   [**`torch.from_numpy()`**](https://pytorch.org/docs/stable/generated/torch.from_numpy.html): Specifically for creating a tensor from a NumPy array. The returned tensor and the NumPy array share the same memory. Changes to one will affect the other.

*   [**`torch.Tensor.clone()`**](https://pytorch.org/docs/stable/generated/torch.Tensor.clone.html): Creates a copy of a tensor, preserving the computation graph for autograd.

*   [**`torch.Tensor.detach()`**](https://pytorch.org/docs/stable/generated/torch.Tensor.detach.html): Creates a new tensor that shares the same storage but is detached from the computation graph, meaning gradients won't be propagated to it.
