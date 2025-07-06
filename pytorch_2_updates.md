# PyTorch 2.0: Key Updates and Cheat Sheet

## Introduction

PyTorch 2.0 marks a significant evolution for the popular deep learning framework, maintaining its beloved Pythonic feel and dynamic nature while introducing powerful under-the-hood changes. The primary goals of this release are to deliver **enhanced performance**, **improved developer experience**, and **greater scalability** for today's demanding AI workloads.

This major update brings `torch.compile` as a flagship feature, enabling substantial speedups with minimal code changes. It also includes stable Accelerated Transformers, more capable MPS backend for Apple Silicon, the integration of `functorch` into the core as `torch.func`, and new tools for easier device and distributed computing management. This document provides an overview of these key updates, complete with code examples and cheat sheets to help you leverage the new capabilities in PyTorch 2.0.

## 1. `torch.compile`
   - **Explanation:** `torch.compile` is the cornerstone of PyTorch 2.0's performance enhancements. It wraps your existing `nn.Module` and returns a compiled version. This process is designed to be largely transparent, making your PyTorch code faster without requiring significant rewrites. It achieves this by JIT (Just-In-Time) compiling parts of your model into optimized kernels. Key technologies underpinning `torch.compile` include:
     - **TorchDynamo:** Safely captures PyTorch programs by analyzing Python bytecode, allowing it to handle more dynamic Python features than previous graph capture methods.
     - **AOTAutograd:** Traces PyTorch's autograd engine to generate an ahead-of-time backward pass graph.
     - **PrimTorch:** Canonicalizes PyTorch's ~2000+ operators into a smaller set of ~250 primitive operators, simplifying the process of creating new backends.
     - **TorchInductor:** A deep learning compiler that generates fast code for various accelerators. For NVIDIA and AMD GPUs, it leverages OpenAI Triton to produce highly performant kernels. For CPUs, it generates C++ code with optimizations like vectorization and multithreading.
     The primary benefit is speed, with reported average speedups of 20-36% across many models. Importantly, `torch.compile` is a fully additive and optional feature, ensuring 100% backward compatibility with existing PyTorch code.
   - **Code Example:**
     ```python
     import torch
     import time

     # Define a simple model
     class MyModel(torch.nn.Module):
         def __init__(self, D_in, H, D_out):
             super().__init__()
             self.linear1 = torch.nn.Linear(D_in, H)
             self.relu = torch.nn.ReLU()
             self.linear2 = torch.nn.Linear(H, D_out)

         def forward(self, x):
             x = self.linear1(x)
             x = self.relu(x)
             x = self.linear2(x)
             return x

     # Model and input parameters
     N, D_in, H, D_out = 640, 1000, 100, 10
     # Ensure model and tensor are on the same device for meaningful comparison
     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
     print(f"Using device: {device}")

     model = MyModel(D_in, H, D_out).to(device)
     input_tensor = torch.randn(N, D_in).to(device)

     # Measure eager mode execution time
     # Warm-up for GPU
     if device.type == 'cuda':
         for _ in range(5):
             _ = model(input_tensor)
         torch.cuda.synchronize()

     start_time = time.time()
     for _ in range(10): # Run multiple times for stable measurement
        output_eager = model(input_tensor)
     if device.type == 'cuda':
        torch.cuda.synchronize()
     end_time = time.time()
     eager_time = (end_time - start_time) / 10
     print(f"Eager mode execution time (avg over 10 runs): {eager_time:.4f} seconds")

     # Compile the model
     # Note: First run of a compiled model includes compilation overhead.
     try:
         compiled_model = torch.compile(model)

         # Measure compiled model execution time (first run includes compile overhead)
         if device.type == 'cuda':
             torch.cuda.synchronize()
         start_time = time.time()
         output_compiled_first_run = compiled_model(input_tensor)
         if device.type == 'cuda':
             torch.cuda.synchronize()
         end_time = time.time()
         print(f"Compiled model execution time (first run with compile): {end_time - start_time:.4f} seconds")

         # Measure compiled model execution time (second run, post-compilation)
         # Warm-up for compiled model on GPU
         if device.type == 'cuda':
             for _ in range(5):
                 _ = compiled_model(input_tensor)
             torch.cuda.synchronize()

         start_time = time.time()
         for _ in range(10): # Run multiple times
            output_compiled_subsequent_run = compiled_model(input_tensor)
         if device.type == 'cuda':
            torch.cuda.synchronize()
         end_time = time.time()
         compiled_time = (end_time - start_time) / 10
         print(f"Compiled model execution time (avg over 10 runs, post-compile): {compiled_time:.4f} seconds")

         if eager_time > 0 and compiled_time > 0:
             print(f"Speedup: {eager_time / compiled_time:.2f}x")

         # Verify outputs are close (optional)
         # print("Output from eager model (sum):", output_eager.sum().item())
         # print("Output from compiled model (sum):", output_compiled_subsequent_run.sum().item())
         # assert torch.allclose(output_eager, output_compiled_subsequent_run, atol=1e-5), "Outputs differ significantly"

     except Exception as e:
         print(f"torch.compile failed or encountered an issue: {e}")
         print("This might be due to specific ops not yet supported by the chosen backend or other reasons.")

     ```
   - **Cheat Sheet:**
     - **Usage:** `compiled_model = torch.compile(model, mode=None, dynamic=False, fullgraph=False, backend='inductor', options=None)`
     - **`model`**: Your `torch.nn.Module` instance.
     - **`mode`** (Optional): Optimizes for specific goals.
       - `None` (default): Balances compilation time and speedup.
       - `'reduce-overhead'`: Minimizes framework overhead for small models or large batches.
       - `'max-autotune'`: Compiles longer to find the fastest configuration, good for very long-running models.
     - **`dynamic=True`** (Optional, Experimental): Enables support for dynamic shapes in your model (e.g., variable sequence lengths). Performance might vary.
     - **`fullgraph=False`** (Optional): If `True`, TorchDynamo will error out if it cannot convert the entire model into a single graph (strict mode). Default `False` allows graph breaks.
     - **`backend`** (Optional): Specifies the compiler backend.
       - `'inductor'` (default): PyTorch's own compiler, uses Triton for GPUs, C++ for CPUs.
       - `'cudagraphs'`: Uses CUDA Graphs for some GPU speedups, often less general than Inductor.
       - `'ipex'`: For Intel CPU optimizations.
       - Other specialized backends exist (e.g., for specific hardware).
     - **`options`** (Optional, dict): Backend-specific options.

## 2. Accelerated Transformers & Scaled Dot Product Attention (SDPA)
   - **Explanation:** PyTorch 2.0 significantly boosts the performance of Transformer models. This is achieved through "Accelerated Transformers," which integrate custom, high-performance kernels, particularly for the scaled dot product attention (SDPA) mechanism. The `torch.nn.functional.scaled_dot_product_attention` function is a key part of this, providing a direct way to access these optimized attention implementations. It can automatically select the best available kernel (like FlashAttention or a memory-efficient xFormers kernel if available and appropriate, or a native C++ version) based on the inputs and hardware. This often leads to substantial speedups and memory savings in Transformer training and inference, without requiring third-party libraries for these specific optimizations. `torch.compile()` can further enhance the performance of models utilizing SDPA.
   - **Code Example (SDPA):**
     ```python
     import torch
     import torch.nn.functional as F
     import math

     # Example parameters
     batch_size = 32
     num_heads = 8
     seq_len_q = 128  # Query sequence length
     seq_len_kv = 256 # Key/Value sequence length (can be same as seq_len_q for self-attention)
     embed_dim_k = 64 # Dimension of key per head
     embed_dim_v = 64 # Dimension of value per head (often same as embed_dim_k)

     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
     print(f"Using device for SDPA: {device}")

     # (Batch, Num_heads, Seq_len, Dim_per_head)
     query = torch.randn(batch_size, num_heads, seq_len_q, embed_dim_k, device=device)
     key = torch.randn(batch_size, num_heads, seq_len_kv, embed_dim_k, device=device)
     value = torch.randn(batch_size, num_heads, seq_len_kv, embed_dim_v, device=device)

     # Optional attention mask (e.g., for padding)
     # Mask should be (Batch, Num_heads, Seq_len_q, Seq_len_kv) or broadcastable
     # True values in mask mean "do not attend" / "mask out". This is different from some conventions.
     # For F.scaled_dot_product_attention, a True value in attn_mask indicates the corresponding key/value position should be IGNORED.
     # A False value means the position should be attended to.
     # If using a float mask, -inf indicates masking.
     # attn_mask = torch.rand(batch_size, num_heads, seq_len_q, seq_len_kv, device=device) > 0.8 # Example random boolean mask

     print(f"Query shape: {query.shape}")
     print(f"Key shape: {key.shape}")
     print(f"Value shape: {value.shape}")

     try:
         # Standard call
         attention_output = F.scaled_dot_product_attention(
             query, key, value,
             # attn_mask=attn_mask, # Optional: for custom padding masks (True means ignore)
             dropout_p=0.1 if True else 0.0, # Enable dropout during training phase
             is_causal=False # Set to True for decoder self-attention (cannot be used with attn_mask)
         )
         print("SDPA output shape:", attention_output.shape)
         # Expected output shape: (batch_size, num_heads, seq_len_q, embed_dim_v)

         # Example with causal masking (for decoder-like self-attention)
         # Here, query, key, value would come from the same source, and seq_len_q == seq_len_kv
         if seq_len_q == seq_len_kv:
             causal_q = torch.randn(batch_size, num_heads, seq_len_q, embed_dim_k, device=device)
             # For self-attention, key and value are typically derived from the same source as query
             causal_k = causal_q # Or a projection of the same source
             causal_v = causal_q # Or a projection of the same source

             causal_attention_output = F.scaled_dot_product_attention(
                 causal_q, causal_k, causal_v,
                 is_causal=True # is_causal=True implies key_padding_mask is False for all S and L.
                                # Cannot provide attn_mask if is_causal=True
             )
             print("Causal SDPA output shape:", causal_attention_output.shape)

     except Exception as e:
         print(f"Error running SDPA: {e}")
         print("This might happen if an unsupported combination of inputs/options is used, or an issue with the environment.")

     # Integration with torch.nn.MultiheadAttention:
     # PyTorch's nn.MultiheadAttention will automatically try to use SDPA internally
     # if certain conditions are met (e.g., inputs are on CUDA, no certain hooks/options are used that prevent it).
     # You typically don't need to change your MHA layers to benefit.
     # For custom attention implementations, directly using F.scaled_dot_product_attention is recommended.
     ```
   - **Cheat Sheet:**
     - **Function:** `torch.nn.functional.scaled_dot_product_attention(query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False)`
     - **Inputs:**
       - `query (Tensor)`: Shape `(N, ..., L, E_k)` where `N` is batch size, `L` is target sequence length, `E_k` is key embedding dim. `...` can be num_heads.
       - `key (Tensor)`: Shape `(N, ..., S, E_k)` where `S` is source sequence length.
       - `value (Tensor)`: Shape `(N, ..., S, E_v)` where `E_v` is value embedding dim.
     - **`attn_mask (Optional[Tensor])`**: Boolean or float mask.
        - If Boolean: `True` indicates the corresponding key/value position should be **ignored/masked out**. `False` means attend. Shape should be broadcastable to `(N, ..., L, S)`.
        - If Float: Mask values are added to the attention scores. `-inf` effectively masks out a position.
        - Cannot be used if `is_causal=True`.
     - **`dropout_p (float)`**: Dropout probability. Default is `0.0`. Applied to attention weights. Only applied during training (`model.train()`).
     - **`is_causal (bool)`**: If `True`, applies a causal mask for autoregressive decoding (prevents attending to future positions). If `True`, `attn_mask` must not be provided.
     - **Benefits:**
       - Automatically selects optimized kernels (e.g., FlashAttention, memory-efficient xFormers) if available and inputs/hardware are compatible.
       - Falls back to a native PyTorch implementation if specialized kernels aren't applicable.
       - Improves speed and memory efficiency for attention computations.
     - **Integration:** Used by `torch.nn.MultiheadAttention` internally by default if conditions permit. Can be used directly for custom Transformer layers.
     - **`torch.compile()`**: Further optimizes models using SDPA for additional speedups.

## 3. MPS Backend (for macOS)
   - **Explanation:** The Metal Performance Shaders (MPS) backend in PyTorch enables GPU-accelerated training and inference on Apple Silicon Macs (M1, M2, M3 series chips). PyTorch 2.0 brought significant improvements in stability, correctness, and operator coverage for the MPS backend, making it a more viable option for Mac users to leverage their GPU capabilities for deep learning tasks. While it may not match dedicated NVIDIA GPUs in raw power for very large models, it provides a substantial speedup over CPU-only operations on Macs.
   - **Code Example:**
     ```python
     import torch

     if torch.backends.mps.is_available():
         print("MPS backend is available on this device!")
         # Check if PyTorch was built with MPS support
         if torch.backends.mps.is_built():
             print("PyTorch was built with MPS support.")
             device = torch.device("mps")
         else:
             print("PyTorch was NOT built with MPS support. Using CPU as fallback.")
             device = torch.device("cpu")
     else:
         print("MPS backend is NOT available on this device. Using CPU.")
         device = torch.device("cpu")

     print(f"Selected device: {device}")

     # Create tensors on the selected device
     try:
         x = torch.randn(5, 5, device=device)
         y = torch.randn(5, 5, device=device)

         # Perform an operation
         z = x @ y # Matrix multiplication

         print("\nTensor x properties:")
         print(f"  x values: \n{x}")
         print(f"  x device: {x.device}")

         print("\nTensor y properties:")
         print(f"  y values: \n{y}")
         print(f"  y device: {y.device}")

         print("\nResult tensor z properties:")
         print(f"  z values (result of x @ y): \n{z}")
         print(f"  z device: {z.device}")

         # Moving tensors between devices
         if device.type == 'mps':
             cpu_tensor = z.to('cpu')
             print(f"\nTensor z moved to CPU: {cpu_tensor.device}, values:\n{cpu_tensor}")
     except Exception as e:
         print(f"\nAn error occurred while using the {device} device: {e}")
         print("This can sometimes happen if an operation is not yet supported on MPS or due to other MPS issues.")
     ```
   - **Cheat Sheet:**
     - **Check Availability:**
       - `torch.backends.mps.is_available() -> bool`: Returns `True` if the current Mac has an MPS-compatible GPU and the necessary drivers/OS support.
       - `torch.backends.mps.is_built() -> bool`: Returns `True` if the installed PyTorch binary was compiled with MPS support. (Both should be true to use MPS).
     - **Set Device:**
       - `device = torch.device("mps")`
       - `tensor.to("mps")` or `tensor.to(torch.device("mps"))`
       - `model.to("mps")`
     - **Usage:**
       - Once a tensor or model is on the "mps" device, operations will generally run on the Apple GPU.
       - Use it like any other PyTorch device (e.g., "cuda", "cpu").
     - **Best Practices:**
       - Keep your macOS updated for the latest Metal drivers and MPS improvements.
       - Not all PyTorch operations are supported on MPS yet, though coverage has improved significantly. If an op is not supported, PyTorch might raise an error or silently fall back to CPU (which can be a performance bottleneck if unnoticed).
       - Performance can vary depending on the model and operation. It's generally most beneficial for compute-bound tasks.

## 4. `torch.func` (formerly Functorch)
   - **Explanation:** `torch.func` (which incorporates the functionality of the former standalone `functorch` library) brings powerful composable function transforms to PyTorch. This includes JAX-like capabilities for vectorization (`vmap`), automatic differentiation (`grad`, `jacrev`, `jacfwd`, `jvp`, `vjp`), and Hessian computations (`hessian`). These transforms allow for more advanced and efficient ways to manipulate and differentiate functions, enabling use cases like per-sample gradient computations, model ensembling, and complex derivative calculations, often with more direct and cleaner code than traditional approaches.
   - **Code Example (`vmap` and `grad`):**
     ```python
     import torch
     from torch.func import vmap, grad

     # --- vmap example: per-sample processing ---
     # Define a function that operates on a single sample
     def process_sample(sample_features, weight_matrix):
         # sample_features: [num_features]
         # weight_matrix: [out_features, num_features]
         return torch.matmul(weight_matrix, sample_features)

     # Create a batch of features and a single weight matrix
     batch_size = 5
     num_features = 10
     out_features = 3

     batched_features = torch.randn(batch_size, num_features) # Batch of 5 samples
     weights = torch.randn(out_features, num_features)       # Single weight matrix for all

     # Vectorize `process_sample` over the batch of features (0-th dimension of batched_features)
     # `in_dims=(0, None)` means:
     #   - map over the 0-th dimension of the first argument (batched_features)
     #   - do not map over the second argument (weights), use it as is for each sample
     vectorized_process = vmap(process_sample, in_dims=(0, None))

     batched_output = vectorized_process(batched_features, weights)

     print("--- vmap Example ---")
     print("Batched features shape:", batched_features.shape) # Expected: (5, 10)
     print("Weights shape:", weights.shape)                   # Expected: (3, 10)
     print("Vmap output shape:", batched_output.shape)       # Expected: (5, 3)

     # --- grad example: per-sample gradients of input---
     # Define a simple model (could be a more complex nn.Module's forward pass)
     # This function takes a single input sample and returns a scalar loss
     def model_loss_single_sample(input_tensor_single, target_tensor_single):
         # A simple linear transformation (conceptually part of a model)
         # For simplicity, using fixed weights and bias here.
         w = torch.tensor([[0.5, -0.2], [0.1, 0.7]], requires_grad=False, device=input_tensor_single.device)
         b = torch.tensor([0.1, -0.1], requires_grad=False, device=input_tensor_single.device)

         prediction = torch.matmul(input_tensor_single.unsqueeze(0), w.T).squeeze(0) + b
         loss = torch.sum((prediction - target_tensor_single)**2) # MSE-like scalar loss
         return loss

     input_dim = 2
     # Note: For `grad` to work on inputs, they need requires_grad=True
     batched_inputs_for_grad = torch.randn(batch_size, input_dim, requires_grad=True)
     batched_targets_for_grad = torch.randn(batch_size, input_dim)

     # We want the gradient of `model_loss_single_sample` w.r.t. its first argument (input_tensor_single)
     grad_fn_single_input = grad(model_loss_single_sample, argnums=0)

     # Use vmap to apply `grad_fn_single_input` to each sample in the batch
     per_sample_input_gradients = vmap(grad_fn_single_input, in_dims=(0, 0))(batched_inputs_for_grad, batched_targets_for_grad)

     print("\n--- grad & vmap Example (Per-Sample Input Gradients) ---")
     print("Batched inputs shape:", batched_inputs_for_grad.shape)
     print("Per-sample input gradients shape:", per_sample_input_gradients.shape)

     # --- grad example: per-sample gradients of model parameters ---
     model = torch.nn.Linear(input_dim, input_dim) # A simple linear layer

     # Function that takes model parameters and a single data sample, returns scalar loss
     def compute_loss_for_functional_call(params, buffers, x_single, y_single):
         # Use functional_call to run the model with explicit params and buffers
         prediction = torch.func.functional_call(model, (params, buffers), (x_single.unsqueeze(0),))
         return torch.sum((prediction.squeeze(0) - y_single)**2)

     # Get initial model parameters and buffers (as dicts)
     params = {name: p for name, p in model.named_parameters()}
     buffers = {name: b for name, b in model.named_buffers()}

     # Grad w.r.t. the first argument of compute_loss_for_functional_call (which is 'params')
     grad_params_fn = grad(compute_loss_for_functional_call, argnums=0)

     # vmap this gradient function.
     # params and buffers are not batched (None). x_single and y_single are batched (0-th dim).
     # `out_dims=0` means the output gradients for params will also be batched along the 0-th dim.
     # The output of vmap(grad_params_fn) will be a dictionary of gradients, matching keys in 'params'.
     per_sample_param_grads_dict = vmap(grad_params_fn, in_dims=(None, None, 0, 0), out_dims=0)(
                                         params, buffers, batched_inputs_for_grad, batched_targets_for_grad)

     print("\n--- grad & vmap Example (Per-Sample Parameter Gradients) ---")
     for name, p_grad in per_sample_param_grads_dict.items():
         print(f"Gradient shape for param '{name}': {p_grad.shape}")
         # Expected: (batch_size, *param_original_shape)
         # e.g., for model.weight: (5, 2, 2)
         # e.g., for model.bias: (5, 2)
     ```
   - **Cheat Sheet:**
     - **Import:** `from torch.func import vmap, grad, jacrev, jacfwd, hessian, jvp, vjp, functional_call`
     - **`vmap(func, in_dims=0, out_dims=0, randomness='error')`**:
       - Vectorizes `func` by mapping it over specified input dimensions.
       - `in_dims`: Tuple/int specifying which dimension to map for each input argument of `func`. `None` means broadcast the argument. `0` is common for batch dimension.
       - `out_dims`: Specifies the output dimension for the mapped axis.
       - `randomness`: How to handle random operations within `func` (`'error'`, `'different'`, `'same'`).
     - **`grad(func, argnums=0, has_aux=False)`**:
       - Computes the gradient of `func` with respect to the argument(s) specified by `argnums`.
       - `func` must return a scalar tensor.
       - `argnums`: Integer or tuple of integers specifying which positional arguments to differentiate.
       - `has_aux=True`: If `func` returns `(output, aux_data)`, `grad` will return `(gradient, aux_data)`.
     - **Jacobians & Hessians:**
       - `jacrev(func, argnums=0)`: Jacobian using reverse-mode AD (efficient for many outputs).
       - `jacfwd(func, argnums=0)`: Jacobian using forward-mode AD (efficient for many inputs).
       - `hessian(func, argnums=0)`: Hessian (Jacobian of the gradient).
     - **`jvp(func, primals, tangents)`**: Computes Jacobian-vector products.
     - **`vjp(func, primals, cotangents)`**: Computes vector-Jacobian products.
     - **`functional_call(module, (parameters, buffers), args, kwargs=None)`**:
       - Calls a `torch.nn.Module` instance in a functional way, using explicitly provided `parameters` and `buffers` (typically dictionaries). Useful for `torch.func` transforms over module parameters.

## 5. `torch.set_default_device` and `torch.device` Context Manager
   - **Explanation:** PyTorch 2.0 introduces more convenient ways to manage the default device for tensor creation. `torch.set_default_device(device)` allows you to globally change the default device (e.g., 'cuda', 'mps', 'cpu') on which factory functions like `torch.empty()`, `torch.randn()`, etc., will allocate new tensors if the `device` argument is not specified. Additionally, `torch.device` can now be used as a context manager (`with torch.device('cuda'): ...`) to temporarily set the default device for such factory functions within a specific block of code. This helps reduce boilerplate `.to(device)` calls and makes device management more streamlined.
   - **Code Example:**
     ```python
     import torch

     print(f"Initial default device (for torch.tensor created without device arg): {torch.tensor([1,2]).device}")
     print(f"Initial default device (for torch.empty created without device arg): {torch.empty(2).device}")

     # --- Using torch.set_default_device ---
     if torch.cuda.is_available():
         print("\n--- Testing with CUDA ---")
         torch.set_default_device('cuda:0') # Or just 'cuda'
         print(f"Default device after set_default_device('cuda:0'): {torch.empty(2).device}")
         cuda_tensor_implicit = torch.tensor([3.0, 4.0]) # Will be on CUDA
         print(f"Tensor created after set_default_device('cuda:0'): {cuda_tensor_implicit.device}")

         # --- Using torch.device as a context manager ---
         print("\n--- Testing torch.device context manager (from CUDA default) ---")
         with torch.device('cpu'):
             cpu_tensor_in_context = torch.tensor([5.0, 6.0])
             print(f"Tensor device inside 'cpu' context manager: {cpu_tensor_in_context.device}")
             print(f"torch.empty(2) device inside 'cpu' context: {torch.empty(2).device}")

         print(f"Tensor device outside 'cpu' context manager (back to CUDA default): {torch.empty(2).device}")

         # Reset to CPU for other examples / tests
         torch.set_default_device('cpu')
         print(f"\nDefault device reset to CPU: {torch.empty(2).device}")

     elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
         print("\n--- Testing with MPS (CUDA not available) ---")
         torch.set_default_device('mps')
         print(f"Default device after set_default_device('mps'): {torch.empty(2).device}")
         mps_tensor_implicit = torch.tensor([3.0, 4.0])
         print(f"Tensor created after set_default_device('mps'): {mps_tensor_implicit.device}")

         print("\n--- Testing torch.device context manager (from MPS default) ---")
         with torch.device('cpu'):
             cpu_tensor_in_context_mps = torch.tensor([5.0, 6.0])
             print(f"Tensor device inside 'cpu' context manager: {cpu_tensor_in_context_mps.device}")
             print(f"torch.empty(2) device inside 'cpu' context: {torch.empty(2).device}")

         print(f"Tensor device outside 'cpu' context manager (back to MPS default): {torch.empty(2).device}")

         # Reset to CPU
         torch.set_default_device('cpu')
         print(f"\nDefault device reset to CPU: {torch.empty(2).device}")
     else:
         print("\nCUDA and MPS not available. Skipping device change demonstrations that require them.")

     print(f"\nFinal default device for factories: {torch.empty(2).device}")
     ```
   - **Cheat Sheet:**
     - **Global Default Device for Factories:**
       - `torch.set_default_device(device_str_or_obj)`
         - Example: `torch.set_default_device('cuda')`, `torch.set_default_device('cuda:1')`, `torch.set_default_device('mps')`, `torch.set_default_device('cpu')`
         - Affects tensor factory functions (e.g., `torch.tensor()`, `torch.randn()`, `torch.zeros()`, `torch.empty()`) when their `device` argument is **not** specified.
       - To check the current factory default: `print(torch.empty(1).device)` (as `torch.get_default_device()` refers to the default CUDA device index, not the global factory default).
     - **Context Manager for Local Default Device for Factories:**
       - `with torch.device(device_str_or_obj): ...`
         - Example: `with torch.device('cuda'): tensor = torch.randn(2,2)` (tensor will be on CUDA if `device` kwarg omitted).
         - Temporarily sets the default device for tensor factories (those not given an explicit `device` kwarg) within the `with` block.
         - The previous factory default device is restored upon exiting the block.
     - **Device Specification:**
       - `device_str_or_obj` can be:
         - A string: `'cpu'`, `'cuda'`, `'cuda:0'`, `'mps'`, `'mps:0'`.
         - A `torch.device` object: `torch.device('cuda')`.
     - **Impact:**
       - Simplifies code by reducing the need for explicit `device=...` for every tensor creation if a consistent device is used for factories.
       - Tensors created with an explicit `device` argument (e.g., `torch.tensor([1], device='cpu')`) are **not** affected by these settings.
       - Be mindful of the scope (global vs. local context) when using these features.

## 6. DTensor (DistributedTensor)
   - **Explanation:** `DTensor` (DistributedTensor) is a prototype feature in PyTorch aimed at simplifying distributed computing using the SPMD (Single Program, Multiple Devices) paradigm. It allows developers to express tensor distributions (sharded or replicated) across a group of devices (a `DeviceMesh`) more naturally. Operations on DTensors automatically handle necessary communication (like all-gather or reduce-scatter) based on their sharding specifications. This abstraction helps in writing cleaner code for complex distributed training strategies, such as tensor parallelism (sharding model layers across devices) and its combination with other parallelism techniques like FSDP (Fully Sharded Data Parallel). While still experimental, DTensor represents a move towards more flexible and intuitive distributed training in PyTorch.
   - **Code Example (Conceptual - Requires Distributed Setup):**
     The following code is designed to be run in a distributed environment, typically using `torchrun`. For example: `torchrun --nproc_per_node=2 your_script_file.py`.
     ```python
     # file: dtensor_example.py
     import torch
     import torch.distributed as dist
     from torch.distributed._tensor import DeviceMesh, Shard, Replicate, distribute_tensor, redistribute_tensor
     import os

     def setup_distributed():
         """Initializes the distributed process group."""
         if not dist.is_initialized():
             backend = "nccl" if torch.cuda.is_available() else "gloo"
             # torchrun usually sets these, but provide defaults for direct execution if needed
             rank = int(os.environ.get("RANK", "0"))
             world_size = int(os.environ.get("WORLD_SIZE", "1"))
             if world_size > 1:
                 dist.init_process_group(backend=backend, rank=rank, world_size=world_size)
                 if backend == "nccl":
                     torch.cuda.set_device(rank) # Crucial for NCCL
             else: # Single process, no real distribution
                 # Mock a single process environment for the example to run without torchrun
                 # This is not true distributed execution.
                 print("Warning: Running DTensor example in a simulated single-process mode.")
                 # No actual init_process_group needed for single process if not using torchrun for it.
                 # However, DeviceMesh expects a group. For CPU single process, gloo can sometimes work.
                 # For simplicity, we'll let the example handle single-process logic.
                 pass


     def cleanup_distributed():
         """Cleans up the distributed process group."""
         if dist.is_initialized() and int(os.environ.get("WORLD_SIZE", "1")) > 1 :
             dist.destroy_process_group()

     def run_dtensor_example_main():
         """Main function to demonstrate DTensor."""

         is_distributed_run = dist.is_initialized() and dist.get_world_size() > 1
         rank = dist.get_rank() if is_distributed_run else 0
         world_size = dist.get_world_size() if is_distributed_run else 1

         if not is_distributed_run and world_size == 1:
             print("Note: DTensor example running in a single-process context. No actual distribution.")

         # Determine device type
         if torch.cuda.is_available() and (not is_distributed_run or torch.cuda.device_count() >= world_size):
             device_type = "cuda"
             # In a real distributed setup (multi-GPU), each rank targets its own GPU
             current_device = f"cuda:{rank}" if is_distributed_run else "cuda:0"
             if is_distributed_run: torch.cuda.set_device(rank) # Ensure correct device for NCCL
         else:
             device_type = "cpu"
             current_device = "cpu"
             if rank == 0:
                 print(f"Using CPU for DTensor example (CUDA not available or not enough devices for all {world_size} ranks).")

         print(f"Rank {rank}/{world_size} on device: {current_device} (mesh device_type: {device_type})")

         # 1. Create a DeviceMesh
         # It defines the set of devices for distributed computation.
         # For a 1D mesh across all ranks:
         device_ids = torch.arange(world_size)
         mesh = DeviceMesh(device_type=device_type, mesh=device_ids)

         # 2. Define global tensor properties and create local shard data
         global_tensor_rows = world_size * 2  # e.g., 4 rows if world_size is 2
         global_tensor_cols = 10

         # Each rank will hold a part of the global tensor.
         # If sharding on dim 0: each rank gets (global_rows / world_size, global_cols)
         local_rows_per_rank = global_tensor_rows // world_size
         local_shard = torch.randn(local_rows_per_rank, global_tensor_cols, device=current_device)
         if rank == 0: print(f"Global tensor would be ({global_tensor_rows}, {global_tensor_cols}). Each rank has local shard of shape ({local_rows_per_rank}, {global_tensor_cols}).")

         # 3. Define sharding placements
         # Shard(0) means sharding along the 0-th dimension.
         placements = [Shard(0)]

         # 4. Create the DTensor using the local shard from each rank
         dtensor = distribute_tensor(local_shard, device_mesh=mesh, placements=placements)
         print(f"Rank {rank}: Created DTensor. Global shape: {dtensor.shape}. Local data shape: {dtensor.to_local().shape}")

         # --- Operations on DTensor ---
         # Element-wise operations are typically local if placements are compatible
         dtensor_doubled = dtensor * 2
         print(f"Rank {rank}: (dtensor * 2). Local shape: {dtensor_doubled.to_local().shape}")

         # Reduction: sum() over a DTensor. Default behavior might depend on sharding.
         # If you sum a sharded DTensor, it may result in a replicated scalar DTensor.
         global_sum_val = dtensor.sum() # This will likely perform a reduce-sum across shards
         # The result `global_sum_val` is a DTensor. It's a scalar tensor, replicated on all ranks.
         print(f"Rank {rank}: dtensor.sum() (replicated scalar DTensor). Local value: {global_sum_val.to_local().item()}")

         # Redistribute: Change sharding, e.g., from Sharded to Replicated
         replicated_dtensor = dtensor.redistribute(mesh, [Replicate()])
         print(f"Rank {rank}: Redistributed to Replicated. Local shape: {replicated_dtensor.to_local().shape}")
         # Now, each rank has the full tensor data.
         # print(f"Rank {rank}: Replicated local data (first row): {replicated_dtensor.to_local()[0] if replicated_dtensor.to_local().numel() > 0 else 'empty'}")

         # Example: Matrix multiplication with a replicated tensor
         # Create a replicated tensor for matmul
         mat_cols = 5
         replicated_matrix_local = torch.randn(global_tensor_cols, mat_cols, device=current_device)
         replicated_matrix_dtensor = distribute_tensor(replicated_matrix_local, mesh, [Replicate()])

         # Matmul: Sharded DTensor @ Replicated DTensor
         # Result should be sharded like the first DTensor if strategies align (e.g., SPMD)
         # dtensor (Sharded[0]) @ replicated_matrix_dtensor (Replicated)
         # This should result in a DTensor sharded on dim 0.
         # (N_shard, K) @ (K, M) -> (N_shard, M) locally
         # Global: (N, K) @ (K, M) -> (N, M)
         try:
             result_dtensor = torch.matmul(dtensor, replicated_matrix_dtensor)
             print(f"Rank {rank}: torch.matmul(dtensor, replicated_matrix_dtensor). Global shape: {result_dtensor.shape}. Local shape: {result_dtensor.to_local().shape}")
         except Exception as e:
             print(f"Rank {rank}: Error during matmul: {e}")


     if __name__ == "__main__":
         # This `if __name__ == "__main__":` block allows the script to be run with `torchrun`
         setup_distributed()
         run_dtensor_example_main()
         cleanup_distributed()
     ```
     **To run this example:**
     1. Save the code above as `dtensor_example.py`.
     2. If you have CUDA and multiple GPUs (e.g., 2):
        `torchrun --nproc_per_node=2 dtensor_example.py`
     3. For CPU testing (simulates 2 processes on CPU):
        `torchrun --nproc_per_node=2 --rdzv-backend=c10d --rdzv-endpoint=localhost:0 dtensor_example.py`
        (You might need to adjust `rdzv_endpoint` or ensure it's unique if running multiple times).
        If `torchrun` for CPU gives issues, you might need to explicitly set `OMP_NUM_THREADS=1` or similar depending on your environment to avoid oversubscription if Gloo backend is used.

   - **Cheat Sheet:**
     - **Core Components:**
       - `from torch.distributed._tensor import DeviceMesh, Shard, Replicate, distribute_tensor, redistribute_tensor`
       - `DeviceMesh(device_type: str, mesh: torch.Tensor)`: Defines the computational grid of devices.
         - `device_type`: e.g., "cuda", "cpu".
         - `mesh`: A 1D or N-D tensor describing the arrangement of global device IDs (e.g., `torch.arange(world_size)` for 1D, `torch.arange(world_size).view(rows, cols)` for 2D).
       - `Shard(dim: int)`: Placement strategy indicating the tensor is sharded along its `dim`-th dimension across the mesh dimension(s) corresponding to that tensor dimension.
       - `Replicate()`: Placement strategy indicating the tensor is replicated on all devices in the mesh (or a sub-mesh if specified).
     - **Creating DTensors:**
       - `dt = distribute_tensor(local_tensor_on_current_rank, device_mesh, placements)`: Creates a DTensor from local tensor data on each rank.
         - `placements`: A list of `Shard` or `Replicate` objects, one for each dimension of the tensor, defining how it's distributed.
     - **Converting DTensors:**
       - `local_tensor = dt.to_local()`: Gets the local tensor component on the current rank.
       - `new_dt = dt.redistribute(device_mesh, new_placements)`: Changes the distribution of the DTensor (e.g., from sharded to replicated), handles necessary communication.
     - **Operations:**
       - Many PyTorch tensor operations can be applied directly to DTensors.
       - The DTensor system manages communication based on the placements of input DTensors to maintain the SPMD paradigm.
       - Example: `dt_c = dt_a + dt_b`, `dt_out = torch.matmul(dt_sharded_col, dt_sharded_row)`.
     - **Use Cases:**
       - Tensor Parallelism (sharding model layers).
       - Sequence Parallelism.
       - Combining with FSDP for 2D/3D parallelism.
       - Writing SPMD-style distributed algorithms with a more tensor-centric API.
     - **Status:** Prototype feature. API and behavior may evolve. Requires a distributed process group to be initialized (typically via `torch.distributed.init_process_group` managed by `torchrun`).

## Conclusion
PyTorch 2.0 brings a wealth of powerful updates focused on performance and usability. `torch.compile` offers a simple path to faster models, Accelerated Transformers and SDPA optimize critical components, `torch.func` provides advanced autograd and vectorization capabilities, MPS backend enhances Mac user experience, and new device/distributed tools like `torch.set_default_device` and `DTensor` streamline development for various hardware setups. Exploring these features can significantly benefit your deep learning workflows. Remember to consult the official PyTorch documentation for the most detailed and up-to-date information.
