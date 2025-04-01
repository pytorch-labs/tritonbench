import torch
import jax.numpy as jnp

JAX_DTYPE_MAPPING = {
    torch.bfloat16: jnp.bfloat16,
    torch.float16: jnp.bfloat16,
}

def torch_to_jax_tensor(t: torch.Tensor) -> jnp.ndarray:
    torch_dtype = t.dtype
    numpy_array = t.float().cpu().detach().numpy()
    return jnp.asarray(jnp.array(numpy_array), dtype=JAX_DTYPE_MAPPING[torch_dtype])
