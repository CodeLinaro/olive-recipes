import math

import torch
from fast_hadamard_transform import hadamard_transform as _hadamard_transform


def retrieve_hadamard_config(size, transpose=False):
    from aimet_torch.experimental.spinquant._hadamard_matrices import get_had12, get_had28

    hadamard_matrix, factor = None, None
    if size % 28 == 0:
        assert is_power_of_two(size // 28)
        factor = 28
        hadamard_matrix = get_had28().T if transpose else get_had28()
    elif size % 12 == 0:
        assert is_power_of_two(size // 12)
        factor = 12
        hadamard_matrix = get_had12().T if transpose else get_had12()
    else:
        assert is_power_of_two(size)
        factor = 1

    return hadamard_matrix, factor


def hadamard_transform(x: torch.Tensor, scale: float = 1.0, handle_no_power_2="factor") -> torch.Tensor:
    """
    This function takes care of the hadamard transform for CUDA, and with different options
    for the dimension not being a power of 2.

    Args:
        x: torch.Tensor
        scale: float
        handle_no_power_2: str
            - "pad": pad to the next power of 2 and then apply the Hadamard Transform
            - "factor": recursively define hadamard transform based on factor
            - "fht": default fast_hadamard_transform behaviour. Equivalent to: pad to the next power of 2 and then apply the Hadamard Transform and reduce the last dimension
            - "error": raise an error if the input is not a power of 2
    """

    log2d = math.log2(x.shape[-1])
    # Define hadamard transform = identity for dimension of 1
    if log2d == 0:
        return x

    # Handle padding when dimension is not a power of 2
    original_size = x.shape[-1]
    if isinstance(log2d, int) or log2d.is_integer():
        pass
    elif handle_no_power_2 in ["pad", "fht"]:
        # We pad and apply the Hadamard Transform. Note that this can increase the last dimension of the representation
        target_size = 2 ** math.ceil(log2d)
        pad_len = int(target_size - original_size)
        x = torch.nn.functional.pad(x, (0, int(pad_len)))
    elif handle_no_power_2 == "error":
        raise ValueError(f"Input shape {x.shape} is not a power of 2")
    elif handle_no_power_2 == "factor":
        try:
            had_matrix, factor = retrieve_hadamard_config(original_size)
        except AssertionError:
            raise ValueError(f"We do not have pre-defined hadamard matrix with dim={original_size}")
        return multiply_hadamard_fast_impl(x, had_matrix, factor)
    else:
        raise ValueError(
            f"Unknown option for handling handle_no_power_2 must \
                         be error', 'pad', or 'ignore', not {handle_no_power_2}"
        )

    out: torch.Tensor = _hadamard_transform(x, scale)  # pyright: ignore[reportAssignmentType]

    # Whether to reduce the size back to the original shape
    if log2d.is_integer() or handle_no_power_2 in ["pad"]:
        return out
    elif handle_no_power_2 in ["fht"]:
        # This is the default fast hadamard transform for non-power-of-2.
        # It is problematic, however, as the inverse would not be correctly defined.
        return out[..., :original_size]


def multiply_hadamard_fast(tensor, had_matrix, factor):
    if tensor.is_cuda:
        return multiply_hadamard_fast_impl(tensor, had_matrix, factor)
    else:
        return multiply_hadamard_slow_impl(tensor, had_matrix, factor)


def is_power_of_two(number):
    return (number & (number - 1) == 0) and (number > 0)


def multiply_hadamard_slow_impl(tensor, had_matrix, factor):
    dim = tensor.shape[-1]
    x = tensor.clone().view(-1, dim)

    H = 1
    while H < dim // factor:
        x = x.view(-1, dim // (2 * H), 2 * H)
        x[:, :, :H] = x[:, :, :H] + x[:, :, H : 2 * H]
        x[:, :, H : 2 * H] = x[:, :, :H] - 2 * x[:, :, H : 2 * H]
        H *= 2
        x = x.view(-1, dim)

    if factor > 1:
        x = x.view(-1, dim // factor, factor)
        had_matrix = had_matrix.to(x.device, x.dtype)
        x = torch.matmul(x, had_matrix.transpose(0, 1))
        x = x.view(-1, dim)

    x = x.view(*tensor.shape)
    return x / torch.sqrt(torch.tensor(dim, dtype=x.dtype, device=x.device))


def multiply_hadamard_fast_impl(tensor, had_matrix, factor):
    dim = tensor.shape[-1]
    if factor == 1:
        return _hadamard_transform(tensor.contiguous()) / torch.tensor(dim).sqrt()
    reshaped = tensor.view(-1, factor, dim // factor)
    transformed = _hadamard_transform(reshaped.contiguous()) / torch.tensor(dim).sqrt()
    transformed = had_matrix.to(transformed.device).to(transformed.dtype) @ transformed
    return transformed.reshape(tensor.shape)


def generate_random_hadamard_matrix(size, device, seed=0):
    generator = torch.Generator()
    generator.manual_seed(seed)
    random_diag = torch.randint(low=0, high=2, size=(size,), generator=generator)
    random_diag = random_diag * 2 - 1
    diagonal_matrix = torch.diag(random_diag)
    had_matrix, factor = retrieve_hadamard_config(size)
    return multiply_hadamard_slow_impl(diagonal_matrix, had_matrix, factor).to(device)


def generate_hadamard_matrix(size, device):
    identity = torch.eye(size)
    had_matrix, factor = retrieve_hadamard_config(size)
    return multiply_hadamard_slow_impl(identity, had_matrix, factor).to(device)


def apply_hadamard_transform_to_linear_r4(weight, in_features):
    weights = weight.data
    original_dtype = weights.dtype
    device = weights.device
    weights = weights.float()
    had_matrix, factor = retrieve_hadamard_config(in_features)
    if had_matrix == None and factor == 1:
        weights = multiply_hadamard_fast(weights, had_matrix, factor)
    else:
        weights = multiply_hadamard_fast(weights, had_matrix.to(device), factor)
    return weights.to(dtype=original_dtype)


def randomized_hadamard_transform(x: torch.Tensor, scale: float, handle_no_power_2="error") -> torch.Tensor:
    s = torch.bernoulli(0.5 * torch.ones(x.shape[-1]).to(x)) * 2 - 1
    return hadamard_transform(x * s, scale=scale, handle_no_power_2=handle_no_power_2)
