from tinychain.collection.tensor import Dense, Tensor
from tinychain.ref import If, While


def norm(tensor: Tensor) -> Tensor:
    """Compute the 2D Frobenius norm.

    Args:
        `tensor`: a matrix or batch of matrices, with shape `[..., M, N]`

    Returns:
        A `Tensor` of shape [...] or a `Number` if the input `tensor` is itself 2-dimensional
    """

    squared = tensor**2
    return If(tensor.ndim() == 2,
              squared.sum()**0.5,
              squared.sum(-1).sum(-1)**0.5)
