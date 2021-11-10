from tinychain.collection.tensor import einsum, Dense, Schema, Sparse, Tensor
from tinychain.decorators import closure, get_op, post_op
from tinychain.ref import After, If, While
from tinychain.state import Map, Stream, Tuple
from tinychain.value import F32, UInt


def identity(size, dtype=F32):
    """Return an identity matrix with dimensions `[size, size]`."""

    schema = Schema([size, size], dtype)
    return Sparse.copy_from(schema, Stream.range((0, size)).map(get_op(lambda i: ((i, i), 1))))


# TODO: vectorize to support a `Tensor` containing a batch of matrices
def householder(a):
    """Returns a Householder transform for the matrix `a`"""

    norm = (a**2).sum()**0.5
    v = (a / (a[0] + norm)).copy()
    tau = 2 / einsum("ji,jk->ik", [v, v])

    return Tuple(After(v.write([0], 1), [v, tau]))


def norm(tensor: Tensor) -> Tensor:
    """Compute the 2D Frobenius (aka Euclidean) norm of the matrices in the given `tensor`.

    Args:
        `tensor`: a matrix or batch of matrices, with shape `[..., M, N]`

    Returns:
        A `Tensor` of shape [...] or a `Number` if the input `tensor` is itself 2-dimensional
    """

    squared = tensor**2
    return If(tensor.ndim == 2,
              squared.sum()**0.5,
              squared.sum(-1).sum(-1)**0.5)


# TODO: vectorize to support a `Tensor` containing a batch of matrices
@post_op
def qr(cxt, matrix: Tensor) -> Tuple:
    """Compute the QR factorization of the given `matrix`.

    Args:
        `a`: a matrix with shape `[M, N]`

    Returns:
        A `Tuple` of `Tensor` objects `(Q, R)` where `A ~= QR` and `Q.transpose() == Q**-1`
    """

    cxt.m = UInt(matrix.shape[0])
    cxt.n = UInt(matrix.shape[1])

    outer_cxt = cxt
    def qr_step(Q: Tensor, R: Tensor, k: UInt) -> Map:
        transform = householder(R[k:, k].expand_dims())
        v = transform[0]
        tau = transform[1]

        H = identity(outer_cxt.m)  # TODO: convert to a Dense tensor of type F32
        H_sub = H[k:, k:] - (tau * einsum("ji,jk->ik", [v, v]))
        return After(H.write([slice(k, None), slice(k, None)], H_sub), {
            "Q": einsum("ij,jk->ik", [H, Q]),
            "R": einsum("ij,jk->ik", [H, R]),
        })

    iterations = cxt.n - 1  # TODO: If(cxt.n == cxt.m, cxt.n - 1, cxt.n)
    return Stream.range(iterations).fold("k", Map(Q=identity(cxt.m), R=matrix), closure(post_op(qr_step)))


@post_op
def svd(cxt, tensor: Tensor) -> Tuple:
    raise NotImplementedError
