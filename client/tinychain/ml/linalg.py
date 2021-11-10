from tinychain.collection.tensor import einsum, Dense, Schema, Sparse, Tensor
from tinychain.decorators import closure, get_op, post_op
from tinychain.ref import After, If, While
from tinychain.state import Map, Stream, Tuple
from tinychain.value import F32, UInt


def identity(size, dtype=F32):
    """Return an identity matrix with dimensions `[size, size]`."""

    schema = Schema([size, size], dtype)
    return Sparse.copy_from(schema, Stream.range((0, size)).map(get_op(lambda i: ((i, i), 1))))


def householder(a):
    """Returns a Householder transform for the matrices in `a`"""

    v = a / (a[:, 0] + (norm(a) * a.sign()))
    tau = 2 / einsum("bji,jk->bik", [v, v])

    return Tuple(After(v.write([slice(None), 0], 1), [v, tau]))


def norm(tensor: Tensor) -> Tensor:
    """Compute the 2D Frobenius (aka Euclidean) norm of the matrices in the given `tensor`.

    Args:
        `tensor`: a matrix or batch of matrices, with shape `[..., M, N]`

    Returns:
        A `Tensor` of shape [...] or a `Number` if the input `tensor` is itself 2-dimensional
    """

    squared = tensor**2
    return If(tensor.ndim() == 2,
              squared.sum()**0.5,
              squared.sum(-1).sum(-1)**0.5)


@post_op
def qr(matrices: Tensor) -> Tuple:
    """Compute the QR factorization of the given `matrices`.

    Args:
        `a`: a batch of matrices with shape `[batch_dim, M, N]`

    Returns:
        A `Tuple` of `Tensor` objects with shapes `([batch_dim, M], [batch_dim, N])`
    """

    batch_dim = matrices.shape()[0]
    n = UInt(matrices.shape()[1])
    m = UInt(matrices.shape()[2])

    R = matrices
    Q = identity(m) + Dense.ones([batch_dim, 1, 1])

    def qr_step(Q: Tensor, R: Tensor, k: UInt) -> Map:
        transform = householder(R[k:, k].expand_dims())
        v = transform[0]
        tau = transform[1]

        H = identity(m)
        H_sub = H[k:, k:] - (tau * einsum("bji,bjk->bik", [v, v]))
        return After(H.write([slice(k, None), slice(k, None)], H_sub), {
            "Q": einsum("bij,bjk->bik", [H, R]),
            "R": einsum("bij,bjk->bik", [H, Q]),
        })

    Q_R = Stream.range((0, n - 1)).fold("k", Map(Q=Q, R=R), closure(post_op(qr_step)))
    Q = Tensor(Q_R[0])
    R = Tensor(Q_R[1])
    return Q[:n].transpose([0, 2, 1]), R[:n]


@post_op
def svd(cxt, tensor: Tensor) -> Tuple:
    raise NotImplementedError
