"""An n-dimensional array of numbers."""

from tinychain import ref
from tinychain.state import Map, State, Stream, Tuple
from tinychain.util import form_of, to_json, uri, URI
from tinychain.value import Bool, I16, F32, Number

from .bound import Range
from .collection import Collection


class Schema(object):
    """
    A `Tensor` schema which comprises a shape and data type.

    The data type must be a subclass of `Number` and defaults to `F32`.
    """

    def __init__(self, shape, dtype=F32):
        self.shape = shape
        self.dtype = dtype

    def __json__(self):
        return to_json([self.shape, self.dtype])


class Tensor(Collection):
    """An n-dimensional array of numbers."""

    __uri__ = uri(Collection) + "/tensor"

    @classmethod
    def load(cls, shape, dtype, data):
        """
        Load a `Tensor` from an existing data set.

        Example:
            .. highlight:: python
            .. code-block:: python

                coords = [[0, 0, 1], [0, 1, 0]]
                values = [1, 2]
                sparse = tc.tensor.Sparse.load([2, 3, 4], tc.I32, zip(coords, values))
                dense = tc.tensor.Dense.load([2, 3, 4], tc.i32, values)
        """

        return super().load(Schema(shape, dtype), data)

    def __getitem__(self, bounds):
        bounds = _handle_bounds(bounds)
        return self._get("", bounds, Tensor)

    def __setitem__(self, bounds, value):
        bounds = _handle_bounds(bounds)
        return self._put("", bounds, value)

    def __add__(self, other):
        return self.add(other)

    def __radd__(self, other):
        return self.add(other)

    def __eq__(self, other):
        return self.eq(other)

    def __gt__(self, other):
        return self.gt(other)

    def __ge__(self, other):
        return self.gte(other)

    def __lt__(self, other):
        return self.lt(other)

    def __le__(self, other):
        return self.lte(other)

    def __mul__(self, other):
        return self.mul(other)

    def __rmul__(self, other):
        return self.mul(other)

    def __ne__(self, other):
        return self.ne(other)

    def __neg__(self):
        return self * -1

    def __pow__(self, other):
        return self.pow(other)

    def __sub__(self, other):
        return self.sub(other)

    def __rsub__(self, other):
        return -(self - other)

    def __truediv__(self, other):
        return self.div(other)

    def __rtruediv__(self, other):
        return other * (self.pow(-1))

    def abs(self):
        """Return the element-wise absolute value of this `Tensor`."""

        return self._get("abs", rtype=self.__class__)

    def add(self, other):
        """Return the element-wise sum of this `Tensor` and another `Tensor` or `Number`."""

        return self._post("add", Map(r=other), Tensor)

    def all(self):
        """Return `True` if all elements in this `Tensor` are nonzero."""

        return self._get("all", rtype=Bool)

    def any(self):
        """Return `True` if any element in this `Tensor` are nonzero."""

        return self._get("any", rtype=Bool)

    def cast(self, number_type):
        """Attempt to cast the data type of `Tensor` into the given `number_type`."""

        return self._get("cast", number_type, self.__class__)

    def copy(self):
        """Return a copy of this `Tensor`"""

        return self.__class__(ref.Post(uri(Tensor) + "/copy_from", {"tensor": self}))

    def div(self, other):
        """Divide this `Tensor` by another `Tensor` or `Number`, broadcasting if necessary."""

        return self._post("div", Map(r=other), Tensor)

    def eq(self, other):
        """Return a boolean `Tensor` with element-wise equality values."""

        return self._post("eq", Map(r=other), Tensor)

    def exp(self):
        """Raise `e` to the power of this `Tensor`."""

        return self._get("exp", rtype=self.__class__)

    def expand_dims(self, axis=None):
        """Return a view of this `Tensor` with an extra dimension of size 1 at the given axis."""

        return self._get("expand_dims", axis, self.__class__)

    def gt(self, other):
        """Return a boolean `Tensor` with element-wise greater-than values."""

        return self._post("gt", Map(r=other), self.__class__)

    def gte(self, other):
        """Return a boolean `Tensor` with element-wise greater-or-equal values."""

        return self._post("gte", Map(r=other), Tensor)

    def lt(self, other):
        """Return a boolean `Tensor` with element-wise less-than values."""

        return self._post("lt", Map(r=other), self.__class__)

    def lte(self, other):
        """Return a boolean `Tensor` with element-wise less-or-equal values."""

        return self._post("lte", Map(r=other), Tensor)

    def logical_and(self, other):
        """Return a boolean `Tensor` with element-wise logical and values."""

        return self._post("and", Map(r=other), self.__class__)

    def logical_not(self):
        """Return a boolean `Tensor` with element-wise logical not values."""

        return self._get("not", rtype=Tensor)

    def logical_or(self, other):
        """Return a boolean `Tensor` with element-wise logical or values."""

        return self._post("or", Map(r=other), Tensor)

    def logical_xor(self, other):
        """Return a boolean `Tensor` with element-wise logical xor values."""

        return self._post("xor", Map(r=other), Tensor)

    def mul(self, other):
        """Multiply this `Tensor` by another `Tensor` or `Number`, broadcasting if necessary."""

        return self._post("mul", Map(r=other), self.__class__)

    def ne(self, other):
        """Return a boolean `Tensor` with element-wise not-equal values."""

        return self._post("ne", Map(r=other), self.__class__)

    def pow(self, other):
        """Raise this `Tensor` to the given power."""

        return self._post("pow", Map(r=other), self.__class__)

    def product(self, axis=None):
        """Calculate the product of this `Tensor` along the given `axis`, or the total product if no axis is given."""

        rtype = Number if axis is None else self.__class__
        return self._get("product", axis, rtype)

    def shape(self):
        """Return the shape of this `Tensor`."""
        return self._get("shape", rtype=Tuple)

    def sub(self, other):
        """Subtract another `Tensor` or `Number` from this one, broadcasting if necessary."""

        return self._post("sub", Map(r=other), Tensor)

    def sum(self, axis=None):
        """Calculate the sum of this `Tensor` along the given `axis`, or the total sum if no axis is given."""

        rtype = Number if axis is None else self.__class__
        return self._get("sum", axis, rtype)

    def transpose(self, permutation=None):
        """
        Return a view of this `Tensor` with its axes transposed according to the given permutation.

        If no permutation is given, the axes will be inverted (e.g. `(0, 1, 2)` inverts to `(2, 1, 0)`).
        """

        return self._get("transpose", permutation, self.__class__)

    def write(self, bounds, value):
        """Write a `Tensor` or `Number` to the given slice of this one, broadcasting if needed."""

        return self.__setitem__(bounds, value)

    def overwrite(self, value):
        """Overwrite all elements of this `Tensor` with the given `value`, broadcasting if needed."""

        return self.__setitem__(None, value)


class Dense(Tensor):
    """
    An n-dimensional array of numbers stored as sequential blocks.

    **IMPORTANT**: for efficiency reasons, serialization of a `Dense` tensor will stop if a non-numeric value
    (NaN or +/- infinity) is encountered. If you receive a `Dense` tensor without enough elements for its shape,
    you can safely treat this response as a divide-by-zero error.
    """

    __uri__ = uri(Tensor) + "/dense"

    @classmethod
    def arange(cls, shape, start, stop):
        """
        Return a `Dense` tensor with the given shape containing a range of numbers
        evenly distributed between `start` and `stop`.
        """

        return cls(ref.Get(uri(cls) + "/range", (shape, start, stop)))

    @classmethod
    def constant(cls, shape, value):
        """Return a `Dense` tensor filled with the given `value`."""

        return cls(ref.Get(uri(cls) + "/constant", (shape, value)))

    @classmethod
    def ones(cls, shape, dtype=F32):
        """
        Return a `Dense` tensor filled with ones.

        If `dtype` is not specified, the data type will be :class:`F32`.
        """

        return cls.constant(shape, dtype(1))

    @classmethod
    def zeros(cls, shape, dtype=F32):
        """
        Return a `Dense` tensor filled with zeros.

        If `dtype` is not specified, the data type will be :class:`F32`.
        """

        return cls.constant(shape, dtype(0))

    def elements(self, bounds):
        """Return a :class:`Stream` of the :class:`Number` elements of this `Dense` tensor."""

        bounds = _handle_bounds(bounds)
        return self._get("elements", bounds, Stream)


class Sparse(Tensor):
    """
    An n-dimensional array of numbers stored as a :class:`Table` of coordinates and values.

    **IMPORTANT**: be careful when broadcasting a `Sparse` tensor--the result may not be so sparse!
    For example, broadcasting a `Sparse` tensor with shape [2, 1] with exactly one element into shape [2, 1000]
    will result in a `Sparse` tensor with 1000 elements.

    The `and`, `div`, and `mul` methods are optimized to avoid this issue by ignoring right-hand values at coordinates
    which are not filled in the left-hand `Tensor`.
    """

    __uri__ = uri(Tensor) + "/sparse"

    @classmethod
    def zeros(cls, shape, dtype=F32):
        """
        Return a `Sparse` tensor with the given shape and data type.

        If `dtype` is not specified, the data type will be :class:`F32`.
        """

        return cls(Schema(shape, dtype))

    def elements(self, bounds=None):
        """Return a :class:`Stream` of this tensor's (:class:`Tuple`, :class:`Number`) coordinate-value elements."""

        bounds = _handle_bounds(bounds)
        return self._get("elements", bounds, Stream)


def einsum(format, tensors):
    """
    Return the Einstein summation of the given `tensors` according the the given `format` string.

    Example: `einsum("ij,kj->ik", [a, b]) # compute the dot product of a and b`

    The tensor product is computed from left to right, so when using any `Sparse` tensors, it's important to put
    the sparsest first in the list to avoid redundant broadcasting.
    """

    return Tensor(ref.Post(uri(Tensor) + "/einsum", {"format": format, "tensors": tensors}))


def _handle_bounds(bounds):
    if bounds is None or isinstance(bounds, ref.Ref) or isinstance(bounds, URI):
        return bounds

    if isinstance(bounds, State):
        form = form_of(bounds)
        if isinstance(form, tuple) or isinstance(form, list):
            bounds = form
        else:
            return bounds

    if hasattr(bounds, "__iter__"):
        bounds = tuple(bounds)
    else:
        bounds = (bounds,)

    return [
        Range.from_slice(x) if isinstance(x, slice)
        else x for x in bounds]
