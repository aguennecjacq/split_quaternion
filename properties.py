from abc import ABC
import numpy as np
from numba import jit


class QuaternionPropertiesMixin(ABC):
    """Basic properties for the split quaternion array class.

    This abstract base class comprises the basic interface for split quaternion
    arrays, including the components (w, x, y, z), the parts (scalar,
    vector), the norm and absolute value, the normalized equivalent, and so
    on.
    """

    @property
    def w(self):
        """The first (scalar) component of the split quaternion"""
        return self.ndarray[..., 0]

    @w.setter
    def w(self, w_new):
        self.ndarray[..., 0] = w_new

    @property
    def x(self):
        """The second component of the split quaternion"""
        return self.ndarray[..., 1]

    @x.setter
    def x(self, x_new):
        self.ndarray[..., 1] = x_new

    @property
    def y(self):
        """The third component of the split quaternion"""
        return self.ndarray[..., 2]

    @y.setter
    def y(self, y_new):
        self.ndarray[..., 2] = y_new

    @property
    def z(self):
        """The fourth component of the split quaternion"""
        return self.ndarray[..., 3]

    @z.setter
    def z(self, z_new):
        self.ndarray[..., 3] = z_new

    @property
    def vector(self):
        """The "vector" part of the split quaternion (final three components)"""
        return self.ndarray[..., 1:]

    @vector.setter
    def vector(self, v):
        self.ndarray[..., 1:] = v

    @property
    @jit
    def norm(self):
        """The norm of the split quaternion."""
        s = self.reshape((-1, 4))
        n = np.empty(s.shape[0], dtype=self.dtype)
        for i in range(s.shape[0]):
            n[i] = s[i, 0] ** 2 + s[i, 1] ** 2 - s[i, 2] ** 2 - s[i, 3] ** 2
        return n.reshape(self.shape[:-1])

    @property
    @jit
    def abs(self):
        """The absolute value of the split quaternion."""
        s = self.reshape((-1, 4))
        n = np.empty(s.shape[0], dtype=self.dtype)
        for i in range(s.shape[0]):
            n[i] = np.sqrt(np.abs(s[i, 0] ** 2 + s[i, 1] ** 2 - s[i, 2] ** 2 - s[i, 3] ** 2))
        return n.reshape(self.shape[:-1])

    @property
    def conjugate(self):
        """The split quaternion conjugate"""
        c = self.copy()
        c.vector *= -1
        return c

    @property
    @jit
    def inverse(self):
        """The multiplicative inverse of the split quaternion"""
        s = self.reshape((-1, 4))
        inv = np.empty(s.shape, dtype=self.dtype)
        for i in range(s.shape[0]):
            n = s[i, 0] ** 2 + s[i, 1] ** 2 - s[i, 2] ** 2 - s[i, 3] ** 2
            inv[i, 0] = s[i, 0] / n
            inv[i, 1] = -s[i, 1] / n
            inv[i, 2] = -s[i, 2] / n
            inv[i, 3] = -s[i, 3] / n
        return inv.reshape(self.shape)

    @property
    def normalized(self):
        """The normalized version of this split quaternion"""
        return self / self.abs

    @property
    def ndarray(self):
        """View this array as a numpy ndarray"""
        return self.view(np.ndarray)

    @property
    def flattened(self):
        """A view of this array with all but the last dimension combined into one"""
        return self.reshape((-1, 4))

    @property
    def iterator(self):
        """Iterate over all but the last dimension of this split quaternion array"""
        s = self.reshape((-1, 4))
        for i in range(s.shape[0]):
            yield s[i]

    def nonzero(self):
        """Return the indices of all nonzero elements.

        This is essentially the same function as numpy.nonzero, except that
        the last dimension is treated as a single split quaternion; if any
        component of the split quaternion is nonzero, the split quaternion is
        considered nonzero.
        """
        return np.nonzero(np.atleast_1d(np.any(self.ndarray, axis=-1)))

    # Aliases
    scalar = w
    i = x
    j = y
    k = z
    modulus = norm
    conj = conjugate
