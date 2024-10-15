import abc
import numpy as np


class QuaternionConvertersMixin(abc.ABC):
    """Converters for the split quaternions array class.
    This abstract base class provides converters for quaternionic arrays.
    """

    @property
    def to_scalar_part(self):
        """The "scalar" part of the split quaternion (first component)."""
        return self.scalar

    @classmethod
    def from_scalar_part(cls, scalars):
        """Create a quaternionic array from its scalar part.

        Essentially, this just inserts three 0s after each scalar part, and
        re-interprets the result as a split quaternion.

        Parameters
        ----------
        scalars : float array
            Array of scalar parts of quaternions.

        Returns
        -------
        q : array of split quaternions
            Split quaternions with scalar parts corresponding to input scalars.  Output shape
            is scalars.shape+(4,).
        """
        q = np.zeros(scalars.shape + (4,), dtype=scalars.dtype)
        q[..., 0] = scalars
        return cls(q)

    @property
    def to_vector_part(self):
        """ The "vector" part of the split quaternion (final three components).
        """
        return self.vector

    @classmethod
    def from_vector_part(cls, vec):
        """Create a quaternionic array from its vector part.

        Essentially, this just inserts a 0 in front of each vector part, and
        re-interprets the result as a split quaternion.

        Parameters
        ----------
        vec : (..., 3) float array

            Array of vector parts of split quaternions.

        Returns
        -------
        q : array of split quaternions
            Split quaternions with vector parts corresponding to input vectors.  Output shape
            is vec.shape[:-1]+(4,).

        """
        return cls(np.insert(vec, 0, 0.0, axis=-1))

    @classmethod
    def random(cls, shape=(4,)):
        """Construct random split quaternions

        Parameters
        ----------
        shape : tuple, optional
            Shape of the output array of split quaternions

        Returns
        -------
        q : array of split quaternions

        """
        if isinstance(shape, int):
            shape = (shape,)
        if len(shape) == 0:
            shape = (4,)
        if shape[-1] != 4:
            shape = shape + (4,)
        q = np.random.uniform(low=-1, high=1, size=shape)
        return cls(q)
