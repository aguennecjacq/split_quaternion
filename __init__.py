import numpy as np
from . import arrays
from . import fourier

array = arrays.SQArray
one = array(1, 0, 0, 0)
one.flags.writeable = False

x = array(0, 1, 0, 0)
x.flags.writeable = False
i = x

y = array(0, 0, 1, 0)
y.flags.writeable = False
j = y

z = array(0, 0, 0, 1)
z.flags.writeable = False
k = z


# Aliases
def ones(shape=(1,)):
    return array.from_scalar_part(np.ones(shape=shape))


def random(shape=(4,)):
    return array.random(shape=shape)