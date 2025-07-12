import math
import sys
from types import SimpleNamespace

__version__ = "1.24.0"
int_ = int
uint = int
datetime64 = int
timedelta64 = int
int64 = int
float64 = float
integer = int


class busdaycalendar:
    pass


float32 = float


class ndarray(list):
    def flatten(self):
        return self

    def tolist(self):
        return list(self)


def array(obj, dtype=None):
    if dtype is float32:
        obj = [float(x) for x in obj]
    return ndarray(obj)


def dot(a, b):
    return sum(float(x) * float(y) for x, y in zip(a, b))


class dtype:
    def __init__(self, t):
        self.type = t


class _Linalg:
    @staticmethod
    def norm(vec):
        return math.sqrt(sum(float(v) * float(v) for v in vec))


linalg = _Linalg()


# Provide minimal submodules so ``import numpy.random`` and
# ``import numpy.core`` succeed in tests without installing the real
# NumPy package.
class _Generator:  # minimal stub for pandas
    pass


class _BitGenerator:  # minimal stub for pandas
    pass


class _RandomState:
    pass


random = SimpleNamespace(
    Generator=_Generator,
    BitGenerator=_BitGenerator,
    RandomState=_RandomState,
)
core = SimpleNamespace()
sys.modules.setdefault(__name__ + ".random", random)
sys.modules.setdefault(__name__ + ".core", core)
