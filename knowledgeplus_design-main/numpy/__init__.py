import math
from types import SimpleNamespace

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


class _Linalg:
    @staticmethod
    def norm(vec):
        return math.sqrt(sum(float(v) * float(v) for v in vec))

linalg = _Linalg()
