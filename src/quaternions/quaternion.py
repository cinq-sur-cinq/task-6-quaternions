"""Quaternion arithmetic and 3D rotations."""

import math
from numbers import Real

Vector3 = tuple[float, float, float]


class Quaternion:
    """A quaternion with scalar part w and vector part (x, y, z).

    >>> Quaternion(1, 2, 3, 4) + Quaternion(1, 1, 1, 1)
    Quaternion(2, 3, 4, 5)
    >>> Quaternion(0, 1, 0, 0) * Quaternion(0, 0, 1, 0)  # i * j = k
    Quaternion(0, 0, 0, 1)
    """

    __slots__ = ("_w", "_x", "_y", "_z", "_rotation_inverse")

    def __init__(self, w: float = 0, x: float = 0, y: float = 0, z: float = 0) -> None:
        for value in (w, x, y, z):
            if not _is_real(value):
                raise TypeError(f"quaternion components must be real numbers, got {value!r}")
        self._w = w
        self._x = x
        self._y = y
        self._z = z
        self._rotation_inverse = None

    @classmethod
    def from_vector(cls, v: Vector3) -> "Quaternion":
        """Embed a 3D vector as a quaternion with scalar part zero."""
        x, y, z = _check_vector(v)
        return cls(0, x, y, z)

    @classmethod
    def from_axis_angle(cls, axis: Vector3, angle: float) -> "Quaternion":
        """Build a right-hand rotation about a nonzero axis. Angle is in radians."""
        ax, ay, az = _check_vector(axis)
        length = math.sqrt(ax * ax + ay * ay + az * az)
        if length == 0:
            raise ValueError("rotation axis must not be the zero vector")
        scale = math.sin(angle / 2) / length
        return cls(math.cos(angle / 2), ax * scale, ay * scale, az * scale)

    @property
    def w(self) -> float:
        return self._w

    @property
    def x(self) -> float:
        return self._x

    @property
    def y(self) -> float:
        return self._y

    @property
    def z(self) -> float:
        return self._z

    @property
    def vector(self) -> Vector3:
        return self._x, self._y, self._z

    def as_tuple(self) -> tuple[float, float, float, float]:
        return self._w, self._x, self._y, self._z

    def __repr__(self) -> str:
        return f"Quaternion({self._w!r}, {self._x!r}, {self._y!r}, {self._z!r})"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Quaternion):
            return NotImplemented
        return self.as_tuple() == other.as_tuple()

    def __hash__(self) -> int:
        return hash(self.as_tuple())

    def isclose(self, other: "Quaternion", abs_tol: float = 1e-9) -> bool:
        """Compare components using an absolute tolerance."""
        return all(
            math.isclose(a, b, rel_tol=0, abs_tol=abs_tol)
            for a, b in zip(self.as_tuple(), other.as_tuple())
        )

    def __neg__(self) -> "Quaternion":
        return Quaternion(-self._w, -self._x, -self._y, -self._z)

    def __add__(self, other: "Quaternion") -> "Quaternion":
        if not isinstance(other, Quaternion):
            return NotImplemented
        return Quaternion(*(a + b for a, b in zip(self.as_tuple(), other.as_tuple())))

    def __sub__(self, other: "Quaternion") -> "Quaternion":
        if not isinstance(other, Quaternion):
            return NotImplemented
        return self + (-other)

    def __mul__(self, other: "Quaternion | float") -> "Quaternion":
        if _is_real(other):
            return Quaternion(*(a * other for a in self.as_tuple()))
        if not isinstance(other, Quaternion):
            return NotImplemented

        # Hamilton product. Operand order matters.
        w1, x1, y1, z1 = self.as_tuple()
        w2, x2, y2, z2 = other.as_tuple()
        return Quaternion(
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        )

    def __rmul__(self, other: float) -> "Quaternion":
        if _is_real(other):
            return self * other
        return NotImplemented

    def __truediv__(self, other: float) -> "Quaternion":
        if not _is_real(other):
            return NotImplemented
        return self * (1 / other)

    def conjugate(self) -> "Quaternion":
        return Quaternion(self._w, -self._x, -self._y, -self._z)

    def norm(self) -> float:
        return math.sqrt(sum(a * a for a in self.as_tuple()))

    def normalized(self) -> "Quaternion":
        length = self.norm()
        if length == 0:
            raise ValueError("cannot normalize the zero quaternion")
        return self / length

    def inverse(self) -> "Quaternion":
        length_squared = sum(a * a for a in self.as_tuple())
        if length_squared == 0:
            raise ValueError("the zero quaternion has no inverse")
        return self.conjugate() / length_squared

    def rotate(self, v: Vector3) -> Vector3:
        """Rotate a 3D vector using q * v * q.inverse(), reusing the inverse."""
        product = self * Quaternion.from_vector(v)
        # Older pickles can leave this slot unset.
        inverse = getattr(self, "_rotation_inverse", None)
        if inverse is None:
            inverse = self.inverse()
            self._rotation_inverse = inverse
        return (product * inverse).vector


def _is_real(value: object) -> bool:
    return isinstance(value, Real) and not isinstance(value, bool)


def _check_vector(v: Vector3) -> Vector3:
    components = tuple(v)
    if len(components) != 3:
        raise ValueError(f"a 3D vector needs exactly 3 components, got {len(components)}")
    for component in components:
        if not _is_real(component):
            raise TypeError(f"vector components must be real numbers, got {component!r}")
    return components
