"""Tests for the Quaternion class: arithmetic, rotations and composition."""

import math

import pytest

from quaternions import Quaternion

ONE = Quaternion(1, 0, 0, 0)
I = Quaternion(0, 1, 0, 0)
J = Quaternion(0, 0, 1, 0)
K = Quaternion(0, 0, 0, 1)

X_AXIS = (1, 0, 0)
Y_AXIS = (0, 1, 0)
Z_AXIS = (0, 0, 1)


@pytest.fixture
def q():
    return Quaternion(1, 2, 3, 4)


@pytest.fixture
def p():
    return Quaternion(-2, 0.5, 1, 3)


def length(v):
    return math.sqrt(sum(c * c for c in v))


# ----- construction --------------------------------------------------------


def test_components(q):
    assert (q.w, q.x, q.y, q.z) == (1, 2, 3, 4)
    assert q.vector == (2, 3, 4)
    assert q.as_tuple() == (1, 2, 3, 4)


def test_default_is_zero():
    assert Quaternion() == Quaternion(0, 0, 0, 0)


def test_components_are_read_only(q):
    with pytest.raises(AttributeError):
        q.w = 10


@pytest.mark.parametrize("bad", ["1", None, True, [1]])
def test_non_number_component_raises(bad):
    with pytest.raises(TypeError):
        Quaternion(bad, 0, 0, 0)


def test_repr(q):
    assert repr(q) == "Quaternion(1, 2, 3, 4)"


def test_from_vector():
    assert Quaternion.from_vector((1, 2, 3)) == Quaternion(0, 1, 2, 3)


@pytest.mark.parametrize("bad", [(1, 2), (1, 2, 3, 4)])
def test_from_vector_wrong_length_raises(bad):
    with pytest.raises(ValueError, match="3 components"):
        Quaternion.from_vector(bad)


# ----- comparison ----------------------------------------------------------


def test_equality(q):
    assert q == Quaternion(1, 2, 3, 4)
    assert q != Quaternion(1, 2, 3, 5)
    assert q != (1, 2, 3, 4)


def test_equal_quaternions_have_equal_hashes(q):
    assert hash(q) == hash(Quaternion(1, 2, 3, 4))


def test_isclose():
    assert Quaternion(1, 0, 0, 0).isclose(Quaternion(1 + 1e-12, 0, 0, 0))
    assert not Quaternion(1, 0, 0, 0).isclose(Quaternion(1.1, 0, 0, 0))


# ----- arithmetic ----------------------------------------------------------


def test_add(q, p):
    assert q + p == Quaternion(-1, 2.5, 4, 7)


def test_sub(q, p):
    assert q - p == Quaternion(3, 1.5, 2, 1)


def test_neg(q):
    assert -q == Quaternion(-1, -2, -3, -4)


def test_scalar_multiplication_and_division(q):
    assert q * 2 == Quaternion(2, 4, 6, 8)
    assert 2 * q == Quaternion(2, 4, 6, 8)
    assert q / 2 == Quaternion(0.5, 1, 1.5, 2)


def test_add_non_quaternion_raises(q):
    with pytest.raises(TypeError):
        q + 1


@pytest.mark.parametrize(
    ("left", "right", "expected"),
    [
        (I, I, -ONE),
        (J, J, -ONE),
        (K, K, -ONE),
        (I, J, K),
        (J, K, I),
        (K, I, J),
        (J, I, -K),
        (K, J, -I),
        (I, K, -J),
    ],
)
def test_unit_multiplication_table(left, right, expected):
    assert left * right == expected


def test_multiplication_is_not_commutative(q, p):
    assert q * p != p * q


def test_multiplication_example(q, p):
    # Checked by hand with q * p = (w1*w2 - v1.v2,  w1*v2 + w2*v1 + v1 x v2):
    # w = -2 - (1 + 3 + 12) = -18
    # v = (0.5, 1, 3) + (-4, -6, -8) + (5, -4, 0.5) = (1.5, -9, -4.5)
    assert q * p == Quaternion(-18, 1.5, -9, -4.5)


def test_one_is_neutral(q):
    assert q * ONE == q
    assert ONE * q == q


def test_multiplication_is_associative(q, p):
    r = Quaternion(0.3, -1, 2, 0.5)
    assert ((q * p) * r).isclose(q * (p * r))


# ----- conjugate, norm, inverse --------------------------------------------


def test_conjugate(q):
    assert q.conjugate() == Quaternion(1, -2, -3, -4)


def test_conjugate_of_product(q, p):
    assert (q * p).conjugate() == p.conjugate() * q.conjugate()


def test_norm(q):
    assert q.norm() == pytest.approx(math.sqrt(30))


def test_norm_is_multiplicative(q, p):
    assert (q * p).norm() == pytest.approx(q.norm() * p.norm())


def test_normalized_has_length_one(q):
    assert q.normalized().norm() == pytest.approx(1)


def test_inverse(q):
    assert (q * q.inverse()).isclose(ONE)
    assert (q.inverse() * q).isclose(ONE)


def test_zero_has_no_inverse():
    with pytest.raises(ValueError, match="no inverse"):
        Quaternion().inverse()


def test_zero_cannot_be_normalized():
    with pytest.raises(ValueError):
        Quaternion().normalized()


# ----- rotations -----------------------------------------------------------


@pytest.mark.parametrize(
    ("axis", "vector", "expected"),
    [
        (Z_AXIS, (1, 0, 0), (0, 1, 0)),
        (Z_AXIS, (0, 1, 0), (-1, 0, 0)),
        (X_AXIS, (0, 1, 0), (0, 0, 1)),
        (Y_AXIS, (0, 0, 1), (1, 0, 0)),
    ],
)
def test_quarter_turn(axis, vector, expected):
    rotation = Quaternion.from_axis_angle(axis, math.pi / 2)
    assert rotation.rotate(vector) == pytest.approx(expected, abs=1e-12)


def test_half_turn():
    rotation = Quaternion.from_axis_angle(Z_AXIS, math.pi)
    assert rotation.rotate((1, 2, 3)) == pytest.approx((-1, -2, 3), abs=1e-12)


def test_rotation_quaternion_is_unit():
    rotation = Quaternion.from_axis_angle((1, 2, 3), 0.7)
    assert rotation.norm() == pytest.approx(1)


def test_axis_length_does_not_matter():
    short = Quaternion.from_axis_angle((0, 0, 1), 1.0)
    long = Quaternion.from_axis_angle((0, 0, 5), 1.0)
    assert short.isclose(long)


def test_zero_angle_changes_nothing():
    rotation = Quaternion.from_axis_angle((1, 1, 0), 0)
    assert rotation.rotate((3, -2, 5)) == pytest.approx((3, -2, 5))


def test_vector_on_axis_does_not_move():
    rotation = Quaternion.from_axis_angle((1, 1, 1), 2.0)
    assert rotation.rotate((2, 2, 2)) == pytest.approx((2, 2, 2))


def test_rotation_keeps_length():
    rotation = Quaternion.from_axis_angle((1, -2, 0.5), 1.3)
    v = (3, 4, 12)
    assert length(rotation.rotate(v)) == pytest.approx(length(v))


def test_full_turn_returns_vector():
    rotation = Quaternion.from_axis_angle((0, 1, 1), 2 * math.pi)
    assert rotation.rotate((1, 2, 3)) == pytest.approx((1, 2, 3))


def test_non_unit_quaternion_rotates_the_same():
    rotation = Quaternion.from_axis_angle((1, 0, 0), 0.8)
    v = (1, 2, 3)
    assert (rotation * 5).rotate(v) == pytest.approx(rotation.rotate(v))


def test_zero_axis_raises():
    with pytest.raises(ValueError, match="zero vector"):
        Quaternion.from_axis_angle((0, 0, 0), 1.0)


def test_rotate_by_zero_quaternion_raises():
    with pytest.raises(ValueError):
        Quaternion().rotate((1, 0, 0))


# ----- composition ---------------------------------------------------------


def test_composition_equals_sequential_rotations():
    first = Quaternion.from_axis_angle((1, 0, 0), 0.4)
    second = Quaternion.from_axis_angle((0, 1, 1), 1.1)
    v = (1, -2, 0.5)
    one_by_one = second.rotate(first.rotate(v))
    combined = (second * first).rotate(v)
    assert combined == pytest.approx(one_by_one)


def test_order_of_rotations_matters():
    about_x = Quaternion.from_axis_angle(X_AXIS, math.pi / 2)
    about_z = Quaternion.from_axis_angle(Z_AXIS, math.pi / 2)
    v = (1, 0, 0)
    assert (about_z * about_x).rotate(v) != pytest.approx((about_x * about_z).rotate(v))


def test_angles_about_same_axis_add_up():
    axis = (1, 2, 2)
    a = Quaternion.from_axis_angle(axis, 0.3)
    b = Quaternion.from_axis_angle(axis, 0.9)
    assert (a * b).isclose(Quaternion.from_axis_angle(axis, 1.2))


def test_inverse_undoes_rotation():
    rotation = Quaternion.from_axis_angle((2, -1, 3), 0.75)
    v = (4, 5, 6)
    assert rotation.inverse().rotate(rotation.rotate(v)) == pytest.approx(v)
