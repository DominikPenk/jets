import math
from dataclasses import dataclass
from typing import Sequence

import pytest

from jets.jet import JetBase

Scalar = float
Vector = Sequence[float]


@dataclass(frozen=True)
class JetUnaryTestData:
    name: str
    a: Scalar | Vector
    da: Scalar | Vector


@dataclass(frozen=True)
class JetBinaryTestData:
    name: str

    # inputs
    a: Scalar | Vector
    da: Scalar | Vector
    b: Scalar | Vector
    db: Scalar | Vector

    # binary ops: (value, grad)
    add: tuple[Scalar | Vector, Scalar | Vector]
    sub: tuple[Scalar | Vector, Scalar | Vector]
    mul: tuple[Scalar | Vector, Scalar | Vector]
    div: tuple[Scalar | Vector, Scalar | Vector]


@dataclass(frozen=True)
class JetReverseBinaryTestData:
    name: str

    # scalar op Jet
    a: Scalar  # scalar on the left
    b: Scalar | Vector  # Jet value
    db: Scalar | Vector  # Jet grad

    add: tuple[Scalar | Vector, Scalar | Vector]
    sub: tuple[Scalar | Vector, Scalar | Vector]
    mul: tuple[Scalar | Vector, Scalar | Vector]
    div: tuple[Scalar | Vector, Scalar | Vector]


@dataclass(frozen=True)
class JetPowerTestData:
    name: str

    # base Jet
    a: Scalar | Vector
    da: Scalar | Vector

    # scalar exponent / base
    p: Scalar

    # expected (value, grad)
    pow: tuple[Scalar | Vector, Scalar | Vector]  # a ** p
    rpow: tuple[Scalar | Vector, Scalar | Vector]  # p ** a


JET_UNARY_TEST_CASES = [
    JetUnaryTestData(name="scalar", a=1.0, da=3.0),
    JetUnaryTestData(name="Vector", a=[1.0, 2.0], da=[3.0, 4.0]),
]


JET_BINARY_TEST_CASES = [
    JetBinaryTestData(
        name="scalar",
        a=2.0,
        da=1.0,
        b=3.0,
        db=4.0,
        add=(5.0, 5.0),
        sub=(-1.0, -3.0),
        mul=(6.0, 11.0),
        div=(2 / 3, (1 * 3 - 2 * 4) / 9),
    ),
    JetBinaryTestData(
        name="Vector",
        a=[2.0, 3.0],
        da=[1.0, 2.0],
        b=[4.0, 5.0],
        db=[2.0, 2.0],
        add=([6.0, 8.0], [3.0, 4.0]),
        sub=([-2.0, -2.0], [-1.0, 0.0]),
        mul=([8.0, 15.0], [8.0, 16.0]),
        div=(
            [0.5, 0.6],
            [
                (1 * 4 - 2 * 2) / 16,
                (2 * 5 - 3 * 2) / 25,
            ],
        ),
    ),
]


JET_REVERSE_BINARY_TEST_CASES = [
    JetReverseBinaryTestData(
        name="scalar-plus-scalar-jet",
        a=5.0,
        b=2.0,
        db=3.0,
        # a + x
        add=(7.0, 3.0),
        # a - x
        sub=(3.0, -3.0),
        # a * x
        mul=(10.0, 15.0),
        # a / x
        div=(2.5, -3.75),  # -a * dx / x^2
    ),
    JetReverseBinaryTestData(
        name="scalar-plus-vector-jet",
        a=10.0,
        b=[2.0, 4.0],
        db=[1.0, 2.0],
        add=([12.0, 14.0], [1.0, 2.0]),
        sub=([8.0, 6.0], [-1.0, -2.0]),
        mul=([20.0, 40.0], [10.0, 20.0]),
        div=(
            [5.0, 2.5],
            [
                -10.0 * 1.0 / 4.0,
                -10.0 * 2.0 / 16.0,
            ],
        ),
    ),
]


JET_POWER_TEST_CASES = [
    JetPowerTestData(
        name="scalar",
        a=2.0,
        da=1.0,
        p=3.0,
        # a ** p = 2^3 = 8
        # d/dx = p * a^(p-1) * da = 3 * 4 * 1 = 12
        pow=(8.0, 12.0),
        # p ** a = 3^2 = 9
        # d/dx = 3^2 * ln(3) * da
        rpow=(9.0, 9.0 * math.log(3.0)),
    ),
    JetPowerTestData(
        name="vector",
        a=[2.0, 3.0],
        da=[1.0, 2.0],
        p=2.0,
        # a ** 2
        # value = [4, 9]
        # grad  = 2 * a * da
        pow=(
            [4.0, 9.0],
            [2 * 2.0 * 1.0, 2 * 3.0 * 2.0],
        ),
        # 2 ** a
        # value = [4, 8]
        # grad  = value * ln(2) * da
        rpow=(
            [4.0, 8.0],
            [
                4.0 * math.log(2.0) * 1.0,
                8.0 * math.log(2.0) * 2.0,
            ],
        ),
    ),
]


class JetContract:
    """
    Contract tests for any Jet implementation.

    Subclasses MUST provide fixtures:
    - jet: factory for creating jets
    """

    # ---------- required fixtures ----------

    @pytest.fixture
    def jet(self) -> type[JetBase]:
        """
        Factory fixture.

        Must return a callable:
            jet(value, grad) -> Jet instance
        """
        raise NotImplementedError

    # ---------- helpers ----------

    def assert_jet(self, x, value, grad):
        assert isinstance(x, JetBase)
        assert x.value == pytest.approx(value)
        assert x.grad == pytest.approx(grad)

    # ---------- construction ----------

    def test_construction(self, jet):
        x = jet(2.0, 1.0)
        self.assert_jet(x, 2.0, 1.0)

    def test_construction_array(self, jet):
        x = jet([1.0, 2.0, 3.0], [1.0, 1.0, 1.0])
        self.assert_jet(x, [1.0, 2.0, 3.0], [1.0, 1.0, 1.0])

    def test_createas(self, jet):
        x = jet(2.0, 1.0)
        y = x.createas(5.0, 3.0)

        self.assert_jet(y, 5.0, 3.0)
        assert type(x) is type(y)

    # ---------- unary operators ----------

    @pytest.mark.parametrize("test_case", JET_UNARY_TEST_CASES, ids=lambda c: c.name)
    def test_neg(self, jet: type[JetBase], test_case: JetUnaryTestData):
        x = jet(test_case.a, test_case.da)

        self.assert_jet(-x, -x.value, -x.grad)

    # ---------- binary operators ----------

    @pytest.mark.parametrize("test_case", JET_BINARY_TEST_CASES, ids=lambda c: c.name)
    def test_add(self, jet: type[JetBase], test_case: JetBinaryTestData):
        x = jet(test_case.a, test_case.da)
        y = jet(test_case.b, test_case.db)

        self.assert_jet(x + y, *test_case.add)

    @pytest.mark.parametrize("test_case", JET_BINARY_TEST_CASES, ids=lambda c: c.name)
    def test_sub(self, jet: type[JetBase], test_case: JetBinaryTestData):
        x = jet(test_case.a, test_case.da)
        y = jet(test_case.b, test_case.db)

        self.assert_jet(x - y, *test_case.sub)

    @pytest.mark.parametrize("test_case", JET_BINARY_TEST_CASES, ids=lambda c: c.name)
    def test_mul(self, jet: type[JetBase], test_case: JetBinaryTestData):
        x = jet(test_case.a, test_case.da)
        y = jet(test_case.b, test_case.db)

        self.assert_jet(x * y, *test_case.mul)

    @pytest.mark.parametrize("test_case", JET_BINARY_TEST_CASES, ids=lambda c: c.name)
    def test_div(self, jet: type[JetBase], test_case: JetBinaryTestData):
        x = jet(test_case.a, test_case.da)
        y = jet(test_case.b, test_case.db)

        self.assert_jet(x / y, *test_case.div)

    # ---------- reverse binary operators ----------

    @pytest.mark.parametrize(
        "test_case",
        JET_REVERSE_BINARY_TEST_CASES,
        ids=lambda case: case.name,
    )
    def test_reverse_add(self, jet, test_case):
        x = jet(test_case.b, test_case.db)
        self.assert_jet(test_case.a + x, *test_case.add)

    @pytest.mark.parametrize(
        "test_case",
        JET_REVERSE_BINARY_TEST_CASES,
        ids=lambda case: case.name,
    )
    def test_reverse_sub(self, jet, test_case):
        x = jet(test_case.b, test_case.db)
        self.assert_jet(test_case.a - x, *test_case.sub)

    @pytest.mark.parametrize(
        "test_case",
        JET_REVERSE_BINARY_TEST_CASES,
        ids=lambda case: case.name,
    )
    def test_reverse_mul(self, jet, test_case):
        x = jet(test_case.b, test_case.db)
        self.assert_jet(test_case.a * x, *test_case.mul)

    @pytest.mark.parametrize(
        "test_case",
        JET_REVERSE_BINARY_TEST_CASES,
        ids=lambda case: case.name,
    )
    def test_reverse_div(self, jet, test_case):
        x = jet(test_case.b, test_case.db)
        self.assert_jet(test_case.a / x, *test_case.div)

    # ---------- power ----------

    @pytest.mark.parametrize(
        "test_case",
        JET_POWER_TEST_CASES,
        ids=lambda c: c.name,
    )
    def test_pow(self, jet, test_case: JetPowerTestData):
        x = jet(test_case.a, test_case.da)
        self.assert_jet(x**test_case.p, *test_case.pow)

    @pytest.mark.parametrize(
        "test_case",
        JET_POWER_TEST_CASES,
        ids=lambda c: c.name,
    )
    def test_rpow(self, jet, test_case: JetPowerTestData):
        x = jet(test_case.a, test_case.da)
        self.assert_jet(test_case.p**x, *test_case.rpow)
