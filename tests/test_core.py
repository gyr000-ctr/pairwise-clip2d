import math

import pytest

import clip2d

def _raises(fn, exc=ValueError):
    try:
        fn()
    except exc:
        return True
    return False

SQ = [(1, 1), (3, 1), (3, 3), (1, 3)]
BIG = [(-1, -1), (5, -1), (5, 5), (-1, 5)]


def _c():
    return clip2d.Clipper()


def _check_inside():
    return _c().clip(SQ, (0, 0, 4, 4)) == SQ


def _check_outside():
    return _c().clip([(10, 10), (11, 10), (11, 11)], (0, 0, 4, 4)) == []


def _check_partial():
    return len(_c().clip([(0, 0), (10, 0), (10, 10), (0, 10)], (0, 0, 4, 4))) >= 3


def _check_bounds():
    out = _c().clip(BIG, (0, 0, 4, 4))
    return all(0 <= x <= 4 and 0 <= y <= 4 for x, y in out)


def _check_area_not_grow():
    out = _c().clip(BIG, (0, 0, 4, 4))
    return abs(_area(out)) <= 16.0 + 1e-9


def _check_empty_poly():
    return _c().clip([], (0, 0, 4, 4)) == []


def _check_two_points():
    return _c().clip([(0, 0), (10, 10)], (0, 0, 4, 4)) == []


def _check_bad_rect():
    return _raises(lambda: _c().clip(SQ, (4, 4, 0, 0)))


def _check_det():
    return _c().clip(SQ, (0, 0, 4, 4)) == _c().clip(SQ, (0, 0, 4, 4))


def _check_type():
    return isinstance(_c().clip(SQ, (0, 0, 4, 4)), list)


def _check_inside_area():
    return abs(_area(_c().clip(SQ, (0, 0, 4, 4))) - 4.0) < 1e-9


def _check_clipped_inside():
    out = _c().clip([(-1, 1), (5, 1), (5, 3), (-1, 3)], (0, 0, 4, 4))
    return all(0 <= x <= 4 for x, y in out)


def _check_no_mutation():
    poly = list(SQ)
    _c().clip(SQ, (0, 0, 4, 4))
    return poly == SQ


def _check_corner():
    out = _c().clip([(2, 2), (8, 2), (8, 8), (2, 8)], (0, 0, 4, 4))
    return len(out) == 4


def _check_idempotent():
    once = _c().clip(SQ, (0, 0, 4, 4))
    return _c().clip(once, (0, 0, 4, 4)) == once


def _area(poly):
    s = 0.0
    for i in range(len(poly)):
        x1, y1 = poly[i]
        x2, y2 = poly[(i + 1) % len(poly)]
        s += x1 * y2 - x2 * y1
    return abs(s) / 2.0

CASES = [("inside", "_check_inside"), ("outside", "_check_outside"), ("partial", "_check_partial"), ("bounds", "_check_bounds"), ("area_not_grow", "_check_area_not_grow"), ("empty_poly", "_check_empty_poly"), ("two_points", "_check_two_points"), ("bad_rect", "_check_bad_rect"), ("det", "_check_det"), ("type", "_check_type"), ("inside_area", "_check_inside_area"), ("clipped_inside", "_check_clipped_inside"), ("no_mutation", "_check_no_mutation"), ("corner", "_check_corner"), ("idempotent", "_check_idempotent")]


@pytest.mark.parametrize("name,check", CASES, ids=[c[0] for c in CASES])
def test_spec(name, check):
    assert globals()[check]() is True
