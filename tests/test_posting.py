import pytest

from researchxtract.core.posting import (PostingList, difference, intersect,
                                         intersect_many, union)


def test_posting_list_requires_increasing_ids():
    p = PostingList()
    p.add(1, 2)
    p.add(4, 1)
    assert p.df == 2
    with pytest.raises(ValueError):
        p.add(3, 1)


def test_set_operations():
    a, b = [1, 3, 5, 7], [3, 4, 5, 9]
    assert intersect(a, b) == [3, 5]
    assert union(a, b) == [1, 3, 4, 5, 7, 9]
    assert difference(a, b) == [1, 7]


def test_set_operations_match_python_sets_and_do_not_mutate():
    a, b = [0, 2, 4, 6, 8], [1, 2, 3, 4, 10]
    a_copy, b_copy = a[:], b[:]
    assert intersect(a, b) == sorted(set(a) & set(b))
    assert union(a, b) == sorted(set(a) | set(b))
    assert difference(a, b) == sorted(set(a) - set(b))
    assert (a, b) == (a_copy, b_copy)


def test_intersect_many_and_empty_cases():
    assert intersect_many([[1, 2, 3], [2, 3], [3, 9]]) == [3]
    assert intersect_many([]) == []
    assert intersect([], [1, 2]) == []
