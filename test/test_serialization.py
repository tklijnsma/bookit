import pytest
import numpy as np

import sys, os, os.path as osp
sys.path.append(osp.dirname(osp.dirname(osp.abspath(__file__))))

import bookit as bk
from bookit.serialization import dumps, loads


def test_basic_dumps():
    ta = bk.TransactionArray([
        bk.Transaction(bk.Date(2021,1,1), 5., 'starbucks'),
        bk.Transaction(bk.Date(2021,1,1), 300., 'costco'),
        bk.Transaction(bk.Date(2021,1,2), 45.23, 'some food', 'unfcu'),
        ])
    t = ta[0]
    assert loads(dumps(t), currencies=[bk.currencies.USD]) == t
    np.testing.assert_array_equal(loads(dumps(ta), currencies=[bk.currencies.USD]), ta)

    root = bk.Node('root', 0)
    child1 = bk.Node('child1', 1, root)
    bk.Node('subchild', 3, child1)
    bk.Node('child2', 2, root)

    reconstr_root = loads(dumps(root), currencies=[bk.currencies.USD])
    assert list(bk.iter_dfs(root)) == list(bk.iter_dfs(reconstr_root))

    categorization = bk.Categorization(ta, category=np.array([0,1,0]), root=root)
    reconstr_categorization = loads(dumps(categorization), currencies=[bk.currencies.USD])

    np.testing.assert_array_equal(categorization.ta, reconstr_categorization.ta)
    np.testing.assert_array_equal(categorization.category, reconstr_categorization.category)
    np.testing.assert_array_equal(categorization._i_max_category, reconstr_categorization._i_max_category)

    assert categorization.root == reconstr_categorization.root
