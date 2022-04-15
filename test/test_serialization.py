import pytest
import numpy as np

import sys, os, os.path as osp
sys.path.append(osp.dirname(osp.dirname(osp.abspath(__file__))))

import dev as tf

from serialization import dumps, loads


def test_basic_dumps():
    ta = tf.TransactionArray([
        tf.Transaction(tf.Date(2021,1,1), 5., 'starbucks'),
        tf.Transaction(tf.Date(2021,1,1), 300., 'costco'),
        tf.Transaction(tf.Date(2021,1,2), 45.23, 'some food'),
        ])
    t = ta[0]
    assert loads(dumps(t), currencies=[tf.USD]) == t
    np.testing.assert_array_equal(loads(dumps(ta), currencies=[tf.USD]), ta)

    root = tf.Node('root', 0)
    child1 = tf.Node('child1', 1, root)
    tf.Node('subchild', 3, child1)
    tf.Node('child2', 2, root)

    reconstr_root = loads(dumps(root), currencies=[tf.USD])
    assert list(tf.iter_dfs(root)) == list(tf.iter_dfs(reconstr_root))

    categorization = tf.Categorization(ta, category=np.array([0,1,0]), root=root)
    reconstr_categorization = loads(dumps(categorization), currencies=[tf.USD])

    np.testing.assert_array_equal(categorization.ta, reconstr_categorization.ta)
    np.testing.assert_array_equal(categorization.category, reconstr_categorization.category)
    assert categorization.root == reconstr_categorization.root
