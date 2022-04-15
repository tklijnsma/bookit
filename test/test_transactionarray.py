import sys, os, os.path as osp
sys.path.append(osp.dirname(osp.dirname(osp.abspath(__file__))))
from dev import Transaction, TransactionArray
import dev as tf
import numpy as np
from datetime import datetime as Date

tf.debug()

def get_test_array():
    return TransactionArray([
        Transaction(Date(2021,1,1), 5., 'starbucks'),
        Transaction(Date(2021,1,1), 300., 'costco'),
        Transaction(Date(2021,1,2), 45.23, 'some food'),
        ])


def test_transaction_inequality():
    t1 = Transaction(Date(2021,1,2), 6., 'starbucks')
    t2 = Transaction(Date(2021,1,2), 6., 'starbucks')
    assert t1 == t2
    t2.amount = 7.
    assert t1 < t2
    t2.date = Date(2021,1,1)
    assert t2 < t1


def test_array_slicing():
    ta = TransactionArray([
        Transaction(Date(2021,1,1), 5., 'starbucks'),
        Transaction(Date(2021,1,1), 300., 'costco'),
        Transaction(Date(2021,1,2), 45.23, 'some food'),
        ])
    ta_1 = TransactionArray([
        Transaction(Date(2021,1,1), 300., 'costco'),
        Transaction(Date(2021,1,2), 45.23, 'some food'),
        ])
    np.testing.assert_array_equal(ta[1:], ta_1)


def test_array_properties():
    ta = TransactionArray([
        Transaction(Date(2021,1,1), 5., 'starbucks'),
        Transaction(Date(2021,1,2), 300., 'costco'),
        ])
    np.testing.assert_array_equal(ta.amount, np.array([5., 300.]))
    np.testing.assert_array_equal(ta.description, np.array(['starbucks', 'costco']))
    np.testing.assert_array_equal(ta.date, np.array(['2021-01-01', '2021-01-02'], dtype='datetime64[D]'))

    
def test_transactionarray_equals():
    ta1 = TransactionArray([
        Transaction(Date(2021,1,1), 5., 'starbucks'),
        Transaction(Date(2021,1,1), 300., 'costco'),
        Transaction(Date(2021,1,2), 45.23, 'some food'),
        ])
    ta2 = TransactionArray([
        Transaction(Date(2021,1,1), 5., 'starbucks'),
        Transaction(Date(2021,1,1), 300., 'costcop'),
        Transaction(Date(2021,1,2), 45.23, 'some food'),
        ])
    np.testing.assert_array_equal((ta1 == ta2), np.array([True, False, True]))


def test_selecting():
    transactions = TransactionArray([
        Transaction(Date(2021,1,1), 5., 'starbucks'),
        Transaction(Date(2021,1,1), 300., 'costco'),
        Transaction(Date(2021,1,2), 7., 'starbucks'),
        ])
    np.testing.assert_equal(transactions.amount < 10., np.array([True, False, True]))
    np.testing.assert_equal(transactions.search_description('starb'), np.array([True, False, True]))
    np.testing.assert_equal(
        transactions[transactions.amount < 10.],
        TransactionArray([
            Transaction(Date(2021,1,1), 5., 'starbucks'),
            Transaction(Date(2021,1,2), 7., 'starbucks'),
            ])
        )
    np.testing.assert_equal(
        transactions[transactions.date > np.datetime64('2021-01-01')],
        TransactionArray([Transaction(Date(2021,1,2), 7., 'starbucks')])
        )


def test_stitching():
    ta1 = TransactionArray([
        Transaction(Date(2021,1,1), 5., 'starbucks'),
        Transaction(Date(2021,1,2), 300., 'costco'),
        Transaction(Date(2021,1,3), 45.23, 'some food'),
        Transaction(Date(2021,1,4), 3.23, 'diapers'),
        ])
    ta2 = TransactionArray([
        Transaction(Date(2021,1,3), 45.23, 'some food'),
        Transaction(Date(2021,1,4), 3.23, 'diapers'),
        Transaction(Date(2021,1,5), 4.85, 'taxes'),
        Transaction(Date(2021,1,6), 70.12, 'oil change'),
        ])
    expected = np.concatenate((ta1, ta2[2:]))
    np.testing.assert_array_equal(tf.stitch(ta1, ta2), expected)


# def test_np_object_array_implementation():
#     ta = TransactionArray([
#         Transaction(Date(2021,1,5), 5., 'starbucks'),
#         Transaction(Date(2021,1,1), 300., 'costco'),
#         Transaction(Date(2021,1,1), 300., 'some food'),
#         ])

#     ta2 = TransactionArray([
#         Transaction(Date(2021,1,1), 5., 'starbucks'),
#         Transaction(Date(2021,1,2), 300., 'costcop'),
#         Transaction(Date(2021,1,3), 45.23, 'some food'),
#         ])
#     # print(ta)
#     # print(ta.shape)
#     # print(np.vstack((ta, ta)))
#     # print(ta.amount)
#     # c = vstack((ta, ta2)).T
#     # print(c)
#     # print(type(c))
#     # print(c.amount)
#     # print(c.camount)
#     # print(c.currency_name)
#     # print(c.date)
#     # print(c.ravel())
#     # print(c.ravel().reshape((2,3)))
#     # print(ta != ta2)
#     # s = c.sort_by_amount(axis=0)
#     # print(s)
#     # print(type(s))
#     # print(c.search_description('star'))
#     # print(c)
#     # c.standard_sort()
#     # print(c)
#     print(ta.sort_1d())
#     print(ta2.sort_1d())
