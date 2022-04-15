import pytest
import sys, os, os.path as osp
sys.path.append(osp.dirname(osp.dirname(osp.abspath(__file__))))
from dev import Categorization, Transaction, TransactionArray, Session, Date, Node
import dev as tf
import numpy as np

tf.debug()


def get_test_array():
    return TransactionArray([
        Transaction(Date(2021,1,1), 5., 'starbucks'),
        Transaction(Date(2021,1,1), 300., 'costco'),
        Transaction(Date(2021,1,2), 45.23, 'some food'),
        ])

def test_categorization():
    ta = get_test_array()
    categorization = Categorization(ta)
    somecat = categorization.new_cat('somecat')
    somesubcat = categorization.new_cat('somesubcat', parent=somecat)
    assert categorization.root.children == [somecat]
    assert somecat.children == [somesubcat]
    
    categorization.category = np.array([0, 1, 2])
    np.testing.assert_array_equal(categorization.select('somecat'), np.array([False, True, False]))
    np.testing.assert_array_equal(categorization.select_multiple(['root', 'somecat']), np.array([True, True, False]))
    np.testing.assert_array_equal(categorization.select_recursively('somecat'), np.array([False, True, True]))


def test_basic_session():
    session = Session(Categorization(get_test_array()))
    root = session.categorization.root

    assert session.exists('category') is False
    cat = session.mkdir('category')
    assert session.exists('category') is True
    assert cat.name == 'category'
    assert cat.parent is root

    assert session.get_dir('/category/..') is root
    assert session.get_dir('category') is cat
    assert session.get_dir('/category') is cat
    assert session.get_dir('category/./..//category/./') is cat

    assert session.pwd == '/'
    session.cd('category')
    assert session.pwd == '/category'
    session.cd('./../category/.././')
    assert session.pwd == '/'

    with pytest.raises(tf.NoSuchPath):
        session.cd('does_not_exist')

    assert session.abspath(root) == '/'


def test_expression_formatting():
    assert list(tf.yield_code_blocks('')) == []
    assert list(tf.yield_code_blocks('a')) == [('a', False)]
    assert list(tf.yield_code_blocks('"a"')) == [('"a"', True)]
    assert list(tf.yield_code_blocks('a"a"')) == [('a', False), ('"a"', True)]
    assert list(tf.yield_code_blocks("a'a'bb")) == [('a', False), ("'a'", True), ('bb', False)]
    assert list(tf.yield_code_blocks("a''bb")) == [('a', False), ("''", True), ('bb', False)]
    assert list(tf.yield_code_blocks("a''''")) == [('a', False), ("''", True), ("''", True)]

    assert tf.format_expression('$somecat') == 'cat("somecat")'
    assert tf.format_expression('$somecat.bla') == 'cat("somecat.bla")'
    assert tf.format_expression('$somecat.bla().foo') == 'cat("somecat.bla")().foo'

    assert tf.format_expression('cd somecat') == 'changedir("somecat")'
    assert tf.format_expression('cd ./..//somecat/bla bla') == 'changedir("./..//somecat/bla") bla'
    assert tf.format_expression('cd somecat; pwd') == 'changedir("somecat"); printworkdir()'

    # assert tf.format_expression('"foo"$somecat"bla"') == '"foo"cat("somecat")"bla"'
    # assert tf.format_expression('"foo"$somecat"bla\\""') == '"foo"cat("somecat")"bla\\""'


def test_expression_evaluating():
    ta = get_test_array()
    session = Session(Categorization(ta))
    somecat = session.mkdir('somecat')
    session.categorization.category = np.array([0, 1, 1])

    assert tf.evaluate(session, '($.).category') is session.categorization.root
    assert tf.evaluate(session, '$somecat.category') is somecat
    np.testing.assert_array_equal(tf.evaluate(session, '$somecat.ta'), ta[1:])
    np.testing.assert_array_equal(tf.evaluate(session, '$/somecat.ta'), ta[1:])
    np.testing.assert_array_equal(tf.evaluate(session, '$/somecat/../somecat.ta'), ta[1:])
    np.testing.assert_array_equal(tf.evaluate(session, '($/somecat/..).ta'), ta[:1])
    np.testing.assert_array_equal(tf.evaluate(session, '$somecat.amount'), ta[1:].amount)
    np.testing.assert_array_equal(tf.evaluate(session, '$somecat.amount.sum()'), ta[1:].amount.sum())

    with pytest.raises(ValueError):
        tf.evaluate(session, 'cat("...amount")')
    with pytest.raises(tf.NoSuchPath):
        tf.evaluate(session, 'cat("/blabla")')

    np.testing.assert_array_equal(tf.evaluate(session, '$somecat[$somecat.amount>100].ta'), ta[1:2])
    np.testing.assert_array_equal(tf.evaluate(session, '$somecat[1:2].ta'), ta[2:])
    np.testing.assert_array_equal(tf.evaluate(session, '$somecat[0].ta'), ta[1])
    np.testing.assert_array_equal(tf.evaluate(session, '($somecat & (amount>100)).ta'), ta[1:2])

    newcat = session.mkdir('newcat')
    tf.evaluate(session, '$somecat >> $newcat')
    np.testing.assert_array_equal(tf.evaluate(session, '$newcat.ta'), ta[1:])





    # category = root.mkdir('category')
    # subcategory = category.mkdir('subcategory')    
    # session = Session(root)

    # assert session.pwd == '/'
    # session.cd('category')
    # assert session.pwd == '/category'

    # assert session.isdir('/category/subcategory')
    # session.rm('/category/subcategory')
    # assert not session.isdir('/category/subcategory')
    

# def test_expressions():
#     root = Directory('/')
#     root.ta = TransactionArray([
#         Transaction(Date(2021,1,1), 5., 'starbucks'),
#         Transaction(Date(2021,1,1), 300., 'costco'),
#         Transaction(Date(2021,1,2), 45.23, 'some food'),
#         ])
#     category = root.mkdir('category')
#     subcategory = category.mkdir('subcategory')    
#     session = Session(root)

#     np.testing.assert_array_equal(session.pwd_dir.ta, root.ta)
#     np.testing.assert_array_equal(session.eval('_'), root.ta)
#     np.testing.assert_array_equal(session.eval('desc == "costco"'), root.ta.description == 'costco')


# def test_expr_split():
#     assert tf.expr_split('') == ['']
#     assert tf.expr_split('aa') == ['aa']
#     assert tf.expr_split('ab|cd|ef') == ['ab', 'cd', 'ef']
#     assert tf.expr_split('ab|c"d"|ef') == ['ab', 'c"d"', 'ef']
#     assert tf.expr_split('ab|c"|d"|\'ef||\'') == ['ab', 'c"|d"', "'ef||'"]
#     assert tf.expr_split('a>>a') == ['a', '>> a']
#     assert tf.expr_split('a | b >> c') == ['a ', ' b ', '>>  c']

# def test_expr_pythonize():
#     assert tf.pythonize_expression('>> bla') == 'categorize("bla")'


# def test_categorizing():
#     root = Directory('/')
#     ta = TransactionArray([
#         Transaction(Date(2021,1,1), 5., 'starbucks'),
#         Transaction(Date(2021,1,1), 300., 'costco'),
#         Transaction(Date(2021,1,2), 45.23, 'some food'),
#         ])
#     root.ta = ta
#     category = root.mkdir('category')
#     session = Session(root)

#     session.eval('_ >> category')
#     # np.testing.assert_array_equal(root.ta, TransactionArray([]))
#     np.testing.assert_array_equal(category.ta, ta)

#     np.testing.assert_array_equal(session.eval('category'), category.ta)