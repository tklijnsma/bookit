from __future__ import annotations
from unicodedata import category
import numpy as np
from dataclasses import dataclass
from datetime import datetime as Date
import pprint
import logging
import os.path as osp, re, sys


DEFAULT_LOGGING_LEVEL = logging.INFO

def setup_logger(name='tf'):
    if name in logging.Logger.manager.loggerDict:
        logger = logging.getLogger(name)
        logger.info('Logger %s is already defined', name)
    else:
        fmt = logging.Formatter(
            fmt = (
                '\033[33m%(levelname)7s:%(asctime)s:%(module)s:%(lineno)s\033[0m'
                + ' %(message)s'
                ),
            datefmt='%Y-%m-%d %H:%M:%S'
            )
        handler = logging.StreamHandler()
        handler.setFormatter(fmt)
        logger = logging.getLogger(name)
        logger.setLevel(DEFAULT_LOGGING_LEVEL)
        logger.addHandler(handler)
    return logger
logger = setup_logger()

def debug(flag=True):
    """Sets the logger level to debug (for True) or warning (for False)"""
    logger.setLevel(logging.DEBUG if flag else DEFAULT_LOGGING_LEVEL)


def set_ps1(path=''):
    sys.ps1 = f"${path}: "


@dataclass
class Currency:
    name: str
    rate: float

USD = Currency('USD', 1.)


@dataclass
class Account:
    name: str

@dataclass
class Transaction:
    date: Date
    amount: float
    description: str
    account: Account = None
    currency: Currency = USD

    @property
    def camount(self):
        return self.amount * self.currency.rate

    def __repr__(self) -> str:        
        if self.amount == 0.:
            color_code = '\033[90m'
            sign = ''
        elif self.amount < 0.:
            color_code = '\033[31m'
            sign = '-'
        elif self.amount > 0.:
            color_code = '\033[32m'
            sign = '+'
        return (
            f"{self.date.strftime('%Y-%m-%d')} {self.currency.name}"
            f" {color_code}{self.amount:{sign}8.2f}\033[0m"
            f" {self.description}"
            )

    def __lt__(self, other):
        if self.date == other.date:
            if self.camount == other.camount:
                if self.description == other.description:
                    return False
                else:
                    return self.description < other.description
            else:
                return self.camount < other.camount
        else:
            return self.date < other.date

    def __add__(self, other) -> float:
        if isinstance(other, Transaction):
            return self.camount + other.camount
        else:
            return self.camount + other

    def __radd__(self, other):
        return self + other



class TransactionArray(np.ndarray):

    def __new__(cls, transactions, info=None):
        return np.asarray(transactions).view(cls)

    def __array_finalize__(self, obj):
        if obj is None: return

    @property
    def amount(self):
        mat = np.empty(self.shape, dtype=float)
        mat.flat = [ t.amount for t in self.flat ]
        return mat

    @property
    def camount(self):
        mat = np.empty(self.shape, dtype=float)
        mat.flat = [ t.camount for t in self.flat ]
        return mat

    @property
    def date(self):
        mat = np.empty(self.shape, dtype='datetime64[D]')
        mat.flat = [ t.date for t in self.flat ]
        return mat

    @property
    def description(self):
        mat = np.empty(self.shape, dtype='<U200')
        mat.flat = [ t.description for t in self.flat ]
        return mat

    @property
    def currency_name(self):
        mat = np.empty(self.shape, dtype='<U10')
        mat.flat = [ t.currency.name for t in self.flat ]
        return mat

    def sum(self, *args, **kwargs):
        return self.camount.sum(*args, **kwargs)

    def __eq__(self, other):
        if isinstance(other, TransactionArray) and self.shape != other.shape:
            raise ValueError(f'Dim mismatch: {self.shape} vs. {other.shape}')
        return super().__eq__(other)

    def sort_by(self, key, axis=-1):
        if key not in {'amount', 'camount', 'date', 'description', 'currency_name'}:
            raise ValueError(f'Unknown sort key \'{key}\'')
        values = getattr(self, key)
        order = np.argsort(values, axis=axis)
        return np.take_along_axis(self, order, axis=axis)

    def sort_1d(self):
        if len(self.shape) > 1:
            raise ValueError('standard_sort_1d only available for 1D TransactionArrays')
        return TransactionArray(sorted(self, key=lambda t: (t.date, t.camount, t.description)))

    def sort_by_date(self, axis=-1):
        return self.sort_by('date', axis)

    def sort_by_amount(self, axis=-1):
        return self.sort_by('amount', axis)

    def sort_by_camount(self, axis=-1):
        return self.sort_by('camount', axis)

    def search_description(self, substring, case_sensitive=False):
        descriptions = self.description
        if not case_sensitive:
            descriptions = np.char.lower(descriptions)
            substring = substring.lower()
        return np.char.find(descriptions, substring) >= 0


# Plug standard numpy array manipulations in scope
np_array_manipulators = [
    'concatenate', 'array_split', 'split', 'hsplit', 'vsplit',
    'dsplit', 'stack', 'block', 'hstack', 'vstack', 'dstack',
    'column_stack'
    ]
for fnname in np_array_manipulators:
    def wrapped(*args, **kwargs):
        return getattr(np, fnname)(*args, 
        **kwargs).view(TransactionArray)
    locals()[fnname] = wrapped


def stitch(ta1: TransactionArray, ta2: TransactionArray):
    # Find the index where the first transaction in ta2 is in ta1
    i = np.argmax(ta1 == ta2[0])
    # Max overlap is however many transactions there are in ta1 after i
    n = ta1.shape[0] - i
    # Check if the rest overlaps
    if np.all(ta1[i:] == ta2[:n]):
        return np.concatenate((ta1[:i], ta2)).view(TransactionArray)
    return np.concatenate((ta1, ta2)).view(TransactionArray)

def merge(tas):
    """
    Merge transaction arrays; filters out duplicates
    """
    return np.unique(np.concatenate(tas)).view(TransactionArray)


# ________________________________________________________________
# ________________________________________________________________
# CATEGORIZATION FRAMEWORK


class NoSuchPath(Exception):
    pass


class Node:
    def __init__(self, name: str, integer: int, parent: Node=None):
        self.name = name
        self.integer = integer
        self.parent = None
        if parent: self.set_parent(parent)
        self.children = []

    def __repr__(self) -> str:
        return super().__repr__().replace('object', '"'+self.name+'"')

    def __eq__(self, other) -> bool:
        return self.name == other.name and self.integer == other.integer

    def add_child(self, other: Node):
        other.parent = self
        self.children.append(other)

    def set_parent(self, other: Node):
        other.add_child(self)

    def get_child(self, name):
        for c in self.children:
            if c.name == name:
                return c
        return None


def iter_dfs(node):
    yield node
    for child in node.children:
        yield from iter_dfs(child)

def iter_dfs_with_depth(node, depth=0):
    yield node, depth
    for child in node.children:
        yield from iter_dfs_with_depth(child, depth+1)

def iter_up(node):
    while True:
        yield node
        if node.parent is None: break
        node = node.parent


class Categorization:
    """
    Main organizing datastructure for categorizing transactions.

    Contains:
    - One TransactionArray of all transactions
    - One np array with the same length as the TransactionArray, mapping
      a transaction to a category (which is simply an integer)
    - A tree of Node's which provide the hierarchy of categories
    """
    def __init__(self, ta: TransactionArray, category: np.ndarray = None, root: Node = None):
        self.ta = ta
        self.category = np.zeros(self.ta.shape[0], dtype=np.int16) if category is None else category
        self.root = Node('root', 0) if root is None else root

        # Internal bookkeeping; mainly dicts for fast lookups
        self._map_int_to_cat = {0: self.root}
        self._map_name_to_cat = {'root': self.root}
        self._map_name_to_int = {'root': 0}

        self._i_max_category = 0

    def new_cat(self, name, parent=None):
        if parent is None: parent = self.root
        self._i_max_category += 1
        cat = Node(name, self._i_max_category)
        cat.set_parent(parent)
        self._map_int_to_cat[cat.integer] = cat
        self._map_name_to_cat[cat.name] = cat
        self._map_name_to_int[cat.name] = cat.integer
        logger.info(
            f'Created new category "{cat.name}" with integer'
            f' {cat.integer} and parent "{parent.name}"'
            )
        return cat

    def select(self, name: Union[str, Node]):
        if isinstance(name, Node):
            integer = name.integer
        else:
            integer = self._map_name_to_int[name]
        return self.category == integer

    def select_multiple(self, names: List[str]):
        integers = [self._map_name_to_int[name] for name in names]
        return np.isin(self.category, integers)

    def select_recursively(self, name: str):
        root = self._map_name_to_cat[name]
        integers = [node.integer for node in iter_dfs(root)]
        return np.isin(self.category, integers)



class Session:
    """
    Provides the linux-like interface to a Categorization instance
    """
    def __init__(self, categorization):
        self.categorization = categorization
        self.curr_dir = categorization.root
        self._filename = None

    def get_dir(self, path):
        base_dir = self.curr_dir
        # First get rid of any needless '.' and '..'
        path = osp.normpath(path)
        # Deal with some easy cases
        if path == '/':
            return self.categorization.root
        elif path == '.':
            return self.curr_dir
        elif path == '..':
            if self.curr_dir.parent is None: raise NoSuchPath(path)
            return self.curr_dir.parent
        # Process any ups:
        while path.startswith('..'):
            if base_dir.parent is None: raise Exception('Cannot go up: reached root')
            base_dir = base_dir.parent
            if path == '..': return base_dir
            path = path.lstrip('./')
        # If path starts with '/', it's an absolute path; start searching from the root
        if path.startswith('/'):
            base_dir = self.categorization.root
        path = path.lstrip('/')
        # Now we have a clean path, and we know we have to start from base_dir
        for subpath in path.split('/'):
            base_dir = base_dir.get_child(subpath)
            if base_dir is None: raise NoSuchPath(path)
        return base_dir


    def abspath(self, directory: Union[str, Node]):
        if isinstance(directory, str): directory = self.get_dir(directory)
        parent_nodes = list(reversed(list(iter_up(directory))))
        parent_nodes = parent_nodes[1:] # Strip off the root and reverse
        return '/' + '/'.join([n.name for n in parent_nodes])


    @property
    def pwd(self):
        return self.abspath(self.curr_dir)


    def cd(self, path: str):
        self.curr_dir = self.get_dir(path)


    def exists(self, path):
        try:
            self.get_dir(path)
            return True
        except NoSuchPath:
            return False


    def isdir(self, path):
        return self.exists(path)


    def rm(self, path: str):
        # TODO: What to do with transactions?
        node = self.get_dir(path)
        if node.parent:
            node.parent.children.remove(node)
        del node


    def mkdir(self, path: str):
        base_dir = self.curr_dir
        path = osp.normpath(path)
        while path.startswith('..'):
            if base_dir.parent is None: raise Exception('Cannot go up: reached root')
            base_dir = base_dir.parent
            path = path.lstrip('./')
        # If path starts with '/', it's an absolute path; start searching from the root
        if path.startswith('/'):
            base_dir = self.categorization.root
        path = path.lstrip('/')
        # Now we have a clean path, and we know we have to start from base_dir
        for subpath in path.split('/'):
            child_dir = base_dir.get_child(subpath)
            if child_dir is None:
                child_dir = self.categorization.new_cat(subpath, parent=base_dir)
            base_dir = child_dir
        return base_dir


def load_session(filename: str):
    import serialization
    with open(filename, 'rb') as f:
        categorization = serialization.load(f)
        print(f'Loaded categorization from {filename}')
    session = Session(categorization)
    session._filename = filename
    return session


class Selection:
    """
    Wrapper around a TransactionArray slice
    """
    def __init__(self, categorization: np.ndarray, selection: np.ndarray, category: Node=None):
        self.categorization = categorization
        self.selection = selection
        self.category = category

    def __repr__(self):
        r = repr(self.ta)
        if self.category is not None:
            r += f' (category: {self.category.name})'
        return r

    @property
    def ta(self):
        return self.categorization.ta[self.selection]

    @property
    def amount(self):
        return self.ta.amount

    def __rshift__(self, other):
        if isinstance(other, Selection):
            if other.category is None:
                raise ValueError('Can only categorize into a specific category!')
            self.categorization.category[self.selection] = other.category.integer
            return
        raise TypeError(f'Operation on type {type(other)} is undefined')


    def __and__(self, other):
        if isinstance(other, Selection):
            other = other.selection
        return Selection(self.categorization, (self.selection & other))


    def __getitem__(self, where):
        new_selection = np.zeros_like(self.selection)
        indices = self.selection.nonzero()[0][where]
        new_selection[indices] = True
        return Selection(self.categorization, new_selection)


def yield_code_blocks(expression: str):
    """
    Takes a string, and splits it into code blocks vs. blocks that are in
    quotation marks (either ' or ").
    Yields a string, and boolean that is True if the block was in quotations
    and False if it was not in quotations
    """
    if not len(expression): return
    in_quote_mode = False
    i_buffer_start = 0
    quote_char = None
    prev_c = None
    for i, c in enumerate(expression):
        if not in_quote_mode and c in ['"', "'"] and prev_c != '\\':
            # If there is something in the buffer, yield it
            if i - i_buffer_start > 0:
                yield expression[i_buffer_start:i], False
            # Beginning of a string
            quote_char = c
            i_buffer_start = i
            in_quote_mode = True
        elif in_quote_mode and c == quote_char and prev_c != '\\':
            # End of a string
            yield expression[i_buffer_start:i+1], True
            # Next block starts at i+1
            i_buffer_start = i+1
            in_quote_mode = False
        prev_c = c
    if in_quote_mode:
        raise ValueError('Missing end of quotation')
    if i_buffer_start <= len(expression)-1:
        yield expression[i_buffer_start:], False


def format_code_block(block):
    block = re.sub(r'\$([\w\.\/]*)', r'selectcategory("\g<1>")', block)
    # posix-like commands
    block = re.sub(r'\bpwd\b', r'printworkdir()', block)
    # ls/tree/cd command: If there is a word behind it, use it as argument, but also
    # allow argumentless
    block = re.sub(r'\bcd\s+([\w\./]+)(\s|$|;)', r'changedir("\g<1>")\g<2>', block)
    block = re.sub(r'\bcd(\s|$)', r'changedir()', block)
    block = re.sub(r'\bls\s+([\w\./]+)(\s|$|;)', r'listdir("\g<1>")\g<2>', block)
    block = re.sub(r'\bls(\s|$)', r'listdir()', block)
    block = re.sub(r'\btree\s+([\w\./]+)(\s|$|;)', r'listtree("\g<1>")\g<2>', block)
    block = re.sub(r'\btree(\s|$)', r'listtree()', block)
    block = re.sub(r'\bsave\s+([\w\./]+)(\s|$|;)', r'savecategorization("\g<1>")\g<2>', block)
    block = re.sub(r'\bsave(\s|$)', r'savecategorization()', block)
    return block


def format_expression(expression):
    return ''.join([
        b if is_quoted else format_code_block(b) \
        for b, is_quoted in yield_code_blocks(expression)
        ])


def make_eval_scope(session: Session):
    """
    Instantiates all the functions and variables that are promised to be
    available in the console.
    """

    def selectcategory(name):
        extension = None
        if '.' in name and re.match(r'^[A-Za-z0-9_]+$', name.rsplit('.',1)[1]):
            # There is an attribute directly following the $, which will need to be eval'd
            path = name.split('/')
            if path[-1].startswith('.'):
                raise ValueError('Calling methods on "$.." or "$." need parentheses; e.g. "($..).amount"')
            path[-1], extension = path[-1].split('.', 1)
            name = '/'.join(path)
            # ($..).amount.sum()
        if name == 'A':
            select = np.ones_like(session.categorization.category, dtype=bool)
            selection = Selection(session.categorization, select)
        else:
            node = session.get_dir(name)
            select = session.categorization.category==node.integer
            selection = Selection(session.categorization, select, node)
        ret = selection
        if extension:
            for method in extension.split('.'):
                ret = getattr(ret, method)
        return ret

    def changedir(name='/'):
        session.cd(name)
        set_ps1(session.abspath(session.curr_dir))
        return

    def printworkdir():
        print(session.pwd)

    def listdir(name=None):
        directory = session.curr_dir if name is None else session.get_dir(name)
        print(session.abspath(directory))
        for child in directory.children:
            print('  '+child.name)
        ta = session.categorization.ta[session.categorization.select(directory)]
        print(ta)

    def listtree(name=None):
        directory = session.curr_dir if name is None else session.get_dir(name)
        count = lambda node: (session.categorization.category == node.integer).sum()
        for node, depth in iter_dfs_with_depth(directory):
            path = session.abspath(node) if depth == 0 else node.name
            if not path.endswith('/'): path += '/'
            print('  '*depth + path + f'  ({count(node)} transactions)')

    def savecategorization(filename=None):
        import serialization
        if filename is None:
            if session._filename is None:
                print('Cannot save: Pass a filename')
                return
            filename = session._filename
        if osp.isfile(filename):
            while True:
                answer = input(f'Overwrite {filename} [y/n]? ').lower()
                if answer == 'y':
                    break
                elif answer == 'n':
                    return
        with open(filename, 'w') as f:
            serialization.dump(session.categorization, f)
            print(f'Saved to {filename}')
            session._filename = filename

    amount = session.categorization.ta.amount
    description = session.categorization.ta.description
    return locals()


def evaluate(session, expression):
    locals().update(make_eval_scope(session))
    return eval(format_expression(expression))
