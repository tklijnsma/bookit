from __future__ import annotations
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


def version():
    version_file = osp.join(osp.dirname(osp.abspath(__file__)), 'include/VERSION')
    with open(version_file) as f:
        return f.read().strip()
__version__ = version()

def debug(flag=True):
    """Sets the logger level to debug (for True) or warning (for False)"""
    logger.setLevel(logging.DEBUG if flag else DEFAULT_LOGGING_LEVEL)


def set_ps1(path=''):
    sys.ps1 = f"${path}: "


DO_VERIFY = True
def set_verification(flag: bool=True):
    global DO_VERIFY
    DO_VERIFY = flag
    logger.info(f'Set verification to {DO_VERIFY}')

def verify(msg):
    if DO_VERIFY:
        while True:
            answer = input(msg).lower()
            if answer == 'y':
                return True
            elif answer == 'n':
                return False
    return True

DO_COLOR = True
def set_coloring(flag: bool=True):
    global DO_COLOR
    DO_COLOR = flag
    logger.info(f'Set coloring to {DO_COLOR}')

colors = {
    'gray' : '\033[90m',
    'red' : '\033[31m',
    'green' : '\033[32m',
    }
def colored(text, color):
    if not DO_COLOR: return text
    return colors[color] + text + '\033[0m'


@dataclass
class Currency:
    name: str
    rate: float


class CurrenciesContainer(dict):
    """
    Like a dict, but with access via attributes
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.__dict__ = self

currencies = CurrenciesContainer(USD=Currency('USD', 1.), EUR=Currency('EUR', 1.0819))


def amount_str(amount: float, width=8):
    if amount == 0.:
        return colored(f'{amount:{width}.2f}', 'gray')
    else:
        return colored(f'{amount:+{width}.2f}', 'red' if amount < 0. else 'green')


@dataclass
class Transaction:
    date: Date
    amount: float
    description: str
    account: str = None
    currency: Currency = currencies.USD

    @property
    def camount(self):
        return self.amount * self.currency.rate

    @property
    def day(self):
        return self.date.day

    @property
    def month(self):
        return self.date.strftime('%b').lower()

    @property
    def imonth(self):
        return self.date.month

    @property
    def year(self):
        return self.date.year

    def __repr__(self) -> str:        
        r = (
            f"{self.date.strftime('%Y-%m-%d')} {self.currency.name}"
            f" {amount_str(self.amount)} {self.description}"
            )
        if self.account:
            r += ' ' + colored('(' + self.account + ')', 'gray')
        return r

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
    def month(self):
        mat = np.empty(self.shape, dtype='<U10')
        mat.flat = [ t.month for t in self.flat ]
        return mat

    @property
    def day(self):
        mat = np.empty(self.shape, dtype=int)
        mat.flat = [ t.day for t in self.flat ]
        return mat

    @property
    def imonth(self):
        mat = np.empty(self.shape, dtype=int)
        mat.flat = [ t.imonth for t in self.flat ]
        return mat

    @property
    def year(self):
        mat = np.empty(self.shape, dtype=int)
        mat.flat = [ t.year for t in self.flat ]
        return mat

    @property
    def monthyear(self):
        mat = np.empty(self.shape, dtype='<U7')
        mat.flat = [ t.month+str(t.year) for t in self.flat ]
        return mat

    @property
    def currency_name(self):
        mat = np.empty(self.shape, dtype='<U10')
        mat.flat = [ t.currency.name for t in self.flat ]
        return mat

    @property
    def account(self):
        mat = np.empty(self.shape, dtype='<U50')
        # mat.flat = [ t.account if t.account else '' for t in self.flat ]
        mat.flat = [ t.account for t in self.flat ]
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

    def search_description(self, *substrings, case_sensitive=False, mode='and'):
        """
        Looks for substrings
        """
        assert mode in ['and', 'or']
        descriptions = self.description
        if not case_sensitive: descriptions = np.char.lower(descriptions)
        mask = np.ones_like(self, dtype=bool) if mode == 'and' else np.zeros_like(self, dtype=bool)
        for substring in substrings:
            if not case_sensitive: substring = substring.lower()
            mask_this_substring = np.char.find(descriptions, substring) >= 0
            if mode == 'and':
                mask &= mask_this_substring
            elif mode == 'or':
                mask |= mask_this_substring
        return mask


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
    Merge transaction arrays; filters out duplicates BETWEEN arrays, but not INSIDE arrays
    """
    base_ta = tas[0]
    for i, ta in enumerate(tas[1:]):
        duplicates = np.isin(ta, base_ta)
        if logger.level >= logging.INFO and duplicates.sum()>0:
            logger.info(
                f'Found {duplicates.sum()} duplicate transactions when adding ta {i+1}:\n{ta[duplicates]}'
                )
        base_ta = np.concatenate((base_ta, ta[~duplicates]))
    return base_ta.view(TransactionArray)


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
        if other.parent: other.parent.children.remove(other)
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
    def __init__(self, ta: TransactionArray, category: np.ndarray = None, root: Node = None, sort: bool=True):
        self.ta = ta
        self.category = np.zeros(self.ta.shape[0], dtype=np.int16) if category is None else category
        if sort: self.resort()
        if root is None:
            self.root = Node('root', 0)
            self._i_max_category = 0
        else:
            self.root = root
            self._i_max_category = max(n.integer for n in iter_dfs(root))

    def resort(self):
        order = np.argsort(self.ta)
        self.ta = self.ta[order]
        self.category = self.category[order]

    def new_cat(self, name, parent=None):
        if parent is None: parent = self.root
        self._i_max_category += 1
        cat = Node(name, self._i_max_category)
        cat.set_parent(parent)
        logger.info(
            f'Created new category "{cat.name}" with integer'
            f' {cat.integer} and parent "{parent.name}"'
            )
        return cat

    def select(self, node: Node):
        return Selection(self, self.category == node.integer, node)

    def select_multiple(self, nodes: List[Node]):
        return Selection(self, np.isin(self.category, [n.integer for n in nodes]))

    def select_recursively(self, node: Node):
        return Selection(self, np.isin(self.category, [n.integer for n in iter_dfs(node)]))

    def add_transactions(self, ta):
        n_old = len(self.ta)
        self.ta = merge((self.ta, ta))
        n_added = len(self.ta) - n_old
        self.category = np.concatenate((self.category, np.zeros(n_added, dtype=int)))
        self.resort()

    def delete_node(self, node: Node):
        # Put any transaction in node or its children in root
        self.select_recursively(node) >> self.root
        if node.parent:
            node.parent.children.remove(node)
        for n in list(iter_dfs(node)):
            del n


def iter_monthyears(min_monthyear, max_monthyear):
    month = min_monthyear.imonth
    year = min_monthyear.year
    while not(year==max_monthyear.year and month==max_monthyear.imonth):
        yield Monthyear(year, month)
        month += 1
        if month == 13:
            year += 1
            month = 1

month_str = {
    1:'jan', 2:'feb', 3:'mar', 4:'apr', 5:'may', 6:'jun',
    7:'jul', 8:'aug', 9:'sep', 10:'oct', 11:'nov', 12:'dec'
    }
month_integer = {v:k for k, v in month_str.items()}

@dataclass
class Monthyear:
    year: int
    imonth: int

    @classmethod
    def from_date(cls, date: Date):
        return cls(date.year, date.month)

    @classmethod
    def from_str(cls, monthyear: str):
        year = int(monthyear[3:])
        if year < 100: year += 2000
        imonth = month_integer[monthyear[:3].lower()]
        return cls(year, imonth)

    @property
    def month(self):
        return month_str[self.imonth]

    @property
    def monthyear(self):
        return month_str[self.imonth] + str(self.year)

    @property
    def date(self):
        return Date(self.year, self.imonth, 1)


def month_table(categorization, root=None, min_monthyear=None, max_monthyear=None, exclude=None):
    if root is None: root = categorization.root
    if min_monthyear is None or max_monthyear is None:
        dates = categorization.select_recursively(root).ta.date.astype(object)
        if min_monthyear is None: min_monthyear = Monthyear.from_date(min(dates))
        if max_monthyear is None: max_monthyear = Monthyear.from_date(max(dates))

    monthyears = list(iter_monthyears(min_monthyear, max_monthyear))

    row_labels = []
    col_labels = [ m.monthyear for m in monthyears]
    table = []
    # Potentially add root itself if it has transactions
    ta = categorization.select(root).ta
    if len(ta):
        row_labels.append('.')
        row = []
        for m in monthyears:
            row.append(ta[(ta.year==m.year) & (ta.imonth==m.imonth)])
        table.append(row)
    for child in root.children:
        if exclude and child.name in exclude: continue
        row_labels.append(child.name)
        ta = categorization.select_recursively(child).ta
        row = []
        for m in monthyears:
            row.append(ta[(ta.year==m.year) & (ta.imonth==m.imonth)])
        table.append(row)
    return tables.Table(table, row_labels, col_labels)


class Session:
    """
    Provides the linux-like interface to a Categorization instance
    """
    def __init__(self, categorization: Categorization):
        self.categorization = categorization
        self.curr_dir = categorization.root
        self._filename = None


    @property
    def gta(self):
        return self.categorization.ta


    def get_dir(self, path) -> Node:
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
        self.update_console_vars()


    @property
    def current_ta(self):
        return self.categorization.ta[self.categorization.category == self.curr_dir.integer]


    @property
    def currdir_selection(self):
        return self.categorization.select(self.curr_dir)


    def exists(self, path):
        try:
            self.get_dir(path)
            return True
        except NoSuchPath:
            return False


    def isdir(self, path):
        return self.exists(path)


    def rm(self, path: Union[str, Node]):
        node = path if isinstance(path, Node) else self.get_dir(path)
        self.categorization.delete_node(node)


    def mv(self, src: str, dst: str):
        """
        Renames a path (potentially giving it a new parent).
        Raises an exception if dst exists.
        """
        if self.exists(dst):
            raise Exception(f'Cannot mv {src} -> {dst}; {dst} exists')
        src_node = self.get_dir(src)
        dst_node = self.mkdir(dst)
        # Move transactions
        self.categorization.select(src_node) >> dst_node
        # Move children
        for c in src_node.children.copy():
            dst_node.add_child(c)
        assert len(src_node.children) == 0
        self.categorization.delete_node(src_node)


    def mkdir(self, path: str) -> Node:
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


    def update_console_vars(self):
        if hasattr(self, '_console_scope'):
            ta = self.current_ta
            self._console_scope['ta'] = ta
            self._console_scope['date'] = ta.date
            self._console_scope['day'] = ta.day
            self._console_scope['month'] = ta.month
            self._console_scope['year'] = ta.year
            self._console_scope['monthyear'] = ta.monthyear
            self._console_scope['amount'] = ta.amount
            self._console_scope['account'] = ta.account
            self._console_scope['description'] = ta.description


def load_session(filename: str):
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
        r = repr(self.ta).replace('TransactionArray', 'Selection')
        if self.category is not None:
            r += f' (category: {self.category.name})'
        return r

    def __invert__(self):
        return Selection(self.categorization, (~self.selection))

    @property
    def ta(self):
        return self.categorization.ta[self.selection]

    def __len__(self):
        return len(self.ta)

    @property
    def amount(self):
        return self.ta.amount

    def __rshift__(self, other: Union[int, Selection, Node]):
        if isinstance(other, Selection):
            if other.category is None:
                raise ValueError('Can only categorize into a specific category!')
            integer = other.category.integer
        elif isinstance(other, Node):
            integer = other.integer
        elif isinstance(other, int):
            integer = other
        else:
            raise TypeError(f'Operation on type {type(other)} is undefined')
        self.categorization.category[self.selection] = integer


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
    block = re.sub(r'\$\$([\w\.\/]*)', r'selectcategoryrecursively("\g<1>")', block)
    block = re.sub(r'\$([\w\.\/]*)', r'selectcategory("\g<1>")', block)
    # posix-like commands
    block = re.sub(r'\bpwd\b', r'printworkdir()', block)
    # ls/tree/cd command: If there is a word behind it, use it as argument, but also
    # allow argumentless
    block = re.sub(r'\brm\s+([\w\./]+)(\s|$|;)', r'removecategory("\g<1>")\g<2>', block)
    block = re.sub(r'\brm(\s|$)', r'removecategory()', block)
    block = re.sub(r'\bcd\s+([\w\./]+)(\s|$|;)', r'changedir("\g<1>")\g<2>', block)
    block = re.sub(r'\bcd(\s|$)', r'changedir()', block)
    block = re.sub(r'\bls\s+([\w\./]+)(\s|$|;)', r'listdir("\g<1>")\g<2>', block)
    block = re.sub(r'\bls(\s|$)', r'listdir()', block)
    block = re.sub(r'\btree\s+([\w\./]+)(\s|$|;)', r'listtree("\g<1>")\g<2>', block)
    block = re.sub(r'\btree(\s|$)', r'listtree()', block)
    block = re.sub(r'\bsave\s+([\w\./]+)(\s|$|;)', r'savecategorization("\g<1>")\g<2>', block)
    block = re.sub(r'\bsave(\s|$)', r'savecategorization()', block)
    block = re.sub(r'\bmkdir\s+([\w\./]+)(\s|$|;)', r'makecategory("\g<1>")\g<2>', block)
    block = re.sub(r'\bmkdir(\s|$)', r'makecategory()', block)
    # Two argument functions
    block = re.sub(r'\bmv\s+([\w\./]+)\s+([\w\./]+)(\s|$|;)', r'renamecategory("\g<1>","\g<2>")', block)
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
    # Insert all directory names in the scope for autocompletion
    all_node_names = [n.name for n in iter_dfs(session.categorization.root)]

    def makecategory(name):
        node = session.mkdir(name)
        all_node_names.append(node.name)

    def renamecategory(src, dst):
        session.mv(src, dst)

    def removecategory(name):
        if verify(f'Remove {name}? [y/n] '):
            session.rm(name)

    def selectcategory(name, recursive=False):
        extension = None
        if '.' in name and re.match(r'^[A-Za-z0-9_]+$', name.rsplit('.',1)[1]):
            # There is an attribute directly following the $, which will need to be eval'd
            path = name.split('/')
            if path[-1].startswith('.'):
                raise ValueError('Calling methods on "$.." or "$." need parentheses; e.g. "($..).amount"')
            path[-1], extension = path[-1].split('.', 1)
            name = '/'.join(path)
            # ($..).amount.sum()
        if name == 'a':
            selection = Selection(session.categorization, np.ones_like(session.categorization.category, dtype=bool))
        elif name == 'h':
            if recursive:
                selection = session.categorization.select_recursively(session.curr_dir)
            else:
                selection = session.categorization.select(session.curr_dir)
        elif name == 'r':
            selection = session.categorization.select_recursively(session.curr_dir)
        else:
            node = session.get_dir(name)
            if recursive:
                selection = session.categorization.select_recursively(node)
            else:
                selection = session.categorization.select(node)
        ret = selection
        if extension:
            for method in extension.split('.'):
                ret = getattr(ret, method)
        return ret

    def selectcategoryrecursively(name):
        return selectcategory(name, True)

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
        print(session.categorization.select(directory).ta)

    def listtree(name=None):
        directory = session.curr_dir if name is None else session.get_dir(name)
        count = lambda node: (session.categorization.category == node.integer).sum()
        for node, depth in iter_dfs_with_depth(directory):
            path = session.abspath(node) if depth == 0 else node.name
            if not path.endswith('/'): path += '/'
            print('  '*depth + path + f'  ({count(node)} transactions)')

    def savecategorization(filename=None):
        if filename is None:
            if session._filename is None:
                print('Cannot save: Pass a filename')
                return
            filename = session._filename
        if osp.isfile(filename) and not verify(f'Overwrite {filename} [y/n]? '):
            return
        with open(filename, 'w') as f:
            serialization.dump(session.categorization, f)
            print(f'Saved to {filename}')
            session._filename = filename

    def search(*substrings, **kw):
        return session.currdir_selection[session.current_ta.search_description(*substrings, **kw)]

    def gsearch(*substrings, **kw):
        return Selection(
            session.categorization,
            session.gta.search_description(*substrings, **kw)
            )

    def table(name=None, begin=None, end=None, exclude=None):
        directory = session.curr_dir if name is None else session.get_dir(name)
        if begin is not None: begin = Monthyear.from_str(begin)
        if end is not None: end = Monthyear.from_str(end)
        print(
            month_table(
                session.categorization, directory,
                min_monthyear=begin, max_monthyear=end, exclude=exclude
                ).str()
            )

    # Convenience variables
    s = search
    gta = session.gta
    gdate = gta.date
    gamount = gta.amount
    gdescription = gta.description
    gday = gta.day
    gmonth = gta.month
    gyear = gta.year
    gmonthyear = gta.monthyear
    gaccount = gta.account

    bk = sys.modules[__name__]
    return locals()


def evaluate(session, expression):
    locals().update(make_eval_scope(session))
    return eval(format_expression(expression))


# __________________________________________________
from . import serialization
from . import console
from . import cli
from . import tables
