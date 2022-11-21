import bookit as bk
import sys, os.path as osp, argparse, importlib
import numpy as np


class Parser:
    def __init__(self, *args, **kwargs):
        self.parser = argparse.ArgumentParser(*args, **kwargs)
        self.parser.add_argument(
            '-y', '--no-verification', action='store_true',
            help='Disables y/n prompts (mostly for file overwriting).'
            )
        self.parser.add_argument(
            '-v', '--verbose', action='store_true',
            help='Enables lower logger level output.'
            )

    def add_argument(self, *args, **kwargs):
        self.parser.add_argument(*args, **kwargs)

    def add_parser_argument(self):
        self.parser.add_argument(
            '-p', '--parser', type=str, default='parsers.py',
            help='Path to a (python) file that defines functions to parse a file to a TransactionArray object.'
            )

    def parse_args(self, *args, **kwargs):
        args = self.parser.parse_args(*args, **kwargs)
        if args.no_verification: bk.set_verification(False)
        if args.verbose: bk.debug()
        return args


def version():
    print(bk.__version__)


def console():
    parser = Parser()
    parser.add_parser_argument()
    parser.add_argument('filenames', type=str, nargs='*')
    parser.add_argument(
        '-t', '--test', action='store_true',
        help='Starts a session with some debug transactions'
        )
    parser.add_argument('--bankname', type=str, help='Only needed if using --parser')
    args = parser.parse_args()

    if args.test and len(args.categoryfile) > 0:
        raise Exception('Do not pass filenames if using the flag --test.')

    if args.bankname:
        read_fn = load_parser(args.parser)

    if args.test:
        bk.logger.info('Loading test session')
        ta = bk.TransactionArray([
            bk.Transaction(bk.Date(2021,1,1), 5., 'starbucks'),
            bk.Transaction(bk.Date(2021,1,1), 300., 'costco'),
            bk.Transaction(bk.Date(2021,1,2), 45.23, 'some food'),
            ])
        session = bk.Session(bk.Categorization(ta))
        somecat = session.mkdir('somecat')
        session.categorization.category = np.array([0, 1, 1])
    else:
        cat = None
        source_filename = None
        for filename in args.filenames:
            obj = read_fn(filename) if args.bankname else bk.serialization.load_from_path(filename)
            if isinstance(obj, bk.TransactionArray):
                if cat is None:
                    cat = bk.Categorization(obj)
                else:
                    cat.add_transactions(obj)
            elif isinstance(obj, bk.Categorization):
                if cat is None:
                    cat = obj
                    source_filename = filename
                else:
                    raise Exception('Loaded multiple categorizations; this is not supported.')
        if cat is None:
            # If no files were passed, just start an empty console
            cat = bk.Categorization(bk.TransactionArray([]))
        session = bk.Session(cat)
        session._filename = source_filename
    bk.console.Console(session).interact(f'Bookit {bk.__version__} console', 'Bye!')


def load_parser(filename, bankname):
    sys.path.append(osp.dirname(osp.abspath(filename)))
    parsers = importlib.import_module(filename.replace('.py',''))
    sys.path.pop()
    try:
        read_fn = getattr(parsers, bankname)
    except AttributeError:
        bk.logger.error(
            f'Could not find a function called `{bankname}` in {filename}.'
            f'\nMake sure there is a function `def {bankname}(filename: str) -> bk.TransactionArray` in {filename}.'
            )
        raise
    return read_fn


def parse_transactions():
    parser = Parser()
    parser.add_argument('bankname', type=str, help='Should match the parse function name in your parsers.py file.')
    parser.add_argument('filenames', type=str, nargs='+', help='Input data files to parse')
    parser.add_argument(
        '-o', '--output', type=str,
        help='Output file to write to. If not specified, parsed transactions are only printed.'
        )
    parser.add_parser_argument()
    args = parser.parse_args()
    
    read_fn = load_parser(args.parser, args.bankname)
    tas = [ read_fn(f) for f in args.filenames ]
    ta = bk.merge(tas)

    if args.output:
        if osp.isfile(args.output) and not bk.verify(f'Overwrite {args.output} [y/n]? '):
            return
        with open(args.output, 'w') as f:
            bk.serialization.dump(ta, f)

    else:
        print('Parsed transactions:')
        print(ta)


def add_transactions():
    parser = Parser()
    parser.add_parser_argument()
    parser.add_argument('cat', type=str)
    parser.add_argument('bankname', type=str)
    parser.add_argument('tas', type=str, nargs='+')
    parser.add_argument('-o', '--output', type=str)
    args = parser.parse_args()

    with open(args.cat, 'rb') as f:
        cat = bk.serialization.load(f)


    read_fn = load_parser(args.parser, args.bankname)
    ta_in = bk.merge([ read_fn(f) for f in args.tas ])

    if isinstance(cat, bk.Categorization):
        cat.add_transactions(ta_in)
        out = cat
    elif isinstance(cat, bk.TransactionArray):
        out = bk.merge((cat, ta_in))
    
    output = args.output if args.output else args.cat

    if osp.isfile(output) and not bk.verify(f'Overwrite {output} [y/n]? '):
        return
                
    with open(output, 'w') as f:
        bk.serialization.dump(out, f)
        print(f'Wrote {out} to {output}')
