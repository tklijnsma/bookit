import bookit as bk
import argparse
import os.path as osp
import importlib.machinery


def extract_read_function(code_file):
    """
    Extracts a `read(filename: str) -> TransactionArray` function from a source file.
    See https://stackoverflow.com/questions/19009932/import-arbitrary-python-source-file-python-3-3
    """
    loader = importlib.machinery.SourceFileLoader('module_with_read', code_file)
    mod = loader.load_module()
    try:
        return mod.read
    except AttributeError:
        raise Exception(
            'Please define a function `def read(filename: str) -> TransactionArray:`'
            ' in your parser module.'
            )

def parse_transactions():
    parser = argparse.ArgumentParser()
    parser.add_argument('filename', type=str)
    parser.add_argument('output', type=str)
    parser.add_argument('-p', '--parser', type=str, required=True)
    args = parser.parse_args()

    read = extract_read_function(args.parser)
    ta = read(args.filename)

    if osp.isfile(args.output):
        while True:
            answer = input(f'Overwrite {args.output} [y/n]? ').lower()
            if answer == 'y':
                break
            elif answer == 'n':
                return
    
    with open(args.output, 'w') as f:
        bk.serialization.dump(ta, f)


def add_transactions():
    parser = argparse.ArgumentParser()
    parser.add_argument('ta', type=str)
    parser.add_argument('cat', type=str)
    parser.add_argument('-o', '--output', type=str)
    parser.add_argument('-p', '--parser', type=str)
    args = parser.parse_args()

    if args.parser:
        # Run args.ta through the parser first
        read = extract_read_function(args.parser)
        ta_in = read(args.ta)
    else:
        with open(args.ta, 'rb') as f:
            ta_in = bk.serialization.load(f)

    with open(args.cat, 'rb') as f:
        cat = bk.serialization.load(f)

    if isinstance(cat, bk.Categorization):
        out = cat.add_transactions(ta_in)
    elif isinstance(cat, bk.TransactionArray):
        out = bk.merge((cat, ta_in))
    
    output = args.output if args.output else args.cat

    if osp.isfile(output):
        while True:
            answer = input(f'Overwrite {output} [y/n]? ').lower()
            if answer == 'y':
                break
            elif answer == 'n':
                return
                
    with open(output, 'w') as f:
        bk.serialization.dump(out, f)
        print(f'Wrote {out} to {output}')
