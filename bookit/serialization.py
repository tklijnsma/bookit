import json
import numpy as np
import dataclasses
import bookit as bk

DATEFMT = '%Y%m%d'

# tree (de)serialization
def serialize(node, ans=[]):
    ans.append({node.name: node.integer})
    if node.children:
        ans.append([])
        for c in node.children:
            serialize(c, ans[-1])
    return ans

def deserialize(serialized_tree, parent=None):
    node = None
    for t in serialized_tree:
        if isinstance(t, dict):
            node = bk.Node(*list(t.items())[0], parent=parent)
        elif isinstance(t, list):
            deserialize(t, parent=node)
    return node


class Encoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, bk.Transaction):
            return dict(
                type = 'Transaction',
                date = obj.date.strftime(DATEFMT),
                amount = obj.amount,
                description = obj.description,
                account = obj.account,
                currency = getattr(obj.currency, 'name', None),
                )

        elif dataclasses.is_dataclass(obj):
            d = dataclasses.asdict(obj)
            if isinstance(obj, bk.Currency):
                d['type'] = 'Currency'
            return d

        elif isinstance(obj, bk.TransactionArray):
            return dict(type='TransactionArray', val=[ self.default(t) for t in obj ])

        elif isinstance(obj, bk.Node):
            return dict(type='Node', val=serialize(obj))

        elif isinstance(obj, bk.Categorization):
            return dict(
                type = 'Categorization',
                ta = self.default(obj.ta),
                category = obj.category.tolist(),
                root = self.default(obj.root)
                )

        return super().default(obj)


class Decoder(json.JSONDecoder):
    def __init__(self, *args, **kwargs):
        self.currencies = bk.currencies.copy()
        self.currencies.update({ c.name : c for c in kwargs.pop('currencies', []) })
        json.JSONDecoder.__init__(self, object_hook=self.object_hook, *args, **kwargs)


    def object_hook(self, d):
        type = d.get('type', None)
        if not type:
            return d
        elif type == 'Currency':
            currency_name = d['name']
            if currency_name in self.currencies:
                self.currencies[currency_name].rate = d['rate']
            else:
                self.currencies[currency_name] = bk.Currency(currency_name, d['rate'])
        elif type == 'Transaction':
            currency = d['currency']
            if currency:
                if currency not in self.currencies:
                    currency = self.currencies.setdefault(currency, bk.Currency(currency, 0.))
                else:
                    currency = self.currencies[currency]
            return bk.Transaction(
                bk.Date.strptime(d['date'], DATEFMT),
                d['amount'],
                d['description'],
                d['account'],
                currency,
                )
        elif type == 'TransactionArray':
            return bk.TransactionArray(d['val'])
        elif type == 'Node':
            return deserialize(d['val'])
        elif type == 'Categorization':
            return bk.Categorization(d['ta'], np.array(d['category']), d['root'])
        return d

def dumps(*args, **kwargs):
    kwargs['cls'] = Encoder
    return json.dumps(*args, **kwargs)

def dump(*args, **kwargs):
    kwargs['cls'] = Encoder
    return json.dump(*args, **kwargs)

def loads(*args, **kwargs):
    kwargs['cls'] = Decoder
    return json.loads(*args, **kwargs)

def load(*args, **kwargs):
    kwargs['cls'] = Decoder
    return json.load(*args, **kwargs)

def load_from_path(path, *args, **kwargs):
    with open(path, 'rb') as f:
        return load(f, *args, **kwargs)