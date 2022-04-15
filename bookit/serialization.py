import json
import numpy as np
import dev as tf
import dataclasses

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
            node = tf.Node(*list(t.items())[0], parent=parent)
        elif isinstance(t, list):
            deserialize(t, parent=node)
    return node


class Encoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, tf.Transaction):
            return dict(
                type = 'Transaction',
                date = obj.date.strftime(DATEFMT),
                amount = obj.amount,
                description = obj.description,
                account = getattr(obj.account, 'name', None),
                currency = getattr(obj.currency, 'name', None),
                )

        elif dataclasses.is_dataclass(obj):
            d = dataclasses.asdict(obj)
            if isinstance(obj, tf.Currency):
                d['type'] = 'Currency'
            elif isinstance(obj, tf.Account):
                d['type'] = 'Account'
            return d

        elif isinstance(obj, tf.TransactionArray):
            return dict(type='TransactionArray', val=[ self.default(t) for t in obj ])

        elif isinstance(obj, tf.Node):
            return dict(type='Node', val=serialize(obj))

        elif isinstance(obj, tf.Categorization):
            return dict(
                type = 'Categorization',
                ta = self.default(obj.ta),
                category = obj.category.tolist(),
                root = self.default(obj.root)
                )

        return super().default(obj)


class Decoder(json.JSONDecoder):
    def __init__(self, *args, **kwargs):
        self.currencies = { c.name : c for c in kwargs.pop('currencies', []) }
        self.accounts = kwargs.pop('accounts', {})
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
                self.currencies[currency_name] = tf.Currency(currency_name, d['rate'])
        elif type == 'Account':
            if d['name'] not in self.accounts:
                self.currencies[d['name']] = tf.Account(d['name'])
        elif type == 'Transaction':
            account = d['account']
            if account:
                if account not in self.accounts:
                    account = self.accounts.setdefault(account, tf.Account(account))
                else:
                    account = self.accounts[account]
            currency = d['currency']
            if currency:
                if currency not in self.currencies:
                    currency = self.currencies.setdefault(currency, tf.Currency(currency, 0.))
                else:
                    currency = self.currencies[currency]
            return tf.Transaction(
                tf.Date.strptime(d['date'], DATEFMT),
                d['amount'],
                d['description'],
                account,
                currency,
                )
        elif type == 'TransactionArray':
            return tf.TransactionArray(d['val'])
        elif type == 'Node':
            return deserialize(d['val'])
        elif type == 'Categorization':
            return tf.Categorization(d['ta'], np.array(d['category']), d['root'])
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
