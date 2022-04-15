import readline # Enables up-arrow in the console!
import rlcompleter
import code
import dev as tf
import numpy as np

class Completer(rlcompleter.Completer):
    def __init__(self, scope, session):
        self.session = session
        # Add keywords for auto-completion
        scope = scope.copy()
        for kw in ['cd', 'pwd', 'ls', 'tree']:
            scope[kw] = 0
        super().__init__(scope)

    def complete(self, text, state):
        # Before running the actual completion, put all the children
        # of the current directory in the namespace
        try:
            _old_namespace = self.namespace.copy()
            self.namespace.update({c.name : 0 for c in self.session.curr_dir.children})
            # TODO:
            # If text.endswith('/'), probably should do some directory expanding
            # Maybe it's better to not complete any builtins?
            # Also: Implement a printtree function
            # See: https://github.com/python/cpython/blob/3.10/Lib/rlcompleter.py
            return super().complete(text, state)
        finally:
            self.namespace = _old_namespace


class Console(code.InteractiveConsole):

    def __init__(self, session):
        self.session = session
        tf.set_ps1(session.abspath(session.curr_dir))
        self.scope = tf.make_eval_scope(session)
        readline.set_completer(Completer(self.scope, session).complete) 
        readline.parse_and_bind("tab: complete")   
        super().__init__(locals=self.scope)

    def push(self, line):
        before = repr(line)
        line = tf.format_expression(line)
        after = repr(line)
        print(f'Formatted line {before} --> {after}')
        return super().push(line)

# Console(locals=dict(answer=42)).interact('Welcome!', 'Bye!')

def test_console():
    import sys
    if len(sys.argv) > 1:
        session = tf.load_session(sys.argv[1])
    else:
        ta = tf.TransactionArray([
            tf.Transaction(tf.Date(2021,1,1), 5., 'starbucks'),
            tf.Transaction(tf.Date(2021,1,1), 300., 'costco'),
            tf.Transaction(tf.Date(2021,1,2), 45.23, 'some food'),
            ])
        session = tf.Session(tf.Categorization(ta))
        somecat = session.mkdir('somecat')
        session.categorization.category = np.array([0, 1, 1])
    Console(session).interact('Welcome!', 'Bye!')

if __name__ == '__main__':
    test_console()