import readline # Enables up-arrow in the console!
import rlcompleter
import code
import numpy as np
import bookit as bk
import re

class Completer(rlcompleter.Completer):
    def __init__(self, scope, session):
        self.session = session
        # Add keywords for auto-completion
        scope = scope.copy()
        for kw in ['cd', 'pwd', 'ls', 'tree', 'mv', 'rm']:
            scope[kw] = 0
        super().__init__(scope)

    def complete(self, text, state):
        # Before running the actual completion, put all the children
        # of the current directory in the namespace
        # Maybe it's better to not complete any builtins?
        # See: https://github.com/python/cpython/blob/3.10/Lib/rlcompleter.py
        try:
            # Add extra options to the namespace
            _old_namespace = self.namespace.copy()
            self.namespace.update({n : 0 for n in self.namespace['all_node_names']})
            return super().complete(text, state)
        finally:
            self.namespace = _old_namespace


class Console(code.InteractiveConsole):

    def __init__(self, session: bk.Session):
        self.session = session
        bk.set_ps1(session.abspath(session.curr_dir))
        self.scope = bk.make_eval_scope(session)
        self.session._console_scope = self.scope
        self.session.update_console_vars()
        readline.set_completer(Completer(self.scope, session).complete) 
        readline.parse_and_bind("tab: complete")   
        np.set_printoptions(threshold=np.inf)
        super().__init__(locals=self.scope)

    def push(self, line):
        try:
            before = repr(line)
            line = bk.format_expression(line)
            after = repr(line)
            if after[1:-1].strip() != "":
                print(f'Formatted line {before} --> {after}')
                r = super().push(line)
                self.session.update_console_vars()
                return r
        except Exception as e:
            print(f'Error:\n{e}')
