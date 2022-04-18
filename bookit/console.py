import readline # Enables up-arrow in the console!
import rlcompleter
import code
import numpy as np
import bookit as bk

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

    def __init__(self, session: bk.Session):
        self.session = session
        bk.set_ps1(session.abspath(session.curr_dir))
        self.scope = bk.make_eval_scope(session)
        self.session._console_scope = self.scope
        readline.set_completer(Completer(self.scope, session).complete) 
        readline.parse_and_bind("tab: complete")   
        super().__init__(locals=self.scope)

    def push(self, line):
        before = repr(line)
        line = bk.format_expression(line)
        after = repr(line)
        if after[1:-1].strip() != "":
            print(f'Formatted line {before} --> {after}')
            r = super().push(line)
            self.session._reset_convenience_variables()
            return r
