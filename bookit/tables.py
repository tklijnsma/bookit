from __future__ import annotations
import bookit as bk
import numpy as np

class Table:
    def __init__(self, ta_table: List[List[bk.Transaction]], row_labels=None, col_labels=None):
        self.ta_table = ta_table
        self.row_labels = row_labels
        self.col_labels = col_labels
        self.add_avg_col = True
        self.add_net_row = True


    def to_list_of_list_of_strs(self):
        n_rows = len(self.ta_table)
        n_cols = len(self.ta_table[0])
        assert len(self.row_labels) == n_rows
        assert len(self.col_labels) == n_cols

        amount_table = np.array(
            [[ self.ta_table[i][j].sum() for j in range(n_cols) ] for i in range(n_rows)]
            )
        avg_col = np.mean(amount_table, axis=1)
        net_row = np.sum(amount_table, axis=0)

        def amount_str(number):
            return bk.amount_str(number, 0)

        table = []
        if self.col_labels:
            if self.row_labels:
                table.append([''] + self.col_labels)
            else:
                table.append(self.col_labels)
            if self.add_avg_col: table[-1].append('avg')
        for i in range(len(self.ta_table)):
            row = []
            if self.row_labels: row.append(self.row_labels[i])
            row.extend([amount_str(ta.sum()) for ta in amount_table[i]])
            if self.add_avg_col: row.append(amount_str(avg_col[i]))
            table.append(row)

        if self.add_net_row:
            row = []
            if self.row_labels: row.append('net')
            row.extend([amount_str(v) for v in net_row])
            if self.add_avg_col: row.append(amount_str(np.sum(avg_col)))
            table.append(row)

        return table

    def str(self):
        return tabelize(self.to_list_of_list_of_strs(), [-1], [-1])


def tabelize(data, horizontal_divisions=None, vertical_divisions=None):
    '''
    Formats a list of lists to a single string (no seps).
    Rows need not be of same length.
    '''
    # Ensure data is strings
    data = [ [ str(i) for i in row ] for row in data ]
    # Determine the row with the most columns
    n_rows = len(data)
    n_columns = max(map(len, data))
    # Process divisions
    if horizontal_divisions is None: horizontal_divisions = []
    if vertical_divisions is None: vertical_divisions = []
    horizontal_divisions = [ (n_rows+v) if v<0 else v for v in horizontal_divisions ]
    vertical_divisions = [ (n_columns+v) if v<0 else v for v in vertical_divisions ]
    horizontal_divisions.sort()
    vertical_divisions.sort()
    # Determine how wide each column should be (max)
    col_widths = [0 for i in range(n_columns)]
    for row in data:
        for i_col, item in enumerate(row):
            l = len(item)
            if item.startswith('\x1b'): l -= 9 # Correct for coloring
            if l > col_widths[i_col]: col_widths[i_col] = l
    # Create a horizontal divider
    h_divider = ['-' for _ in range(sum(col_widths) + n_columns-1)]
    for i in reversed(sorted(vertical_divisions)):
        p = sum(col_widths[:i])+(i)
        h_divider.insert(p, '|-')
    h_divider = ''.join(h_divider) + '\n'
    # Format
    s = ''
    for i_row, row in enumerate(data):
        if i_row in horizontal_divisions: s += h_divider
        for i_col, item in enumerate(row):
            if vertical_divisions and i_col in vertical_divisions:
                s += '| '
            w = '>' + str(col_widths[i_col] + (9 if item.startswith('\x1b') else 0))
            s += format(item, w)
            if i_col < n_columns-1: s += ' '
        if i_row < n_rows-1: s += '\n'
    return s
