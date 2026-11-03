"""Desktop explorer for the frozen classifiers and their recorded evidence.

Only presentation changes in this release: the saved model and test results
remain unchanged. No prediction triggers training or test-set evaluation.
"""
import json
import tkinter as tk
from tkinter import messagebox, ttk

from car_ai import ROOT, FEATURES, LABELS
from predict import compare
from presentation import (FIELD_NAMES, LABEL_NAMES, TOKENS, display_value,
                          display_conditions, input_summary)

BG = '#f1f4f8'
WHITE = '#ffffff'
INK = '#142438'
MUTED = '#596b80'
NAVY = '#11283f'
TEAL = '#087e80'
BORDER = '#dce4ed'
COLOURS = {'unacc': '#a63740', 'acc': '#946014', 'good': '#16716a', 'vgood': '#245db1'}
FONT = 'Segoe UI'


def label(parent, text='', size=11, colour=INK, bold=False, bg=WHITE, **kwargs):
    """Consistent typography; wrap width follows the widget's available width."""
    widget = tk.Label(parent, text=text, bg=bg, fg=colour, anchor='w', justify='left',
                      font=(FONT, size, 'bold' if bold else 'normal'), wraplength=420, **kwargs)
    widget.bind('<Configure>', lambda event: widget.configure(wraplength=max(100, event.width - 4)))
    return widget


def panel(parent, **kwargs):
    return tk.Frame(parent, bg=WHITE, highlightbackground=BORDER,
                    highlightthickness=1, padx=20, pady=18, **kwargs)


class ScrollPage(ttk.Frame):
    """A vertically scrollable tab, including keyboard and wheel navigation."""
    def __init__(self, parent):
        super().__init__(parent)
        self.canvas = tk.Canvas(self, bg=BG, highlightthickness=0)
        scrollbar = ttk.Scrollbar(self, orient='vertical', command=self.canvas.yview)
        self.canvas.configure(yscrollcommand=scrollbar.set)
        scrollbar.pack(side='right', fill='y')
        self.canvas.pack(side='left', fill='both', expand=True)
        self.body = tk.Frame(self.canvas, bg=BG, padx=22, pady=20)
        self.item = self.canvas.create_window((0, 0), window=self.body, anchor='nw')
        self.canvas.bind('<Configure>', self.resize)
        self.body.bind('<Configure>', lambda _e: self.canvas.configure(scrollregion=self.canvas.bbox('all')))

    def resize(self, event):
        self.canvas.itemconfigure(self.item, width=event.width)


class DecisionCard:
    def __init__(self, parent, title, subtitle):
        self.frame = panel(parent)
        label(self.frame, title.upper(), size=10, colour=MUTED, bold=True).pack(fill='x')
        label(self.frame, subtitle, size=10, colour=MUTED).pack(fill='x', pady=(2, 10))
        self.result = label(self.frame, size=24, bold=True)
        self.result.pack(fill='x', pady=(0, 12))
        self.rule = label(self.frame, size=11, bold=True)
        self.rule.pack(fill='x')
        self.conditions = label(self.frame, size=11)
        self.conditions.pack(fill='x', pady=(8, 10))
        self.note = label(self.frame, size=10, colour=MUTED)
        self.note.pack(fill='x')

    def set(self, result, rule_text, conditions, note):
        self.result.configure(text=LABEL_NAMES[result], fg=COLOURS[result])
        self.rule.configure(text=rule_text)
        self.conditions.configure(text=conditions)
        self.note.configure(text=note)


class CarApp:
    """UI controller; classifier and display-formatting logic live separately."""
    def __init__(self, root):
        self.root = root
        self.model = json.loads((ROOT / 'results/model.json').read_text(encoding='utf-8'))
        self.metrics = json.loads((ROOT / 'results/final_metrics.json').read_text(encoding='utf-8'))
        self.rules = {r['id']: r for r in self.model['rules']}
        self.last_summary = ''
        root.title('Car Evaluation — Decision Explorer')
        width = min(1160, max(800, root.winfo_screenwidth() - 80))
        height = min(880, max(620, root.winfo_screenheight() - 100))
        root.geometry(f'{width}x{height}')
        root.minsize(800, 600)
        root.configure(bg=BG)
        self.configure_styles()
        self.build_header()
        self.tabs = ttk.Notebook(root)
        self.tabs.pack(fill='both', expand=True, padx=14, pady=(14, 0))
        self.pages = [ScrollPage(self.tabs) for _ in range(3)]
        for page, name in zip(self.pages, ('  Evaluate a car  ', '  Test evidence  ', '  Rule explorer  ')):
            self.tabs.add(page, text=name)
        self.build_evaluation(self.pages[0].body)
        self.build_evidence(self.pages[1].body)
        self.build_rules(self.pages[2].body)
        label(root, 'CPU5006  •  Synthetic dataset classification  •  Not real-world purchasing advice',
              size=9, colour=MUTED, bg=BG).pack(fill='x', padx=26, pady=9)
        root.bind('<Return>', lambda _e: self.evaluate())
        root.bind_all('<MouseWheel>', self.scroll)
        root.bind_all('<Button-4>', lambda e: self.scroll(e, -1))
        root.bind_all('<Button-5>', lambda e: self.scroll(e, 1))
        root.bind('<Prior>', lambda _e: self.current_page().canvas.yview_scroll(-1, 'pages'))
        root.bind('<Next>', lambda _e: self.current_page().canvas.yview_scroll(1, 'pages'))
        self.evaluate()

    def configure_styles(self):
        style = ttk.Style()
        style.theme_use('clam')
        style.configure('.', font=(FONT, 10))
        style.configure('TFrame', background=BG)
        style.configure('TNotebook', background=BG, borderwidth=0)
        style.configure('TNotebook.Tab', padding=(16, 11), background='#e3eaf2', foreground=MUTED)
        style.map('TNotebook.Tab', background=[('selected', WHITE)], foreground=[('selected', INK)])
        style.configure('TCombobox', padding=(9, 7), foreground=INK, fieldbackground=WHITE)
        style.map('TCombobox', fieldbackground=[('readonly', WHITE)], foreground=[('readonly', INK)])
        style.configure('TButton', padding=(12, 8), background='#e7edf4', foreground=INK)
        style.configure('Primary.TButton', background=TEAL, foreground=WHITE)
        style.map('Primary.TButton', background=[('active', '#096569')])
        style.configure('Treeview', rowheight=31, background=WHITE, fieldbackground=WHITE, foreground=INK)
        style.configure('Treeview.Heading', background='#e7edf4', foreground=INK, font=(FONT, 10, 'bold'))
        style.map('Treeview', background=[('selected', '#d5eceb')], foreground=[('selected', INK)])

    def build_header(self):
        header = tk.Frame(self.root, bg=NAVY, padx=28, pady=20)
        header.pack(fill='x')
        label(header, 'DECISION EXPLORER', size=10, colour='#8bd5d0', bold=True, bg=NAVY).pack(fill='x')
        label(header, 'Car evaluation', size=25, colour=WHITE, bold=True, bg=NAVY).pack(fill='x', pady=(3, 4))
        label(header, 'Explore a prediction. Read the rule. Check the evidence.',
              size=11, colour='#c5d5e5', bg=NAVY).pack(fill='x')

    def current_page(self):
        return self.pages[self.tabs.index(self.tabs.select())]

    def scroll(self, event, direction=None):
        # Let the rule list consume its own scroll events instead of scrolling twice.
        if isinstance(event.widget, (ttk.Treeview, ttk.Combobox)):
            return
        delta = direction if direction is not None else (-1 if event.delta > 0 else 1)
        self.current_page().canvas.yview_scroll(delta * 3, 'units')

    def build_evaluation(self, parent):
        form = panel(parent)
        form.pack(fill='x')
        label(form, '01  Describe the car', size=14, bold=True).pack(fill='x')
        label(form, 'Results update when you change a value. All inputs use the dataset’s categories.',
              size=10, colour=MUTED).pack(fill='x', pady=(4, 16))
        fields = tk.Frame(form, bg=WHITE)
        fields.pack(fill='x')
        self.values = []
        defaults = ('low', 'low', '4', '4', 'big', 'high')
        for i, (feature, tokens, default) in enumerate(zip(FEATURES, TOKENS, defaults)):
            column, row = i % 3, (i // 3) * 2
            fields.columnconfigure(column, weight=1, uniform='inputs')
            label(fields, FIELD_NAMES[feature], size=10, bold=True).grid(row=row, column=column,
                sticky='ew', padx=(0, 16), pady=(0, 5))
            variable = tk.StringVar(value=display_value(default))
            self.values.append(variable)
            box = ttk.Combobox(fields, textvariable=variable, state='readonly',
                               values=[display_value(t) for t in tokens], width=14)
            box.grid(row=row + 1, column=column, sticky='ew', padx=(0, 16), pady=(0, 16))
            box.bind('<<ComboboxSelected>>', lambda _e: self.evaluate())
        actions = tk.Frame(form, bg=WHITE)
        actions.pack(fill='x')
        ttk.Button(actions, text='Evaluate car', style='Primary.TButton', command=self.evaluate).pack(side='left')
        ttk.Button(actions, text='Reset example', command=lambda: self.preset(defaults)).pack(side='left', padx=8)
        ttk.Button(actions, text='Try low safety', command=self.low_safety).pack(side='left')
        label(parent, '02  Compare the decisions', size=14, bold=True, bg=BG).pack(fill='x', pady=(22, 10))
        cards = tk.Frame(parent, bg=BG)
        cards.pack(fill='x')
        cards.columnconfigure(0, weight=1, uniform='cards')
        cards.columnconfigure(1, weight=1, uniform='cards')
        self.manual_card = DecisionCard(cards, 'Manual rules', 'Six authored rules • first match wins')
        self.learned_card = DecisionCard(cards, 'Learned rules', f"{len(self.rules)} exported rules • one matching path")
        self.manual_card.frame.grid(row=0, column=0, sticky='nsew', padx=(0, 8))
        self.learned_card.frame.grid(row=0, column=1, sticky='nsew', padx=(8, 0))
        self.agreement = label(parent, size=11, bold=True, bg=BG)
        self.agreement.pack(fill='x', pady=(12, 8))
        row = tk.Frame(parent, bg=BG)
        row.pack(fill='x')
        ttk.Button(row, text='Copy explanation', command=self.copy_explanation).pack(side='left')
        ttk.Button(row, text='Inspect matched rule', command=self.inspect_match).pack(side='left', padx=8)
        label(parent, 'Agreement does not prove correctness. Compare held-out performance in Test evidence.',
              size=10, colour=MUTED, bg=BG).pack(fill='x', pady=(12, 0))

    def values_as_tokens(self):
        return tuple(tokens[[display_value(t) for t in tokens].index(v.get())]
                     for tokens, v in zip(TOKENS, self.values))

    def preset(self, tokens):
        for variable, token in zip(self.values, tokens):
            variable.set(display_value(token))
        self.evaluate()

    def low_safety(self):
        self.values[-1].set(display_value('low'))
        self.evaluate()

    def evaluate(self):
        try:
            values = self.values_as_tokens()
            result = compare(values)
            manual, learned = result['manual'], result['learned']
            self.matched_id = learned['rule_id']
            rule = self.rules[self.matched_id]
            manual_id, manual_reason = manual['explanation'].split(': ', 1)
            self.manual_card.set(manual['label'], f'Rule {manual_id}', manual_reason.capitalize() + '.',
                'Priority matters: this is the first condition that matched.')
            conditions = '\n'.join('• ' + item for item in display_conditions(rule))
            self.learned_card.set(learned['label'], f"Rule {learned['rule_id']} · all conditions match", conditions,
                f"{learned['training_support']} training records reached this rule. "
                + ('Parent-majority fallback. ' if learned['fallback'] else '')
                + 'Support is a count, not a confidence probability.')
            agree = manual['label'] == learned['label']
            self.agreement.configure(text=('✓  Both methods agree on this input.' if agree else
                                           'Different decisions — inspect the conditions above.'),
                                     fg=TEAL if agree else '#946014')
            self.last_summary = (input_summary(values) + '\n\nManual: ' + LABEL_NAMES[manual['label']]
                + '\n' + manual['explanation'] + '\n\nLearned: ' + LABEL_NAMES[learned['label']]
                + '\n' + learned['explanation'] + f"\nTraining support: {learned['training_support']} (not confidence).")
        except (OSError, ValueError, KeyError) as exc:
            messagebox.showerror('Cannot evaluate', str(exc))

    def copy_explanation(self):
        self.root.clipboard_clear()
        self.root.clipboard_append(self.last_summary)
        self.agreement.configure(text='Explanation copied to the clipboard.', fg=TEAL)

    def inspect_match(self):
        self.rule_filter.set('All classifications')
        self.populate_rules()
        self.tabs.select(2)
        self.tree.selection_set(self.matched_id)
        self.tree.see(self.matched_id)
        self.show_rule()
        self.pages[2].canvas.yview_moveto(0)

    def build_evidence(self, parent):
        label(parent, 'Evidence from the reserved test set', size=18, bold=True, bg=BG).pack(fill='x')
        label(parent, f"{self.metrics['test_size']} records • settings frozen before testing • no retraining in this app",
              size=10, colour=MUTED, bg=BG).pack(fill='x', pady=(5, 16))
        table = panel(parent)
        table.pack(fill='x')
        rows = [('Method', 'Accuracy', 'Macro F1', 'Balanced accuracy')]
        for key, title in [('majority_baseline', 'Majority baseline'), ('manual_rules', 'Manual rules'),
                           ('learned_rules', 'Learned rules')]:
            data = self.metrics[key]
            rows.append((title, f"{data['accuracy']:.2%}", f"{data['macro_f1']:.3f}", f"{data['balanced_accuracy']:.2%}"))
        for i, row in enumerate(rows):
            for j, value in enumerate(row):
                table.columnconfigure(j, weight=2 if j == 0 else 1)
                label(table, value, size=10 if i == 0 else 12, bold=i in (0, 3),
                      colour=TEAL if i == 3 else INK).grid(row=i, column=j, sticky='ew', padx=6, pady=9)
        label(parent, 'Accuracy counts correct predictions. Macro F1 weights each class equally. '
              'Balanced accuracy averages recall across classes.', size=10, colour=MUTED, bg=BG).pack(fill='x', pady=(10, 18))
        matrix_panel = panel(parent)
        matrix_panel.pack(fill='x')
        label(matrix_panel, 'Where predictions go wrong', size=14, bold=True).pack(fill='x')
        self.matrix_choice = tk.StringVar(value='Learned rules')
        picker = ttk.Combobox(matrix_panel, textvariable=self.matrix_choice,
                             values=['Learned rules', 'Manual rules', 'Majority baseline'], state='readonly', width=22)
        picker.pack(anchor='w', pady=12)
        picker.bind('<<ComboboxSelected>>', lambda _e: self.draw_matrix())
        label(matrix_panel, 'Rows = actual class  ·  Columns = predicted class  ·  Diagonal = correct',
              size=10, colour=MUTED).pack(fill='x', pady=(0, 10))
        self.matrix = tk.Frame(matrix_panel, bg=WHITE)
        self.matrix.pack(fill='x')
        self.matrix_note = label(matrix_panel, size=10, colour=MUTED)
        self.matrix_note.pack(fill='x', pady=(12, 0))
        self.draw_matrix()
        label(parent, 'Read the limits as well as the score', size=14, bold=True, bg=BG).pack(fill='x', pady=(22, 8))
        label(parent, 'Only 14 good and 13 very-good cases are in the test set. '
              'These results describe one split of a synthetic dataset, not real vehicle safety or buyer satisfaction. '
              'The learned model has 156 rules, so improved accuracy comes with a larger rule base to maintain.',
              size=11, colour=MUTED, bg=BG).pack(fill='x')

    def draw_matrix(self):
        for child in self.matrix.winfo_children():
            child.destroy()
        key = {'Learned rules': 'learned_rules', 'Manual rules': 'manual_rules',
               'Majority baseline': 'majority_baseline'}[self.matrix_choice.get()]
        data = self.metrics[key]
        matrix = data['confusion_matrix_rows_actual_columns_predicted']
        names = [LABEL_NAMES[token] for token in LABELS]
        for j, name in enumerate(['Actual / predicted'] + names):
            self.matrix.columnconfigure(j, weight=1)
            label(self.matrix, name, size=9, bold=True).grid(row=0, column=j, sticky='ew', padx=4, pady=7)
        for i, row in enumerate(matrix):
            label(self.matrix, names[i], size=10, bold=True).grid(row=i+1, column=0, sticky='ew', padx=4, pady=4)
            for j, count in enumerate(row):
                cell = tk.Label(self.matrix, text=str(count), font=(FONT, 13, 'bold'),
                                fg=TEAL if i == j else INK,
                                bg='#e0f0ed' if i == j else '#f0f3f7', pady=12)
                cell.grid(row=i+1, column=j+1, sticky='nsew', padx=3, pady=3)
        total = sum(sum(row) for row in matrix)
        correct = sum(matrix[i][i] for i in range(4))
        self.matrix_note.configure(text=f'{correct} correct of {total}  ·  {total-correct} errors. '
                                  'All counts come from the supplied evaluation results.')

    def build_rules(self, parent):
        label(parent, 'Explore the learned rule base', size=18, bold=True, bg=BG).pack(fill='x')
        label(parent, 'Each rule is one path through the learned tree. Select a row to read its conditions.',
              size=10, colour=MUTED, bg=BG).pack(fill='x', pady=(5, 14))
        self.rule_filter = tk.StringVar(value='All classifications')
        box = ttk.Combobox(parent, textvariable=self.rule_filter, state='readonly', width=25,
                          values=['All classifications'] + list(LABEL_NAMES.values()))
        box.pack(anchor='w', pady=(0, 12))
        box.bind('<<ComboboxSelected>>', lambda _e: self.populate_rules())
        list_frame = tk.Frame(parent, bg=WHITE)
        list_frame.pack(fill='x')
        self.tree = ttk.Treeview(list_frame, columns=('label', 'conditions', 'support'), show='tree headings', height=8)
        self.tree.heading('#0', text='Rule')
        self.tree.column('#0', width=95, stretch=False)
        for key, title, width in [('label', 'Classification', 180), ('conditions', 'Conditions', 130),
                                   ('support', 'Training records', 160)]:
            self.tree.heading(key, text=title)
            self.tree.column(key, width=width, anchor='w' if key == 'label' else 'center')
        scrollbar = ttk.Scrollbar(list_frame, orient='vertical', command=self.tree.yview)
        self.tree.configure(yscrollcommand=scrollbar.set)
        scrollbar.pack(side='right', fill='y')
        self.tree.pack(fill='x', expand=True)
        self.tree.bind('<<TreeviewSelect>>', lambda _e: self.show_rule())
        self.rule_count = label(parent, size=10, colour=MUTED, bg=BG)
        self.rule_count.pack(fill='x', pady=8)
        detail = panel(parent)
        detail.pack(fill='x', pady=(4, 0))
        self.rule_title = label(detail, 'Select a rule', size=15, bold=True)
        self.rule_title.pack(fill='x')
        self.rule_detail = label(detail, size=11)
        self.rule_detail.pack(fill='x', pady=(10, 0))
        self.populate_rules()

    def populate_rules(self):
        for child in self.tree.get_children():
            self.tree.delete(child)
        selected = self.rule_filter.get()
        for rule in self.rules.values():
            if selected == 'All classifications' or LABEL_NAMES[rule['label']] == selected:
                self.tree.insert('', 'end', iid=rule['id'], text=rule['id'], values=(
                    LABEL_NAMES[rule['label']], len(rule['conditions']), rule['support']))
        self.rule_count.configure(text=f'{len(self.tree.get_children())} of {len(self.rules)} rules shown. Support counts training records, not test records.')
        self.rule_title.configure(text='Select a rule')
        self.rule_detail.configure(text='Use the list above, or choose “Inspect matched rule” after evaluating a car.')

    def show_rule(self):
        selected = self.tree.selection()
        if not selected:
            return
        rule = self.rules[selected[0]]
        self.rule_title.configure(text=f"{rule['id']}  →  {LABEL_NAMES[rule['label']]}", fg=COLOURS[rule['label']])
        distribution = ', '.join(f'{LABEL_NAMES[token]}: {rule["counts"].get(token, 0)}' for token in LABELS)
        self.rule_detail.configure(text='ALL of these conditions must match:\n\n'
            + '\n'.join('• ' + line for line in display_conditions(rule))
            + f"\n\nTraining support: {rule['support']}\nTraining labels: {distribution}\n"
            + ('Fallback: parent-majority decision.' if rule['fallback'] else 'Fallback: no.')
            + '\nSupport is not a probability that a new prediction is correct.')


def main():
    root = tk.Tk()
    try:
        CarApp(root)
    except (OSError, ValueError, KeyError) as exc:
        messagebox.showerror('Project files unavailable',
                             f'{exc}\n\nExtract the entire ZIP before starting the application.')
        root.destroy()
        return
    root.mainloop()


if __name__ == '__main__':
    main()
