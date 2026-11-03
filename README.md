# Car Evaluation AI

Polished local prototype for CPU5006 Assessment S1. Updated interface release. Compare a six-rule manual classifier with 156 learned symbolic rules. Both return a classification and the conditions behind it. The saved model is ready to use: opening the app does not train or evaluate anything.

## Start on Windows

1. Extract the whole ZIP first.
2. Install Python 3.10 or newer if it is not already installed, including Tcl/Tk (normally included in the standard Windows installer).
3. Double-click `START_WINDOWS.bat`, or open a terminal in this folder and run:

```bash
py app.py
```

Choose the six attributes in **Evaluate a car**. Separate cards show both predictions and the conditions behind them, and results update as inputs change. **Try low safety** changes one input; **Reset example** restores the starting case. **Copy explanation** copies the current comparison, and **Inspect matched rule** opens its details.

**Test evidence** compares all three methods and lets you switch confusion matrices. **Rule explorer** filters the 156 learned rules by classification and shows their conditions and training-label counts. All tabs scroll; Page Up/Page Down and the mouse wheel are supported. Category names are displayed in plain English. `vhigh` means very high; `med` means medium; `5more` means five or more doors; passenger `more` is the dataset's category above four.

The CLI and evaluation need only Python's standard library. The desktop app additionally needs the optional standard-library Tkinter component and a graphical display. On Linux, the operating system's `python3-tk` package may be required. No pip packages are needed.

The prediction engine, CLI and revised evaluation event-handler logic have been tested. The desktop window could not be launched in the preparation environment; its visual layout and Windows launcher still need a local check. See `docs/Verification.md`.

## Other commands

Run from this folder. Use `python` or `python3` instead of `py` where appropriate.

```bash
py predict.py low low 4 4 big high
py -m unittest -v
py evaluate.py
```

The first command compares a single car using the saved model. Attribute order is purchase cost, maintenance cost, doors, persons, luggage, safety. Invalid categories produce an error instead of an unreliable guess.

The second command runs 18 tests (the event-handler test is skipped if Tkinter is unavailable). The final command reproduces model selection and reserved-test evaluation, replacing the generated files in `results/`. It reproduces the same experiment, not a new independent test. Do not use the test results to tune the model further.

## Final reserved test results

| Approach | Accuracy | Macro F1 | Balanced accuracy |
| --- | ---: | ---: | ---: |
| Majority baseline | 69.94% | 0.2058 | 25.00% |
| Manual rules | 82.66% | 0.5826 | 61.34% |
| Learned rules | 92.49% | 0.8476 | 84.37% |

These are measurements on the original 346 reserved records, not the starter's development scores. The learned model classifies 320 correctly and makes 26 errors. The test set contains 242 unacceptable, 77 acceptable, 14 good and 13 very-good cases.

Macro F1 is the arithmetic mean of the four class F1 scores. Balanced accuracy averages class recall. Zero-denominator metrics are defined as zero. Confusion-matrix rows are actual labels, columns predicted labels, ordered `unacc`, `acc`, `good`, `vgood`.

## Algorithms and decisions

**Manual decision list:** `car_ai.classify` checks six ordered rules. Low safety or two-person capacity fails the initial suitability policy. Subsequent rules consider cost and higher-category features. The final rule defaults to acceptable. These are provisional policies, not reconstructed DEX decision tables or automotive expertise. Door count is not used by this method.

**Learned symbolic rules:** `rule_learning.fit` implements categorical ID3-style information-gain splitting. At each node it chooses the feature giving the largest reduction in label entropy. Class-count ties use `LABELS` order; gain ties use original feature order. Stop at a pure node, the selected depth, exhausted attributes, no positive gain, or no eligible split. A split is ineligible if any non-empty child has fewer than the selected minimum records. Empty domain branches get an explicitly marked parent-majority fallback. This is not C4.5 and has no post-pruning.

`export_rules` turns every leaf path into an IF–THEN rule. `predict` scans the exported rules and requires exactly one match. Rules cover valid categories, including locally unobserved branches. Invalid tokens are rejected. The final model happens to have no fallback leaves. Training support indicates a count, not a calibrated probability.

The manual algorithm fulfils the explicit rule-based approach. The comparison learns rules using supervised induction; it must not be described as a wholly hand-authored expert system. This distinction is discussed in the report.

## Evaluation protocol

The original development/test allocation is preserved in `data/split_indices.json` and verified with a SHA-256 dataset hash. The development portion has 1,382 rows. A stratified seed-17 split gives 1,036 training and 346 validation rows. The manual prototype had previously been evaluated on all development rows; only the new learner is selected using internal validation.

Eight predeclared learner configurations use depths 3, 4, 5, 6 and minimum non-empty branch sizes 1, 3. Select highest validation macro F1, breaking ties by fewer rules, smaller depth, then larger minimum size. Depth 5/minimum 1 wins; depth 6/minimum 1 ties at validation macro F1 0.6262 but uses 241 rather than 160 validation-stage rules. The chosen configuration is refitted on all development rows, producing 156 rules. `selection.json` and `model.json` are written before reserved test scoring. No model adjustment was made after reading test results.

The larger training sample changes learned partitions, so the refitted test score is higher than the selection-stage validation score. The observed difference does not establish the same improvement on every split. These are finite-sample estimates from one synthetic benchmark, not real-world performance guarantees.

## Project files

| File | Purpose |
| --- | --- |
| `app.py` | Three-tab desktop explorer |
| `presentation.py` | Human-readable field and category names |
| `predict.py` | Single-case CLI and shared inference entry point |
| `car_ai.py` | Validation, manual rules, metrics and original split helper |
| `rule_learning.py` | Information gain, induction, rule export and inference |
| `evaluate.py` | Selection, frozen model generation and final evaluation |
| `test_car_ai.py`, `test_learning.py`, `test_release.py` | Eighteen behavioural and release tests |
| `data/` | Original supplied data/documentation and fixed split indices |
| `results/model.json`, `results/rules.txt` | Frozen machine-readable and readable rules |
| `results/selection.json` | All eight configurations, validation results and indices |
| `results/final_metrics.json` | Final metrics, class scores, matrices and hashes |
| `results/test_predictions.csv` | Every test prediction and explanation |
| `docs/Supporting_Report.docx` | Supporting report with Harvard references |
| `docs/Report.md` | Editable text equivalent of the report |
| `docs/Walkthrough_Guide.md` | Timed prompts and practice questions |
| `docs/Verification.md` | Checks performed and environment limits |
| `docs/Technical_Notes.md` | Worked information gain, real errors and algorithm explanations |
| `docs/Assessment_Checklist.md` | Criterion-by-criterion evidence and submission steps |

## Sources and attribution

Dataset: Bohanec, M. (1988), *Car Evaluation*, UCI Machine Learning Repository, https://doi.org/10.24432/C5JP48. UCI lists CC BY 4.0: https://creativecommons.org/licenses/by/4.0/. Data bytes are unchanged from the supplied archive. UCI's preferred citation year is 1988; the archive's `car.names` gives a 1997 donation/documentation date. The final report uses the repository's preferred citation, correcting the starter's working reference.

Algorithm: Quinlan, J.R. (1986), ‘Induction of decision trees’, *Machine Learning*, 1, pp. 81–106. https://doi.org/10.1007/BF00116251.

Metric definitions: scikit-learn developers, https://scikit-learn.org/stable/modules/model_evaluation.html. Scikit-learn was used only as an independent verification tool during preparation, not as a runtime dependency or model implementation.