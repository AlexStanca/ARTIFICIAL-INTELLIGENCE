# Eight minute walkthrough guide

Use these prompts to practise. Do not read this file aloud as a script. Open the application, source code and results before recording. The finished video must be your recording, demonstrating your understanding.

| Time | On screen | Points to explain naturally |
| --- | --- | --- |
| 0:00–0:45 | Desktop app | Six attributes, four outputs; reproducing dataset judgements, not advising real buyers |
| 0:45–1:20 | `data/car.names` and README | 1,728 categorical combinations; synthetic origin; roughly 70% unacceptable |
| 1:20–2:10 | `car_ai.py`, `classify` | First-match priority; why exclusions precede quality rules; risk of the acceptable default |
| 2:10–3:25 | `rule_learning.py`, `fit` and `export_rules` | Entropy as uncertainty; gain as its reduction; recursive splits; leaf paths become IF–THEN rules |
| 3:25–4:15 | App | Evaluate default case; change safety to low; open the matched rule; explain support count |
| 4:15–5:15 | `evaluate.py`, `selection.json` | Original test split retained; internal validation; eight candidates; tie-break favours fewer rules; refit and freeze |
| 5:15–6:20 | Test evidence tab | Compare accuracy/macro F1; 10/13 very-good cases recognised; 26 total learned-rule errors |
| 6:20–7:20 | `rules.txt` and report | Six versus 156 rules; one split and few rare-class cases; synthetic labels; pruning as future work |
| 7:20–8:00 | Terminal tests | Run tests; show reproduction command; summarise what works and what remains uncertain |

## Trace the code before recording

- `validate_features` checks the six inputs before any prediction.
- `classify` returns a label plus a reason immediately when a condition matches.
- `fit` counts classes, calculates information gain, selects an attribute, then repeats on each branch.
- `export_rules` collects every path's conditions and leaf label.
- `predict` finds the one matching symbolic rule. Zero or multiple matches are treated as implementation errors.
- `evaluate.main` separates model choice from reserved test scoring. It records the selected model before producing test predictions.

## Questions to answer without memorising

**Why use a majority baseline?** To reveal how much accuracy comes from imbalance alone: 69.94% with no useful discrimination.

**What does macro F1 add?** It gives each of the four classes equal weight and combines precision with recall within each class.

**Why is this rule-based if a learner is involved?** The manual model applies authored rules. The second method learns a symbolic structure, exports IF–THEN rules and uses those rules for inference. Rule acquisition and rule execution are separate stages.

**Why depth five?** Depth five and six tie on validation macro F1 at 0.6262, but depth five produces fewer validation-stage rules. Test data did not make this choice.

**Why does the final rule count differ from validation?** The selected learner is refitted with 1,382 rather than 1,036 records. More training data changes partitions and stopping points.

**Why is test performance higher than validation performance?** Different records and a larger refit training sample. The result is observed, not proof that test sets should score higher.

**Is training support confidence?** No. It counts records reaching the learned leaf; it does not provide a calibrated probability for a new case.

**Do all six features have to occur in every rule?** No. A path can stop early when its cases are pure or another stopping condition applies.

**What is the largest remaining risk?** The model predicts a synthetic labelling policy, not real vehicle quality. Explainability cannot establish that the policy itself is appropriate.

## Before submission

Add the video URL and correct assignment repository URL to the Word report. Make sure your assessor can open both links. Keep the video within eight minutes and explain only what you can genuinely justify. If asked about assistance, describe it accurately.

## Interface rehearsal

Before recording, check the window at your normal display scaling. Try the three tabs, scroll to the lower sections, filter rules, inspect the current prediction and copy an explanation. Reset the example before your walkthrough. Read `Technical_Notes.md` for a numerical information-gain example and a real misclassification you can explain. Use `Assessment_Checklist.md` to check the required submission links.
