# Bias Audit Report

_Generated: 2026-03-20 21:16:44 UTC_

## Run summary

- Accuracy: **0.7240**
- Train size: **750**
- Test size: **250**
- Fairness threshold: **0.05**
- Minimum group size: **30**

## Fairness by sex

### Disparity summary

| Metric | Difference (max-min) | Ratio (min/max) | Flag |
|---|---:|---:|---|
| selection_rate | 0.1036 | 0.8692 | YES |
| true_positive_rate | 0.1088 | 0.8761 | YES |
| false_positive_rate | 0.0600 | 0.8966 | YES |

### By-group metrics

| Group | false_positive_rate | selection_rate | true_positive_rate |
|---|---:|---:|---:|
| female | 0.5200 | 0.6883 | 0.7692 |
| male | 0.5800 | 0.7919 | 0.8780 |

## Fairness by age group

### Disparity summary

| Metric | Difference (max-min) | Ratio (min/max) | Flag |
|---|---:|---:|---|
| selection_rate | 0.3415 | 0.6585 | YES |
| true_positive_rate | 0.3182 | 0.6818 | YES |
| false_positive_rate | 0.6190 | 0.3810 | YES |

### By-group metrics

| Group | false_positive_rate | selection_rate | true_positive_rate |
|---|---:|---:|---:|
| adult | 0.6061 | 0.8000 | 0.8627 |
| middle | 0.3810 | 0.7031 | 0.8605 |
| senior | 1.0000 | 1.0000 | 1.0000 |
| young | 0.6316 | 0.6585 | 0.6818 |

## Findings

The following disparities exceeded the configured threshold:

- **sex**: **selection_rate** difference = **0.1036**
- **sex**: **true_positive_rate** difference = **0.1088**
- **sex**: **false_positive_rate** difference = **0.0600**
- **age_group**: **selection_rate** difference = **0.3415**
- **age_group**: **true_positive_rate** difference = **0.3182**
- **age_group**: **false_positive_rate** difference = **0.6190**

## Diagnostics run

### Executed diagnostics

- `run_threshold_sensitivity` args={"attribute": "age_group"}
- `run_slice_scan` args={}

### Diagnostic highlights

- `run_threshold_sensitivity`
- `age_group`: lowest TPR gap at threshold 0.1 (TPR diff=0.0233, FPR diff=0.0952)
- `run_slice_scan`
- `checking_account_status=A14` (n=93): selection=1.0000, TPR=1.0000, FPR=1.0000
- `checking_account_status=A13` (n=15): selection=1.0000, TPR=1.0000, FPR=1.0000
- `credit_history=A34` (n=66): selection=0.9394, TPR=0.9608, FPR=0.8667
- `savings_account=A64` (n=14): selection=0.9286, TPR=1.0000, FPR=0.5000
- `savings_account=A63` (n=22): selection=0.9091, TPR=0.8824, FPR=1.0000

## Agent explanation

- Summary: Fairness Audit Report
- Likely cause: TPR/FPR disparities are flagged for age_group.
- Limit: group_sizes.sex.male < audit_config.min_group_size

### Narrative

# Fairness Audit Narrative
## What we found
We detected demographic disparities in selection rates for sex and true positive rates for age group.
## Evidence (numbers)
* Sex: selection_rate = 0.10359582613917873, true_positive_rate = 0.6818181818181818, false_positive_rate = 0.05999999999999994
* Age Group: true_positive_rate = 0.6818181818181818
## Likely causes (hypotheses)
TPR/FPR disparities are flagged for age_group.
## Recommended next tests
- No additional diagnostics recommended (existing diagnostics already executed).
## Mitigations to consider
None
## Limits / Unknowns
Group sizes for sex.male are below audit_config.min_group_size, which may lead to unstable estimates.
