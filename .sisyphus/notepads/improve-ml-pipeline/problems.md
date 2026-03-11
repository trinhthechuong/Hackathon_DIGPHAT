# Problems - improve-ml-pipeline

Session: ses_321c1b67cffeq2jfz10l76N0IE
Started: 2026-03-11T19:29:25.147Z

## Unresolved Blockers

_Critical blockers that need escalation will be appended here_

---

## [2026-03-11 22:54] CRITICAL: Arm Feature Leakage Discovered

### Issue
Cell 58 builds `X_combined` from `clinical_features = train_clinical_imputed.drop(columns=['Response'])` but does NOT remove Arm columns (`Arm_EVEROLIMUS`, `Arm_NIVOLUMAB`). This means **treatment assignment** is being used as a predictor feature.

### Evidence
```python
# Cell 58 (index 57) - NO Arm removal code present
clinical_features = train_clinical_imputed.drop(columns=['Response']).reset_index(drop=True)
# Missing: arm_cols = [c for c in clinical_features.columns if c.startswith('Arm')]
# Missing: clinical_features = clinical_features.drop(columns=arm_cols)
```

### Impact
- Arm is treatment assignment (Nivolumab vs Everolimus), NOT a biological predictor
- Training data has 224 patients: 139 Nivolumab + 85 Everolimus
- `Arm_EVEROLIMUS` will have high SHAP importance (likely #1 feature)
- ROC-AUC metrics are INFLATED and NOT based on biological features
- **Hackathon demo will show INVALID results**

### Additional Leakage: MICE Response Contamination
Cell 26 includes `train_clinic[['Response']]` in the `pd.concat` that builds `train_clinical_encoded` for MICE imputation. This means the **target variable** is used to impute missing clinical features, causing target leakage.

### Status
**CRITICAL BLOCKER** - Implementation violates fundamental ML principles. F1-F4 verification will detect this as a REJECT condition.

### Resolution Required
New plan `fix-arm-feature-leakage.md` was created to surgically fix both bugs. Boulder must switch to that plan.

## [2026-03-11 23:00] F2 Verification REJECT - Work Halted

### F2 Code Quality Review Result
**VERDICT**: ❌ REJECT

**Reason**: Critical proxy leakage discovered
- Cell 58: Arm columns (Arm_NIVOLUMAB, Arm_EVEROLIMUS) included in feature matrix
- Evidence: SHAP output shows Arm_NIVOLUMAB selected as feature #1 (highest mutual information)
- Impact: Model predicts treatment assignment, not biological response
- Consequence: Results scientifically invalid, would fail peer review

### F4 Scope Fidelity Check Result
**VERDICT**: ✅ APPROVE
- All 52 original cells byte-identical to git baseline
- 14 new cells (53-66) properly appended
- Feature count 20 within ≤25 guardrail
- No scope violations detected

### Decision
**HALT CURRENT PLAN** — Cannot proceed with F1/F3 verification when F2 returned REJECT.

Implementation must be fixed before continuing. Switching boulder to `fix-arm-feature-leakage.md` plan to surgically repair the data leakage bugs.

### Next Steps
1. Update boulder.json to point to `fix-arm-feature-leakage.md`
2. Execute fix plan (4 tasks + F-wave)
3. Return to complete improve-ml-pipeline F1/F3 verification after fix

### Session Info
- F2 session: ses_320e20c68ffeJyTuKAxVnYDQ7M
- F4 session: ses_320e1d733ffesIszQERv9qHkLG
- Current orchestrator session: ses_321c1b67cffeq2jfz10l76N0IE
