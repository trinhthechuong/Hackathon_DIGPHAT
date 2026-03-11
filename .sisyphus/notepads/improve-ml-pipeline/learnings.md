# Learnings - improve-ml-pipeline

Session: ses_321c1b67cffeq2jfz10l76N0IE
Started: 2026-03-11T19:29:25.147Z

## Conventions & Patterns

_Accumulated wisdom from task execution will be appended here_

---
## Task 1: Dependency Installation & Notebook Setup

### nbformat for Safe Notebook Modification
**Pattern**: Use `nbformat` library instead of direct JSON manipulation
- **Why**: Preserves notebook metadata, cell structure, and execution history
- **Key steps**:
  1. Read notebook with `nbformat.read(fp, as_version=4)`
  2. Determine next execution_count from existing cells
  3. Create cells with `nbformat.v4.new_markdown_cell()` / `new_code_cell()`
  4. Append to `nb.cells` array
  5. Write with `nbformat.write(nb, fp)`

### Version Pin Strategy for Package Compatibility
**Problem**: NumPy 2.x broke scikit-learn/pandas compatibility
**Solution**: Pin compatible versions across dependency chain
- SHAP 0.49.1 (compatible with numpy<2)
- XGBoost 3.2.0 (no issues with numpy 1.x)

### Execution Count Management
**Convention**: Execution counts are sequential integers
- Check max existing: `max(cell.execution_count for cell in nb.cells if cell.execution_count)`
- Increment by 1 for new code cells
- Markdown cells don't have execution_count

### Dependencies Verified
| Package | Version | Min Required | Status |
|---------|---------|--------------|--------|
| SHAP | 0.49.1 | 0.40 | ✅ TreeExplainer compatible |
| XGBoost | 3.2.0 | 1.5 | ✅ scale_pos_weight supported |

### Blockers & Technical Debt
- NumPy version conflict in environment (mixed versions due to constraints)
- Warnings appear but don't break functionality
- Future: Consider full environment rebuild

### ⚠️ CRITICAL FIX: nbformat Normalization Issue
**Problem**: nbformat library normalizes whitespace and line endings when read/write
- Caused silent modifications to all cell source arrays during round-trip
- Violates G1 guardrail: existing cells must remain BYTE-IDENTICAL

**Solution**: Use raw JSON with minimal indent for append-only operations
```python
import json
with open('notebook.ipynb', 'r') as f:
    nb = json.load(f)  # Raw JSON, no nbformat normalization
# Append new cells as dicts
nb['cells'].append(new_cell_dict)
with open('notebook.ipynb', 'w') as f:
    json.dump(nb, f, indent=1)  # Minimal formatting
```

**Verification**: Deep JSON comparison against `git show HEAD:` to ensure original cells unchanged

