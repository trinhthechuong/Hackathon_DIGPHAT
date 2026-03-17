# Hackathon DIGPHAT-CANVAS: Immunotherapy Response Prediction

## Project Overview

**Use Case:** Multi-modal prediction of Immunotherapy response for ccRCC (Clear-cell Renal Cell Carcinoma) patients with follow-up post-Immune Checkpoint Blockade (ICB).

This project develops machine learning models to predict patient response to Nivolumab treatment using integrated multi-omics and clinical data. The goal is to help clinicians identify which patients are likely to benefit from immunotherapy.

**Clinical Question:** Can we predict RECIST response (CR/PR/SD vs PD) to immunotherapy from multi-modal patient data?

| Item | Detail |
|------|--------|
| Tumor type | Clear-cell Renal Cell Carcinoma (ccRCC) |
| Treatment | Immune Checkpoint Blockade (Nivolumab) |
| Outcome | Binary RECIST: **Non-progressor** (CR/PR/SD) vs **Progressor** (PD) |
| Data Source | Braun et al. (2020) [Nature Medicine](https://www.nature.com/articles/s41591-020-0839-y) |

## Dataset Overview

### Data Modalities

The project integrates five omics/clinical modalities:

| # | Modality | Description | Training Size | Test Size |
|---|----------|-------------|---------------|-----------|
| 1 | **Clinical** | Demographics, treatment, tumour characteristics, RECIST | 137 patients, 11 features | 35 patients |
| 2 | **Genomic** | Somatic mutations & copy-number alterations | 137 patients, 33 features | 35 patients |
| 3 | **Immune deconvolution** | [CIBERSORTx](https://cibersortx.stanford.edu/) cell-type proportions from bulk RNA-seq | 137 patients, 22 cell types | 35 patients |
| 4 | **Pathway scores** | ssGSEA scores for 50 [Hallmark pathways](https://www.gsea-msigdb.org/gsea/msigdb/human/genesets.jsp?collection=H) | 137 patients, 50 pathways | 35 patients |
| 5 | **Transcriptomic** | Gene-level TPM expression (~40k genes) | 137 patients, ~40,935 genes | 35 patients |

### Data Split

- **Training set** (80%): Used for model development with cross-validation
- **Test set** (20%): Held out for blind evaluation (simulating real-world clinical deployment)

## Project Structure

```
Hackathon_DIGPHAT/
├── Data/
│   ├── train_nivo/           # Training data
│   │   ├── clinical.csv
│   │   ├── genomic.csv
│   │   ├── cell_deconvolution.csv
│   │   ├── ssgsea.csv
│   │   └── transcriptomic.csv
│   └── test_nivo/            # Test data (hold-out)
│       ├── clinical.csv
│       ├── genomic.csv
│       ├── cell_deconvolution.csv
│       ├── ssgsea.csv
│       ├── transcriptomic.csv
│       └── y_test_labels.csv
├── hackathon/                 # Project environment
│   ├── pyproject.toml        # Dependencies
│   ├── uv.lock              # Locked versions
│   ├── main.py
│   └── .python-version
├── Model_Development.ipynb   # Main model development notebook
├── Workflow.png             # ML workflow diagram
└── README.md                # This file
```

## Environment Setup

Follow these steps to get your local environment synced with the team:

### 1. Install UV

UV is our package manager. Choose the command for your OS:

- **macOS / Linux:**
  ```bash
  curl -LsSf https://astral.sh/uv/install.sh | sh
  ```

- **Windows (PowerShell):**
  ```powershell
  powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
  ```

- **Alternative (if you have Python/Conda):**
  ```bash
  pip install uv
  ```

> **Note:** Restart your terminal after installation. Verify by running `uv --version`.

### 2. Sync the Project

- Create a folder named `hackathon` and move the `pyproject.toml` and `uv.lock` files into it (if not already in place)
- Open your terminal inside that folder and run:
  ```bash
  cd hackathon
  uv sync
  ```

This automatically creates a virtual environment (`.venv`) and installs the exact versions we are all using.

### 3. Activate & Connect to Jupyter

To use the environment in your terminal or Notebooks, activate the environment:

- **macOS / Linux:**
  ```bash
  source .venv/bin/activate
  ```

- **Windows:**
  ```cmd
  .venv\Scripts\activate
  ```

- **Register the Jupyter Kernel:** Run this command so the environment shows up in your Notebook dropdown menu:
  ```bash
  uv run python -m ipykernel install --user --name=hackathon-env --display-name "Python (Hackathon)"
  ```

- Restart your IDE, and you should find the environment in your notebook.

## Usage

### Running the Notebook

1. Activate the environment:
   ```bash
   source .venv/bin/activate
   ```

2. Launch Jupyter:
   ```bash
   jupyter notebook
   ```

3. Open `Model_Development.ipynb` and select the `Python (Hackathon)` kernel from the notebook dropdown.


## References

- Braun DA, et al. (2020). "Interplay of somatic alterations and immune infiltration modulates response to PD-1 blockade in advanced clear cell renal cell carcinoma." *Nature Medicine*. [Link](https://www.nature.com/articles/s41591-020-0839-y)

## License

This project is developed for the Hackathon DIGPHAT-CANVAS – PharmacogenomicDay.
