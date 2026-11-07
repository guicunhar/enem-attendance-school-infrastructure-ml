# School Infrastructure and ENEM Attendance

A **Machine Learning** project covering the full pipeline from data preparation to **model training**, developed for the course **Introduction to Data Science** (Introdução à Ciência dos Dados), taught by Professor **Adriano Branco** at **PUC-Rio**, from August to December 2025.

## Team

All team members are Computer Science students at PUC-Rio:

| Name | Email |
|---|---|
| Guilherme Cunha | guicunharib@gmail.com |
| José Carlos Neto | jcsneto16@gmail.com |
| Lucio Oliveira | EMAIL_LUCIO |
| Luidgi Colimerio |luidgicolimerio@gmail.com |
| Rafael Estevao | rafael6297@gmail.com |

## Goal

This project investigates how **school infrastructure** affects **student attendance** at ENEM, the Brazilian national high school exam. We combine two public datasets from INEP:

- **ENEM 2024 microdata:** whether each student showed up for each exam day.
- **School Census 2024:** each school's infrastructure (library, labs, internet, sports court, staff, and so on).

For each school we compute the share of its students who attended every ENEM test (`PRESENCA_COMPLETA`). Then we train models that predict whether a school has **High** attendance (above 90%) or not. Finally, we use interpretability techniques (coefficients, SHAP) to find which infrastructure items have the most impact. The aim is to help guide public investment: when a school is renovated, which improvements are likely to bring the biggest attendance gains?

Our success metric was accuracy above 75%. The best model, a Random Forest Classifier, reached about **89% accuracy**.

The full plan is in [EXECUTION_PLAN_en.pdf](EXECUTION_PLAN_en.pdf) (English) and [PLANO_EXECUCAO_ptbr.pdf](PLANO_EXECUCAO_ptbr.pdf) (Portuguese).

## Notebooks

Run the notebooks in order. Each one reads the files the previous one saved in `data/`.

| Notebook | What it does | Reads | Writes |
|---|---|---|---|
| `01_CreateDataset.ipynb` | Selects the relevant columns from both raw datasets, creates derived features (such as students per class), merges them by school and computes the attendance rate per school. | `RESULTADOS_2024.csv`, `microdados_ed_basica_2024.csv` | `data_set.csv` |
| `02_ExploratoryAnalysis.ipynb` | Exploratory analysis: distributions by region, state and school type, data quality checks, the impact of each infrastructure item on attendance, and correlation matrices. | `data_set.csv` | — |
| `03_FeaturePreprocessing.ipynb` | Preprocessing: fills null values, encodes regions and states, removes outliers and combines related features. | `data_set.csv` | `data_set_final.csv` |
| `04_InitialModelTraining.ipynb` | Trains and compares models: Random Forest Regressor, HistGradientBoosting Regressor, Random Forest Classifier and Logistic Regression. Saves the models and a metrics table. | `data_set_final.csv` | `*.joblib`, `*.pkl`, `model_comparison_metrics.csv` |
| `05_ModelComparing_Visualization.ipynb` | Charts comparing the models' accuracy, F1-score, recall and precision. | `model_comparison_metrics.csv` | — |
| `06_Interpretability.ipynb` | Explains the models: Logistic Regression coefficients and odds ratios, plus SHAP plots for the tree models. | `*.joblib`, `data_set_final.csv` | — |

## How to Run

1. **Install Python 3.10+** and the dependencies:

   ```bash
   pip install pandas numpy matplotlib seaborn plotly scikit-learn joblib shap xgboost jupyter
   ```

2. **Download the raw data** from the INEP open data portal:
   - [ENEM microdata](https://www.gov.br/inep/pt-br/acesso-a-informacao/dados-abertos/microdados/enem): download the 2024 edition and use the results file `RESULTADOS_2024.csv`
   - [School Census microdata](https://www.gov.br/inep/pt-br/acesso-a-informacao/dados-abertos/microdados/censo-escolar): download the 2024 edition and use the basic education file `microdados_ed_basica_2024.csv`

   > **Note:** The project was originally developed and tested with the **2024** data only. Other years may have different columns or file formats, so we can't say how the notebooks will behave with them.

3. **Put both CSV files in a `data/` folder** next to the notebooks:

   ```
   ciencia_dados_enem/
   ├── data/
   │   ├── RESULTADOS_2024.csv
   │   └── microdados_ed_basica_2024.csv
   ├── 01_CreateDataset.ipynb
   ├── ...
   └── 06_Interpretability.ipynb
   ```

4. **Open Jupyter** in the project folder and run the notebooks in order, from `01` to `06`:

   ```bash
   jupyter notebook
   ```

   All intermediate files (datasets, models and metrics) are saved to and read from `data/`, so each notebook can also be rerun on its own once the earlier ones have run.

> **Note:** The raw files are large, and notebook `01` needs a good amount of RAM to load them.
