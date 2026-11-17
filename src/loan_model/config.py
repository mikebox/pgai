"""Central configuration: paths, column names, constants."""
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA = DATA_DIR / "training_data.csv"
PROCESSED_DIR = DATA_DIR / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

RANDOM_STATE = 42
TARGET = "Decision"
POSITIVE_LABEL = "DECLINED"  # the model predicts P(DECLINED); class 1 = risky
ID_COL = "ID"

RAW_COLUMNS = [
    "ID", "Qualifications", "Employment_Type", "Age", "Pincode", "Gender",
    "Product_Code", "Loan_Amount", "LTV", "Resident_Type", "Net_salary",
    "Final_Tier", "Make_Code", "Model_Description", "Model_Variant",
    "PAST_LOANS_ACTIVE", "Decision",
]

# Business policy limits (from the brief) and plausibility bounds (from EDA).
POLICY_AGE_MIN, POLICY_AGE_MAX = 18, 60
MIN_PLAUSIBLE_MONTHLY_SALARY = 5_000   # below this a monthly salary is almost certainly a data-entry error
MIN_PLAUSIBLE_LTV = 1.0                # LTV of 0 is impossible for a financed vehicle
MAX_PLAUSIBLE_LTV = 100.0

