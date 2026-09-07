Login Forms Detection
=====================

Detects malicious phishing login forms by analysing the structural signature of a
page's HTML DOM. Binary classification: LOGIN_FORM_MALICIOUS vs NO_FORM.


Setup
-----
    pip install -r requirements.txt


Run Order
---------
Start with EDA (optional):
    notebooks/01_data_exploration.ipynb

Structural model (recommended):
    04_xgboost_structural.ipynb
    05_random_forest_structural.ipynb
    06_xgboost_analysis.ipynb       (requires 04)
    07_random_forest_analysis.ipynb  (requires 05)

Sequence model:
    02_xgboost_detector.ipynb
    03_random_forest_detector.ipynb

Comparison (optional, requires all above):
    09_accuracy_comparison.ipynb


