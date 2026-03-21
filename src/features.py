# BUSINESS PURPOSE: This file engineers predictive signals from patient data.
# Raw clinical data (diagnosis codes, lab values, length of stay) is not immediately
# useful to a prediction model — we need to transform it into meaningful variables
# that capture readmission risk. For example, "number of prior admissions in the last
# 6 months" is a much stronger risk signal than a raw admission date. This file
# converts clinical facts into the features the model actually learns from.

# TODO: implement feature engineering functions
