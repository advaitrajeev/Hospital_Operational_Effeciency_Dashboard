# Exploratory Data Analysis (EDA) Summary

## Overview
Phase 4 focused on understanding the core drivers of operational inefficiency through visual and statistical exploration of the cleaned encounters dataset. We analyzed Average Length of Stay (ALOS), bed utilization patterns, and 30-day readmission risks across various dimensions including DRG code, Department, and Payer type.

## Key Insights

### 1. Bimodal ALOS Distributions within DRGs
**Crucial Discovery:** When plotting the distribution of Length of Stay for high-volume conditions (e.g., Heart Failure - DRG291), the data revealed a distinct **bimodal distribution**. 
- **Fast Track Population:** ~40% of patients are discharged within 1–3 days.
- **Complex Population:** ~60% of patients experience extended stays of 6–10 days.
*Implication:* Using a simple "average" or linear model for ALOS is fundamentally flawed because the mean sits in the trough between the two peaks. This strongly motivates a segmented machine learning approach where we first classify a patient's trajectory (Fast Track vs. Complex) before predicting the exact discharge date.

### 2. Predictable Weekly Bottlenecks
The admission volume heatmap (Day of Week vs. Hour) highlights severe cyclical pressure:
- **Peak Admissions:** Mondays and Tuesdays between 10:00 AM and 2:00 PM see the highest influx.
- **Weekend Drop-off:** Volumes plummet on Saturdays and Sundays.
*Implication:* Bed utilization feels tighter early in the week. Staffing models must shift from flat static ratios to dynamic scheduling that mirrors this cyclical demand.

### 3. Readmission Variance by Payer and Age
- Medicare patients consistently exhibit the highest readmission rates, confirming alignment with CMS benchmarks.
- A strong positive correlation exists between age and readmission probability.

## Next Steps
These findings directly inform **Phase 5 (Feature Engineering)**. We will construct target variables that capture the bimodal nature of ALOS and generate cyclical time-based features (e.g., sine/cosine of admission hour and day) to empower the ML models.
