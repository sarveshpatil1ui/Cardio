
# Phase 1 Graph Analysis Update

CARDIO AI now includes an optional Graph Analysis feature.

## Where it appears

- Main menu: `[6] Graph Analysis`
- After a prediction: `VISUAL ANALYSIS` asks whether the user wants graph analysis.

## Available visualizations

1. Age Distribution
2. Cholesterol vs Disease
3. Resting Blood Pressure vs Disease
4. Maximum Heart Rate vs Disease
5. Target Distribution
6. Correlation Heatmap
7. Model Comparison
8. Feature Importance

The feature opens the existing project-generated PNG visualizations from the `visualizations/` directory using the operating system's default image viewer. It does not change the ML prediction logic.

These graphs are educational EDA/model-analysis views and should not be interpreted as individual medical diagnoses.
