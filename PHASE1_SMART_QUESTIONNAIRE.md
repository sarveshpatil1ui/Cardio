# Phase 1 — Smart Questionnaire Integration

This version combines the original Phase 1 animated terminal application with the Smart Questionnaire implementation.

## Included
- Animated CARDIO AI terminal UI
- Patient profiles
- Report-based 13-feature assessment
- Simple-language assessment for users without reports
- Transparent fallback for unavailable technical fields
- Prediction history
- Patient-specific Logistic Regression feature contributions
- Model performance comparison
- System diagnostics
- Personalized educational recommendations
- Detailed prediction reports

## Smart Questionnaire principle
The simple mode avoids asking users to understand technical ML fields such as `slope`, `ca`, `thal`, and `oldpeak`. It asks understandable questions first. Fields that cannot be reliably derived from those answers are filled with reference values from the processed project dataset and are clearly disclosed.

This is an educational software feature and does not convert the model into a clinical diagnostic system.
