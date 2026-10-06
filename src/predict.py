"""Interactive CARDIO AI application with smart questionnaire and Phase 1 analytics."""
from __future__ import annotations

import csv
import os
import sys
import uuid
import subprocess
from datetime import datetime

import joblib
import numpy as np
import pandas as pd

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from recommendation_engine import generate_recommendations
from terminal_ui import (
    boot_sequence, model_loaded, menu, input_section, feature_scan,
    prediction_animation, probability_animation, risk_meter, result_panel,
    recommendations_panel, patient_summary, explainability_panel,
    model_performance_table, diagnostics_panel, about_panel, report_saved,
    history_empty, error, goodbye, section
)

FEATURE_ORDER = [
    "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
    "thalach", "exang", "oldpeak", "slope", "ca", "thal"
]

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "final_model.joblib")
HISTORY_DIR = os.path.join(PROJECT_ROOT, "data", "history")
HISTORY_PATH = os.path.join(HISTORY_DIR, "prediction_history.csv")
REPORT_DIR = os.path.join(PROJECT_ROOT, "reports")
PERF_PATH = os.path.join(PROJECT_ROOT, "model_comparison.csv")
FEATURE_PATH = os.path.join(PROJECT_ROOT, "feature_importance.csv")
DATASET_FILE = os.path.join(PROJECT_ROOT, "data", "processed", "heart_processed.csv")

CHOICES = {
    "sex": {0: "Female", 1: "Male"},
    "cp": {1: "Typical Angina", 2: "Atypical Angina", 3: "Non-anginal Pain", 4: "Asymptomatic"},
    "fbs": {0: "No", 1: "Yes"},
    "restecg": {0: "Normal", 1: "ST-T Wave Abnormality", 2: "Left Ventricular Hypertrophy"},
    "exang": {0: "No", 1: "Yes"},
    "slope": {1: "Upsloping", 2: "Flat", 3: "Downsloping"},
    "ca": {0: "0", 1: "1", 2: "2", 3: "3"},
    "thal": {3: "Normal", 6: "Fixed Defect", 7: "Reversible Defect"},
}


def safe_input(prompt: str) -> str:
    try:
        return input(prompt).strip()
    except (KeyboardInterrupt, EOFError):
        print("\nExiting program...")
        sys.exit(0)


def numeric_input(label, description, min_val=None, max_val=None, integer=False):
    print(f"\n  {label}")
    print(f"    ↳ {description}")
    while True:
        raw = safe_input("    Enter value: ")
        try:
            value = float(raw)
            if integer and not value.is_integer():
                raise ValueError
            if min_val is not None and value < min_val:
                print(f"    [!] Value must be ≥ {min_val}.")
                continue
            if max_val is not None and value > max_val:
                print(f"    [!] Value must be ≤ {max_val}.")
                continue
            return int(value) if integer else value
        except ValueError:
            print("    [!] Please enter a valid number.")


def choice_input(title, description, options):
    print(f"\n  {title}")
    print(f"    ↳ {description}")
    for key, text, detail in options:
        print(f"      [{key}] {text}")
        if detail:
            print(f"          {detail}")
    valid = {str(item[0]): item[0] for item in options}
    while True:
        raw = safe_input("    Select option: ")
        if raw in valid:
            return valid[raw]
        print("    [!] Invalid choice. Select one of the displayed numbers.")


def yes_no(title, description):
    return choice_input(title, description, [(1, "Yes", ""), (0, "No", "")])


def collect_report_data():
    """Full report path. Preserves the original model-compatible feature encoding."""
    input_section()
    print("  📋 REPORT MODE")
    print("  Enter values from an available medical report or measured clinical record.")
    print("  Friendly descriptions are shown while the original ML encoding is preserved.\n")

    data = {}
    data["age"] = numeric_input("01/13 • Age", "Age in years.", 1, 120, True)
    data["sex"] = choice_input(
        "02/13 • Sex", "Select the category used by the dataset.",
        [(0, "Female", ""), (1, "Male", "")]
    )
    data["cp"] = choice_input(
        "03/13 • Chest Pain Type",
        "Choose the category shown in the clinical record.",
        [(1, "Typical Angina", "Typical angina category."),
         (2, "Atypical Angina", "Atypical angina category."),
         (3, "Non-Anginal Pain", "Pain not classified as angina."),
         (4, "Asymptomatic", "No typical chest-pain symptom category.")]
    )
    data["trestbps"] = numeric_input(
        "04/13 • Resting Blood Pressure",
        "Enter the systolic/resting value in mm Hg. Example: 120/80 → 120.",
        50, 260
    )
    data["chol"] = numeric_input(
        "05/13 • Total Cholesterol",
        "Total serum cholesterol in mg/dL.", 80, 700
    )
    data["fbs"] = choice_input(
        "06/13 • Fasting Blood Sugar",
        "Was fasting blood sugar above 120 mg/dL in the recorded test?",
        [(0, "No", "120 mg/dL or below."),
         (1, "Yes", "Above 120 mg/dL.")]
    )
    data["restecg"] = choice_input(
        "07/13 • Resting ECG Result",
        "Select the category stated in the ECG report.",
        [(0, "Normal", ""),
         (1, "ST-T Wave Abnormality", ""),
         (2, "Left Ventricular Hypertrophy", "")]
    )
    data["thalach"] = numeric_input(
        "08/13 • Maximum Heart Rate Achieved",
        "Maximum heart rate recorded during the test, in beats per minute.",
        50, 230
    )
    data["exang"] = choice_input(
        "09/13 • Exercise-Induced Angina",
        "Did chest discomfort/angina occur during exercise in the recorded test?",
        [(0, "No", ""), (1, "Yes", "")]
    )
    data["oldpeak"] = numeric_input(
        "10/13 • ST Depression",
        "ST depression measurement shown in the exercise test/report.",
        0.0, 10.0
    )
    data["slope"] = choice_input(
        "11/13 • Exercise ST-Segment Slope",
        "Select the slope category reported by the test.",
        [(1, "Upsloping", ""), (2, "Flat", ""), (3, "Downsloping", "")]
    )
    data["ca"] = choice_input(
        "12/13 • Number of Major Vessels",
        "Number of major vessels reported in the relevant test field.",
        [(0, "0 vessels", ""), (1, "1 vessel", ""),
         (2, "2 vessels", ""), (3, "3 vessels", "")]
    )
    data["thal"] = choice_input(
        "13/13 • Thalassemia / Thal Category",
        "Select the category stated in the dataset-compatible report field.",
        [(3, "Normal", "Dataset encoding 3."),
         (6, "Fixed Defect", "Dataset encoding 6."),
         (7, "Reversible Defect", "Dataset encoding 7.")]
    )
    return data, "report"


def simple_cp_mapping(location, trigger, duration):
    """Transparent UI mapping only; not a clinical diagnostic rule."""
    if trigger in (1, 2) and location in (1, 2) and duration in (3, 4):
        return 1
    if trigger in (1, 2) and location in (1, 2):
        return 2
    if trigger in (3, 4, 5) and location in (3, 4, 5):
        return 3
    return 2


def load_reference_values():
    """Reference values used only when basic mode cannot supply a model field."""
    df = pd.read_csv(DATASET_FILE)
    reference = {}
    for col in ["trestbps", "chol", "thalach", "oldpeak"]:
        reference[col] = float(df[col].median())
    for col in ["restecg", "slope", "ca", "thal"]:
        reference[col] = int(df[col].mode().iloc[0])
    return reference


def collect_basic_data():
    """
    Smart/simple-language path.

    The UCI model requires 13 fields, but many users will not know technical
    report terminology. This path asks everyday questions first and explicitly
    uses project-dataset reference values only for fields that cannot be
    responsibly inferred from those questions.
    """
    input_section()
    print("  💬 SIMPLE QUESTION MODE")
    print("  No medical report? No problem.")
    print("  Answer the questions in everyday language.\n")

    data = {}
    data["age"] = numeric_input("01 • How old are you?", "Your age in years.", 1, 120, True)
    data["sex"] = choice_input(
        "02 • Sex", "Select one option.",
        [(0, "Female", ""), (1, "Male", "")]
    )

    has_chest_pain = yes_no(
        "03 • Do you currently or recently get chest discomfort/pain?",
        "Choose Yes if you have experienced it; choose No if not."
    )

    if has_chest_pain:
        location = choice_input(
            "04 • Where do you usually feel it?",
            "Pick the closest description.",
            [(1, "Center of the chest", ""),
             (2, "Left side of the chest", ""),
             (3, "Right side of the chest", ""),
             (4, "Upper chest", ""),
             (5, "Somewhere else / not sure", "")]
        )
        trigger = choice_input(
            "05 • When does it usually happen?",
            "Think about walking, stairs, exercise and rest.",
            [(1, "During exercise / walking", ""),
             (2, "After climbing stairs", ""),
             (3, "At rest", ""),
             (4, "During stress", ""),
             (5, "Randomly", "")]
        )
        duration = choice_input(
            "06 • How long does it usually last?",
            "Choose the closest duration.",
            [(1, "A few seconds", ""),
             (2, "Less than 5 minutes", ""),
             (3, "5–15 minutes", ""),
             (4, "More than 15 minutes", ""),
             (5, "Not sure", "")]
        )
        data["cp"] = simple_cp_mapping(location, trigger, duration)
        data["exang"] = 1 if trigger in (1, 2) else 0
    else:
        data["cp"] = 4
        data["exang"] = 0
        print("\n  No chest discomfort selected; the interface records the asymptomatic category.\n")

    data["fbs"] = choice_input(
        "07 • Have you ever been told your fasting blood sugar is high?",
        "You do not need to know the exact value.",
        [(0, "No", ""),
         (1, "Yes", ""),
         (0, "I'm not sure", "For this model input, treated as No.")]
    )

    bp_known = yes_no(
        "08 • Do you know your recent blood pressure?",
        "If yes, enter the upper/systolic number. If no, a dataset reference value is used."
    )
    if bp_known:
        data["trestbps"] = numeric_input(
            "Blood Pressure", "Example: 120/80 → enter 120.", 50, 260
        )
    else:
        data["trestbps"] = None

    chol_known = yes_no(
        "09 • Do you know your recent cholesterol value?",
        "If yes, enter total cholesterol from your report."
    )
    if chol_known:
        data["chol"] = numeric_input(
            "Total Cholesterol", "Enter total cholesterol in mg/dL.", 80, 700
        )
    else:
        data["chol"] = None

    exercise = choice_input(
        "10 • Does walking, running or climbing stairs ever bring on chest discomfort?",
        "Answer based on your usual experience.",
        [(0, "No", ""),
         (1, "Yes", ""),
         (0, "Sometimes / not sure", "For the model input, treated as No.")]
    )
    data["exang"] = max(data.get("exang", 0), exercise)

    print("\n  [i] The model also expects technical test fields such as ECG,")
    print("      maximum heart rate, ST depression, slope, vessels and thal.")
    print("      These cannot be reliably determined from simple questions.")
    print("      CARDIO AI will use reference values from the project's processed")
    print("      dataset for those unavailable fields.")
    print("      This is an educational demonstration fallback — not a clinical measurement.\n")

    reference = load_reference_values()
    for key in FEATURE_ORDER:
        if key not in data or data[key] is None:
            data[key] = reference[key]

    return data, "basic"


def patient_profile():
    name = safe_input("\n  Patient Name (optional): ") or "Anonymous Patient"
    patient_id = "HD-" + datetime.now().strftime("%Y%m%d") + "-" + uuid.uuid4().hex[:4].upper()
    date = datetime.now().strftime("%d-%m-%Y %H:%M")
    return {"patient_id": patient_id, "name": name, "date": date, "data": {}}


def patient_contributions(model, data):
    """Compute patient-specific Logistic Regression transformed-feature contributions."""
    try:
        df = pd.DataFrame([data])[FEATURE_ORDER]
        preprocessor = model.named_steps["preprocessor"]
        classifier = model.named_steps["classifier"]
        transformed = preprocessor.transform(df)
        if hasattr(transformed, "toarray"):
            transformed = transformed.toarray()
        values = np.asarray(transformed[0], dtype=float)
        coefficients = np.asarray(classifier.coef_[0], dtype=float)
        names = preprocessor.get_feature_names_out()

        items = []
        for name, value in zip(names, values * coefficients):
            clean = name.replace("num__", "").replace("cat__", "")
            items.append({"label": clean, "contribution": float(value)})
        return sorted(items, key=lambda item: abs(item["contribution"]), reverse=True)
    except Exception:
        return []


def ensure_history():
    os.makedirs(HISTORY_DIR, exist_ok=True)
    if not os.path.exists(HISTORY_PATH):
        with open(HISTORY_PATH, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                "patient_id", "name", "date", "input_mode",
                *FEATURE_ORDER, "probability", "risk_tier", "prediction"
            ])


def save_history(profile, prob, tier, pred, input_mode):
    ensure_history()
    with open(HISTORY_PATH, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            profile["patient_id"], profile["name"], profile["date"], input_mode,
            *[profile["data"][feature] for feature in FEATURE_ORDER],
            f"{prob:.6f}", tier, pred
        ])


def history_view():
    section("PREDICTION HISTORY", "▤")
    if not os.path.exists(HISTORY_PATH):
        return history_empty()

    try:
        rows = list(csv.DictReader(open(HISTORY_PATH, encoding="utf-8")))
    except Exception as exc:
        error(f"Could not read prediction history: {exc}")
        return

    if not rows:
        return history_empty()

    print(f'  {"ID":<20}{"Date":<18}{"Probability":<14}{"Risk":<18}{"Mode":<10}')
    print("  " + "─" * 80)
    for row in rows[-12:]:
        try:
            probability = float(row.get("probability", 0))
        except (ValueError, TypeError):
            probability = 0.0
        print(
            f'  {row.get("patient_id",""):<20}'
            f'{row.get("date",""):<18}'
            f'{probability:<14.2%}'
            f'{row.get("risk_tier",""):<18}'
            f'{row.get("input_mode",""):<10}'
        )
    print(f"\n  Showing {min(12, len(rows))} most recent of {len(rows)} assessments.\n")


def save_report(profile, prob, tier, pred, recs, contributions, input_mode):
    os.makedirs(REPORT_DIR, exist_ok=True)
    path = os.path.join(REPORT_DIR, f'{profile["patient_id"]}_report.txt')

    with open(path, "w", encoding="utf-8") as f:
        f.write("CARDIO AI — HEART DISEASE RISK ASSESSMENT\n")
        f.write("=" * 70 + "\n")
        f.write(f'Patient ID: {profile["patient_id"]}\n')
        f.write(f'Patient Name: {profile["name"]}\n')
        f.write(f'Assessment: {profile["date"]}\n')
        f.write(f'Input Mode: {input_mode}\n\n')
        f.write(f"Predicted Risk Probability: {prob:.2%}\n")
        f.write(f"Risk Tier: {tier}\n")
        f.write(
            f'Model Classification: '
            f'{"Disease Risk Detected" if pred == 1 else "No Disease Risk Detected"}\n\n'
        )

        f.write("CLINICAL INPUTS\n" + "-" * 70 + "\n")
        for key in FEATURE_ORDER:
            f.write(f"{key}: {profile['data'][key]}\n")

        f.write("\nMODEL SIGNALS\n" + "-" * 70 + "\n")
        for item in contributions[:8]:
            f.write(f'{item["label"]}: {item["contribution"]:+.4f}\n')

        f.write("\nGENERAL EDUCATIONAL RECOMMENDATIONS\n" + "-" * 70 + "\n")
        for index, recommendation in enumerate(recs, 1):
            f.write(f"{index}. {recommendation}\n")

        if input_mode == "basic":
            f.write("\nBASIC-INPUT NOTE\n" + "-" * 70 + "\n")
            f.write(
                "Some unavailable clinical test fields were filled with reference "
                "values from the processed project dataset.\n"
            )
            f.write(
                "This fallback is for educational demonstration only and is not a "
                "substitute for measured clinical data.\n"
            )

        f.write("\nDISCLAIMER\n" + "-" * 70 + "\n")
        f.write(
            "This is an educational risk prediction system, not a medical diagnosis. "
            "Consult a qualified healthcare professional for medical advice.\n"
        )

    report_saved(path)
    return path



def open_file(path):
    """Open a generated visualization using the user's default desktop viewer."""
    try:
        if os.name == "nt":
            os.startfile(os.path.abspath(path))
        elif sys.platform == "darwin":
            subprocess.Popen(["open", os.path.abspath(path)])
        else:
            subprocess.Popen(["xdg-open", os.path.abspath(path)])
        print(f"  ✓ Opened → {path}\n")
    except Exception as exc:
        print(f"  ✓ Visualization saved → {path}")
        print(f"    Open it manually if your terminal cannot launch image files: {exc}\n")


def graph_menu():
    section("GRAPH ANALYSIS", "📊")
    print("  Explore visual patterns from the project's Cleveland heart-disease dataset.")
    print("  These are educational EDA/model-analysis visualizations, not clinical diagnoses.\n")
    options = [
        ("1", "Age Distribution", "How ages are distributed in the dataset"),
        ("2", "Cholesterol vs Disease", "Cholesterol distribution by target class"),
        ("3", "Resting BP vs Disease", "Resting blood pressure by target class"),
        ("4", "Maximum Heart Rate vs Disease", "Maximum heart rate by target class"),
        ("5", "Target Distribution", "Disease vs no-disease class balance"),
        ("6", "Correlation Heatmap", "Relationships among numeric clinical variables"),
        ("7", "Model Comparison", "Accuracy and other evaluation metrics"),
        ("8", "Feature Importance", "Model-derived feature coefficients"),
        ("0", "Back", "Return to the main menu"),
    ]
    for key, title, desc in options:
        print(f"  [{key}] {title:<30} {desc}")
    print()


def graph_analysis_view():
    """Open existing project graphs without changing the ML pipeline."""
    graph_dir = os.path.join(PROJECT_ROOT, "visualizations")
    available = {
        "1": ("Age Distribution", "age_distribution.png"),
        "2": ("Cholesterol vs Disease", "cholesterol_vs_target.png"),
        "3": ("Resting BP vs Disease", "resting_bp_vs_target.png"),
        "4": ("Maximum Heart Rate vs Disease", "max_heart_rate_vs_target.png"),
        "5": ("Target Distribution", "target_distribution.png"),
        "6": ("Correlation Heatmap", "correlation_heatmap.png"),
        "7": ("Model Comparison", "model_comparison.png"),
        "8": ("Feature Importance", "feature_importance.png"),
    }
    while True:
        graph_menu()
        choice = safe_input("  Select graph: ")
        if choice == "0":
            return
        if choice not in available:
            error("Choose one of the displayed graph options.")
            continue
        title, filename = available[choice]
        path = os.path.join(graph_dir, filename)
        if not os.path.exists(path):
            error(f"{filename} was not found. Run the project's EDA/model analysis first.")
            continue
        print(f"\n  📊 Preparing {title}...")
        pause = 0.15
        for _ in range(3):
            print("  " + "█" * 28, end="\r", flush=True)
            import time; time.sleep(pause)
            print("  " + "░" * 28, end="\r", flush=True)
            import time; time.sleep(pause)
        print("  " + " " * 35, end="\r")
        open_file(path)
        safe_input("  Press Enter to return to graph analysis...")


def ask_graph_analysis_after_prediction():
    section("VISUAL ANALYSIS", "📊")
    print("  Would you like to see graph analysis of the project data?")
    print("  [1] Yes — open the graph analysis menu")
    print("  [2] No — continue to assessment options")
    while True:
        choice = safe_input("  Select: ")
        if choice == "1":
            graph_analysis_view()
            return
        if choice == "2":
            return
        error("Choose 1 or 2.")

def run_assessment(model):
    profile = patient_profile()

    section("HOW WOULD YOU LIKE TO PROVIDE INFORMATION?", "♥")
    print("  [1] 📋 I HAVE MEDICAL REPORTS")
    print("      → Enter the actual report/test values.\n")
    print("  [2] 💬 I DON'T HAVE MEDICAL REPORTS")
    print("      → Answer simple everyday questions.\n")
    print("  [0] Return to menu\n")

    mode = safe_input("  Select an option: ")
    if mode == "0":
        return "menu"

    if mode == "1":
        data, input_mode = collect_report_data()
    elif mode == "2":
        data, input_mode = collect_basic_data()
    else:
        error("Invalid option. Choose 0, 1 or 2.")
        return "menu"

    profile["data"] = data

    patient_summary(profile)
    feature_scan()
    prediction_animation()

    frame = pd.DataFrame([data])[FEATURE_ORDER]
    probability = float(model.predict_proba(frame)[0, 1])
    prediction = int(model.predict(frame)[0])

    recommendations = generate_recommendations(data, probability)
    contributions = patient_contributions(model, data)
    tier = recommendations["risk_tier"]

    probability_animation(probability)
    risk_meter(probability, tier)
    result_panel(probability, tier, prediction, profile["patient_id"])
    recommendations_panel(recommendations["recommendations"])
    explainability_panel(contributions)
    ask_graph_analysis_after_prediction()

    save_history(profile, probability, tier, prediction, input_mode)

    while True:
        print("  [1] Save detailed report")
        print("  [2] New assessment")
        print("  [3] Main menu")
        action = safe_input("  Choose: ")

        if action == "1":
            save_report(
                profile, probability, tier, prediction,
                recommendations["recommendations"], contributions, input_mode
            )
        elif action == "2":
            return "new"
        elif action == "3":
            return "menu"
        else:
            error("Choose 1, 2 or 3.")


def model_performance_view():
    if not os.path.exists(PERF_PATH):
        return error("model_comparison.csv not found.")

    df = pd.read_csv(PERF_PATH)
    rows = df.to_dict("records")
    model_performance_table(rows)

    if "ROC-AUC" in df.columns and not df.empty:
        best = df.loc[df["ROC-AUC"].idxmax()]
        print(
            f'  Highest recorded ROC-AUC in this project table: '
            f'{best["Model"]} ({best["ROC-AUC"]:.2%})\n'
        )


def explainability_view(model):
    section("EXPLAINABLE AI", "🧠")
    print("  Run an assessment to see patient-specific model signals.")
    print("  The system calculates transformed-feature contributions using")
    print("  the final Logistic Regression pipeline.")
    print("\n  Important: model associations are not proof of medical causation.\n")


def diagnostics_view(model):
    tests = [
        ("Trained final model", os.path.exists(MODEL_PATH)),
        ("Processed dataset", os.path.exists(os.path.join(
            PROJECT_ROOT, "data", "processed", "heart_processed.csv"
        ))),
        ("Recommendation engine", os.path.exists(os.path.join(
            PROJECT_ROOT, "src", "recommendation_engine.py"
        ))),
        ("Terminal UI", os.path.exists(os.path.join(
            PROJECT_ROOT, "src", "terminal_ui.py"
        ))),
        ("Model comparison data", os.path.exists(PERF_PATH)),
        ("Feature importance data", os.path.exists(FEATURE_PATH)),
        ("Loaded model object", model is not None),
    ]
    diagnostics_panel(tests)
    print("  Smart questionnaire: report mode + simple-language mode")
    print("  Automated suite: src/test_system.py\n")


def main():
    boot_sequence()

    if not os.path.exists(MODEL_PATH):
        error(f"Model not found: {MODEL_PATH}")
        return

    model = joblib.load(MODEL_PATH)
    model_loaded("models/final_model.joblib")

    while True:
        menu()
        try:
            choice = safe_input("  Enter choice: ")
        except SystemExit:
            goodbye()
            break

        try:
            if choice == "1":
                while run_assessment(model) == "new":
                    pass
            elif choice == "2":
                history_view()
            elif choice == "3":
                model_performance_view()
            elif choice == "4":
                explainability_view(model)
            elif choice == "5":
                diagnostics_view(model)
            elif choice == "6":
                graph_analysis_view()
            elif choice == "7":
                about_panel()
            elif choice == "0":
                goodbye()
                break
            else:
                error("Invalid option. Choose 0–7.")

            if choice != "0":
                safe_input("  Press Enter to continue...")
        except SystemExit:
            goodbye()
            break
        except KeyboardInterrupt:
            print()
            goodbye()
            break
        except Exception as exc:
            error(f"Operation failed: {exc}")


if __name__ == "__main__":
    main()
