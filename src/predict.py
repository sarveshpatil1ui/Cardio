"""Interactive CARDIO AI application with prediction, analytics and history."""
from __future__ import annotations
import os, sys, csv, uuid
from datetime import datetime
import joblib, pandas as pd

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from recommendation_engine import generate_recommendations
from terminal_ui import *

FEATURE_ORDER=['age','sex','cp','trestbps','chol','fbs','restecg','thalach','exang','oldpeak','slope','ca','thal']
PROJECT_ROOT=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH=os.path.join(PROJECT_ROOT,'models','final_model.joblib')
HISTORY_DIR=os.path.join(PROJECT_ROOT,'data','history')
HISTORY_PATH=os.path.join(HISTORY_DIR,'prediction_history.csv')
REPORT_DIR=os.path.join(PROJECT_ROOT,'reports')
PERF_PATH=os.path.join(PROJECT_ROOT,'model_comparison.csv')
FEATURE_PATH=os.path.join(PROJECT_ROOT,'feature_importance.csv')

CHOICES={
'sex':{0:'Female',1:'Male'}, 'cp':{1:'Typical Angina',2:'Atypical Angina',3:'Non-anginal Pain',4:'Asymptomatic'},
'fbs':{0:'False/No',1:'True/Yes'}, 'restecg':{0:'Normal',1:'ST-T Wave Abnormality',2:'Left Ventricular Hypertrophy'},
'exang':{0:'No',1:'Yes'}, 'slope':{1:'Upsloping',2:'Flat',3:'Downsloping'}, 'ca':{0:'0',1:'1',2:'2',3:'3'},
'thal':{3:'Normal',6:'Fixed Defect',7:'Reversable Defect'}}


def safe_float(prompt,min_val=None,max_val=None):
    while True:
        try:
            v=float(input(prompt).strip())
            if min_val is not None and v<min_val: raise ValueError(f'Must be at least {min_val}.')
            if max_val is not None and v>max_val: raise ValueError(f'Must be at most {max_val}.')
            return v
        except ValueError as e: error(str(e) if str(e) else 'Enter a valid number.')
        except (KeyboardInterrupt,EOFError): raise SystemExit

def choice(prompt, options):
    desc=' / '.join(f'{k}:{v}' for k,v in options.items())
    while True:
        try:
            v=int(float(input(f'{prompt} ({desc}): ').strip()))
            if v in options:return float(v)
            error(f'Valid choices: {list(options)}')
        except ValueError:error('Enter one of the listed numbers.')
        except (KeyboardInterrupt,EOFError):raise SystemExit

def collect_patient_data():
    input_section(); data={}
    data['age']=safe_float('  [01/13] Age (years): ',1,120)
    data['sex']=choice('  [02/13] Sex',CHOICES['sex']); data['cp']=choice('  [03/13] Chest Pain Type',CHOICES['cp'])
    data['trestbps']=safe_float('  [04/13] Resting Blood Pressure (mm Hg): ',50,260)
    data['chol']=safe_float('  [05/13] Serum Cholesterol (mg/dl): ',80,700)
    data['fbs']=choice('  [06/13] Fasting Blood Sugar > 120 mg/dl',CHOICES['fbs'])
    data['restecg']=choice('  [07/13] Resting ECG Results',CHOICES['restecg'])
    data['thalach']=safe_float('  [08/13] Maximum Heart Rate Achieved (bpm): ',50,230)
    data['exang']=choice('  [09/13] Exercise-Induced Angina',CHOICES['exang'])
    data['oldpeak']=safe_float('  [10/13] ST Depression (oldpeak): ',0,10)
    data['slope']=choice('  [11/13] Slope of Peak Exercise ST Segment',CHOICES['slope'])
    data['ca']=choice('  [12/13] Number of Major Vessels Colored by Fluoroscopy',CHOICES['ca'])
    data['thal']=choice('  [13/13] Thalassemia',CHOICES['thal'])
    print(); return data

def new_profile():
    section('PATIENT PROFILE','◉')
    name=input('  Patient Name (optional): ').strip() or 'Anonymous Patient'
    pid='HD-'+datetime.now().strftime('%Y%m%d')+'-'+uuid.uuid4().hex[:4].upper()
    return {'patient_id':pid,'name':name,'date':datetime.now().strftime('%d-%m-%Y %H:%M'),'data':collect_patient_data()}

def ensure_history():
    os.makedirs(HISTORY_DIR,exist_ok=True)
    if not os.path.exists(HISTORY_PATH):
        with open(HISTORY_PATH,'w',newline='',encoding='utf-8') as f:
            csv.writer(f).writerow(['patient_id','name','date','probability','risk_tier','prediction'])

def save_history(profile,prob,tier,pred):
    ensure_history()
    with open(HISTORY_PATH,'a',newline='',encoding='utf-8') as f:
        csv.writer(f).writerow([profile['patient_id'],profile['name'],profile['date'],f'{prob:.6f}',tier,pred])

def history_view():
    section('PREDICTION HISTORY','▤')
    if not os.path.exists(HISTORY_PATH): return history_empty()
    rows=list(csv.DictReader(open(HISTORY_PATH,encoding='utf-8')))
    if not rows:return history_empty()
    print(f'  {"ID":<20}{"Date":<18}{"Probability":<14}{"Risk":<18}')
    print('  '+'─'*70)
    for r in rows[-12:]: print(f'  {r["patient_id"]:<20}{r["date"]:<18}{float(r["probability"]):<14.2%}{r["risk_tier"]:<18}')
    print(f'\n  Showing {min(12,len(rows))} most recent of {len(rows)} assessments.\n')

def patient_contributions(model, data):
    """Compute logistic transformed-feature contributions for the current patient."""
    try:
        import numpy as np
        df=pd.DataFrame([data])[FEATURE_ORDER]
        pre=model.named_steps['preprocessor']; clf=model.named_steps['classifier']
        transformed=pre.transform(df)
        names=pre.get_feature_names_out()
        vals=np.asarray(transformed[0],dtype=float)
        coef=np.asarray(clf.coef_[0],dtype=float)
        contrib=vals*coef
        items=[]
        for name,v in zip(names,contrib):
            clean=name.replace('num__','').replace('cat__','')
            items.append({'label':clean,'contribution':float(v)})
        return sorted(items,key=lambda x:abs(x['contribution']),reverse=True)
    except Exception:
        return []

def explainability_view(model):
    section('EXPLAINABLE AI','🧠')
    print('  Run an assessment first to see patient-specific model signals.')
    print('  The feature analysis file also contains the trained Logistic Regression coefficients.\n')

def model_performance_view():
    if not os.path.exists(PERF_PATH): return error('model_comparison.csv not found.')
    df=pd.read_csv(PERF_PATH)
    rows=df.to_dict('records'); model_performance_table(rows)
    best=df.loc[df['ROC-AUC'].idxmax()]
    print(c(f'  Highest recorded ROC-AUC in this project table: {best["Model"]} ({best["ROC-AUC"]:.2%})',CYAN)); print()

def diagnostics_view():
    tests=[
      ('Trained final model',os.path.exists(MODEL_PATH)),('Processed dataset',os.path.exists(os.path.join(PROJECT_ROOT,'data','processed','heart_processed.csv'))),
      ('Recommendation engine',os.path.exists(os.path.join(PROJECT_ROOT,'src','recommendation_engine.py'))),('Terminal UI',os.path.exists(os.path.join(PROJECT_ROOT,'src','terminal_ui.py'))),
      ('Model comparison data',os.path.exists(PERF_PATH)),('Feature importance data',os.path.exists(FEATURE_PATH)),
    ]
    diagnostics_panel(tests)
    print('  Automated suite: src/test_system.py (run manually for full verification).\n')

def about_view(): about_panel()

def save_report(profile,prob,tier,pred,recs,contrib):
    os.makedirs(REPORT_DIR,exist_ok=True)
    path=os.path.join(REPORT_DIR,f'{profile["patient_id"]}_report.txt')
    with open(path,'w',encoding='utf-8') as f:
        f.write('CARDIO AI — HEART DISEASE RISK ASSESSMENT\n'+'='*64+'\n')
        f.write(f'Patient ID: {profile["patient_id"]}\nPatient Name: {profile["name"]}\nAssessment: {profile["date"]}\n\n')
        f.write(f'Predicted Risk Probability: {prob:.2%}\nRisk Tier: {tier}\nModel Classification: {"Disease Risk Detected" if pred==1 else "No Disease Risk Detected"}\n\n')
        f.write('CLINICAL INPUTS\n'+'-'*64+'\n')
        for k,v in profile['data'].items(): f.write(f'{k}: {v}\n')
        f.write('\nMODEL SIGNALS\n'+'-'*64+'\n')
        for x in contrib[:8]: f.write(f'{x["label"]}: {x["contribution"]:+.4f}\n')
        f.write('\nGENERAL EDUCATIONAL RECOMMENDATIONS\n'+'-'*64+'\n')
        for i,r in enumerate(recs,1):f.write(f'{i}. {r}\n')
        f.write('\nDISCLAIMER\n'+recs and '' or '')
        f.write('This is an educational risk prediction system, not a medical diagnosis. Consult a qualified healthcare professional for medical advice.\n')
    report_saved(path); return path

def run_assessment(model):
    profile=new_profile(); patient_summary(profile)
    feature_scan(); prediction_animation()
    df=pd.DataFrame([profile['data']])[FEATURE_ORDER]
    prob=float(model.predict_proba(df)[0,1]); pred=int(model.predict(df)[0])
    rec=generate_recommendations(profile['data'],prob); contrib=patient_contributions(model,profile['data'])
    probability_animation(prob); risk_meter(prob,rec['risk_tier']); result_panel(prob,rec['risk_tier'],pred,profile['patient_id'])
    recommendations_panel(rec['recommendations']); explainability_panel(contrib)
    save_history(profile,prob,rec['risk_tier'],pred)
    while True:
        print('  [1] Save detailed report   [2] New assessment   [3] Main menu')
        choice=input('  Choose: ').strip()
        if choice=='1': save_report(profile,prob,rec['risk_tier'],pred,rec['recommendations'],contrib)
        elif choice=='2': return 'new'
        elif choice=='3': return 'menu'
        else: error('Choose 1, 2 or 3.')

def main():
    boot_sequence()
    if not os.path.exists(MODEL_PATH): error(f'Model not found: {MODEL_PATH}'); return
    model=joblib.load(MODEL_PATH); model_loaded('models/final_model.joblib')
    while True:
        menu()
        try: ch=input('  Enter choice: ').strip()
        except (KeyboardInterrupt,EOFError): goodbye(); break
        try:
            if ch=='1':
                while run_assessment(model)=='new': pass
            elif ch=='2': history_view()
            elif ch=='3': model_performance_view()
            elif ch=='4': explainability_view(model)
            elif ch=='5': diagnostics_view()
            elif ch=='6': about_view()
            elif ch=='0': goodbye(); break
            else: error('Invalid option. Choose 0–6.')
            if ch!='0': input('  Press Enter to continue...')
        except SystemExit: goodbye(); break
        except KeyboardInterrupt: print(); goodbye(); break
        except Exception as exc: error(f'Operation failed: {exc}')

if __name__=='__main__': main()
