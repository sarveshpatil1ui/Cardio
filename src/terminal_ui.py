"""Rich animated terminal interface for the Heart Disease Risk Prediction project."""
from __future__ import annotations
import os, sys, time, textwrap
from typing import Iterable

RESET='\033[0m'; BOLD='\033[1m'; DIM='\033[2m'
PURPLE='\033[95m'; CYAN='\033[96m'; GREEN='\033[92m'; YELLOW='\033[93m'; RED='\033[91m'; BLUE='\033[94m'; WHITE='\033[97m'; GRAY='\033[90m'
COLOR_ENABLED = sys.stdout.isatty() or os.environ.get('TERM_PROGRAM') == 'vscode'
FAST = os.environ.get('CARDIO_FAST_MODE') == '1' or not COLOR_ENABLED

def c(text, color=''):
    return f'{color}{text}{RESET}' if COLOR_ENABLED and color else text

def pause(seconds=.15):
    if not FAST: time.sleep(seconds)

def clear_screen():
    if COLOR_ENABLED: print('\033[2J\033[H', end='')
    else: print('\n' * 2)

def hide_cursor():
    if COLOR_ENABLED: print('\033[?25l', end='')

def show_cursor():
    if COLOR_ENABLED: print('\033[?25h', end='')

LOGO=[
'  ██████╗ █████╗ ██████╗ ██████╗ ██╗ ██████╗',
' ██╔════╝██╔══██╗██╔══██╗██╔══██╗██║██╔═══██╗',
' ██║     ███████║██████╔╝██║  ██║██║██║   ██║',
' ██║     ██╔══██║██╔══██╗██║  ██║██║██║   ██║',
' ╚██████╗██║  ██║██║  ██║██████╔╝██║╚██████╔╝',
'  ╚═════╝╚═╝  ╚═╝╚═╝  ╚═╝╚═════╝ ╚═╝ ╚═════╝',
]

def show_logo(animated=True):
    clear_screen(); print()
    cols=[PURPLE,CYAN,GREEN,PURPLE,CYAN,GREEN]
    for line,col in zip(LOGO,cols):
        print(c(line,col)); pause(.06 if animated else 0)
    print(c('\n          ♥  CARDIO AI  •  RISK PREDICTION ENGINE', BOLD+CYAN))
    print(c('             MACHINE LEARNING • ANALYTICS • INSIGHTS', DIM+WHITE)); print()

def heartbeat(cycles=2):
    frames=['___/\\_________/\\__________/\\________________',
            '__/  \\_______/  \\________/  \\_______________',
            '_/    \\_____/    \\______/    \\____/\\________',
            '/       \\__/       \\____/       \\/  \\_______']
    for _ in range(cycles):
        for frame in frames:
            print('\r'+c('♥ ',RED)+c(frame,GREEN),end='',flush=True); pause(.06)
    print('\r'+' '*65+'\r',end='')

def section(title, icon='◆', width=72):
    title=f' {icon} {title} '
    print(c('╭'+title+'─'*max(1,width-len(title))+'╮',CYAN))
    print(c('╰'+'─'*width+'╯',CYAN))

def progress_bar(label, duration=.45, color=CYAN, width=34):
    print(f'  {label}')
    steps=24
    for i in range(steps+1):
        filled=int(width*i/steps); bar='█'*filled+'░'*(width-filled)
        print(f'  [{c(bar,color)}] {i*100//steps:3d}%',end='\r',flush=True); pause(duration/steps)
    print()

def boot_sequence():
    try: hide_cursor(); show_logo(); heartbeat(2)
    except Exception: pass
    section('SYSTEM INITIALIZATION','⚙')
    for label in ['Loading clinical data engine','Loading preprocessing pipeline','Loading trained classifier','Loading recommendation engine','Preparing analytics modules']:
        progress_bar(label,.25,GREEN)
    print(c('\n  ✓ CARDIO AI ENGINE → ONLINE\n',BOLD+GREEN))
    pause(.3); show_cursor()

def model_loaded(path):
    print(c(f'  ✓ Model loaded: {path}',GREEN)); print(c('  ✓ Final classifier: Logistic Regression',CYAN)); print()

def menu():
    section('CARDIO AI CONTROL CENTER','♥')
    rows=[
      ('1','New Risk Assessment','Enter a patient profile and run prediction'),
      ('2','Prediction History','View saved local assessment history'),
      ('3','Model Performance','Compare the five evaluated ML models'),
      ('4','Explainable AI','Inspect model feature signals'),
      ('5','System Diagnostics','Check project components and test status'),
      ('6','About Project','View project architecture and scope'),
      ('0','Exit','Close CARDIO AI'),]
    for n,title,desc in rows:
        print(f'  {c("["+n+"]",BOLD+PURPLE)} {c(title,WHITE):<28} {c(desc,DIM)}')
    print()

def input_section(): section('PATIENT CLINICAL DATA','♥')

def feature_scan():
    section('CLINICAL FEATURE SCAN','⌁')
    for label in ['Age','Blood pressure','Cholesterol','Chest pain','ECG','Maximum heart rate','Exercise response','ST analysis']:
        print(f'  {c("✓",GREEN)} Scanning {label:<24}',end='\r'); pause(.08)
    print('  '+c('✓ Clinical feature scan complete','GREEN')+' '*10); print()

def prediction_animation():
    section('AI INFERENCE ENGINE','◆')
    for label in ['Transforming feature vector','Running Logistic Regression classifier','Calculating probability','Building risk assessment']:
        progress_bar(label,.22,PURPLE)
    print()

def probability_animation(probability):
    section('RISK PROBABILITY','♥'); target=max(0,min(1,float(probability)))
    for i in range(26):
        value=target*i/25
        print(f'\r                 {c(f"{value:6.2%}",BOLD+CYAN)}',end='',flush=True); pause(.025)
    print('\n')

def risk_meter(probability,tier):
    p=max(0,min(1,float(probability))); width=50; marker=min(width-1,int(p*width))
    color=RED if 'High' in tier else YELLOW if 'Moderate' in tier else GREEN
    bar='━'*marker+'●'+'─'*(width-marker-1)
    section('CARDIOVASCULAR RISK METER','◈')
    print(c('  LOW                 MODERATE                    HIGH',DIM+WHITE))
    print('  '+c(bar,color)); print(c('  0%                 30%                       70%        100%',DIM))
    print(c(f'\n                    {p:.2%}  →  {tier.upper()}',BOLD+color)); print()

def result_panel(probability,tier,pred_class,patient_id=None):
    color=RED if 'High' in tier else YELLOW if 'Moderate' in tier else GREEN
    symbol='▲' if 'High' in tier else '◆' if 'Moderate' in tier else '✓'
    line='═'*70
    print(c('╔'+line+'╗',color)); print(c('║'+ ' '*70 +'║',color))
    print(c('║'+'                 ANALYSIS COMPLETE'.center(70)+'║',BOLD+color))
    print(c('║'+' '*70+'║',color)); print(c('║'+'             HEART DISEASE RISK'.center(70)+'║',color))
    print(c('║'+' '*70+'║',color)); print(c(f'║{probability:>35.2%}{" ":>35}║',BOLD+color))
    print(c('║'+' '*70+'║',color)); print(c(f'║{(symbol+"  "+tier.upper()):^70}║',BOLD+color))
    if patient_id: print(c(f'║{("Patient ID: "+patient_id):^70}║',color))
    print(c('║'+' '*70+'║',color)); print(c('╚'+line+'╝',color))
    outcome='Disease Risk Detected' if pred_class==1 else 'No Disease Risk Detected'
    print(c(f'\n  MODEL CLASSIFICATION → {outcome}',WHITE)); print()

def recommendations_panel(recommendations):
    section('PERSONALIZED HEALTH INSIGHTS','✦')
    for i,rec in enumerate(recommendations,1):
        lines=textwrap.wrap(rec,width=64) or ['']
        print(f'  {c("✓",GREEN)} {i}. {lines[0]}')
        for x in lines[1:]: print('      '+x)
        pause(.08)
    print()

def patient_summary(profile):
    section('PATIENT PROFILE','◉')
    print(f'  Patient ID       : {c(profile["patient_id"],BOLD+CYAN)}')
    print(f'  Patient Name     : {profile["name"]}')
    print(f'  Assessment Date  : {profile["date"]}')
    print(f'  Age              : {profile["data"]["age"]:.0f} years')
    print()

def explainability_panel(contributions):
    section('EXPLAINABLE AI • FEATURE SIGNALS','🧠')
    print(c('  These are model-derived signals, not causal explanations or a diagnosis.\n',DIM+YELLOW))
    max_abs=max([abs(x['contribution']) for x in contributions] or [1])
    for item in contributions[:8]:
        v=item['contribution']; length=max(1,int(abs(v)/max_abs*22)); bar='█'*length
        col=RED if v>0 else GREEN
        direction='↑' if v>0 else '↓'
        print(f'  {item["label"]:<25} {c(bar,col):<24} {direction} {v:+.3f}')
    print()

def model_performance_table(rows):
    section('MODEL PERFORMANCE','▣')
    print('  '+c(f'{"Model":<21}{"Accuracy":>10}{"Precision":>11}{"Recall":>10}{"F1":>10}{"ROC-AUC":>11}',BOLD+WHITE))
    print('  '+'─'*73)
    for r in rows:
        print(f'  {r["Model"]:<21}{r["Accuracy"]:>10.2%}{r["Precision"]:>11.2%}{r["Recall"]:>10.2%}{r["F1-Score"]:>10.2%}{r["ROC-AUC"]:>11.2%}')
    print()

def diagnostics_panel(items):
    section('SYSTEM DIAGNOSTICS','⚙')
    for name,status in items:
        symbol='✓' if status else '✗'; col=GREEN if status else RED
        print(f'  {c(symbol,col)} {name:<38} {c("ONLINE" if status else "CHECK REQUIRED",col)}')
    print()

def about_panel():
    section('ABOUT CARDIO AI','♥')
    print('  Heart Disease Risk Prediction & Personalized Recommendation System')
    print('  Dataset        : UCI Heart Disease — Cleveland processed dataset')
    print('  Records        : 303')
    print('  Input features : 13')
    print('  Models         : Logistic Regression, Decision Tree, Random Forest, SVM, KNN')
    print('  Purpose        : Educational / academic risk screening')
    print(c('\n  Medical note: predictions are not medical diagnoses.',YELLOW)); print()

def report_saved(path): print(c(f'  ✓ Report saved → {path}\n',GREEN))
def history_empty(): print(c('  No prediction history found yet. Run a new assessment first.\n',YELLOW))
def disclaimer(text):
    section('IMPORTANT','!'); print(c('  '+text,YELLOW)); print()
def error(message): print(c(f'\n  ✗ {message}',RED))
def goodbye():
    print(); section('SESSION TERMINATED','♥'); heartbeat(1); print(c('  CARDIO AI ENGINE → OFFLINE',DIM+CYAN)); print(c('  Stay informed. Stay healthy. ♥',GREEN)); print()
