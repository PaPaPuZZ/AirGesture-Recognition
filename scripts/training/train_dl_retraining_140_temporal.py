import json,pickle,warnings
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,balanced_accuracy_score,classification_report,confusion_matrix,f1_score
from sklearn.model_selection import StratifiedKFold,cross_val_predict,cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

REPO_ROOT = Path(__file__).resolve().parents[2]
(REPO_ROOT / "models").mkdir(parents=True, exist_ok=True)
(REPO_ROOT / "results/letters").mkdir(parents=True, exist_ok=True)

DATA_FILE=REPO_ROOT / "data/letters/dl_retraining_140_combined_raw_data.csv"
MODEL_FILE=REPO_ROOT / "models/dl_retrained_140_temporal_best_model.pkl"
METADATA_FILE=REPO_ROOT / "results/letters/dl_retrained_140_temporal_metadata.json"
COMPARISON_FILE=REPO_ROOT / "results/letters/dl_retrained_140_temporal_model_comparison.csv"
CV_PREDICTIONS_FILE=REPO_ROOT / "results/letters/dl_retrained_140_temporal_cv_predictions.csv"
CONFUSION_FILE=REPO_ROOT / "results/letters/dl_retrained_140_temporal_confusion_matrix.csv"
SUMMARY_FILE=REPO_ROOT / "results/letters/dl_retrained_140_temporal_training_summary.txt"

CLASSES=list("DEFGHIJKL")
NUM_SAMPLES=100
NUM_CHANNELS=6
NUM_SEGMENTS=4
EXPECTED_SENSOR_COLUMNS=600
EXPECTED_FEATURES=216
RANDOM_STATE=42
N_SPLITS=5

def section(title):
    print("\n"+"="*78)
    print(title)
    print("="*78)

def extract_temporal_features(recordings):
    rows=[]
    for recording in recordings:
        features=[]
        for segment in np.array_split(recording,NUM_SEGMENTS,axis=0):
            for ch in range(segment.shape[1]):
                x=segment[:,ch]
                dx=np.diff(x)
                features.extend([
                    np.mean(x),np.std(x),np.min(x),np.max(x),np.ptp(x),
                    np.sqrt(np.mean(x**2)),np.mean(np.abs(dx)),
                    np.max(np.abs(dx)),np.sum(np.abs(dx))
                ])
        rows.append(features)
    return np.asarray(rows,dtype=float)

def macro_f1(y_true,y_pred):
    return f1_score(y_true,y_pred,labels=CLASSES,average="macro",zero_division=0)

section("D-L 140-RECORDING TEMPORAL MODEL TRAINING")
path=Path(DATA_FILE)
if not path.exists():
    raise SystemExit(f"ERROR: {DATA_FILE} not found.")
df=pd.read_csv(path)
print("Dataset:",DATA_FILE)
print("Shape:",df.shape)

if "actual" not in df.columns:
    raise SystemExit("ERROR: missing 'actual' column.")
df["actual"]=df["actual"].astype(str).str.strip().str.upper()

sensor_cols=[c for c in df.columns if c.startswith("sensor_")]
sensor_cols=sorted(sensor_cols,key=lambda c:int(c.split("_")[1]))
if len(df)!=140:
    raise SystemExit(f"ERROR: expected 140 rows, found {len(df)}.")
if len(sensor_cols)!=600:
    raise SystemExit(f"ERROR: expected 600 sensor columns, found {len(sensor_cols)}.")

section("DATASET VALIDATION")
expected={"D":20,"E":20,"F":20,"G":20,"H":20,"I":10,"J":10,"K":10,"L":10}
counts=df["actual"].value_counts().reindex(CLASSES,fill_value=0)
for c in CLASSES:
    print(f"{c}: {int(counts[c])}/{expected[c]}")
    if int(counts[c])!=expected[c]:
        raise SystemExit(f"ERROR: bad count for {c}.")

numeric=df[sensor_cols].apply(pd.to_numeric,errors="coerce")
missing=int(numeric.isna().sum().sum())
print("Missing sensor values:",missing)
if missing:
    raise SystemExit("ERROR: missing/non-numeric sensor values.")
values=numeric.to_numpy(dtype=float)
if not np.all(np.isfinite(values)):
    raise SystemExit("ERROR: non-finite sensor values.")
dups=int(numeric.duplicated(keep=False).sum())
print("Rows involved in exact 600-value duplicates:",dups)
if dups:
    raise SystemExit("ERROR: exact duplicate recordings detected.")

recordings=values.reshape(len(df),NUM_SAMPLES,NUM_CHANNELS)
y=df["actual"].to_numpy()
mag=np.sqrt(np.sum(recordings[:,:,:3]**2,axis=2))
print("Raw shape:",recordings.shape)
print(f"Acceleration magnitude mean: {mag.mean():.4f} m/s^2")
print(f"Acceleration magnitude median: {np.median(mag):.4f} m/s^2")

section("TEMPORAL FEATURE EXTRACTION")
X=extract_temporal_features(recordings)
print("Feature matrix:",X.shape)
if X.shape!=(140,216):
    raise SystemExit(f"ERROR: expected (140,216), got {X.shape}.")
if not np.all(np.isfinite(X)):
    raise SystemExit("ERROR: non-finite features.")

models={
"Logistic Regression":Pipeline([
    ("scaler",StandardScaler()),
    ("classifier",LogisticRegression(max_iter=20000,random_state=RANDOM_STATE))
]),
"RBF SVM":Pipeline([
    ("scaler",StandardScaler()),
    ("classifier",SVC(kernel="rbf",C=1.0,gamma="scale",probability=True,random_state=RANDOM_STATE))
]),
"Random Forest":RandomForestClassifier(
    n_estimators=600,class_weight="balanced",random_state=RANDOM_STATE,n_jobs=-1
)
}

cv=StratifiedKFold(n_splits=N_SPLITS,shuffle=True,random_state=RANDOM_STATE)
scoring={"accuracy":"accuracy","balanced_accuracy":"balanced_accuracy","macro_f1":"f1_macro"}
rows=[]
oof={}

section("5-FOLD STRATIFIED CROSS-VALIDATION")
for name,model in models.items():
    print("\nEvaluating:",name)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        scores=cross_validate(model,X,y,cv=cv,scoring=scoring,n_jobs=-1)
        pred=cross_val_predict(model,X,y,cv=cv,n_jobs=-1)
    oof[name]=pred
    row={
        "model":name,
        "mean_cv_accuracy":scores["test_accuracy"].mean(),
        "std_cv_accuracy":scores["test_accuracy"].std(),
        "mean_cv_balanced_accuracy":scores["test_balanced_accuracy"].mean(),
        "std_cv_balanced_accuracy":scores["test_balanced_accuracy"].std(),
        "mean_cv_macro_f1":scores["test_macro_f1"].mean(),
        "std_cv_macro_f1":scores["test_macro_f1"].std(),
        "oof_accuracy":accuracy_score(y,pred),
        "oof_balanced_accuracy":balanced_accuracy_score(y,pred),
        "oof_macro_f1":macro_f1(y,pred)
    }
    rows.append(row)
    print(f"Mean CV accuracy: {row['mean_cv_accuracy']:.4f}")
    print(f"Mean CV balanced accuracy: {row['mean_cv_balanced_accuracy']:.4f}")
    print(f"Mean CV macro F1: {row['mean_cv_macro_f1']:.4f}")
    print(f"OOF accuracy: {row['oof_accuracy']:.4f}")
    print(f"OOF macro F1: {row['oof_macro_f1']:.4f}")

comparison=pd.DataFrame(rows).sort_values(
    ["mean_cv_macro_f1","mean_cv_balanced_accuracy","mean_cv_accuracy"],
    ascending=False
).reset_index(drop=True)
comparison.to_csv(COMPARISON_FILE,index=False)

section("MODEL COMPARISON")
print(comparison.to_string(index=False))
selected_name=comparison.loc[0,"model"]
selected=clone(models[selected_name])
pred=oof[selected_name]
print("\nSelected candidate:",selected_name)

section("SELECTED MODEL OUT-OF-FOLD DIAGNOSTICS")
report=classification_report(y,pred,labels=CLASSES,digits=3,zero_division=0)
print(report)
cm=confusion_matrix(y,pred,labels=CLASSES)
cm_df=pd.DataFrame(cm,index=[f"actual_{c}" for c in CLASSES],columns=[f"pred_{c}" for c in CLASSES])
print(cm_df)
cm_df.to_csv(CONFUSION_FILE)

mis=int(np.sum(pred!=y))
print(f"\nMisclassifications: {mis}/{len(y)}")
print(f"OOF accuracy: {100*accuracy_score(y,pred):.2f}%")
print(f"OOF balanced accuracy: {100*balanced_accuracy_score(y,pred):.2f}%")
print(f"OOF macro F1: {100*macro_f1(y,pred):.2f}%")

pred_df=pd.DataFrame({"actual":y,"prediction":pred,"correct":pred==y})
if "trial_number" in df.columns:
    pred_df.insert(1,"trial_number",df["trial_number"].values)
if "source" in df.columns:
    pred_df.insert(2 if "trial_number" in pred_df.columns else 1,"source",df["source"].values)
pred_df.to_csv(CV_PREDICTIONS_FILE,index=False)

section("FIT SELECTED MODEL ON ALL 140 RECORDINGS")
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    selected.fit(X,y)
print("Selected:",selected_name)
print("Classes:",list(selected.classes_))
print("Features:",selected.n_features_in_)
if list(selected.classes_)!=CLASSES or selected.n_features_in_!=216:
    raise SystemExit("ERROR: fitted model validation failed.")

section("SAVE NEW MODEL")
with open(MODEL_FILE,"wb") as f:
    pickle.dump(selected,f)

metadata={
    "model_name":selected_name,
    "model_file":MODEL_FILE,
    "training_data_file":DATA_FILE,
    "num_recordings":140,
    "classes":CLASSES,
    "class_counts":{c:int(counts[c]) for c in CLASSES},
    "num_samples_per_recording":100,
    "num_channels":6,
    "num_segments":4,
    "num_temporal_features":216,
    "features_per_segment_channel":[
        "mean","std","min","max","peak_to_peak","rms",
        "mean_abs_diff","max_abs_diff","sum_abs_diff"
    ],
    "units":{"acceleration":"m/s^2","gyroscope":"rad/s"},
    "channel_order":["accel_x","accel_y","accel_z","gyro_x","gyro_y","gyro_z"],
    "cv":{"type":"StratifiedKFold","n_splits":5,"shuffle":True,"random_state":42},
    "selection_rule":"highest mean CV macro F1, then balanced accuracy, then accuracy",
    "selected_metrics":{k:(float(v) if isinstance(v,(float,np.floating)) else v)
                        for k,v in comparison.iloc[0].to_dict().items()}
}
with open(METADATA_FILE,"w") as f:
    json.dump(metadata,f,indent=2)

summary=[]
summary.append("D-L 140-RECORDING TEMPORAL TRAINING SUMMARY")
summary.append("="*60)
summary.append(f"Dataset: {DATA_FILE}")
summary.append("Class counts: "+str({c:int(counts[c]) for c in CLASSES}))
summary.append("Raw shape: "+str(recordings.shape))
summary.append("Feature shape: "+str(X.shape))
summary.append(f"Acceleration magnitude mean: {mag.mean():.6f} m/s^2")
summary.append("")
summary.append("MODEL COMPARISON")
summary.append(comparison.to_string(index=False))
summary.append("")
summary.append(f"SELECTED: {selected_name}")
summary.append(f"OOF accuracy: {accuracy_score(y,pred):.6f}")
summary.append(f"OOF balanced accuracy: {balanced_accuracy_score(y,pred):.6f}")
summary.append(f"OOF macro F1: {macro_f1(y,pred):.6f}")
summary.append("")
summary.append("CLASSIFICATION REPORT")
summary.append(report)
summary.append("")
summary.append("IMPORTANT:")
summary.append("This is a new candidate model.")
summary.append("The old frozen model was NOT overwritten.")
summary.append("Run a completely fresh live D-L holdout before replacing the old model.")
Path(SUMMARY_FILE).write_text("\n".join(summary))

print("Saved:")
for f in [MODEL_FILE,METADATA_FILE,COMPARISON_FILE,CV_PREDICTIONS_FILE,CONFUSION_FILE,SUMMARY_FILE]:
    print(" ",f)

section("TRAINING COMPLETE")
print("New candidate model:",MODEL_FILE)
print("Old frozen model was NOT overwritten.")
print("NEXT: run a completely fresh live 45-recording D-L holdout.")
