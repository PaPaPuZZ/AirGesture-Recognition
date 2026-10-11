#!/usr/bin/env python3
"""
TRAIN EXPANDED UNIFIED 0-9 CLASSIFIER

Input:
  data/numbers/numbers_0_9_expanded_training_raw_data.csv

Feature extractor:
  4 temporal segments x 6 channels x 9 statistics = 216 features

Candidates:
  Logistic Regression
  RBF SVM
  Random Forest

Primary current comparison:
  5-fold stratified CV + out-of-fold metrics.

Additional diagnostics:
  - GroupKFold by session_id when >= 2 sessions exist.
  - GroupKFold by participant_id when >= 2 participants exist.

As multiple people contribute data, participant-independent metrics become
the most important generalization estimate.
"""

import pickle
from pathlib import Path
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import (
    StratifiedKFold,
    GroupKFold,
    cross_val_predict,
    cross_validate,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

REPO_ROOT=Path(__file__).resolve().parents[2]
DATA_FILE=REPO_ROOT/"data/numbers/numbers_0_9_expanded_training_raw_data.csv"
MODEL_FILE=REPO_ROOT/"models/mpu6050_numbers_0_9_expanded_best_model.pkl"
RESULTS_DIR=REPO_ROOT/"results/numbers"
RESULTS_DIR.mkdir(parents=True,exist_ok=True)

COMPARISON_FILE=RESULTS_DIR/"numbers_0_9_expanded_model_comparison.csv"
OOF_FILE=RESULTS_DIR/"numbers_0_9_expanded_oof_predictions.csv"
CONFUSION_FILE=RESULTS_DIR/"numbers_0_9_expanded_oof_confusion_matrix.csv"
SUMMARY_FILE=RESULTS_DIR/"numbers_0_9_expanded_training_summary.txt"

NUM_SAMPLES=100
NUM_CHANNELS=6
NUM_SEGMENTS=4
SAMPLES_PER_SEGMENT=25
EXPECTED_FEATURES=216
CLASSES=[str(i) for i in range(10)]
RANDOM_STATE=42

FEATURE_DEFINITION=[
    "mean","std","min","max","range","rms",
    "avg_abs_change","max_abs_change","total_movement"
]

def normalize_label(v):
    return str(v).replace(".0","").strip()

def get_sensor_columns(df):
    cols=[c for c in df.columns if c.startswith("value_")]
    def key(c):
        a,b=c.replace("value_","").split("_")
        return int(a),int(b)
    return sorted(cols,key=key)

def extract_features(recording):
    features=[]
    for segment in np.array_split(recording,NUM_SEGMENTS,axis=0):
        for ch in range(NUM_CHANNELS):
            x=segment[:,ch]
            dx=np.diff(x)
            features.extend([
                np.mean(x),
                np.std(x),
                np.min(x),
                np.max(x),
                np.ptp(x),
                np.sqrt(np.mean(x**2)),
                np.mean(np.abs(dx)),
                np.max(np.abs(dx)),
                np.sum(np.abs(dx)),
            ])
    out=np.asarray(features,dtype=float)
    if out.shape!=(EXPECTED_FEATURES,):
        raise RuntimeError(f"Expected 216 features, got {out.shape}")
    return out

def build_models():
    return {
        "Logistic Regression":Pipeline([
            ("scaler",StandardScaler()),
            ("classifier",LogisticRegression(
                max_iter=7500,C=1.0,random_state=RANDOM_STATE
            ))
        ]),
        "RBF SVM":Pipeline([
            ("scaler",StandardScaler()),
            ("classifier",SVC(
                kernel="rbf",C=10.0,gamma="scale",
                probability=True,random_state=RANDOM_STATE
            ))
        ]),
        "Random Forest":RandomForestClassifier(
            n_estimators=1000,
            random_state=RANDOM_STATE,
            n_jobs=-1
        )
    }

def metrics(y,p):
    return {
        "accuracy":accuracy_score(y,p),
        "balanced_accuracy":balanced_accuracy_score(y,p),
        "macro_f1":f1_score(y,p,average="macro",zero_division=0),
    }

def grouped_diagnostic(model,X,y,groups,label):
    unique=np.unique(groups)
    if len(unique)<2:
        return None
    n_splits=min(5,len(unique))
    cv=GroupKFold(n_splits=n_splits)
    pred=cross_val_predict(
        model,X,y,groups=groups,cv=cv,n_jobs=-1
    )
    m=metrics(y,pred)
    return {
        "group_type":label,
        "groups":len(unique),
        "splits":n_splits,
        **m
    }

def main():
    print("="*72)
    print("EXPANDED UNIFIED 0-9 TRAINING")
    print("="*72)

    if not DATA_FILE.exists():
        raise FileNotFoundError(DATA_FILE)

    df=pd.read_csv(DATA_FILE)
    if "actual" not in df.columns:
        raise RuntimeError("Missing actual column")

    y=df["actual"].map(normalize_label).to_numpy(dtype=str)
    cols=get_sensor_columns(df)
    if len(cols)!=600:
        raise RuntimeError(f"Expected 600 sensor columns, found {len(cols)}")

    values=df[cols].apply(pd.to_numeric,errors="coerce").to_numpy(dtype=float)
    if not np.all(np.isfinite(values)):
        raise RuntimeError("Non-finite/missing sensor values")

    recordings=values.reshape(len(df),NUM_SAMPLES,NUM_CHANNELS)
    X=np.vstack([extract_features(r) for r in recordings])

    print("Rows:",len(df))
    print("Feature matrix:",X.shape)
    print("Class counts:")
    for digit in CLASSES:
        print(f"  {digit}: {int(np.sum(y==digit))}")

    cv=StratifiedKFold(
        n_splits=5,shuffle=True,random_state=RANDOM_STATE
    )

    rows=[]
    fitted={}
    preds={}

    for name,model in build_models().items():
        print()
        print("-"*72)
        print(name)
        print("-"*72)

        scoring={
            "accuracy":"accuracy",
            "balanced_accuracy":"balanced_accuracy",
            "macro_f1":"f1_macro",
        }

        scores=cross_validate(
            model,X,y,cv=cv,scoring=scoring,n_jobs=-1,error_score="raise"
        )
        pred=cross_val_predict(model,X,y,cv=cv,n_jobs=-1)
        m=metrics(y,pred)

        row={
            "model":name,
            "mean_cv_accuracy":float(np.mean(scores["test_accuracy"])),
            "mean_cv_balanced_accuracy":float(np.mean(scores["test_balanced_accuracy"])),
            "mean_cv_macro_f1":float(np.mean(scores["test_macro_f1"])),
            "oof_accuracy":m["accuracy"],
            "oof_balanced_accuracy":m["balanced_accuracy"],
            "oof_macro_f1":m["macro_f1"],
        }
        rows.append(row)
        preds[name]=pred

        print(f"OOF accuracy: {m['accuracy']:.4f}")
        print(f"OOF balanced accuracy: {m['balanced_accuracy']:.4f}")
        print(f"OOF macro F1: {m['macro_f1']:.4f}")

        model.fit(X,y)
        fitted[name]=model

    comparison=pd.DataFrame(rows).sort_values(
        ["oof_balanced_accuracy","oof_macro_f1","oof_accuracy"],
        ascending=False
    ).reset_index(drop=True)

    best_name=str(comparison.iloc[0]["model"])
    best_model=fitted[best_name]
    best_pred=np.asarray(preds[best_name],dtype=str)
    best_metrics=metrics(y,best_pred)

    print()
    print("="*72)
    print("FINAL MODEL COMPARISON")
    print("="*72)
    print(comparison.to_string(index=False))

    print()
    print("="*72)
    print("SELECTED MODEL")
    print("="*72)
    print("Best model:",best_name)
    print(f"OOF accuracy: {best_metrics['accuracy']:.4f}")
    print(f"OOF balanced accuracy: {best_metrics['balanced_accuracy']:.4f}")
    print(f"OOF macro F1: {best_metrics['macro_f1']:.4f}")
    print()
    print(classification_report(
        y,best_pred,labels=CLASSES,digits=4,zero_division=0
    ))

    group_results=[]

    if "session_id" in df.columns:
        r=grouped_diagnostic(
            build_models()[best_name],X,y,
            df["session_id"].astype(str).to_numpy(),
            "session_id"
        )
        if r:
            group_results.append(r)

    if "participant_id" in df.columns:
        r=grouped_diagnostic(
            build_models()[best_name],X,y,
            df["participant_id"].astype(str).to_numpy(),
            "participant_id"
        )
        if r:
            group_results.append(r)

    print()
    print("="*72)
    print("GROUP GENERALIZATION DIAGNOSTICS")
    print("="*72)

    if not group_results:
        print("Not enough independent groups yet.")
    else:
        for r in group_results:
            print(
                f"{r['group_type']}: groups={r['groups']} "
                f"accuracy={r['accuracy']:.4f} "
                f"balanced={r['balanced_accuracy']:.4f} "
                f"macro_f1={r['macro_f1']:.4f}"
            )

    comparison.to_csv(COMPARISON_FILE,index=False)

    oof=pd.DataFrame({
        "actual":y,
        "prediction":best_pred,
        "correct":y==best_pred,
    })
    for col in ["participant_id","session_id","dominant_hand","trial_number"]:
        if col in df.columns:
            oof[col]=df[col].to_numpy()
    oof.to_csv(OOF_FILE,index=False)

    cm=pd.DataFrame(
        confusion_matrix(y,best_pred,labels=CLASSES),
        index=[f"actual_{x}" for x in CLASSES],
        columns=[f"pred_{x}" for x in CLASSES]
    )
    cm.to_csv(CONFUSION_FILE)

    saved={
        "model":best_model,
        "model_name":best_name,
        "classes":CLASSES,
        "num_samples":NUM_SAMPLES,
        "num_channels":NUM_CHANNELS,
        "num_segments":NUM_SEGMENTS,
        "expected_features":EXPECTED_FEATURES,
        "feature_definition":FEATURE_DEFINITION,
        "training_file":str(DATA_FILE),
        "training_rows":len(df),
        "training_class_counts":{
            d:int(np.sum(y==d)) for d in CLASSES
        },
        "oof_accuracy":best_metrics["accuracy"],
        "oof_balanced_accuracy":best_metrics["balanced_accuracy"],
        "oof_macro_f1":best_metrics["macro_f1"],
        "group_diagnostics":group_results,
    }

    with open(MODEL_FILE,"wb") as handle:
        pickle.dump(saved,handle)

    lines=[
        "EXPANDED UNIFIED 0-9 TRAINING SUMMARY\n",
        "="*70+"\n",
        f"Rows: {len(df)}\n",
        f"Selected model: {best_name}\n",
        f"OOF accuracy: {best_metrics['accuracy']:.4f}\n",
        f"OOF balanced accuracy: {best_metrics['balanced_accuracy']:.4f}\n",
        f"OOF macro F1: {best_metrics['macro_f1']:.4f}\n\n",
        "MODEL COMPARISON\n",
        comparison.to_string(index=False)+"\n\n",
        "GROUP DIAGNOSTICS\n",
    ]
    if group_results:
        for r in group_results:
            lines.append(str(r)+"\n")
    else:
        lines.append("Not enough independent groups yet.\n")

    SUMMARY_FILE.write_text("".join(lines))

    print()
    print("Saved model:",MODEL_FILE)
    print("Saved comparison:",COMPARISON_FILE)
    print("Saved OOF:",OOF_FILE)
    print("Saved confusion matrix:",CONFUSION_FILE)
    print("Saved summary:",SUMMARY_FILE)

if __name__=="__main__":
    main()
