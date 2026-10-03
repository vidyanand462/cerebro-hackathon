import sys
import os
import pandas as pd
import numpy as np

# Set standard output to handle utf-8 if possible
if sys.platform.startswith("win"):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

from src.preprocessing import DataPreprocessor
from src.feature_engineering import FeatureEngineer
from src.onset_detector import OnsetDetector
from src.model import ProcrastinationModelPipeline
from src.explanation import ExplanationEngine

def main():
    print("=========================================================")
    print("  ProcrastiGuard AI - Academic Procrastination Detector  ")
    print("=========================================================")

    # 1. Load and clean raw dataset
    raw_path = "data/raw/student_academic_activities.csv"
    print(f"\n[1/5] Loading data from {raw_path}...")
    preprocessor = DataPreprocessor(raw_path)
    df_clean = preprocessor.load_data()
    print(f"      Loaded {len(df_clean)} records for {df_clean['student_id'].nunique()} students.")

    # 2. Extract Temporal and Sequential Behavioral Features
    print("\n[2/5] Extracting sequential behavioral features...")
    engineer = FeatureEngineer()
    df_features = engineer.compute_sequential_features(df_clean)
    
    # Save processed features
    os.makedirs("data/processed", exist_ok=True)
    df_features.to_csv("data/processed/featured_academic_activities.csv", index=False)
    print("      Features calculated: delay ratios, deadline pressure, velocity, last-minute severity.")

    # 3. Onset Change-Point Detection
    print("\n[3/5] Executing change-point onset detection across student sequences...")
    detector = OnsetDetector()
    onsets_df = detector.detect_all_students(df_features)
    print("      Onset Trajectory Breakdown:")
    for traj, count in onsets_df['trajectory_type'].value_counts().items():
        print(f"        - {traj}: {count} students ({count/len(onsets_df)*100:.1f}%)")

    # 4. Train Model Pipeline
    print("\n[4/5] Training Machine Learning Classifier (Random Forest)...")
    pipeline = ProcrastinationModelPipeline(model_dir="models")
    metrics = pipeline.train(df_features)
    print(f"      Model Accuracy: {metrics['accuracy']*100:.2f}%")
    print(f"      5-Fold Cross-Validation Weighted F1: {metrics['cv_f1_mean']*100:.2f}% (+/- {metrics['cv_f1_std']*100:.2f}%)")
    print("      Top 5 Most Influential Features:")
    for feat, imp in list(metrics['feature_importances'].items())[:5]:
        print(f"        - {feat}: {imp*100:.2f}%")

    # 5. Test Explanation Engine
    print("\n[5/5] Testing Grounded Explanation Generation on Sample Student (S001)...")
    explainer = ExplanationEngine(model_pipeline=pipeline)
    sample_student = df_features[df_features['student_id'] == 'S001']
    sample_onset = detector.analyze_student_sequence(sample_student)
    sample_pred = pipeline.predict(sample_student)[-1]
    
    explanation = explainer.generate_explanation(sample_student, onset_info=sample_onset, pred_info=sample_pred)
    print(f"      Risk Level: {explanation['risk_label']}")
    print(f"      Headline: {explanation['headline']}")
    print(f"      Onset Event: {explanation['onset_summary']}")
    print("      Evidence Grounding:")
    for ev in explanation['evidence']:
        print(f"        {ev}")

    print("\n=========================================================")
    print("   [OK] SYSTEM PIPELINE VERIFIED AND ARTIFACTS PERSISTED   ")
    print("=========================================================")

if __name__ == "__main__":
    main()
