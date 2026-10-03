import pandas as pd
import numpy as np
import pickle
import json
import os
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, f1_score, confusion_matrix

class ProcrastinationModelPipeline:
    """
    ML training and inference pipeline for academic procrastination pattern classification.
    """
    def __init__(self, model_dir="models"):
        self.model_dir = model_dir
        os.makedirs(self.model_dir, exist_ok=True)
        self.classifier = None
        self.feature_columns = None
        self.target_encoder = None
        self.task_type_encoder = None
        self.model_metadata = {}

    def prepare_dataset(self, df: pd.DataFrame):
        df = df.copy()

        # Target encoding
        if 'procrastination_label' in df.columns:
            if self.target_encoder is None:
                self.target_encoder = LabelEncoder()
                df['target'] = self.target_encoder.fit_transform(df['procrastination_label'])
            else:
                df['target'] = self.target_encoder.transform(df['procrastination_label'])
        else:
            df['target'] = None

        # Encode task_type
        if 'task_type' in df.columns:
            if self.task_type_encoder is None:
                self.task_type_encoder = LabelEncoder()
                df['task_type_encoded'] = self.task_type_encoder.fit_transform(df['task_type'].astype(str))
            else:
                # Handle unseen task types gracefully
                known = set(self.task_type_encoder.classes_)
                df['task_type_clean'] = df['task_type'].astype(str).apply(lambda x: x if x in known else self.task_type_encoder.classes_[0])
                df['task_type_encoded'] = self.task_type_encoder.transform(df['task_type_clean'])

        # Numeric features list
        candidate_features = [
            'task_number',
            'duration_minutes',
            'hours_before_deadline',
            'submission_delay_hours',
            'last_minute_flag',
            'late_submission_flag',
            'inactivity_gap_hours',
            'survey_behavior_score',
            'start_delay_hours',
            'delay_ratio',
            'deadline_pressure',
            'critical_last_minute',
            'inactivity_ratio',
            'burst_intensity',
            'delta_deadline_distance',
            'cum_last_minute_count',
            'cum_late_count',
            'expanding_avg_deadline_distance',
            'expanding_avg_delay_ratio',
            'expanding_last_minute_rate',
            'deterioration_rate',
            'task_type_encoded'
        ]

        self.feature_columns = [col for col in candidate_features if col in df.columns]
        X = df[self.feature_columns].fillna(0.0)
        y = df['target'] if 'target' in df.columns else None

        return X, y

    def train(self, df: pd.DataFrame, test_size=0.2, random_state=42):
        X, y = self.prepare_dataset(df)
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )

        # Train primary Random Forest Classifier
        self.classifier = RandomForestClassifier(
            n_estimators=150,
            max_depth=8,
            min_samples_split=4,
            random_state=random_state,
            class_weight='balanced'
        )
        self.classifier.fit(X_train, y_train)

        # Validation metrics
        y_pred = self.classifier.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='weighted')
        report = classification_report(y_test, y_pred, target_names=self.target_encoder.classes_, output_dict=True)

        # Cross validation
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
        cv_scores = cross_val_score(self.classifier, X, y, cv=cv, scoring='f1_weighted')

        # Feature importances
        feature_importances = dict(zip(self.feature_columns, self.classifier.feature_importances_.tolist()))
        sorted_fi = dict(sorted(feature_importances.items(), key=lambda item: item[1], reverse=True))

        self.model_metadata = {
            "model_type": "RandomForestClassifier",
            "accuracy": round(float(acc), 4),
            "weighted_f1": round(float(f1), 4),
            "cv_f1_mean": round(float(cv_scores.mean()), 4),
            "cv_f1_std": round(float(cv_scores.std()), 4),
            "classes": self.target_encoder.classes_.tolist(),
            "feature_importances": sorted_fi,
            "feature_columns": self.feature_columns,
            "classification_report": report
        }

        self.save_model()
        return self.model_metadata

    def predict(self, df: pd.DataFrame):
        """
        Predicts procrastination class and class probabilities for input dataframe.
        """
        X, _ = self.prepare_dataset(df)
        preds = self.classifier.predict(X)
        probs = self.classifier.predict_proba(X)
        pred_labels = self.target_encoder.inverse_transform(preds)

        results = []
        for i in range(len(df)):
            class_probs = {cls: round(float(probs[i][idx]), 3) for idx, cls in enumerate(self.target_encoder.classes_)}
            results.append({
                "predicted_label": pred_labels[i],
                "confidence": round(float(np.max(probs[i])), 3),
                "probabilities": class_probs
            })
        return results

    def save_model(self):
        model_path = os.path.join(self.model_dir, "procrastination_classifier.pkl")
        with open(model_path, "wb") as f:
            pickle.dump({
                "classifier": self.classifier,
                "feature_columns": self.feature_columns,
                "target_encoder": self.target_encoder,
                "task_type_encoder": self.task_type_encoder
            }, f)

        meta_path = os.path.join(self.model_dir, "model_metadata.json")
        with open(meta_path, "w") as f:
            json.dump(self.model_metadata, f, indent=2)

    def load_model(self):
        model_path = os.path.join(self.model_dir, "procrastination_classifier.pkl")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found at {model_path}. Train the model first.")

        with open(model_path, "rb") as f:
            saved = pickle.load(f)
            self.classifier = saved["classifier"]
            self.feature_columns = saved["feature_columns"]
            self.target_encoder = saved["target_encoder"]
            self.task_type_encoder = saved["task_type_encoder"]

        meta_path = os.path.join(self.model_dir, "model_metadata.json")
        if os.path.exists(meta_path):
            with open(meta_path, "r") as f:
                self.model_metadata = json.load(f)


if __name__ == "__main__":
    from preprocessing import DataPreprocessor
    from feature_engineering import FeatureEngineer

    df_raw = DataPreprocessor("data/raw/student_academic_activities.csv").load_data()
    df_feat = FeatureEngineer().compute_sequential_features(df_raw)

    pipeline = ProcrastinationModelPipeline()
    metrics = pipeline.train(df_feat)
    print("Training Complete!")
    print(f"Accuracy: {metrics['accuracy']*100:.2f}% | CV Weighted F1: {metrics['cv_f1_mean']*100:.2f}%")
    print("\nTop 5 Predictive Features:")
    for feat, imp in list(metrics['feature_importances'].items())[:5]:
        print(f"  - {feat}: {imp*100:.2f}%")
