import sys
import os
import unittest
import pandas as pd

# Add workspace root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.preprocessing import DataPreprocessor
from src.feature_engineering import FeatureEngineer
from src.onset_detector import OnsetDetector
from src.model import ProcrastinationModelPipeline
from src.explanation import ExplanationEngine

class TestProcrastinationPipeline(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.data_path = os.path.join("data", "raw", "student_academic_activities.csv")
        cls.preprocessor = DataPreprocessor(cls.data_path)
        cls.cleaned_df = cls.preprocessor.load_data()
        cls.engineer = FeatureEngineer()
        cls.feature_df = cls.engineer.compute_sequential_features(cls.cleaned_df)
        cls.detector = OnsetDetector()
        cls.pipeline = ProcrastinationModelPipeline(model_dir="models")
        cls.explainer = ExplanationEngine()

    def test_01_preprocessing(self):
        self.assertIsNotNone(self.cleaned_df)
        self.assertGreater(len(self.cleaned_df), 0)
        self.assertIn('student_id', self.cleaned_df.columns)
        self.assertIn('hours_before_deadline', self.cleaned_df.columns)
        self.assertIn('activity_start', self.cleaned_df.columns)
        print("[PASSED] Test 1: Preprocessing data integrity passed.")

    def test_02_feature_engineering(self):
        self.assertIn('delay_ratio', self.feature_df.columns)
        self.assertIn('deadline_pressure', self.feature_df.columns)
        self.assertIn('behavioral_procrastination_score', self.feature_df.columns)
        self.assertIn('cum_last_minute_count', self.feature_df.columns)
        
        # Verify score ranges [0, 100]
        scores = self.feature_df['behavioral_procrastination_score']
        self.assertTrue((scores >= 0).all() and (scores <= 100).all())
        print("[PASSED] Test 2: Feature engineering validation passed.")

    def test_03_onset_detection(self):
        onsets_df = self.detector.detect_all_students(self.cleaned_df)
        self.assertGreater(len(onsets_df), 0)
        self.assertIn('onset_detected', onsets_df.columns)
        self.assertIn('trajectory_type', onsets_df.columns)

        # Test single student
        sample = self.cleaned_df[self.cleaned_df['student_id'] == 'S001']
        single_res = self.detector.analyze_student_sequence(sample)
        self.assertIn('trajectory_type', single_res)
        self.assertIn('onset_explanation', single_res)
        print("[PASSED] Test 3: Onset detection algorithm passed.")

    def test_04_model_training_and_inference(self):
        metrics = self.pipeline.train(self.feature_df)
        self.assertIn('accuracy', metrics)
        self.assertGreater(metrics['accuracy'], 0.70)
        self.assertIn('feature_importances', metrics)
        
        # Prediction
        preds = self.pipeline.predict(self.feature_df.head(5))
        self.assertEqual(len(preds), 5)
        self.assertIn('predicted_label', preds[0])
        self.assertIn('confidence', preds[0])
        print(f"[PASSED] Test 4: Model training & prediction passed (Accuracy: {metrics['accuracy']*100:.1f}%).")

    def test_05_explanation_generation(self):
        sample = self.cleaned_df[self.cleaned_df['student_id'] == 'S001']
        onset_info = self.detector.analyze_student_sequence(sample)
        explanation = self.explainer.generate_explanation(sample, onset_info=onset_info)
        
        self.assertIn('headline', explanation)
        self.assertIn('evidence', explanation)
        self.assertGreater(len(explanation['evidence']), 0)
        self.assertIn('recommendations', explanation)
        print("[PASSED] Test 5: Grounded explanation engine passed.")

if __name__ == '__main__':
    unittest.main()
