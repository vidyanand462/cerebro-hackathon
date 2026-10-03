import pandas as pd
import numpy as np
import os

class DataPreprocessor:
    """
    Handles data ingestion, validation, timestamp normalization,
    and sequence sorting for student academic logs.
    """
    def __init__(self, filepath=None):
        self.filepath = filepath
        self.df = None

    def load_data(self, filepath=None):
        path = filepath or self.filepath
        if not path or not os.path.exists(path):
            raise FileNotFoundError(f"Data file not found at: {path}")
        
        self.df = pd.read_csv(path)
        return self.clean_and_format()

    def clean_and_format(self, df=None):
        if df is not None:
            self.df = df.copy()

        # Clean string whitespace from column names
        self.df.columns = [col.strip() for col in self.df.columns]

        # Parse date columns
        date_cols = ['assigned_at', 'activity_start', 'submission_time', 'deadline']
        for col in date_cols:
            if col in self.df.columns:
                self.df[col] = pd.to_datetime(self.df[col], errors='coerce')

        # Fill missing values for numericals
        num_defaults = {
            'duration_minutes': 30,
            'hours_before_deadline': 48.0,
            'submission_delay_hours': 0.0,
            'last_minute_flag': 0,
            'late_submission_flag': 0,
            'inactivity_gap_hours': 40.0,
            'survey_behavior_score': 0.5,
            'synthetic_behavior_score': 0.5
        }
        for col, default_val in num_defaults.items():
            if col in self.df.columns:
                self.df[col] = pd.to_numeric(self.df[col], errors='coerce').fillna(default_val)

        # Categorical defaults
        if 'task_type' in self.df.columns:
            self.df['task_type'] = self.df['task_type'].fillna('assignment').astype(str).str.lower()

        if 'procrastination_label' in self.df.columns:
            self.df['procrastination_label'] = self.df['procrastination_label'].fillna('moderate').astype(str).str.lower()

        # Sort sequences by student_id and task_number or activity_start
        sort_cols = ['student_id']
        if 'task_number' in self.df.columns:
            sort_cols.append('task_number')
        elif 'activity_start' in self.df.columns:
            sort_cols.append('activity_start')

        self.df = self.df.sort_values(by=sort_cols).reset_index(drop=True)
        return self.df


if __name__ == "__main__":
    preprocessor = DataPreprocessor("data/raw/student_academic_activities.csv")
    cleaned_df = preprocessor.load_data()
    print("Preprocessing successful! Shape:", cleaned_df.shape)
    print("Columns:", cleaned_df.columns.tolist())
