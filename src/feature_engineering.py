import pandas as pd
import numpy as np

class FeatureEngineer:
    """
    Transforms raw temporal academic activity logs into rich behavioral,
    sequential, and trend indicators for procrastination detection.
    """
    def __init__(self):
        pass

    def compute_task_level_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        # 1. Total available time window in hours
        if 'assigned_at' in df.columns and 'deadline' in df.columns:
            window_hours = (df['deadline'] - df['assigned_at']).dt.total_seconds() / 3600.0
            # Fallback if window is 0 or NaN
            window_hours = np.where(window_hours <= 0, 168.0, window_hours) # default 7 days
            df['task_window_hours'] = window_hours
        else:
            df['task_window_hours'] = 168.0

        # 2. Time elapsed before starting (Start Delay in hours)
        if 'activity_start' in df.columns and 'assigned_at' in df.columns:
            start_delay = (df['activity_start'] - df['assigned_at']).dt.total_seconds() / 3600.0
            start_delay = np.maximum(0.0, start_delay)
            df['start_delay_hours'] = start_delay
        else:
            df['start_delay_hours'] = df['task_window_hours'] - df['hours_before_deadline']

        # 3. Delay Ratio: fraction of available time wasted before starting
        df['delay_ratio'] = np.clip(df['start_delay_hours'] / df['task_window_hours'], 0.0, 1.0)

        # 4. Deadline Pressure Index: nonlinear pressure scale [0, 1]
        # Pressure peaks exponentially as hours_before_deadline approaches 0
        df['deadline_pressure'] = 1.0 / (np.log2(np.maximum(0.1, df['hours_before_deadline']) + 2.0))

        # 5. Granular Last-Minute Flags
        df['critical_last_minute'] = (df['hours_before_deadline'] < 6.0).astype(int)
        df['extreme_panic_start'] = (df['hours_before_deadline'] < 2.0).astype(int)

        # 6. Inactivity Ratio
        if 'inactivity_gap_hours' in df.columns:
            df['inactivity_ratio'] = np.clip(df['inactivity_gap_hours'] / df['task_window_hours'], 0.0, 1.0)
        else:
            df['inactivity_ratio'] = df['delay_ratio']

        # 7. Burst Intensity: work duration relative to remaining time
        duration_hours = df['duration_minutes'] / 60.0
        remaining_hrs_safe = np.maximum(0.5, df['hours_before_deadline'])
        df['burst_intensity'] = np.clip(duration_hours / remaining_hrs_safe, 0.0, 5.0)

        return df

    def compute_sequential_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Computes sequential lag, rolling differences, and temporal trend metrics
        for each student across their task sequence.
        """
        df = self.compute_task_level_features(df)
        
        # Sort by student and task number
        df = df.sort_values(by=['student_id', 'task_number']).reset_index(drop=True)

        # Calculate lag features per student
        df['prev_hours_before_deadline'] = df.groupby('student_id')['hours_before_deadline'].shift(1)
        df['prev_hours_before_deadline'] = df['prev_hours_before_deadline'].fillna(df['hours_before_deadline'])

        # Difference in deadline distance: negative means student is starting closer to deadline than before
        df['delta_deadline_distance'] = df['hours_before_deadline'] - df['prev_hours_before_deadline']

        # Cumulative / Rolling features
        grouped = df.groupby('student_id')
        
        df['cum_last_minute_count'] = grouped['last_minute_flag'].cumsum()
        df['cum_late_count'] = grouped['late_submission_flag'].cumsum()
        
        # Rolling averages (expanding window up to current task)
        df['expanding_avg_deadline_distance'] = grouped['hours_before_deadline'].expanding().mean().reset_index(level=0, drop=True)
        df['expanding_avg_delay_ratio'] = grouped['delay_ratio'].expanding().mean().reset_index(level=0, drop=True)
        df['expanding_last_minute_rate'] = grouped['last_minute_flag'].expanding().mean().reset_index(level=0, drop=True)

        # Procrastination Velocity (rate of deterioration)
        # Difference between first task deadline distance and current task deadline distance
        first_dist = grouped['hours_before_deadline'].transform('first')
        df['distance_drop_from_baseline'] = first_dist - df['hours_before_deadline']
        df['deterioration_rate'] = df['distance_drop_from_baseline'] / np.maximum(1, df['task_number'] - 1)
        df['deterioration_rate'] = df['deterioration_rate'].fillna(0.0)

        # Composite Behavioral Procrastination Score (0 - 100)
        # Weights: Last Minute (30%), Delay Ratio (25%), Inactivity (20%), Deadline Pressure (15%), Lateness (10%)
        comp_score = (
            df['last_minute_flag'] * 25.0 +
            df['critical_last_minute'] * 15.0 +
            df['delay_ratio'] * 25.0 +
            df['inactivity_ratio'] * 15.0 +
            df['deadline_pressure'] * 15.0 +
            df['late_submission_flag'] * 15.0 +
            np.clip(df['deterioration_rate'] / 2.0, 0.0, 15.0)
        )
        df['behavioral_procrastination_score'] = np.clip(comp_score, 0.0, 100.0).round(1)

        return df

    def get_student_aggregated_profiles(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Aggregates sequential features into a single summary profile per student.
        """
        features_df = self.compute_sequential_features(df)
        
        agg_funcs = {
            'hours_before_deadline': ['mean', 'min', 'std'],
            'delay_ratio': ['mean', 'max'],
            'inactivity_gap_hours': ['mean', 'max'],
            'last_minute_flag': ['sum', 'mean'],
            'late_submission_flag': ['sum', 'mean'],
            'duration_minutes': ['mean', 'sum'],
            'deadline_pressure': ['mean', 'max'],
            'behavioral_procrastination_score': ['mean', 'last'],
            'deterioration_rate': ['last'],
            'survey_behavior_score': ['first'],
            'procrastination_label': ['last']
        }
        
        # Flatten multi-level columns
        summary = features_df.groupby('student_id').agg(agg_funcs)
        summary.columns = ['_'.join(col).strip() for col in summary.columns.values]
        summary = summary.reset_index()

        # Add overall risk classification
        def classify_risk(score):
            if score >= 65:
                return 'high'
            elif score >= 38:
                return 'moderate'
            else:
                return 'low'

        summary['overall_risk_category'] = summary['behavioral_procrastination_score_last'].apply(classify_risk)
        return summary


if __name__ == "__main__":
    from preprocessing import DataPreprocessor
    preprocessor = DataPreprocessor("data/raw/student_academic_activities.csv")
    cleaned_df = preprocessor.load_data()

    engineer = FeatureEngineer()
    fe_df = engineer.compute_sequential_features(cleaned_df)
    print("Feature Engineering successful! Shape:", fe_df.shape)
    print(fe_df[['student_id', 'task_number', 'hours_before_deadline', 'delay_ratio', 'behavioral_procrastination_score']].head(10))
