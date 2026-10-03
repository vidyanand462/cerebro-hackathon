import pandas as pd
import numpy as np

class OnsetDetector:
    """
    Identifies the exact sequential point (Task Number & Date) where
    procrastination patterns, repeated postponement, or last-minute rushes
    first emerge in a student's academic history.
    """
    def __init__(self, last_minute_threshold_hours=24.0, drop_ratio_threshold=0.35):
        self.last_minute_threshold = last_minute_threshold_hours
        self.drop_ratio_threshold = drop_ratio_threshold

    def analyze_student_sequence(self, student_df: pd.DataFrame) -> dict:
        """
        Analyzes a single student's chronological tasks and determines onset point.
        """
        student_df = student_df.sort_values(by='task_number').reset_index(drop=True)
        student_id = student_df['student_id'].iloc[0] if len(student_df) > 0 else "Unknown"

        if len(student_df) == 0:
            return {
                "student_id": student_id,
                "onset_detected": False,
                "onset_task_number": None,
                "trajectory_type": "Insufficient Data",
                "onset_explanation": "No tasks recorded."
            }

        # Extract sequence values
        hrs_list = student_df['hours_before_deadline'].tolist()
        task_nums = student_df['task_number'].tolist()
        task_types = student_df['task_type'].tolist()
        dates = student_df['activity_start'].astype(str).tolist()
        task_ids = student_df['task_id'].tolist()
        last_minute_flags = student_df['last_minute_flag'].tolist()
        late_flags = student_df['late_submission_flag'].tolist()

        n_tasks = len(hrs_list)

        # Baseline: first task deadline distance
        baseline_hrs = hrs_list[0]

        # Classification check: Chronic vs Proactive vs Emerging
        all_last_minute = all(h < self.last_minute_threshold for h in hrs_list)
        none_last_minute = all(h >= self.last_minute_threshold for h in hrs_list)
        avg_hrs = np.mean(hrs_list)

        if none_last_minute and avg_hrs > 60.0:
            return {
                "student_id": student_id,
                "onset_detected": False,
                "onset_task_number": None,
                "onset_task_id": None,
                "onset_date": None,
                "trajectory_type": "Proactive / Consistent Habit",
                "severity": "Low",
                "onset_explanation": f"Student maintains proactive study habits. Average start time is {avg_hrs:.1f} hours before deadlines with 0 last-minute tasks."
            }

        if all_last_minute or (hrs_list[0] < self.last_minute_threshold and len(hrs_list) >= 2 and hrs_list[1] < self.last_minute_threshold):
            return {
                "student_id": student_id,
                "onset_detected": True,
                "onset_task_number": 1,
                "onset_task_id": task_ids[0],
                "onset_date": dates[0][:10] if dates[0] else "N/A",
                "trajectory_type": "Chronic / Persistent Last-Minute",
                "severity": "High",
                "onset_explanation": f"Procrastination pattern was already established from Task 1 ({task_types[0]}), started only {hrs_list[0]:.1f}h before deadline."
            }

        # Step-change / Change-Point Detection across sequence (Emerging Procrastination)
        detected_idx = -1
        trigger_reason = ""

        for i in range(1, n_tasks):
            curr_h = hrs_list[i]
            prev_h = hrs_list[i - 1]
            drop_from_prev = prev_h - curr_h
            drop_ratio = drop_from_prev / max(1.0, prev_h)

            # Condition 1: Transition across the 24h last-minute barrier
            if prev_h >= self.last_minute_threshold and curr_h < self.last_minute_threshold:
                detected_idx = i
                trigger_reason = f"Crossed last-minute threshold (<24h). Start time dropped from {prev_h:.1f}h to {curr_h:.1f}h before deadline."
                break

            # Condition 2: Severe drop in preparation lead-time (>40% drop and <36h left)
            elif drop_ratio >= self.drop_ratio_threshold and curr_h < 40.0:
                detected_idx = i
                trigger_reason = f"Sharp preparation delay ({drop_ratio*100:.0f}% drop). Lead-time reduced from {prev_h:.1f}h to {curr_h:.1f}h."
                break

            # Condition 3: Late submission occurs
            elif late_flags[i] == 1 and late_flags[i-1] == 0:
                detected_idx = i
                trigger_reason = f"First late submission recorded at Task {task_nums[i]} ({task_types[i]})."
                break

        if detected_idx != -1:
            t_num = task_nums[detected_idx]
            t_id = task_ids[detected_idx]
            t_type = task_types[detected_idx]
            t_date = dates[detected_idx][:10] if dates[detected_idx] else "N/A"

            return {
                "student_id": student_id,
                "onset_detected": True,
                "onset_task_number": int(t_num),
                "onset_task_id": t_id,
                "onset_task_type": t_type,
                "onset_date": t_date,
                "trajectory_type": f"Emerging Procrastination (Onset at Task {t_num})",
                "severity": "High" if hrs_list[-1] < 20.0 else "Moderate",
                "trigger_reason": trigger_reason,
                "onset_explanation": f"Pattern began appearing at Task {t_num} ({t_type.capitalize()} on {t_date}): {trigger_reason}"
            }

        # If fluctuation exists without steep single break
        return {
            "student_id": student_id,
            "onset_detected": False,
            "onset_task_number": None,
            "onset_task_id": None,
            "onset_date": None,
            "trajectory_type": "Fluctuating / Mild Risk",
            "severity": "Moderate",
            "onset_explanation": f"Student displays fluctuating start times without a distinct single breaking point. Average deadline distance is {avg_hrs:.1f}h."
        }

    def detect_all_students(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Runs onset detection over every student in the dataset.
        """
        results = []
        for s_id, group in df.groupby('student_id'):
            res = self.analyze_student_sequence(group)
            results.append(res)
        return pd.DataFrame(results)


if __name__ == "__main__":
    from preprocessing import DataPreprocessor
    from feature_engineering import FeatureEngineer

    df = DataPreprocessor("data/raw/student_academic_activities.csv").load_data()
    detector = OnsetDetector()
    onsets_df = detector.detect_all_students(df)
    print("Onset detection summary:")
    print(onsets_df['trajectory_type'].value_counts())
    print("\nSample Emerging Student Cases:")
    print(onsets_df[onsets_df['trajectory_type'].str.contains('Emerging')].head(5)[['student_id', 'onset_task_number', 'onset_explanation']])
