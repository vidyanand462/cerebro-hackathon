import pandas as pd
import numpy as np

class ExplanationEngine:
    """
    Generates human-readable, grounded, evidence-based explanations
    and actionable recommendations from student activity histories.
    """
    def __init__(self, model_pipeline=None):
        self.model_pipeline = model_pipeline

    def generate_explanation(self, student_df: pd.DataFrame, onset_info: dict = None, pred_info: dict = None) -> dict:
        """
        Generates structured explanation with headline, evidence bullets,
        root-cause context, and personalized recommendation.
        """
        student_df = student_df.sort_values(by='task_number').reset_index(drop=True)
        student_id = student_df['student_id'].iloc[0] if len(student_df) > 0 else "Student"

        total_tasks = len(student_df)
        hrs = student_df['hours_before_deadline'].values
        last_minute_count = int(student_df['last_minute_flag'].sum())
        late_count = int(student_df['late_submission_flag'].sum())
        avg_hrs = float(np.mean(hrs))
        min_hrs = float(np.min(hrs))
        max_hrs = float(np.max(hrs))
        first_hrs = float(hrs[0])
        last_hrs = float(hrs[-1])
        drop_hrs = first_hrs - last_hrs
        drop_pct = (drop_hrs / max(1.0, first_hrs)) * 100.0

        # Most common distraction/reason if present in survey
        reasons_col = 'procrastination_reasons'
        survey_reasons = student_df[reasons_col].iloc[0] if reasons_col in student_df.columns and pd.notna(student_df[reasons_col].iloc[0]) else "General task postponement"

        # Predicted label or heuristic risk
        risk_label = pred_info.get('predicted_label', 'moderate') if pred_info else ('high' if last_minute_count >= 2 else ('low' if avg_hrs > 60 else 'moderate'))

        # Evidence generation
        evidence_points = []

        if last_minute_count > 0:
            evidence_points.append(
                f"⏰ **Last-Minute Activity**: {last_minute_count} of {total_tasks} tasks ({int(last_minute_count/total_tasks*100)}%) were started within 24 hours of the deadline."
            )

        if drop_hrs > 15.0:
            evidence_points.append(
                f"📉 **Increasing Postponement**: Start lead-time decreased from **{first_hrs:.1f} hours** on Task 1 down to **{last_hrs:.1f} hours** on Task {total_tasks} (a **{drop_pct:.0f}% reduction** in preparation margin)."
            )
        elif avg_hrs < 30.0:
            evidence_points.append(
                f"⚡ **Consistently Compressed Window**: Average start time was **{avg_hrs:.1f} hours** before deadlines, leaving little room for revisions."
            )
        else:
            evidence_points.append(
                f"✅ **Early Preparation Habit**: Maintained an average lead-time of **{avg_hrs:.1f} hours** across all academic tasks."
            )

        if min_hrs < 6.0:
            evidence_points.append(
                f"🚨 **Panic Start Event**: The most compressed task was started just **{min_hrs:.1f} hours** before the deadline."
            )

        if late_count > 0:
            evidence_points.append(
                f"⚠️ **Submission Delays**: {late_count} task(s) resulted in late submissions past the formal deadline."
            )

        # Onset Context
        onset_str = ""
        if onset_info and onset_info.get("onset_detected"):
            t_num = onset_info.get("onset_task_number")
            t_type = onset_info.get("onset_task_type", "task")
            t_date = onset_info.get("onset_date", "earlier in term")
            onset_str = f"Behavioral shift emerged at **Task {t_num} ({t_type.capitalize()} on {t_date})**."
        elif onset_info and not onset_info.get("onset_detected"):
            onset_str = "No sudden behavioral shift detected; habits remained steady across the recorded tasks."

        # Headline synthesis
        if risk_label == 'high':
            headline = f"⚠️ High Procrastination Pattern Detected: Persistent deadline-oriented compression observed across recent tasks."
        elif risk_label == 'moderate':
            headline = f"🟡 Moderate Procrastination Risk: Increasing postponement trend detected with vulnerability to deadline pressure."
        else:
            headline = f"🟢 Healthy Proactive Workflow: Regular, early task initiation with low deadline proximity risk."

        # Actionable recommendations
        recommendations = []
        if risk_label == 'high':
            recommendations.append("Break down assignments into 24-hour milestone checkpoints immediately upon release.")
            recommendations.append("Implement the 10-Minute Rule: commit to working on new tasks for just 10 minutes on Day 1.")
            if "distraction" in survey_reasons.lower():
                recommendations.append("Use app blockers or Pomodoro timers during initial reading sessions to counter phone distractions.")
        elif risk_label == 'moderate':
            recommendations.append("Schedule a fixed 'Start Day' 72 hours before any deadline to prevent gradual deterioration.")
            recommendations.append("Set personal artificial deadlines 24 hours ahead of actual course submission deadlines.")
        else:
            recommendations.append("Keep maintaining your early start routines and balanced study intervals.")

        return {
            "student_id": student_id,
            "risk_label": risk_label.capitalize(),
            "headline": headline,
            "onset_summary": onset_str,
            "evidence": evidence_points,
            "reported_distraction_factors": survey_reasons,
            "recommendations": recommendations,
            "plain_text_explanation": f"{headline}\n\nKey History Insights:\n" + "\n".join([f"• {e}" for e in evidence_points]) + (f"\n\nOnset Details: {onset_str}" if onset_str else "")
        }


if __name__ == "__main__":
    from preprocessing import DataPreprocessor
    from feature_engineering import FeatureEngineer
    from onset_detector import OnsetDetector

    df = DataPreprocessor("data/raw/student_academic_activities.csv").load_data()
    student_sample = df[df['student_id'] == 'S001']
    
    detector = OnsetDetector()
    onset_info = detector.analyze_student_sequence(student_sample)
    
    engine = ExplanationEngine()
    explanation = engine.generate_explanation(student_sample, onset_info=onset_info)
    
    print("Generated Explanation for S001:")
    print("Headline:", explanation['headline'])
    print("Onset:", explanation['onset_summary'])
    print("\nEvidence:")
    for ev in explanation['evidence']:
        print(" ", ev)
