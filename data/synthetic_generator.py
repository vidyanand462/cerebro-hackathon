import pandas as pd
import numpy as np
import datetime
import random
import os

# Ensure directories exist
os.makedirs("data/raw", exist_ok=True)
os.makedirs("data/processed", exist_ok=True)
os.makedirs("models", exist_ok=True)

CSV_RAW = "data/raw/student_academic_activities.csv"

def generate_full_synthetic_data(num_students=100, tasks_per_student=5, seed=42):
    """
    Generates a full synthetic academic activity dataset with realistic behavioral archetypes:
    1. Consistent/Proactive (Low Risk)
    2. Developing/Emerging Procrastinator (Onset shift from T2/T3/T4)
    3. Chronic Last-Minute Worker (Consistently High Risk)
    4. Sporadic/Fluctuating Procrastinator (Moderate Risk)
    """
    np.random.seed(seed)
    random.seed(seed)
    
    task_types = ["assignment", "quiz", "lab", "project"]
    study_years = ["First Year", "Second Year", "Third Year", "Fourth Year"]
    socio_econ = ["Low", "Lower-middle", "Middle", "Upper-middle", "High"]
    reasons_list = [
        "Lack of interest",
        "Distractions (e.g., social media)",
        "Poor time management",
        "Stress and anxiety",
        "Procrastination due to lack of resources",
        "Health issues",
        "Unclear instructions from professors",
        "Personal family problems",
        "Overconfidence"
    ]
    delay_freqs = ["Never", "Occasionally", "Sometimes", "Often", "Always"]
    cgpa_ranges = ["Below 2.50", "2.50 - 2.99", "3.00 - 3.49", "3.50 - 3.74", "3.75 - 4.00"]
    study_hour_ranges = ["0-5 hours", "6-10 hours", "11-15 hours", "16+ hours"]
    mobile_ranges = ["1-2 hours", "3-4 hours", "More than 4 hours"]
    distractions = ["Never", "Occasionally", "Sometimes", "Often", "Always"]

    records = []
    
    for i in range(1, num_students + 1):
        s_id = f"S{i:03d}"
        archetype = np.random.choice(["proactive", "emerging", "chronic", "sporadic"], p=[0.25, 0.35, 0.25, 0.15])
        
        # Demographic & survey baseline attributes
        year = random.choice(study_years)
        socio = random.choice(socio_econ)
        num_reasons = random.randint(1, 3)
        reasons = ", ".join(random.sample(reasons_list, num_reasons))
        cgpa = random.choice(cgpa_ranges)
        study_hours = random.choice(study_hour_ranges)
        mobile_hrs = random.choice(mobile_ranges)
        distract = random.choice(distractions)
        
        if archetype == "proactive":
            delay_freq = random.choice(["Never", "Occasionally"])
            survey_score = round(random.uniform(0.0, 0.35), 3)
            time_mgmt = random.choice(["Always", "Often"])
            cgpa = random.choice(["3.50 - 3.74", "3.75 - 4.00"])
        elif archetype == "emerging":
            delay_freq = random.choice(["Sometimes", "Often"])
            survey_score = round(random.uniform(0.40, 0.65), 3)
            time_mgmt = random.choice(["Sometimes", "Never"])
        elif archetype == "chronic":
            delay_freq = random.choice(["Often", "Always"])
            survey_score = round(random.uniform(0.65, 0.95), 3)
            time_mgmt = random.choice(["Never", "Occasionally"])
            cgpa = random.choice(["Below 2.50", "2.50 - 2.99"])
        else: # sporadic
            delay_freq = random.choice(["Occasionally", "Sometimes", "Often"])
            survey_score = round(random.uniform(0.35, 0.60), 3)
            time_mgmt = random.choice(["Sometimes", "Often"])
            
        base_date = datetime.date(2026, 8, 1) + datetime.timedelta(days=random.randint(0, 20))
        
        # Onset point for emerging type
        onset_task = random.randint(2, 4) if archetype == "emerging" else (1 if archetype == "chronic" else 999)
        
        for t in range(1, tasks_per_student + 1):
            t_id = f"{s_id}_T{t}"
            t_type = random.choice(task_types)
            task_window_days = 7
            assigned_at = base_date + datetime.timedelta(days=(t-1)*7)
            deadline = assigned_at + datetime.timedelta(days=task_window_days)
            
            # Determine hours before deadline based on archetype and progression
            if archetype == "proactive":
                hrs_before = round(random.uniform(70.0, 120.0), 2)
            elif archetype == "chronic":
                hrs_before = round(random.uniform(0.5, 30.0), 2)
            elif archetype == "emerging":
                if t < onset_task:
                    hrs_before = round(random.uniform(65.0, 110.0), 2)
                elif t == onset_task:
                    hrs_before = round(random.uniform(28.0, 48.0), 2)
                else: # post-onset
                    hrs_before = round(random.uniform(1.0, 24.0), 2)
            else: # sporadic
                hrs_before = round(random.uniform(25.0, 85.0), 2)
                
            deadline_dt = datetime.datetime(deadline.year, deadline.month, deadline.day, 23, 59, 0)
            activity_start_dt = deadline_dt - datetime.timedelta(hours=hrs_before)
            duration_minutes = random.randint(15, 65)
            
            # Submission calculation
            is_last_minute = 1 if hrs_before < 24.0 else 0
            is_late = 1 if (hrs_before < 2.0 and random.random() < 0.6) else 0
            
            if is_late:
                sub_delay_hours = round(random.uniform(1.0, 14.0), 2)
                submission_dt = deadline_dt + datetime.timedelta(hours=sub_delay_hours)
            else:
                sub_delay_hours = 0.0
                hours_after_start = random.uniform(0.5, min(hrs_before, 36.0))
                submission_dt = activity_start_dt + datetime.timedelta(hours=hours_after_start)
                if submission_dt > deadline_dt:
                    submission_dt = deadline_dt - datetime.timedelta(minutes=random.randint(5, 60))
            
            inactivity_gap_hours = round(max(0.0, hrs_before - (duration_minutes / 60.0)), 2)
            
            # Synthetic behavior score calculation
            # Normalized pressure & lateness
            urgency_factor = max(0.0, min(1.0, (120.0 - hrs_before) / 120.0))
            synth_score = round(0.5 * urgency_factor + 0.3 * (1.0 if is_last_minute else 0.0) + 0.2 * (1.0 if is_late else 0.0), 3)
            
            if synth_score < 0.35:
                label = "low"
            elif synth_score < 0.68:
                label = "moderate"
            else:
                label = "high"
                
            records.append({
                "student_id": s_id,
                "task_id": t_id,
                "task_number": t,
                "task_type": t_type,
                "assigned_at": str(assigned_at),
                "activity_start": str(activity_start_dt),
                "submission_time": str(submission_dt),
                "deadline": str(deadline),
                "duration_minutes": duration_minutes,
                "hours_before_deadline": hrs_before,
                "submission_delay_hours": sub_delay_hours,
                "last_minute_flag": is_last_minute,
                "late_submission_flag": is_late,
                "inactivity_gap_hours": inactivity_gap_hours,
                "survey_behavior_score": survey_score,
                "synthetic_behavior_score": synth_score,
                "procrastination_label": label,
                "timestamp": f"11/{random.randint(4, 25)}/2024 {random.randint(9, 23)}:{random.randint(10, 59):02d}:00",
                "study_year": year,
                "socio_economic_background": socio,
                "assignment_delay_frequency": delay_freq,
                "procrastination_reasons": reasons,
                "assignment_submission_timing": "Never" if is_late else ("Sometimes" if is_last_minute else "Always"),
                "last_minute_exam_preparation": "Yes" if synth_score > 0.5 else "No",
                "effect_of_procrastination_on_grades": "Significantly" if synth_score > 0.6 else "Moderately",
                "procrastination_and_grade_outcome": "Yes" if synth_score > 0.5 else "No",
                "stress_due_to_procrastination": "Significantly" if synth_score > 0.65 else ("Moderately" if synth_score > 0.4 else "Not at all"),
                "study_hours_per_week": study_hours,
                "cgpa": cgpa,
                "use_of_time_management": time_mgmt,
                "procrastination_management_training": random.choice(["Yes", "No"]),
                "procrastination_recovery_strategies": random.choice(["Yes", "No"]),
                "hours_spent_on_mobile_non_academic": mobile_hrs,
                "study_session_distractions": distract
            })
            
    df = pd.DataFrame(records)
    return df

if __name__ == "__main__":
    df = generate_full_synthetic_data(num_students=120)
    print(f"Generated {len(df)} records across {df['student_id'].nunique()} students.")
    df.to_csv("data/raw/synthetic_full_trajectory.csv", index=False)
