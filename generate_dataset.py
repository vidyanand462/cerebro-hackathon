import pandas as pd
from data.synthetic_generator import generate_full_synthetic_data

if __name__ == "__main__":
    df = generate_full_synthetic_data(num_students=451, tasks_per_student=5, seed=42)
    df.to_csv("data/raw/student_academic_activities.csv", index=False)
    print(f"Successfully generated and saved complete academic dataset with {len(df)} rows across {df['student_id'].nunique()} students (S001 - S451)!")
