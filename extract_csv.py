import json
import glob
import os

all_logs = glob.glob(r"C:\Users\DELL\.gemini\antigravity-ide\brain\*\.system_generated\logs\*.jsonl")
found = False

for log in all_logs:
    try:
        with open(log, "r", encoding="utf-8", errors="ignore") as f:
            for idx, line in enumerate(f):
                if "S451" in line and "student_id,task_id" in line:
                    data = json.loads(line)
                    # Search all string values
                    for k, v in data.items():
                        if isinstance(v, str) and "student_id,task_id" in v:
                            start = v.find("student_id,task_id")
                            end = v.find("</USER_REQUEST>") if "</USER_REQUEST>" in v else len(v)
                            csv_data = v[start:end].strip()
                            with open("data/raw/student_academic_activities.csv", "w", encoding="utf-8") as out:
                                out.write(csv_data)
                            print(f"Successfully extracted {len(csv_data.splitlines())} lines from {log} at line {idx}!")
                            found = True
                            break
                    if found:
                        break
    except Exception as e:
        print(f"Error reading {log}: {e}")
    if found:
        break

if not found:
    print("Could not find in transcript logs.")
