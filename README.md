# 🎓 ProcrastiGuard AI — Academic Procrastination Detection System
**CEREBRO Hackathon — Problem Statement 1**  
*Department of Computer Science & Engineering, Institute of Engineering & Management (IEM), Kolkata*

---

## 📌 Executive Summary
This system is an end-to-end Machine Learning and Behavioral Analytics platform designed to identify patterns of **academic procrastination** from student activity sequences. 

Rather than treating student activity as static tabular rows, this architecture analyzes **temporal sequences** (study sessions, assignment lead times, submission delays, inactivity gaps) to:
1. **Detect Procrastination Patterns**: Identify chronic, emerging, and sporadic procrastination habits.
2. **Identify Onset Points**: Pinpoint the exact chronological task ($T_k$) and date where repeated postponement or last-minute behavior first appeared.
3. **Generate Explainable AI (XAI) Narratives**: Produce clear, evidence-based, human-readable explanations grounded in the student's historical timeline.
4. **Interactive Dashboard**: Provide an intuitive web application for student tracking, visual Gantt sequences, and habit remediation advice.

---

## 🏛️ Architecture & Pipeline Overview

```
                          STUDENT ACTIVITY LOGS
              (Timestamps, Deadlines, Starts, Inactivity)
                                  │
                                  ▼
                   ┌──────────────────────────────┐
                   │   Data Cleaning & Sorting    │
                   │   (src/preprocessing.py)     │
                   └──────────────┬───────────────┘
                                  │
                                  ▼
                   ┌──────────────────────────────┐
                   │ Feature Engineering Engine   │
                   │ (src/feature_engineering.py) │
                   └──────────────┬───────────────┘
                                  │
         ┌────────────────────────┴────────────────────────┐
         ▼                                                 ▼
┌──────────────────────────────┐          ┌──────────────────────────────┐
│  Change-Point Onset Detector │          │ Machine Learning Classifier  │
│    (src/onset_detector.py)   │          │       (src/model.py)         │
└──────────────┬───────────────┘          └──────────────┬───────────────┘
               │                                         │
               └──────────────────┬──────────────────────┘
                                  ▼
                   ┌──────────────────────────────┐
                   │ Grounded Explanation Engine  │
                   │     (src/explanation.py)     │
                   └──────────────┬───────────────┘
                                  │
                                  ▼
                   ┌──────────────────────────────┐
                   │  Streamlit Web Application   │
                   │         (app/app.py)         │
                   └──────────────────────────────┘
```

---

## 📐 Mathematical Formulations & Behavioral Metrics

### 1. Delay Ratio
Measures the fraction of the total available preparation window that was postponed before the first active session:
$$\text{Delay Ratio} = \frac{\text{Activity Start} - \text{Assigned Date}}{\text{Deadline} - \text{Assigned Date}}$$

### 2. Nonlinear Deadline Pressure Index
Models the escalating psychological pressure as the deadline approaches:
$$P(t) = \frac{1}{\log_2(\max(0.1, \text{Hours Before Deadline}) + 2.0)}$$

### 3. Procrastination Velocity & Deterioration Rate
Measures the rate of lead-time reduction between the student's baseline and current task:
$$\Delta_{\text{rate}} = \frac{\text{LeadTime}_{T_1} - \text{LeadTime}_{T_k}}{\max(1, k - 1)}$$

### 4. Composite Behavioral Procrastination Score (0–100)
Combines last-minute urgency, delay ratio, inactivity gap penalty, and submission delays into a normalized behavioral index:
$$\text{BPI} = w_1 \cdot \text{LastMinute} + w_2 \cdot \text{DelayRatio} + w_3 \cdot \text{InactivityRatio} + w_4 \cdot P(t) + w_5 \cdot \text{Lateness} + w_6 \cdot \Delta_{\text{rate}}$$

---

## 📂 Repository Structure

```text
procrastination-detector/
├── app/
│   └── app.py                     # Streamlit interactive dashboard
├── data/
│   ├── raw/
│   │   └── student_academic_activities.csv  # Raw student activity sequence logs
│   ├── processed/
│   │   └── featured_academic_activities.csv # Extracted behavioral features
│   └── synthetic_generator.py     # Trajectory data generator
├── models/
│   ├── procrastination_classifier.pkl       # Serialized Random Forest model
│   └── model_metadata.json        # Evaluation metrics & feature importances
├── src/
│   ├── __init__.py
│   ├── preprocessing.py           # Ingestion & timestamp normalization
│   ├── feature_engineering.py     # Temporal & sequential feature extraction
│   ├── onset_detector.py          # Change-point emergence detection algorithm
│   ├── model.py                   # Classifier training & inference pipeline
│   └── explanation.py             # Grounded natural language generator
├── tests/
│   └── test_pipeline.py           # Automated unit test suite
├── train_and_evaluate.py          # Complete training & validation entrypoint
├── requirements.txt               # Dependencies list
├── README.md                      # Comprehensive project documentation
└── .gitignore                     # Git ignore file
```

---

## 🚀 Quickstart & Installation

### 1. Clone or Open the Repository
```bash
cd procastination
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Run Pipeline & Train Model
```bash
python train_and_evaluate.py
```

### 4. Run Automated Test Suite
```bash
python -m unittest tests/test_pipeline.py
```

### 5. Launch the Web Application
```bash
streamlit run app/app.py
```

---

## 📊 Evaluation & Performance Benchmark

| Metric | Result |
| :--- | :--- |
| **Model Algorithm** | Balanced Random Forest (150 Estimators) |
| **Overall Classification Accuracy** | **83.33%** |
| **5-Fold Cross-Validation Weighted F1** | **84.57% (± 5.61%)** |
| **Primary Predictors** | Hours Before Deadline, Deadline Pressure, Start Delay, Delay Ratio, Inactivity Gap |

---

## 🎯 Sample Grounded Explanation Output

> **Pattern Detected**: ⚠️ High Procrastination Pattern Detected: Persistent deadline-oriented compression observed across recent tasks.
>
> **Onset Event**: Behavioral shift emerged at **Task 4 (Lab on 2026-08-27)**.
>
> **Timeline Evidence Grounding**:
> • ⏰ **Last-Minute Activity**: 2 of 5 tasks (40%) were started within 24 hours of the deadline.
> • 📉 **Increasing Postponement**: Start lead-time decreased from **53.8 hours** on Task 1 down to **24.3 hours** on Task 5 (a **55% reduction** in preparation margin).
> • 💡 **Recommended Habit Strategy**: Implement the 10-Minute Rule: commit to working on new tasks for just 10 minutes on Day 1.

---
*Created for CEREBRO Hackathon — Department of Computer Science & Engineering, IEM Kolkata.*
