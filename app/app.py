import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import datetime
import os
import sys

# Add project root to python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.preprocessing import DataPreprocessor
from src.feature_engineering import FeatureEngineer
from src.onset_detector import OnsetDetector
from src.model import ProcrastinationModelPipeline
from src.explanation import ExplanationEngine

st.set_page_config(
    page_title="ProcrastiGuard AI | Academic Procrastination Detection System",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.35rem;
        font-weight: 800;
        background: linear-gradient(90deg, #1E40AF, #6D28D9, #BE185D);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 2px;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #475569;
        margin-bottom: 1.5rem;
        font-weight: 500;
    }
    .kpi-card {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 14px;
        padding: 18px;
        box-shadow: 0 2px 6px rgba(0,0,0,0.04);
        transition: transform 0.2s ease;
    }
    .kpi-title {
        font-size: 0.85rem;
        font-weight: 600;
        color: #64748B;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    .kpi-value {
        font-size: 1.85rem;
        font-weight: 700;
        margin-top: 4px;
    }
    .explanation-box {
        background: linear-gradient(135deg, #F8FAFC 0%, #EEF2FF 100%);
        border: 1px solid #C7D2FE;
        border-left: 6px solid #4F46E5;
        border-radius: 12px;
        padding: 22px;
        margin: 18px 0;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05);
    }
    .preset-card {
        background: #F1F5F9;
        border-radius: 8px;
        padding: 10px 14px;
        margin-bottom: 8px;
        border-left: 3px solid #3B82F6;
        font-size: 0.85rem;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_system():
    data_path = os.path.join("data", "raw", "student_academic_activities.csv")
    preprocessor = DataPreprocessor(data_path)
    raw_df = preprocessor.load_data()
    
    engineer = FeatureEngineer()
    feat_df = engineer.compute_sequential_features(raw_df)
    
    pipeline = ProcrastinationModelPipeline(model_dir="models")
    if not os.path.exists(os.path.join("models", "procrastination_classifier.pkl")):
        pipeline.train(feat_df)
    else:
        pipeline.load_model()
        
    detector = OnsetDetector()
    explainer = ExplanationEngine(model_pipeline=pipeline)
    
    return raw_df, feat_df, pipeline, detector, explainer, engineer, preprocessor

raw_df, feat_df, pipeline, detector, explainer, engineer, preprocessor = load_system()

# Sidebar Navigation
with st.sidebar:
    st.image("https://img.icons8.com/isometric/100/brain.png", width=64)
    st.markdown("### **🎓 ProcrastiGuard AI**")
    st.markdown("**Academic Procrastination Detection**")
    st.caption("CEREBRO Hackathon • Dept. of CSE, IEM Kolkata")
    st.divider()
    
    app_mode = st.radio(
        "Mode Selection",
        [
            "✍️ Interactive User Input & Live Diagnosis",
            "🔍 Explore Student Database (S001 - S451)",
            "📊 Cohort Analytics & Model Benchmark",
            "📤 Upload Custom Activity CSV"
        ]
    )
    st.divider()
    st.markdown("#### **System Status**")
    st.success(f"🟢 **Trained Model Active** (Accuracy: {pipeline.model_metadata.get('accuracy', 1.0)*100:.1f}%)")
    st.info(f"📁 **Dataset Records**: {len(feat_df)} tasks across {feat_df['student_id'].nunique()} students")

# ==============================================================================
# MODE 1: INTERACTIVE USER INPUT & LIVE DIAGNOSIS
# ==============================================================================
if app_mode == "✍️ Interactive User Input & Live Diagnosis":
    st.markdown('<div class="main-header">🎓 ProcrastiGuard AI: Live Student Diagnosis</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Automated detection of academic procrastination patterns, preparation lead-time deterioration, and onset emergence points.</div>', unsafe_allow_html=True)

    # Preset Quick-Loaders for Judges / Users
    st.markdown("##### ⚡ Quick Presets (Click to pre-fill test scenarios):")
    p_col1, p_col2, p_col3, p_col4 = st.columns(4)
    
    # Initialize session state for user input tasks if not set
    if "user_tasks" not in st.session_state:
        # Default scenario: Emerging Procrastination
        st.session_state.user_tasks = [
            {"task_type": "assignment", "assigned_offset": 0, "start_offset": 3, "deadline_offset": 7, "duration": 40, "late_hrs": 0.0},
            {"task_type": "lab", "assigned_offset": 7, "start_offset": 10, "deadline_offset": 14, "duration": 35, "late_hrs": 0.0},
            {"task_type": "quiz", "assigned_offset": 14, "start_offset": 18, "deadline_offset": 21, "duration": 45, "late_hrs": 0.0},
            {"task_type": "project", "assigned_offset": 21, "start_offset": 27, "deadline_offset": 28, "duration": 30, "late_hrs": 0.0}, # <24h start
            {"task_type": "assignment", "assigned_offset": 28, "start_offset": 34.8, "deadline_offset": 35, "duration": 50, "late_hrs": 4.5} # late
        ]
        st.session_state.student_name = "Custom Student A"
        st.session_state.study_year = "Third Year"
        st.session_state.distractions = "Distractions (e.g., social media), Poor time management"

    with p_col1:
        if st.button("📉 Load: Emerging Procrastinator"):
            st.session_state.user_tasks = [
                {"task_type": "assignment", "assigned_offset": 0, "start_offset": 3, "deadline_offset": 7, "duration": 40, "late_hrs": 0.0},
                {"task_type": "lab", "assigned_offset": 7, "start_offset": 10, "deadline_offset": 14, "duration": 35, "late_hrs": 0.0},
                {"task_type": "quiz", "assigned_offset": 14, "start_offset": 18.5, "deadline_offset": 21, "duration": 45, "late_hrs": 0.0},
                {"task_type": "project", "assigned_offset": 21, "start_offset": 27.2, "deadline_offset": 28, "duration": 30, "late_hrs": 0.0}, # Onset T4
                {"task_type": "assignment", "assigned_offset": 28, "start_offset": 34.8, "deadline_offset": 35, "duration": 50, "late_hrs": 6.0}
            ]
            st.session_state.student_name = "Rahul Sharma (Emerging)"
            st.session_state.distractions = "Distractions (e.g., social media), Stress and anxiety"
            st.rerun()

    with p_col2:
        if st.button("🟢 Load: Proactive Achiever"):
            st.session_state.user_tasks = [
                {"task_type": "assignment", "assigned_offset": 0, "start_offset": 1.5, "deadline_offset": 7, "duration": 50, "late_hrs": 0.0},
                {"task_type": "quiz", "assigned_offset": 7, "start_offset": 8.5, "deadline_offset": 14, "duration": 45, "late_hrs": 0.0},
                {"task_type": "lab", "assigned_offset": 14, "start_offset": 15.5, "deadline_offset": 21, "duration": 60, "late_hrs": 0.0},
                {"task_type": "project", "assigned_offset": 21, "start_offset": 22.5, "deadline_offset": 28, "duration": 55, "late_hrs": 0.0},
                {"task_type": "assignment", "assigned_offset": 28, "start_offset": 29.5, "deadline_offset": 35, "duration": 40, "late_hrs": 0.0}
            ]
            st.session_state.student_name = "Ananya Roy (Proactive)"
            st.session_state.distractions = "None / High Discipline"
            st.rerun()

    with p_col3:
        if st.button("🔴 Load: Chronic Last-Minute"):
            st.session_state.user_tasks = [
                {"task_type": "assignment", "assigned_offset": 0, "start_offset": 6.7, "deadline_offset": 7, "duration": 30, "late_hrs": 0.0},
                {"task_type": "quiz", "assigned_offset": 7, "start_offset": 13.8, "deadline_offset": 14, "duration": 25, "late_hrs": 2.0},
                {"task_type": "project", "assigned_offset": 14, "start_offset": 20.7, "deadline_offset": 21, "duration": 35, "late_hrs": 0.0},
                {"task_type": "lab", "assigned_offset": 21, "start_offset": 27.8, "deadline_offset": 28, "duration": 20, "late_hrs": 8.0},
                {"task_type": "quiz", "assigned_offset": 28, "start_offset": 34.9, "deadline_offset": 35, "duration": 30, "late_hrs": 3.5}
            ]
            st.session_state.student_name = "Vikram Sen (Chronic)"
            st.session_state.distractions = "Poor time management, Distractions (e.g., social media), Overconfidence"
            st.rerun()

    with p_col4:
        if st.button("🔄 Reset / Clear Tasks"):
            st.session_state.user_tasks = [
                {"task_type": "assignment", "assigned_offset": 0, "start_offset": 2, "deadline_offset": 7, "duration": 30, "late_hrs": 0.0}
            ]
            st.session_state.student_name = "New Student"
            st.session_state.distractions = "Distractions (e.g., social media)"
            st.rerun()

    st.markdown("<hr style='margin: 15px 0;'>", unsafe_allow_html=True)

    # Student Info Row
    f_col1, f_col2, f_col3 = st.columns([1.5, 1, 1.5])
    with f_col1:
        student_name_input = st.text_input("Student Identifier / Name", value=st.session_state.get("student_name", "Student User"), key="stud_name_key")
    with f_col2:
        num_tasks_input = st.selectbox("Number of Sequential Tasks", [1, 2, 3, 4, 5], index=len(st.session_state.user_tasks)-1, key="num_tasks_key")
    with f_col3:
        distraction_input = st.text_input("Reported Reasons / Distractions", value=st.session_state.get("distractions", "Distractions (e.g., social media)"), key="distract_key")

    st.markdown("#### 📝 Sequential Task Activity Logs (Live Calendar Inputs):")
    
    base_date = datetime.date(2026, 8, 1)
    input_records = []

    for t_idx in range(num_tasks_input):
        # Get default or prefilled values
        default_t = st.session_state.user_tasks[t_idx] if t_idx < len(st.session_state.user_tasks) else {
            "task_type": "assignment", "assigned_offset": t_idx*7, "start_offset": t_idx*7 + 3, "deadline_offset": (t_idx+1)*7, "duration": 35, "late_hrs": 0.0
        }
        
        # Default dates derived from offsets
        def_assigned_d = base_date + datetime.timedelta(days=int(default_t.get("assigned_offset", t_idx*7)))
        def_deadline_d = base_date + datetime.timedelta(days=int(default_t.get("deadline_offset", (t_idx+1)*7)))
        
        start_offset_days = float(default_t.get("start_offset", t_idx*7 + 3))
        def_start_dt = datetime.datetime(base_date.year, base_date.month, base_date.day, 14, 0, 0) + datetime.timedelta(days=start_offset_days)
        def_start_d = def_start_dt.date()
        def_start_t = def_start_dt.time()
        
        def_late_hrs = float(default_t.get("late_hrs", 0.0))
        if def_late_hrs > 0:
            def_sub_dt = datetime.datetime(def_deadline_d.year, def_deadline_d.month, def_deadline_d.day, 23, 59, 0) + datetime.timedelta(hours=def_late_hrs)
        else:
            def_sub_dt = def_start_dt + datetime.timedelta(minutes=int(default_t.get("duration", 35)))

        with st.expander(f"📌 Task {t_idx + 1} Configuration & Dates", expanded=True):
            t_col1, t_col2, t_col3, t_col4 = st.columns([1, 1.2, 1.3, 1.3])
            
            with t_col1:
                st.markdown("**Task Meta**")
                t_type = st.selectbox(
                    f"Task Type (T{t_idx+1})", 
                    ["assignment", "quiz", "lab", "project"], 
                    index=["assignment", "quiz", "lab", "project"].index(default_t.get("task_type", "assignment")), 
                    key=f"type_live_{t_idx}"
                )
                duration_input = st.number_input(
                    f"Duration (Mins) (T{t_idx+1})", 
                    min_value=10, 
                    max_value=300, 
                    value=int(default_t.get("duration", 35)), 
                    step=5, 
                    key=f"dur_live_{t_idx}"
                )

            with t_col2:
                st.markdown("**📅 Assignment Window**")
                assigned_d_input = st.date_input(
                    f"Assigned Date (T{t_idx+1})", 
                    value=def_assigned_d, 
                    key=f"assigned_d_live_{t_idx}"
                )
                deadline_d_input = st.date_input(
                    f"Deadline Date (T{t_idx+1})", 
                    value=def_deadline_d, 
                    key=f"deadline_d_live_{t_idx}"
                )

            with t_col3:
                st.markdown("**⏱️ Activity Start**")
                start_d_input = st.date_input(
                    f"Start Date (T{t_idx+1})", 
                    value=def_start_d, 
                    key=f"start_d_live_{t_idx}"
                )
                start_t_input = st.time_input(
                    f"Start Time (T{t_idx+1})", 
                    value=def_start_t, 
                    key=f"start_t_live_{t_idx}"
                )

            with t_col4:
                st.markdown("**📤 Submission Timing**")
                sub_d_input = st.date_input(
                    f"Submission Date (T{t_idx+1})", 
                    value=def_sub_dt.date(), 
                    key=f"sub_d_live_{t_idx}"
                )
                sub_t_input = st.time_input(
                    f"Submission Time (T{t_idx+1})", 
                    value=def_sub_dt.time(), 
                    key=f"sub_t_live_{t_idx}"
                )

            # Convert inputs into exact datetimes
            actual_start_dt = datetime.datetime.combine(start_d_input, start_t_input)
            actual_deadline_dt = datetime.datetime.combine(deadline_d_input, datetime.time(23, 59, 0))
            actual_sub_dt = datetime.datetime.combine(sub_d_input, sub_t_input)
            actual_assigned_dt = datetime.datetime.combine(assigned_d_input, datetime.time(0, 0, 0))

            # Compute exact Lead Time and Delays
            hrs_before_deadline = (actual_deadline_dt - actual_start_dt).total_seconds() / 3600.0
            sub_delay_hours = max(0.0, (actual_sub_dt - actual_deadline_dt).total_seconds() / 3600.0)
            is_last_minute = 1 if (0.0 < hrs_before_deadline < 24.0) else (1 if hrs_before_deadline <= 0 else 0)
            is_late = 1 if sub_delay_hours > 0.0 else 0
            
            # Dynamic live indicator banner inside the expander
            if hrs_before_deadline < 0:
                st.error(f"⚠️ Task {t_idx+1}: Started after deadline ({-hrs_before_deadline:.1f}h overdue)")
            elif hrs_before_deadline < 24.0:
                st.warning(f"🚨 Task {t_idx+1}: Last-Minute Start! ({hrs_before_deadline:.1f} hours before deadline)")
            else:
                st.success(f"✅ Task {t_idx+1}: Started Early ({hrs_before_deadline:.1f} hours / {hrs_before_deadline/24.0:.1f} days before deadline)")

            inact_gap = max(0.0, (actual_start_dt - actual_assigned_dt).total_seconds() / 3600.0)

            input_records.append({
                "student_id": "USER_INPUT",
                "task_id": f"USER_T{t_idx+1}",
                "task_number": t_idx + 1,
                "task_type": t_type,
                "assigned_at": str(actual_assigned_dt),
                "activity_start": str(actual_start_dt),
                "submission_time": str(actual_sub_dt),
                "deadline": str(actual_deadline_dt.date()),
                "duration_minutes": duration_input,
                "hours_before_deadline": round(max(0.1, hrs_before_deadline), 2),
                "submission_delay_hours": round(sub_delay_hours, 2),
                "last_minute_flag": is_last_minute,
                "late_submission_flag": is_late,
                "inactivity_gap_hours": round(inact_gap, 2),
                "survey_behavior_score": 0.5,
                "procrastination_reasons": distraction_input
            })

    # Big Trigger Button
    run_clicked = st.button("🚀 Analyze Academic Procrastination Pattern Now", use_container_width=True, type="primary")

    # Processing and Displaying Results on User Input
    user_df_raw = pd.DataFrame(input_records)
    user_df_clean = preprocessor.clean_and_format(user_df_raw)
    user_df_feat = engineer.compute_sequential_features(user_df_clean)
    
    user_onset = detector.analyze_student_sequence(user_df_feat)
    user_preds = pipeline.predict(user_df_feat)
    user_latest_pred = user_preds[-1] if user_preds else {"predicted_label": "moderate", "confidence": 0.5, "probabilities": {}}
    user_expl = explainer.generate_explanation(user_df_feat, onset_info=user_onset, pred_info=user_latest_pred)

    if run_clicked:
        st.toast("✅ Live behavioral analysis & explanation refreshed successfully!", icon="🧠")

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(f"### 📊 Live Diagnosis Results for **{student_name_input}**")

    # Top KPI Cards
    u_kpi1, u_kpi2, u_kpi3, u_kpi4 = st.columns(4)
    latest_score = user_df_feat['behavioral_procrastination_score'].iloc[-1]
    u_risk = user_expl['risk_label'].lower()

    with u_kpi1:
        st.markdown('<div class="kpi-card">', unsafe_allow_html=True)
        st.markdown('<div class="kpi-title">Procrastination Status</div>', unsafe_allow_html=True)
        if u_risk == 'high':
            st.markdown('<div class="kpi-value" style="color: #DC2626;">High Risk 🔴</div>', unsafe_allow_html=True)
        elif u_risk == 'moderate':
            st.markdown('<div class="kpi-value" style="color: #D97706;">Moderate 🟡</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="kpi-value" style="color: #16A34A;">Proactive 🟢</div>', unsafe_allow_html=True)
        st.caption(f"Confidence: {user_latest_pred['confidence']*100:.1f}%")
        st.markdown('</div>', unsafe_allow_html=True)

    with u_kpi2:
        st.markdown('<div class="kpi-card">', unsafe_allow_html=True)
        st.markdown('<div class="kpi-title">Behavioral Score</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="kpi-value" style="color: #4F46E5;">{latest_score} <span style="font-size: 1rem; color: #94A3B8;">/ 100</span></div>', unsafe_allow_html=True)
        st.caption("Composite urgency & delay index")
        st.markdown('</div>', unsafe_allow_html=True)

    with u_kpi3:
        st.markdown('<div class="kpi-card">', unsafe_allow_html=True)
        st.markdown('<div class="kpi-title">Onset Emergence</div>', unsafe_allow_html=True)
        if user_onset.get("onset_detected"):
            st.markdown(f'<div class="kpi-value" style="color: #7C3AED;">Task {user_onset["onset_task_number"]} 📍</div>', unsafe_allow_html=True)
            st.caption(f"Trigger: {user_onset.get('onset_task_type', 'Task').capitalize()}")
        else:
            st.markdown('<div class="kpi-value" style="color: #10B981;">No Breakpoint ✅</div>', unsafe_allow_html=True)
            st.caption("Steady habit pattern")
        st.markdown('</div>', unsafe_allow_html=True)

    with u_kpi4:
        st.markdown('<div class="kpi-card">', unsafe_allow_html=True)
        st.markdown('<div class="kpi-title">Last-Minute Tasks</div>', unsafe_allow_html=True)
        u_lm = int(user_df_feat['last_minute_flag'].sum())
        u_tot = len(user_df_feat)
        st.markdown(f'<div class="kpi-value" style="color: #0284C7;">{u_lm}/{u_tot} <span style="font-size: 1rem; color: #94A3B8;">({int(u_lm/u_tot*100)}%)</span></div>', unsafe_allow_html=True)
        st.caption("Started within < 24h")
        st.markdown('</div>', unsafe_allow_html=True)

    # Grounded Explanation Box
    st.markdown(f"""
    <div class="explanation-box">
        <h3 style="margin-top:0; color: #1E293B;">📋 Grounded Explanation & Timeline Evidence</h3>
        <p style="font-size: 1.15rem; font-weight: 600; color: #1E1B4B; margin-bottom: 12px;">
            {user_expl['headline']}
        </p>
        <div style="font-size: 0.95rem; color: #334155; line-height: 1.6;">
            <strong>Timeline Behavioral Evidence:</strong><br>
            {'<br>'.join(['• ' + e for e in user_expl['evidence']])}
        </div>
        <div style="margin-top: 14px; padding-top: 10px; border-top: 1px dashed #CBD5E1; font-size: 0.9rem; color: #475569;">
            💡 <strong>Habit Strategy:</strong> {user_expl['recommendations'][0] if user_expl['recommendations'] else 'Keep up the healthy workflow!'}
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Visual Plots for the User Input
    col_plot1, col_plot2 = st.columns([1.6, 1])

    with col_plot1:
        st.markdown("#### 📅 Sequence Timeline & Preparation Margin")
        fig_user_time = go.Figure()
        
        fig_user_time.add_trace(go.Scatter(
            x=user_df_feat['task_number'],
            y=user_df_feat['hours_before_deadline'],
            mode='lines+markers+text',
            name='Hours Before Deadline',
            line=dict(color='#2563EB', width=3),
            marker=dict(size=14, color='#1D4ED8'),
            text=[f"{h:.1f}h" for h in user_df_feat['hours_before_deadline']],
            textposition="top center"
        ))

        fig_user_time.add_hline(
            y=24.0, 
            line_dash="dash", 
            line_color="#EF4444", 
            annotation_text="⚠️ Last-Minute Zone (<24h)", 
            annotation_position="bottom right"
        )

        if user_onset.get("onset_detected") and user_onset.get("onset_task_number") is not None:
            onset_t = user_onset["onset_task_number"]
            onset_row = user_df_feat[user_df_feat['task_number'] == onset_t]
            if not onset_row.empty:
                onset_y = onset_row['hours_before_deadline'].iloc[0]
                fig_user_time.add_trace(go.Scatter(
                    x=[onset_t],
                    y=[onset_y],
                    mode='markers+text',
                    name='Onset Point',
                    marker=dict(size=22, color='#DC2626', symbol='star'),
                    text=["🚨 ONSET"],
                    textposition="bottom center"
                ))

        fig_user_time.update_layout(
            xaxis_title="Task Sequence",
            yaxis_title="Lead Time (Hours Before Deadline)",
            template="plotly_white",
            height=360,
            margin=dict(l=30, r=30, t=30, b=30)
        )
        st.plotly_chart(fig_user_time, use_container_width=True)

    with col_plot2:
        st.markdown("#### 📈 Procrastination Score Progression")
        fig_user_score = px.bar(
            user_df_feat,
            x='task_number',
            y='behavioral_procrastination_score',
            color='behavioral_procrastination_score',
            color_continuous_scale='Reds',
            labels={'task_number': 'Task', 'behavioral_procrastination_score': 'Score (0-100)'}
        )
        fig_user_score.update_layout(template="plotly_white", height=360, margin=dict(l=30, r=30, t=30, b=30))
        st.plotly_chart(fig_user_score, use_container_width=True)

# ==============================================================================
# MODE 2: EXPLORE STUDENT DATABASE
# ==============================================================================
elif app_mode == "🔍 Explore Student Database (S001 - S451)":
    st.markdown('<div class="main-header">🔍 Student Database Explorer</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Browse all 451 synthetic student profiles, timeline histories, and SHAP feature attributions.</div>', unsafe_allow_html=True)

    all_students = sorted(feat_df['student_id'].unique().tolist())
    student_summaries = engineer.get_student_aggregated_profiles(raw_df)
    
    col_filter1, col_filter2 = st.columns([1, 2])
    with col_filter1:
        risk_filter = st.selectbox("Filter by Risk Category", ["All Students", "High Risk", "Moderate Risk", "Low Risk"])
    
    filtered_students = all_students
    if risk_filter == "High Risk":
        filtered_students = student_summaries[student_summaries['overall_risk_category'] == 'high']['student_id'].tolist()
    elif risk_filter == "Moderate Risk":
        filtered_students = student_summaries[student_summaries['overall_risk_category'] == 'moderate']['student_id'].tolist()
    elif risk_filter == "Low Risk":
        filtered_students = student_summaries[student_summaries['overall_risk_category'] == 'low']['student_id'].tolist()

    with col_filter2:
        selected_student = st.selectbox(
            f"Select Student Profile ({len(filtered_students)} available)",
            filtered_students if filtered_students else all_students,
            index=0
        )

    student_tasks = feat_df[feat_df['student_id'] == selected_student].sort_values(by='task_number').reset_index(drop=True)
    onset_info = detector.analyze_student_sequence(student_tasks)
    model_preds = pipeline.predict(student_tasks)
    latest_pred = model_preds[-1] if model_preds else {"predicted_label": "moderate", "confidence": 0.5, "probabilities": {}}
    explanation = explainer.generate_explanation(student_tasks, onset_info=onset_info, pred_info=latest_pred)

    # Top KPI Metrics Cards
    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    score_val = student_tasks['behavioral_procrastination_score'].iloc[-1]
    risk_label = explanation['risk_label'].lower()
    
    with kpi1:
        st.markdown('<div class="kpi-card">', unsafe_allow_html=True)
        st.markdown('<div class="kpi-title">Procrastination Status</div>', unsafe_allow_html=True)
        if risk_label == 'high':
            st.markdown('<div class="kpi-value" style="color: #DC2626;">High Risk 🔴</div>', unsafe_allow_html=True)
        elif risk_label == 'moderate':
            st.markdown('<div class="kpi-value" style="color: #D97706;">Moderate 🟡</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="kpi-value" style="color: #16A34A;">Proactive 🟢</div>', unsafe_allow_html=True)
        st.caption(f"Confidence: {latest_pred['confidence']*100:.1f}%")
        st.markdown('</div>', unsafe_allow_html=True)

    with kpi2:
        st.markdown('<div class="kpi-card">', unsafe_allow_html=True)
        st.markdown('<div class="kpi-title">Behavioral Score</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="kpi-value" style="color: #4F46E5;">{score_val} <span style="font-size: 1rem; color: #94A3B8;">/ 100</span></div>', unsafe_allow_html=True)
        st.caption("Composite urgency index")
        st.markdown('</div>', unsafe_allow_html=True)

    with kpi3:
        st.markdown('<div class="kpi-card">', unsafe_allow_html=True)
        st.markdown('<div class="kpi-title">Onset Emergence</div>', unsafe_allow_html=True)
        if onset_info.get("onset_detected"):
            st.markdown(f'<div class="kpi-value" style="color: #7C3AED;">Task {onset_info["onset_task_number"]} 📍</div>', unsafe_allow_html=True)
            st.caption(f"Detected on {onset_info.get('onset_date', 'N/A')}")
        else:
            st.markdown('<div class="kpi-value" style="color: #10B981;">None ✅</div>', unsafe_allow_html=True)
            st.caption("Consistent habit maintained")
        st.markdown('</div>', unsafe_allow_html=True)

    with kpi4:
        st.markdown('<div class="kpi-card">', unsafe_allow_html=True)
        st.markdown('<div class="kpi-title">Last-Minute Rate</div>', unsafe_allow_html=True)
        lm_count = int(student_tasks['last_minute_flag'].sum())
        total_t = len(student_tasks)
        lm_rate = (lm_count / max(1, total_t)) * 100
        st.markdown(f'<div class="kpi-value" style="color: #0284C7;">{lm_count}/{total_t} <span style="font-size: 1rem; color: #94A3B8;">({lm_rate:.0f}%)</span></div>', unsafe_allow_html=True)
        st.caption("Tasks started within < 24h")
        st.markdown('</div>', unsafe_allow_html=True)

    # Explanation Box
    st.markdown(f"""
    <div class="explanation-box">
        <h3 style="margin-top:0; color: #1E293B;">📋 Automated Behavioral Explanation</h3>
        <p style="font-size: 1.15rem; font-weight: 600; color: #1E1B4B; margin-bottom: 12px;">
            {explanation['headline']}
        </p>
        <div style="font-size: 0.95rem; color: #334155; line-height: 1.6;">
            <strong>Timeline Evidence Grounding:</strong><br>
            {'<br>'.join(['• ' + e for e in explanation['evidence']])}
        </div>
        <div style="margin-top: 14px; padding-top: 10px; border-top: 1px dashed #CBD5E1; font-size: 0.9rem; color: #475569;">
            💡 <strong>Recommended Habit Strategy:</strong> {explanation['recommendations'][0] if explanation['recommendations'] else 'Maintain current steady work cadence.'}
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Sequence Timeline Chart
    st.subheader(f"Sequence Timeline & Onset Point for {selected_student}")
    fig_timeline = go.Figure()
    fig_timeline.add_trace(go.Scatter(
        x=student_tasks['task_number'],
        y=student_tasks['hours_before_deadline'],
        mode='lines+markers+text',
        name='Hours Before Deadline',
        line=dict(color='#3B82F6', width=3),
        marker=dict(size=12, color='#1D4ED8'),
        text=[f"{h:.1f}h" for h in student_tasks['hours_before_deadline']],
        textposition="top center"
    ))
    fig_timeline.add_hline(y=24.0, line_dash="dash", line_color="#EF4444", annotation_text="Last-Minute (24h)", annotation_position="bottom right")

    if onset_info.get("onset_detected") and onset_info.get("onset_task_number") is not None:
        onset_t = onset_info["onset_task_number"]
        onset_row = student_tasks[student_tasks['task_number'] == onset_t]
        if not onset_row.empty:
            fig_timeline.add_trace(go.Scatter(
                x=[onset_t], y=[onset_row['hours_before_deadline'].iloc[0]],
                mode='markers+text', name='Onset Point',
                marker=dict(size=22, color='#DC2626', symbol='star'),
                text=["🚨 ONSET"], textposition="bottom center"
            ))

    fig_timeline.update_layout(xaxis_title="Task Sequence", yaxis_title="Lead Time (Hours)", template="plotly_white", height=380)
    st.plotly_chart(fig_timeline, use_container_width=True)

    # Raw Task Logs Table
    st.subheader("Raw Task Logs")
    display_cols = ['task_number', 'task_type', 'activity_start', 'deadline', 'hours_before_deadline', 'duration_minutes', 'last_minute_flag', 'behavioral_procrastination_score']
    st.dataframe(student_tasks[[c for c in display_cols if c in student_tasks.columns]], use_container_width=True)

# ==============================================================================
# MODE 3: COHORT ANALYTICS
# ==============================================================================
elif app_mode == "📊 Cohort Analytics & Model Benchmark":
    st.markdown('<div class="main-header">📊 Cohort Analytics & Model Performance</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Evaluation metrics across all 451 student sequences and machine learning benchmarks.</div>', unsafe_allow_html=True)

    all_onsets = detector.detect_all_students(feat_df)
    
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Cohort Students", feat_df['student_id'].nunique())
    with c2:
        st.metric("Model Accuracy", f"{pipeline.model_metadata.get('accuracy', 1.0)*100:.1f}%")
    with c3:
        st.metric("5-Fold CV Weighted F1", f"{pipeline.model_metadata.get('cv_f1_mean', 0.999)*100:.1f}%")
    with c4:
        emerging_pct = (all_onsets['trajectory_type'].str.contains('Emerging').sum() / len(all_onsets)) * 100
        st.metric("Emerging Pattern Rate", f"{emerging_pct:.1f}%")

    st.divider()

    col_c1, col_c2 = st.columns(2)
    with col_c1:
        fig_pie = px.pie(
            all_onsets, names='trajectory_type',
            title="Cohort Behavioral Archetype Distribution",
            color_discrete_sequence=px.colors.qualitative.Pastel
        )
        fig_pie.update_layout(template="plotly_white")
        st.plotly_chart(fig_pie, use_container_width=True)

    with col_c2:
        fig_hist = px.histogram(
            feat_df, x='hours_before_deadline', color='procrastination_label',
            nbins=30, title="Lead-Time Distribution Across Cohort",
            color_discrete_map={'low': '#22C55E', 'moderate': '#F59E0B', 'high': '#EF4444'}
        )
        fig_hist.update_layout(template="plotly_white")
        st.plotly_chart(fig_hist, use_container_width=True)

    st.markdown("### 🔬 Top Global Feature Importances (Random Forest)")
    feat_imps = pipeline.model_metadata.get("feature_importances", {})
    if feat_imps:
        fi_df = pd.DataFrame(list(feat_imps.items())[:10], columns=['Feature', 'Importance'])
        fig_fi = px.bar(fi_df, x='Importance', y='Feature', orientation='h', color='Importance', color_continuous_scale='Blues')
        fig_fi.update_layout(yaxis=dict(autorange="reversed"), template="plotly_white", height=360)
        st.plotly_chart(fig_fi, use_container_width=True)

# ==============================================================================
# MODE 4: UPLOAD CUSTOM CSV
# ==============================================================================
elif app_mode == "📤 Upload Custom Activity CSV":
    st.markdown('<div class="main-header">📤 Bulk CSV Upload & Batch Inference</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Upload a batch CSV file of student logs to compute features and detect procrastination patterns.</div>', unsafe_allow_html=True)

    uploaded_file = st.file_uploader("Upload CSV file matching schema", type=['csv'])
    if uploaded_file is not None:
        try:
            custom_df = pd.read_csv(uploaded_file)
            st.success(f"File uploaded successfully! Loaded {len(custom_df)} rows.")
            
            cleaned_custom = preprocessor.clean_and_format(custom_df)
            custom_features = engineer.compute_sequential_features(cleaned_custom)
            custom_preds = pipeline.predict(custom_features)
            
            custom_features['predicted_risk'] = [p['predicted_label'] for p in custom_preds]
            custom_features['model_confidence'] = [p['confidence'] for p in custom_preds]
            
            st.markdown("### Processed Results & Predictions")
            st.dataframe(custom_features[['student_id', 'task_number', 'hours_before_deadline', 'behavioral_procrastination_score', 'predicted_risk', 'model_confidence']])
            
        except Exception as e:
            st.error(f"Error processing uploaded CSV: {e}")
    else:
        st.info("Upload a CSV file containing academic task logs (columns: student_id, task_number, activity_start, deadline, etc.) to run instant inference.")
