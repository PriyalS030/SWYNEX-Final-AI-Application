# 🎓 Student Performance Risk Prediction — Final AI Application

**SWYNEX Technologies — AI Internship Task 4: Final AI Application**

---

## 📋 Executive Summary

This project represents the final publication-grade release of the **Student Performance Risk Prediction System**, developed as Task 4 of the SWYNEX Technologies AI Internship. 

The application integrates machine learning classification, model interpretability through relative feature importance analysis, decoupled external Large Language Model (LLM) guidance, an interactive qualitative evaluation dashboard, and robust error handling into a clean, web-based decision-support prototype.

### Key Highlights:
- **Core Classifier:** Random Forest model trained on the UCI Student Performance Dataset ($92.41\%$ Accuracy, $0.8800$ At-Risk F1 score).
- **Intelligent Factor Analysis:** Extracts relative feature contributions directly from `model.feature_importances_` to explain model decision weights without claiming causal mechanics.
- **Decoupled AI API Integration:** Connects securely to Groq API (`qwen/qwen3.8-27b`) for natural-language educational guidance, featuring automated fallback handling if the API is offline or unconfigured.
- **Qualitative Evaluation Dashboard (`/evaluate`):** Predefined benchmark examples, model weight visual chart, and failure case robustness audit.
- **Security & Secret Management:** Zero hardcoded API keys; `.env` credentials strictly excluded from Git.

> **⚠️ Decision-Support Tool Disclaimer:** This application is designed strictly as an educational decision-support tool to assist educators in identifying students who may benefit from early support. It must **not** be used as a definitive judgment of a student's ability, potential, or future performance.

---

## 🎯 Problem Statement

Secondary school academic underperformance often goes unnoticed until period exams occur, leaving limited time for educational intervention. Early identification of students who may be at risk allows educators and school administrators to provide timely academic support, tutoring, and resource allocation.

The objective of this project is to build a transparent, beginner-friendly binary classification system that estimates whether a student is **At Risk** ($G3 < 10$) or **Low Risk** ($G3 \ge 10$) based on academic, behavioral, demographic, and support indicators.

---

## 📊 Dataset Overview

The system uses the **UCI Student Performance Dataset** (Mathematics subject: `student-mat.csv`).

- **Source:** [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/320/student+performance)
- **Sample Size:** 395 secondary school students
- **Target Variable:** Final grade `G3` (range 0–20) mapped to binary classification:
  - **At Risk (1):** $G3 < 10$ — 130 students ($32.9\%$)
  - **Low Risk (0):** $G3 \ge 10$ — 265 students ($67.1\%$)
- **Attributes:** 14 selected features covering prior grades, attendance, study habits, past failures, and support systems.

---

## 🔧 Feature Dictionary & Form Inputs

| Feature Name | UI Label | Data Type | Valid Range / Options |
|--------------|----------|-----------|-----------------------|
| `studytime` | Weekly Study Time | Categorical (Ordinal) | 1: <2h, 2: 2-5h, 3: 5-10h, 4: >10h |
| `failures` | Past Class Failures | Numeric (Discrete) | 0, 1, 2, 3+ |
| `absences` | Number of School Absences | Numeric (Continuous) | 0 to 93 days |
| `schoolsup` | Extra Educational Support | Binary | 1 (Yes), 0 (No) |
| `famsup` | Family Educational Support | Binary | 1 (Yes), 0 (No) |
| `paid` | Extra Paid Classes | Binary | 1 (Yes), 0 (No) |
| `activities` | Extracurricular Activities | Binary | 1 (Yes), 0 (No) |
| `higher` | Wants Higher Education | Binary | 1 (Yes), 0 (No) |
| `internet` | Internet Access at Home | Binary | 1 (Yes), 0 (No) |
| `freetime` | Free Time After School | Categorical (Ordinal) | 1 (Very Low) to 5 (Very High) |
| `goout` | Going Out with Friends | Categorical (Ordinal) | 1 (Very Low) to 5 (Very High) |
| `health` | Current Health Status | Categorical (Ordinal) | 1 (Very Bad) to 5 (Very Good) |
| `G1` | First Period Grade | Numeric (Continuous) | 0 to 20 |
| `G2` | Second Period Grade | Numeric (Continuous) | 0 to 20 |

---

## 🤖 Machine Learning Approach & Pipeline

1. **Preprocessing & Encoding:**
   - Binary text columns (`yes`/`no`) mapped to integer indicators (`1`/`0`).
   - Numerical features standardized using `scikit-learn` `StandardScaler`.
2. **Train/Test Split:**
   - Stratified 80/20 train/test split ($316$ training samples, $79$ test samples) preserving target class proportions ($32.9\%$ At Risk).
3. **Model Exploration:**
   - Evaluated 4 distinct model families across 2 experimental setups (Model A: With G1 & G2 vs. Model B: Without G1 & G2).

---

## 📈 Model Comparison & Benchmark Results

Benchmarking results evaluated on the 20% holdout test set ($79$ students):

### Model A Setup (With Period Grades G1 & G2)

| Classifier Algorithm | Accuracy | Precision (At Risk) | Recall (At Risk) | F1-Score (At Risk) | 5-Fold CV F1 Mean |
|----------------------|----------|---------------------|-------------------|---------------------|-------------------|
| **Random Forest** | **92.41%** | **91.67%** | **84.62%** | **0.8800** | **0.8636** |
| Logistic Regression | 91.14% | 85.19% | 88.46% | 0.8679 | 0.8174 |
| Decision Tree | 86.08% | 77.78% | 80.77% | 0.7925 | 0.7768 |
| Support Vector Machine (SVM) | 86.08% | 80.00% | 76.92% | 0.7843 | 0.7942 |

### Model B Setup (Without Period Grades G1 & G2 — Early Warning Mode)

| Classifier Algorithm | Accuracy | Precision (At Risk) | Recall (At Risk) | F1-Score (At Risk) | 5-Fold CV F1 Mean |
|----------------------|----------|---------------------|-------------------|---------------------|-------------------|
| Decision Tree | 75.95% | 65.22% | 57.69% | 0.6122 | 0.4261 |
| Random Forest | 70.89% | 58.82% | 38.46% | 0.4651 | 0.4173 |
| Logistic Regression | 70.89% | 61.54% | 30.77% | 0.4103 | 0.3713 |
| Support Vector Machine (SVM) | 70.89% | 66.67% | 23.08% | 0.3429 | 0.4214 |

---

## ✅ Selected Model Rationale

**Selection:** **Random Forest Classifier (Model A Setup)**

- **Justification:** Achieved the highest overall test accuracy ($92.41\%$) and the strongest F1-score ($0.8800$) for the At-Risk target class, combined with the highest cross-validation stability ($0.8636$).
- **G1/G2 Analysis:** Removing period grades (Model B) results in a ~27 percentage point drop in F1 performance, demonstrating that prior period grades provide crucial statistical signals for predicting final outcomes.

---

## 💡 Relative Feature Importance Analysis

The application extracts feature importance weights directly from the trained Random Forest model's `feature_importances_` attribute to rank feature contributions.

### Relative Feature Weight Breakdown:
- **Second Period Grade (G2):** $41.2\%$ ($0.4118$) relative feature weight
- **First Period Grade (G1):** $27.0\%$ ($0.2697$) relative feature weight
- **School Absences:** $6.3\%$ ($0.0633$) relative feature weight
- **Past Class Failures:** $4.8\%$ ($0.0482$) relative feature weight
- **Going Out with Friends:** $4.0\%$ ($0.0400$) relative feature weight
- **Other Attributes (Health, Freetime, Studytime, Support, etc.):** $16.7\%$ total combined weight

> **Methodological Note:** G1 and G2 have the highest relative feature importance in the trained Random Forest model. Feature importance indicates the relative contribution of features to the model's predictions; it does not imply causation or determine an individual student's outcome.

---

## 🔌 Decoupled AI API Architecture & Fallback Security

The architecture cleanly decouples machine learning classification from natural-language AI explanations:

```text
User Input → Flask Web App → Random Forest Model → Risk Prediction (At Risk / Low Risk)
                   │
                   ├──> Compute Factor Analysis (scikit-learn feature importances)
                   │
                   └──> Call External Groq LLM API (services/ai_api.py)
                              │
                              ├──> [API Online]  → Returns Educational Explanation
                              └──> [API Offline] → Displays User Fallback Banner
```

### Decoupling Benefits:
1. **Model Independence:** The ML classifier determines risk status independently. The LLM does *not* alter or replace the classification.
2. **API Resiliency:** If the API key is missing, network calls time out, or rate limits occur, the UI displays a clear fallback notice while the ML prediction operates normally.

---

## 🚀 Predefined Evaluation Benchmark Cases (Qualitative Examples)

The application includes 5 predefined evaluation profiles to demonstrate model output consistency across qualitative scenario examples:

> **Important Evaluation Note:** These 5 predefined profiles serve as qualitative evaluation examples and illustrative benchmark cases to verify system functionality; they do **not** constitute a representative test-set accuracy measurement.

| Profile Name | Category | Scenario Description | Expected | Model Output | Match Status |
|--------------|----------|----------------------|----------|--------------|--------------|
| **Consistent High Performer** | Low-Risk Baseline | High G1/G2 (16/17), 0 failures, 2 absences | Low Risk | **Low Risk** | ✅ PASS |
| **Severely Struggling Student** | At-Risk Baseline | Low G1/G2 (5/6), 3 failures, 18 absences | At Risk | **At Risk** | ✅ PASS |
| **Borderline Student** | Mixed / Threshold | Middling G1/G2 (10/9), 1 failure, 8 absences | At Risk | **At Risk** | ✅ PASS |
| **High Grades + High Absences** | Behavioral Contradiction | High G1/G2 (15/15) but 14 absences & high going out | Low Risk | **Low Risk** | ✅ PASS |
| **Low Grades + High Support** | Academic Contradiction | Weak G1/G2 (7/8) despite max study time (4) & full support | At Risk | **At Risk** | ✅ PASS |

---

## ⚠️ Failure Cases & Robustness Audit Matrix

System stability verified under edge-case and failure scenarios:

| Failure Scenario | Input Trigger | Expected Behavior | Actual System Result |
|------------------|---------------|-------------------|----------------------|
| **Missing Required Input** | Omitted `G1` field | Validation catch | Alert: *"Missing value for 'First Period Grade (G1)'"* |
| **Out-of-Range Numeric** | `absences = 150` (>93 max) | Range check catch | Alert: *"'Number of School Absences' must be between 0 and 93"* |
| **Type Mismatch** | `G1 = "abc"` | Conversion catch | Alert: *"Could not convert string to float"* |
| **API Disruption** | Invalid / Missing API Key | Fallback message | Returns: *"AI explanation temporarily unavailable"* |

---

## 🛠️ Step-by-Step Installation & Setup Guide

### Prerequisites
- Python 3.8 or higher
- `pip` package manager

### 1. Clone & Navigate
```bash
cd SWYNEX-Final-AI-Application
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Configure API Key (Optional for LLM Guidance)
```bash
# Copy example configuration template
cp .env.example .env

# Edit .env and enter your Groq API key
# AI_API_KEY=gsk_your_groq_api_key_here
```

### 4. Run Evaluation Suite & Generate Artifacts
```bash
python evaluate.py
```

### 5. Run Verification Audit Script
```bash
python run_audit.py
```

### 6. Launch Web Application
```bash
python app.py
```
Access the application in your browser:
- **Prediction Interface:** `http://127.0.0.1:5000/`
- **Evaluation Dashboard:** `http://127.0.0.1:5000/evaluate`

---

## 💡 User Guide & Live Demo Flow

### Beginner Quick Demo Shortcuts:
- On the home page, click **"⚡ Load Sample At-Risk Student"** or **"⚡ Load Sample Low-Risk Student"** to auto-fill the form with 1 click.
- Click **"Predict Risk"** to see real-time classification, relative feature importance breakdown, and AI educational insight.

---

## 🔐 API Key Security & `.env` Best Practices

- **Zero Hardcoded Secrets:** No API keys are embedded in Python code, HTML, CSS, or documentation.
- **Git Protection:** `.env` is explicitly listed in `.gitignore` and excluded from repository tracking.
- **Safe Template:** `.env.example` provides safe placeholder values (`AI_API_KEY=your_api_key_here`).
- **Data Privacy Confirmation:** Form data entered into the application is processed in-memory during the Flask request lifecycle to generate real-time risk predictions and is **not** persisted to a database.

---

## ⚖️ Responsible AI & Ethical Considerations

1. **Educational Decision-Support Only:** This system is an early-warning aid for educators, **not** a definitive judgment of a student's intelligence, character, or future capabilities.
2. **Preventing Labeling Stigmas:** Risk classifications must be handled with discretion to prevent self-fulfilling prophecies or biased treatment.
3. **Data Privacy:** Inputs are processed strictly in-memory per request and are not stored or sold.
4. **Contextual Awareness:** Automated models cannot capture unobserved qualitative factors such as personal motivation, home emergencies, or learning differences.

---

## ⚠️ Technical Limitations & Scope

- **Sample Size & Scope:** Trained on $395$ students from 2 Portuguese secondary schools; findings may not generalize across different countries or school systems.
- **Period Grade Dominance:** G1 and G2 account for over $68\%$ of relative model feature weight ($41.2\%$ and $27.0\%$ respectively), reducing early-warning predictive power before period exams take place.
- **Cross-Sectional Dataset:** Captures snapshot data rather than multi-year longitudinal trends.

---

## 📚 Dataset Reference & Citation

P. Cortez and A. Silva. "Using Data Mining to Predict Secondary School Student Performance." In A. Brito and J. Teixeira (Eds.), *Proceedings of 5th Future Business Technology Conference (FUBUTEC 2008)*, pp. 5-12, Porto, Portugal, April, 2008, EUROSIS, ISBN 978-9077381-39-7.

Available at: [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/320/student+performance)

---

*Final AI Application developed for the SWYNEX Technologies AI Internship Program (Task 4).*
