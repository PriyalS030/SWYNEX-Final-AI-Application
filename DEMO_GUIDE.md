# 📹 Silent Screen-Recording Demo Guide & Video Script

**SWYNEX Technologies — Final AI Application (Task 4)**

This guide provides a step-by-step presentation structure for recording a **silent screen-recording video** with on-screen text overlays (captions) for your LinkedIn submission.

---

## 🎬 Video Recording Checklist

- **Format:** Silent screen recording (1080p widescreen or 16:9 vertical format for LinkedIn).
- **Target Duration:** ~90 seconds to 2 minutes.
- **Tools Needed:** OBS Studio, Loom, Windows Snipping Tool (Video), or QuickTime + any basic video editor (CapCut, Premiere, or Canva for adding text overlays).

---

## ⏱️ Step-by-Step Screen-Recording Sequence

### Scene 1: Introduction & Project Overview (0:00 - 0:15)
- **Visual Action:** 
  - Start on the home page (`http://127.0.0.1:5000/`) showing the header: **"🎓 Student Performance Risk Prediction — SWYNEX Task 4"**.
  - Smoothly scroll down past the three organized form sections (*Academic Performance*, *Support & Education*, *Lifestyle & Well-being*).
- **On-Screen Text Overlay Suggestions:**
  > 📌 **SWYNEX AI Internship Task 4: Final AI Application**
  > 🎓 Student Performance Risk Prediction System
  > 📊 ML Classifier (Random Forest) + Decoupled AI Explanation API

---

### Scene 2: Beginner Quick-Demo & Real-Time Prediction (0:15 - 0:45)
- **Visual Action:** 
  - Click the **"⚡ Load Sample At-Risk Student"** shortcut button at the top of the form.
  - Show the form fields auto-filling (e.g., G1=5, G2=6, Absences=18, Past Failures=3).
  - Scroll down and click the prominent **"Predict Risk"** button.
  - The page reloads showing the Result Card:
    - ⚠️ **"At Risk"** red status badge.
    - **Relative Feature Importance Breakdown** (G2 Grade: 38.4%, G1 Grade: 18.2%, Absences: 11.8%).
    - **AI Insight Box** (LLM generated educational context).
- **On-Screen Text Overlay Suggestions:**
  > ⚡ **Quick Demo Mode:** Loaded sample at-risk student profile.
  > 🤖 **Random Forest Model:** Classifies student as **"At Risk"** ($92.4\%$ Accuracy).
  > 💡 **Feature Importance Analysis:** G1 & G2 period grades hold highest relative feature weight.
  > 💬 **Decoupled AI API:** Provides natural-language educational guidance.

---

### Scene 3: Testing Low-Risk Student Profile (0:45 - 1:05)
- **Visual Action:** 
  - Click the **"⚡ Load Sample Low-Risk Student"** button.
  - Show form fields auto-filling (G1=16, G2=17, Absences=2, 0 Failures, High Study Time).
  - Click **"Predict Risk"**.
  - Show the updated Result Card with the green **"Low Risk"** status badge.
- **On-Screen Text Overlay Suggestions:**
  > 🟩 **Sample Low-Risk Profile:** High period grades, low absences, high study time.
  > ✅ **Prediction Result:** **"Low Risk"** with tailored AI educational feedback.

---

### Scene 4: Interactive Evaluation & Benchmark Dashboard (1:05 - 1:30)
- **Visual Action:** 
  - Click **"Evaluation & Analysis"** in the top navigation bar (`http://127.0.0.1:5000/evaluate`).
  - Scroll through the **Predefined Benchmark Matrix** showing the 5 qualitative evaluation cases.
  - Scroll down to highlight the embedded **Random Forest Feature Importance Bar Chart** (`results/feature_importance.png`).
- **On-Screen Text Overlay Suggestions:**
  > 📊 **Evaluation Dashboard:** 5 qualitative benchmark profiles.
  > 📈 **Feature Weight Visual:** Random Forest `feature_importances_` extraction.
  > 📌 *Note: Feature importance indicates relative model contribution; it does not imply causation.*

---

### Scene 5: Error Handling & Responsible AI Disclaimer (1:30 - 1:50)
- **Visual Action:** 
  - Return to Prediction tab (`/`).
  - Enter `Absences = 150` (>93 max) and click **"Predict Risk"**.
  - Show the red validation error banner: *"Number of School Absences must be between 0 and 93"*.
  - Scroll down to highlight the **Responsible AI Disclaimer** on the Result Card.
- **On-Screen Text Overlay Suggestions:**
  > 🛡️ **Robust Validation:** Prevents invalid inputs without crashing.
  > ⚖️ **Responsible AI:** Decision-support tool only — not a definitive judgment of student ability.

---

### Scene 6: Closing & Submission Attribution (1:50 - 2:00)
- **Visual Action:** 
  - Return to header/footer showing GitHub repository link and dataset citation.
- **On-Screen Text Overlay Suggestions:**
  > 🌐 **Full Codebase & Documentation on GitHub:** `SWYNEX-Final-AI-Application`
  > 📊 **Dataset:** UCI Student Performance Dataset (Mathematics)
  > 🎓 Built for **@SWYNEX Technologies** AI Internship Task 4

---

## 💡 Quick Tips for Recording

1. **Mouse Movements:** Keep cursor movements slow, steady, and deliberate.
2. **Text Font:** Use clean, high-contrast text overlays (white text with dark semi-transparent background box).
3. **LinkedIn Post Text:** When posting on LinkedIn, tag `@SWYNEX Technologies`, include your GitHub repository URL, and add relevant hashtags: `#MachineLearning #Python #Flask #AI #EducationTech #DataScience #SWYNEX`.
