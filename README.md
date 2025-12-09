# 🧠 AI-Based Disease Prediction System

This project is a **dual-input AI disease prediction tool** that uses both  
**machine learning (symptom-based)** and **deep learning (MRI scan-based CNN)**  
to predict possible diseases. Users can either **enter symptoms** or **upload a scan image**,  
making the system helpful for **early diagnosis** and improved decision-making accuracy.

---

## 🎯 Features
- 📝 Symptom-based prediction using ML  
- 🧠 MRI/Scan-based prediction using CNN  
- 🤖 Hybrid AI system (ML + DL)  
- 📊 Shows prediction probability  
- 🖥️ Simple and user-friendly interface  
- ⚡ Fast and automated diagnosis support  

---

## 🛠️ Technologies Used

| Technology / Tool       | Purpose                                      |
|-------------------------|-----------------------------------------------|
| **Python**              | Core development language                    |
| **TensorFlow / Keras**  | CNN model for scan prediction                 |
| **Scikit-learn**        | Symptom-based ML model                        |
| **OpenCV**              | Image preprocessing                           |
| **Tkinter / Streamlit** | User interface                                |
| **NumPy, Pandas**       | Data processing                               |

---

## 📁 Project Structure

| File / Folder             | Description                                      |
|---------------------------|--------------------------------------------------|
| `predict_symptoms.py`     | Symptom-based disease prediction logic           |
| `predict_scan.py`         | MRI/CT scan prediction using CNN                 |
| `model.h5`                | Trained CNN model (not included in repo)         |
| `symptom_model.pkl`       | ML model for symptoms (not included in repo)     |
| `ui.py`                   | Application UI                                   |
| `dataset/`                | Training dataset (not included)                  |
| `sample_images/`          | Few sample images for demonstration              |

---

## ▶️ How It Works

### **1️⃣ Symptom-Based Prediction**
- User selects/input symptoms  
- ML model analyzes patterns  
- Predicts the most likely disease  

### **2️⃣ Scan-Based Prediction**
- User uploads MRI/CT scan  
- Image is preprocessed using OpenCV  
- CNN model predicts disease category  

---

## 📌 Important Note 
```markdown
⚠️ The full dataset and trained model files are not included in this repository.  
Only the core project source code and sample images are provided.
