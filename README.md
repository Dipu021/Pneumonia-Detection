# 🫁 PneumoScan AI — Chest X-ray Pneumonia Detection

> A production-ready deep learning application that detects **Pneumonia** from chest X-ray images using a fine-tuned **ResNet-18** model — built with PyTorch and deployed with Streamlit.


## 📌 Overview

PneumoScan AI is an end-to-end machine learning project demonstrating the full pipeline from raw medical imaging data to a working web application. It uses **transfer learning** on a ResNet-18 backbone to classify chest X-rays as **Normal** or **Pneumonia**, achieving ~95% accuracy and ~96% recall on pneumonia cases.

> ⚠️ **Disclaimer:** This tool is for educational and research purposes only. It is **not a certified medical device** and must not replace professional clinical diagnosis.

---

## ✨ Features

- 📤 Upload chest X-ray images (JPG / PNG)
- 🔍 Real-time AI inference — Normal vs Pneumonia
- 📊 Confidence score & class probability breakdown
- 🩺 Do's & Don'ts care guidance
- 🥗 Diet & nutrition recommendations for recovery
- 🏥 Nearest hospital & specialist finder (Google Maps integration)
- 🌑 Professional clinical dark-theme UI

---

## 🧠 Model Details

| Property | Value |
|---|---|
| Architecture | ResNet-18 |
| Training strategy | Transfer learning (ImageNet pretrained) |
| Output classes | Normal · Pneumonia |
| Input size | 224 × 224 px |
| Loss function | CrossEntropyLoss |
| Optimizer | Adam |
| Framework | PyTorch |

---

## 📊 Performance

| Metric | Value |
|---|---|
| Accuracy | ~95% |
| Pneumonia Recall | ~96% |
| False Negatives | 32 |

> In medical AI, **minimizing false negatives is critical.** A missed pneumonia diagnosis can be life-threatening. This model prioritizes high recall on pneumonia cases to reduce the chance of missed diagnoses.

---

## 🗂 Project Structure

```
pneumonia-detection/
│
├── app.py                  # Streamlit web application
├── src/
│   ├── train.py            # Model training script
│   ├── evaluate.py         # Evaluation & metrics
│   └── dataset.py          # Data loading & transforms
├── model/
│   └── pneumonia_model.pth # Trained weights (not included — see below)
├── requirements.txt
├── README.md
└── screenshot.png
```

---

## ⚙️ Installation & Usage

**1. Clone the repository**
```bash
git clone https://github.com/Dipu021/Pneumonia-Detection.git
cd pneumonia-detection
```

**2. Install dependencies**
```bash
pip install -r requirements.txt
```

**3. Train the model** *(skip if you have weights)*
```bash
python src/train.py
```

**4. Run the app**
```bash
streamlit run app.py
```

---

## 📁 Dataset

- Source: [Chest X-Ray Images (Pneumonia) — Kaggle](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia)
- Classes: `NORMAL` / `PNEUMONIA`
- Dataset is **not included** in this repo due to size. Download it from Kaggle and place it under `data/`.

---

## 🔮 Future Improvements

- [ ] Reduce false negatives further with focal loss
- [ ] Grad-CAM heatmap visualization for explainability
- [ ] Deploy app on Streamlit Cloud / Hugging Face Spaces
- [ ] Extend to multi-class (Bacterial vs Viral Pneumonia)
- [ ] Add DICOM format support for real clinical images

---

## 🧠 Key Learnings

- Built an end-to-end ML pipeline from data loading to web deployment
- Applied transfer learning with ResNet-18 for medical image classification
- Understood why **recall > accuracy** matters in healthcare AI
- Deployed a real-time inference model using Streamlit

---

## 👨‍💻 Author

**Dipu Thakur ** — https://github.com/Dipu021
