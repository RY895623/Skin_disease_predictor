# 🔬 DermAI — AI-Powered Skin Disease Classifier

<div align="center">

![Python](https://img.shields.io/badge/Python-3.10-blue?style=for-the-badge&logo=python)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.15-orange?style=for-the-badge&logo=tensorflow)
![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-green?style=for-the-badge&logo=fastapi)
![Groq](https://img.shields.io/badge/Groq-LLaMA3-purple?style=for-the-badge)
![License](https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge)

**An end-to-end AI web application that classifies skin diseases from dermoscopic images using MobileNetV2 + Transfer Learning, with LLM-powered explanations via Groq.**

[Demo](#demo) • [Features](#features) • [Tech Stack](#tech-stack) • [Installation](#installation) • [Model](#model-details) • [Results](#results)

</div>

---

## 📌 Overview

DermAI is a full-stack machine learning application that detects **7 skin conditions** from dermoscopic images. It combines a fine-tuned **MobileNetV2** deep learning model trained on the **HAM10000** dataset with a **Groq LLaMA3** language model that explains the diagnosis in plain English.

> ⚠️ **Medical Disclaimer:** DermAI is an educational tool only. It is not a substitute for professional medical advice. Always consult a qualified dermatologist.

---

## ✨ Features

- 🧠 **MobileNetV2** with two-phase transfer learning (frozen → fine-tuned)
- 📊 **7-class skin disease classification** on HAM10000 dataset
- ⚖️ **Class imbalance handling** via sqrt-softened class weights
- 🎯 **Confidence thresholding** — returns "uncertain" when confidence < 60%
- 🤖 **Groq LLaMA3 explanations** — structured AI explanation for each diagnosis
- 📈 **Real-time probability bars** for all 7 classes
- 🖼️ **Drag & drop image upload** with live preview
- ⚡ **Fast inference** — ~278ms per prediction
- 🏥 **Medical disclaimer** and responsible AI design

---

## 🎯 Detected Conditions

| Code | Condition | Severity |
|------|-----------|----------|
| `nv` | Melanocytic Nevi (Mole) | 🟢 Low |
| `mel` | Melanoma | 🔴 High |
| `bkl` | Benign Keratosis | 🟢 Low |
| `bcc` | Basal Cell Carcinoma | 🔴 High |
| `akiec` | Actinic Keratosis / Intraepithelial Carcinoma | 🟡 Moderate |
| `df` | Dermatofibroma | 🟢 Low |
| `vasc` | Vascular Lesion | 🟡 Moderate |

---

## 🛠️ Tech Stack

| Layer | Technology |
|-------|------------|
| **ML Framework** | TensorFlow 2.15, Keras |
| **Model** | MobileNetV2 (ImageNet pretrained) |
| **Dataset** | HAM10000 (10,015 dermoscopic images) |
| **Backend** | FastAPI + Uvicorn |
| **LLM** | Groq API (LLaMA3-8B) |
| **Frontend** | HTML, CSS, JavaScript (Jinja2 templates) |
| **Data Processing** | Pandas, NumPy, Scikit-learn, Pillow |

---

## 🧠 Model Details

### Architecture
```
Input (128×128×3)
    ↓
MobileNetV2 (ImageNet weights)
    ↓
GlobalAveragePooling2D
    ↓
BatchNormalization
    ↓
Dense(256, relu) → Dropout(0.4)
    ↓
Dense(128, relu) → Dropout(0.3)
    ↓
Dense(7, softmax)
```

### Training Strategy
- **Phase 1** — Base model fully frozen, head trained for 8 epochs (`lr=1e-3`)
- **Phase 2** — Top 20 layers unfrozen, fine-tuned for 12 epochs (`lr=1e-5`)
- **Class weights** — sqrt-softened balanced weights to handle severe class imbalance
- **Augmentation** — rotation, zoom, flips, brightness shifts (training only)
- **Callbacks** — EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

### Dataset — HAM10000

| Class | Images | Weight |
|-------|--------|--------|
| nv (Mole) | 6,705 | 0.46 |
| mel (Melanoma) | 1,113 | 1.13 |
| bkl (Benign Keratosis) | 1,099 | 1.14 |
| bcc (Basal Cell) | 514 | 1.67 |
| akiec | 327 | 2.09 |
| vasc | 142 | 3.17 |
| df | 115 | 3.54 |

---

## 📊 Results

| Metric | Value |
|--------|-------|
| Validation Accuracy | **72.1%** |
| Best Val Loss | 0.7774 |
| Inference Speed | ~278ms |
| Training Image Size | 128×128 |
| Classes | 7 |
| Random Baseline | 14.3% |

> The model achieves **5× better than random** on a severely imbalanced 7-class medical dataset, trained entirely on CPU.

---

## 🚀 Installation

### 1. Clone the repository
```bash
git clone https://github.com/yourusername/dermAI.git
cd dermAI
```

### 2. Create virtual environment
```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Mac/Linux
source venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Set up environment variables
Create a `.env` file in the root directory:
```env
GROQ_API_KEY=your_groq_api_key_here
```
Get a free Groq API key at [console.groq.com](https://console.groq.com)

### 5. Download HAM10000 dataset (for training)
Download from [Kaggle](https://www.kaggle.com/datasets/kmader/skin-lesion-analysis-toward-melanoma-detection) and place in:
```
dermAI/
└── data/
    ├── HAM10000_images_part_1/
    ├── HAM10000_images_part_2/
    └── HAM10000_metadata.csv
```

### 6. Train the model
```bash
python train_model.py
```

### 7. Run the app
```bash
python run.py
```

Open **http://127.0.0.1:8000** in your browser.

---

## 📁 Project Structure

```
dermAI/
├── app/
│   ├── main.py              # FastAPI app + routes
│   ├── models/
│   │   ├── skin_model.h5    # Trained model
│   │   └── class_mapping.json
│   ├── templates/
│   │   └── index.html       # Frontend UI
│   ├── static/              # CSS / JS assets
│   └── utils/
│       ├── predict.py       # Image preprocessing
│       └── explanation.py   # Groq LLM integration
├── data/                    # HAM10000 dataset (not committed)
├── train_model.py           # Full training pipeline
├── create_dummy_model.py    # Quick test model
├── organize_data.py         # Dataset preparation script
├── requirements.txt
├── run.py
└── .env                     # API keys (not committed)
```

---

## 🔮 Future Improvements

- [ ] Increase image size to 224×224 for higher accuracy (+5–8%)
- [ ] Train EfficientNetB3 on Google Colab GPU (~82% accuracy)
- [ ] Add Test-Time Augmentation (TTA) for better confidence
- [ ] Deploy to HuggingFace Spaces for public access
- [ ] Add confusion matrix & per-class metrics dashboard
- [ ] Convert to TFLite for mobile deployment
- [ ] Expand to ISIC 2019 dataset (25,000+ images)

---

## 🤝 Contributing

Pull requests are welcome! For major changes, please open an issue first to discuss what you would like to change.

---

## 📄 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

---

## 👨‍💻 Author

**Your Name**
- GitHub: [@yourusername](https://github.com/yourusername)
- LinkedIn: [linkedin.com/in/yourprofile](https://linkedin.com/in/yourprofile)

---

<div align="center">

⭐ **Star this repo if you found it helpful!** ⭐

*Built with ❤️ as a fresher AI/ML project*

</div>
