# 📊 Advanced Data Dashboard

A multi-page interactive data analytics dashboard built with **Streamlit**, featuring sales analytics, machine learning insights, and PDF report generation.

---

## 🚀 Features

- 📈 **Interactive Visualizations** — Dynamic charts and graphs powered by Plotly
- 🤖 **ML-Powered Insights** — Predictive analytics using scikit-learn
- 🔍 **SHAP Explainability** — Understand model predictions with SHAP values
- 📄 **PDF Report Export** — Generate downloadable reports via ReportLab
- 🗂️ **Multi-Page Layout** — Clean sidebar navigation across multiple pages
- 📅 **Date-Aware Data** — Time-series support with automatic date parsing

---

## 🗃️ Project Structure

```
dashboard_project/
│
├── app.py                  # Main entry point, Streamlit config & homepage
├── utils.py                # Shared utilities (data loading & preprocessing)
├── requirements.txt        # Python dependencies
│
├── pages/                  # Streamlit multi-page app pages
├── data/                   # Raw and processed datasets
│   └── data.csv            # Sample sales dataset (date, region, category, sales, profit)
└── screenshots/            # App screenshots for documentation
```

---

## 📦 Installation

**1. Clone the repository**
```bash
git clone https://github.com/your-username/dashboard-project.git
cd dashboard-project
```

**2. Create and activate a virtual environment**
```bash
python -m venv venv

# On Windows
venv\Scripts\activate

# On macOS/Linux
source venv/bin/activate
```

**3. Install dependencies**
```bash
pip install -r requirements.txt
```

---

## ▶️ Running the App

```bash
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`.

---

## 📊 Sample Data

The dashboard uses a CSV dataset with the following columns:

| Column     | Type     | Description                        |
|------------|----------|------------------------------------|
| `date`     | datetime | Date of the transaction            |
| `region`   | string   | Geographic region (North/South/East/West) |
| `category` | string   | Product category                   |
| `sales`    | float    | Sales amount                       |
| `profit`   | float    | Profit amount                      |

You can replace `data/data.csv` with your own dataset — just ensure it follows the same column structure, or update `utils.py` accordingly.

---

## 🛠️ Tech Stack

| Library        | Purpose                              |
|----------------|--------------------------------------|
| `streamlit`    | Web app framework                    |
| `pandas`       | Data manipulation                    |
| `plotly`       | Interactive charts                   |
| `scikit-learn` | Machine learning models              |
| `shap`         | Model explainability                 |
| `matplotlib`   | Static visualizations                |
| `reportlab`    | PDF report generation                |

---

## 📋 Requirements

- Python 3.8+
- See `requirements.txt` for all dependencies

---

## 🤝 Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a new branch (`git checkout -b feature/your-feature`)
3. Commit your changes (`git commit -m 'Add your feature'`)
4. Push to the branch (`git push origin feature/your-feature`)
5. Open a Pull Request

---

## 📄 License

This project is licensed under the [MIT License](LICENSE).
