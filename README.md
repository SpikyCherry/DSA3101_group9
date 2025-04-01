# 🧠 Machine Learning for Personalized Marketing Campaigns in Banking

This repository contains a collaborative data science project aimed at helping a bank optimize its marketing strategy using machine learning. The goal is to develop personalized, data-driven marketing campaigns that increase customer engagement, conversion rates, and marketing ROI.

A team of 10 data scientists collaborated to explore various supervised and unsupervised learning tasks using real-world and synthetic customer data.

---

## 📁 Repository Structure

```
├── data/
│   ├── raw/                      # Place manually downloaded datasets here
│   ├── synthesized/              # Synthetic data generated for specific tasks
│   ├── processed/                # Cleaned data saved by each question
│   ├── generate_synthetic_data.py  # Script to generate synthetic data
│   ├── data_dictionary.md        # Descriptions of dataset features
│   └── data_instructions.md      # Instructions to download/generate data
│
├── utils/                        # Shared functions (e.g., encoding)
│
├── models/
│   ├── question_1/               # Trained models saved per question
│   ├── question_2/
│   └── ...
│
├── EDA/                          # Exploratory notebooks for global understanding
│
├── question_x/
│   ├── preprocess.py             # Load and clean data for this question
│   ├── train_model.py            # Final model implementation
│   └── analysis.ipynb            # Full workflow: EDA, training, visualization
```

---

## 🔄 Project Workflow

1. **Data Collection**
   - Two real-world datasets from Kaggle are used.
   - Synthetic data is generated to augment the dataset where necessary.

2. **Preprocessing**
   - Shared and question-specific transformations are applied.
   - Final datasets are saved in `data/processed/`.

3. **Modeling**
   - Each question runs a separate training pipeline using their preprocessed dataset.
   - Trained models are saved under `models/question_x/`.

4. **Notebook Reporting**
   - Each `analysis.ipynb` provides an end-to-end record: data exploration, model comparisons, visualization, and insights.

---

## 📚 Data Access and Documentation

- 📄 [Data Download & Generation Instructions](data/data_instructions.md)
- 📘 [Data Dictionary](data/data_dictionary.md)

Please follow these documents **before running any scripts**.

---

## 📝 Reproducibility

To run a specific question:

```bash
python question_x/preprocess.py
python question_x/train_model.py
# or open question_x/analysis.ipynb
```

