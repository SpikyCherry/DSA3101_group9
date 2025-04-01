# 📥 Data Instructions for Personalized Marketing Campaigns in Banking

## 1. External Dataset Downloads (Kaggle)

This project uses **two external datasets**, which must be manually downloaded from Kaggle due to file size and access limitations.

### 🔗 Dataset 1: Bank Churn Prediction
- **Kaggle Source:** [Bank Churn](https://www.kaggle.com/datasets/sakshigoyal7/credit-card-customers/data)
- **Download Instructions:**
   1. Go to the Kaggle link above.
   2. Download the CSV files manually.
   3. Place them in:
      ```
      data/raw/
      ```

### 🔗 Dataset 2: Banking Marketing Targets
- **Kaggle Source:** [Banking Dataset - Marketing Targets](https://www.kaggle.com/datasets/prakharrathi25/banking-dataset-marketing-targets)
- **Download Instructions:**
   1. Navigate to the dataset page.
   2. Download the CSV file(s).
   3. Save them to:
      ```
      data/raw/
      ```

> ✅ **Note:** You do not need to unzip or rename the files. Just ensure the raw data is correctly placed in `data/raw/`.

---

## 2. Generating Partly Synthetic Data

Some parts of the project require synthetic datasets, especially for questions that involve custom features or data augmentation.

### ⚙️ Generate Synthetic Data
Run the following command from the root of the repository:
```bash
python data/generate_synthetic_data.py --output ./data/synthesized/
```

- This script outputs synthetic datasets used in various tasks.
- Make sure the outputs appear in:
  ```
  data/synthesized/
  ```

---

## 3. Data Folder Structure
```
data/
├── raw/                      # Manually downloaded Kaggle datasets
├── synthesized/              # Synthesized datasets saved here
├── processed/                # Preprocessed datasets saved by question_x/preprocess.py
├── generate_synthetic_data.py  # Script to create synthetic data
└── data_dictionary.md        # Description of all variables in datasets
```

---

## 4. Next Steps
After placing the datasets in the correct folders, proceed with the question-specific preprocessing scripts:
```bash
cd question_1/
python preprocess.py
```

Preprocessed outputs will be saved in `data/processed/` and used by each question's modeling pipeline.
