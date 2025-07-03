### Repository of the MIPT 2025 Hackathon's team "The piglets rubbed each other's backs" 

```
git clone https://github.com/...
cd pigs_mipt_2025_hackathon
```

---

### Environment Setup
```bash
# Create virtual environment
python3 -m venv .venv

# Activate environment (Linux/Mac)
source .venv/bin/activate

# Activate environment (Windows)
.venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

---

### Repository Structure
```
ldl-classification/
├── data/
│   ├── ppgs/            # Folder with PPG signals (.npy files)
│   ├── train.csv        # Training metadata (ID, ЛПНП)
│   └── test.csv         # Test metadata (ID only)
│
├── outputs/             # Submission files
│   └── baseline_submit.csv
│
├── model/               # Trained model weights
│   └── best_model.pth
│
├── first_try.ipynb      # Solution notebook
├── baseline.ipynb       # Baseline solution notebook from Sber
└── requirements.txt     # Python dependencies
```

