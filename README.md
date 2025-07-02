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
├── first_try.ipynb      # Solution notebook from Deepseek and Linar
├── baseline.ipynb       # Baseline solution notebook from Sber
└── requirements.txt     # Python dependencies
```

---

### Key Features
1. **Preprocessing Pipeline**:
   - Bandpass filtering (0.8-5 Hz)
   - Signal normalization
   - Length standardization (2560 samples)
   
2. **CNN Architecture**:
   - 4 residual blocks with skip connections
   - Multi-head attention mechanism
   - Adaptive max pooling

3. **Training Configuration**:
   - 5 epochs with early stopping
   - AdamW optimizer (lr=0.001)
   - Batch size 10
   - Stratified 80/20 train-val split

---

### Expected Output
```
Epoch 1/5
Train Loss: 0.6214, Val Loss: 0.5892, Val AUC: 0.7013
Epoch 2/5
Train Loss: 0.5728, Val Loss: 0.5611, Val AUC: 0.7236
...
Best val AUC: 0.7619

Test predictions saved to outputs/baseline_submit.csv
```

---

### Customization Options
1. **Modify architecture**:
   ```python
   # In PPG_NN class
   self.rb_4 = MyResidualBlock(downsample=True)  # Add 5th residual block
   ```
   
2. **Adjust training**:
   ```python
   # Before training
   optimizer = torch.optim.AdamW(model.parameters(), 
                                 lr=0.0005, 
                                 weight_decay=0.01)
   train_loader = DataLoader(..., batch_size=16)
   ```

3. **Add augmentations**:
   ```python
   # In PPGDataset.__getitem__
   ppg += np.random.normal(0, 0.01, len(ppg))  # Gaussian noise
   ppg = np.roll(ppg, np.random.randint(-100,100))  # Random shift
   ```

---

### Clinical Notes
This solution analyzes PPG waveforms for LDL prediction by detecting:
1. **Arterial stiffness changes**: Affects pulse wave velocity
2. **Vascular compliance**: Alters diastolic wave characteristics
3. **Autonomic regulation**: Modulates heart rate variability

For optimal performance with limited data (553 samples):
- Uses small kernel sizes (5-9 samples)
- Implements dropout (p=0.5)
- Employs attention-based feature selection
- Includes residual connections for stable training

---

### Troubleshooting
1. **CUDA out of memory**:
   - Reduce batch size (8 or 4)
   - Use `torch.cuda.empty_cache()`
   
2. **Missing dependencies**:
   ```bash
   pip install --upgrade -r requirements.txt
   ```
   
3. **Data path issues**:
   Verify file structure matches:
   ```
   data/ppgs/k31__1__1.npy
   data/train.csv
   ```
