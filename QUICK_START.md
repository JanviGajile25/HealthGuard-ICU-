# Quick Start Guide

## ⚡ TL;DR

```bash
# First time ONLY (one time setup)
python main.py all

# Every other time (daily use)
python main.py dashboard
```

## 📋 Detailed Guide

### First Time Setup (ONE TIME ONLY)

When you first clone/download the project:

```bash
# Step 1: Install dependencies
pip install -r requirements.txt

# Step 2: Setup everything (preprocess + train)
python main.py all
```

**This takes 5-10 minutes and creates:**
- `data/X_train.npy` - Training data
- `data/y_train.npy` - Training labels
- `data/X_test.npy` - Test data
- `data/y_test.npy` - Test labels
- `data/sepsis_lstm.pth` - Trained model

**You only do this ONCE!**

### Daily Usage (After Setup)

After the first-time setup, you can use the system immediately:

```bash
# Launch the dashboard (most common)
python main.py dashboard

# Or make a prediction
python main.py predict
```

**No preprocessing or training needed!** The system uses existing data/model.

### Check Status

Not sure if you've set up the system?

```bash
python main.py
```

Output shows:
```
Status:
  Data preprocessed: ✓ Yes  (or ✗ No)
  Model trained:     ✓ Yes  (or ✗ No)
```

## 🔄 When to Re-run Setup

You only need to re-run preprocessing/training if:

1. **New data**: You have updated patient data
2. **Better model**: You want to retrain with different parameters
3. **Missing files**: Data or model files were deleted

To re-run:
```bash
# Re-preprocess data
python main.py preprocess

# Re-train model
python main.py train

# Or both
python main.py all
```

The system will ask for confirmation before overwriting existing files.

## 📊 Common Workflows

### Workflow 1: First Time User
```bash
pip install -r requirements.txt
python main.py all              # Wait 5-10 minutes
python main.py dashboard        # Use the system
```

### Workflow 2: Daily User
```bash
python main.py dashboard        # That's it!
```

### Workflow 3: Developer/Researcher
```bash
# Check status
python main.py

# Make changes to code...

# Test prediction
python main.py predict

# Test dashboard
python main.py dashboard

# Retrain if needed
python main.py train
```

## ❓ FAQ

### Q: Do I need to run `python main.py all` every time?
**A: NO!** Only the first time. After that, just use `python main.py dashboard`.

### Q: How long does setup take?
**A: 5-10 minutes** for preprocessing + training. But you only do it once!

### Q: Can I skip preprocessing?
**A: Yes**, if you already have the data files in `data/` directory.

### Q: Can I skip training?
**A: Yes**, if you already have `data/sepsis_lstm.pth` model file.

### Q: What if I get an error?
**A: Run** `python main.py all` to regenerate everything.

### Q: How do I update the model?
**A: Run** `python main.py train` - it will ask for confirmation.

### Q: Where is my data stored?
**A: In** `data/` directory (X_train.npy, y_train.npy, etc.)

### Q: Where is my model stored?
**A: In** `data/sepsis_lstm.pth`

## 🎯 Summary

| Task | Command | Frequency |
|------|---------|-----------|
| **First time setup** | `python main.py all` | Once |
| **Daily use** | `python main.py dashboard` | Every time |
| **Check status** | `python main.py` | When unsure |
| **Re-train** | `python main.py train` | Rarely |
| **Re-preprocess** | `python main.py preprocess` | Rarely |

## 🚀 Ready to Go!

After first-time setup, you can use the dashboard anytime:

```bash
python main.py dashboard
```

The dashboard will open at: http://localhost:8501

**No preprocessing or training needed!** 🎉
