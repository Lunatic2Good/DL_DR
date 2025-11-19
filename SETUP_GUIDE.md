# Setup Guide - Installation Instructions

## Step 1: Install Python

If Python is not installed on your system, follow these steps:

### Option A: Install from Python.org (Recommended)
1. Go to https://www.python.org/downloads/
2. Download Python 3.10 or 3.11 (64-bit)
3. **IMPORTANT**: During installation, check the box "Add Python to PATH"
4. Click "Install Now"

### Option B: Install from Microsoft Store
1. Open Microsoft Store
2. Search for "Python 3.11" or "Python 3.10"
3. Click "Install"

### Verify Installation
After installation, open a **new** PowerShell or Command Prompt window and run:
```bash
python --version
```

You should see something like: `Python 3.11.x`

## Step 2: Install Dependencies

Once Python is installed, navigate to the project directory and run:

```bash
pip install -r requirements.txt
```

### If `pip` command doesn't work, try:
```bash
python -m pip install -r requirements.txt
```

### If you have multiple Python versions, use:
```bash
python3 -m pip install -r requirements.txt
```

## Step 3: Troubleshooting

### Issue: "pip is not recognized"
**Solution**: Use `python -m pip` instead of just `pip`

### Issue: "Permission denied" or "Access denied"
**Solution**: Run PowerShell/Command Prompt as Administrator, or use:
```bash
pip install --user -r requirements.txt
```

### Issue: TensorFlow installation fails
**Solution**: TensorFlow requires Python 3.8-3.11. Make sure you have a compatible version.

### Issue: "No module named 'tensorflow'"
**Solution**: 
1. Make sure you're using the correct Python version
2. Try: `python -m pip install tensorflow==2.15.0`
3. If you have GPU, you might need: `pip install tensorflow-gpu`

### Issue: Package version conflicts
**Solution**: Create a virtual environment (recommended):
```bash
# Create virtual environment
python -m venv venv

# Activate it (Windows PowerShell)
.\venv\Scripts\Activate.ps1

# Or (Windows Command Prompt)
venv\Scripts\activate.bat

# Then install requirements
pip install -r requirements.txt
```

## Step 4: Verify Installation

Test if everything is installed correctly:
```bash
python -c "import flask; import tensorflow; print('All packages installed successfully!')"
```

## Alternative: Using Conda (if you have Anaconda/Miniconda)

```bash
# Create conda environment
conda create -n btp_project python=3.10
conda activate btp_project

# Install packages
pip install -r requirements.txt
```

## Need Help?

If you continue to have issues:
1. Make sure Python is added to your system PATH
2. Try using a virtual environment (see above)
3. Check that you're using Python 3.8-3.11 (TensorFlow requirement)
4. Ensure you have internet connection for downloading packages

