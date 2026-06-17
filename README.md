# Digit Classification: SVM vs. Decision Tree

This repository contains a Python implementation comparing two classic Machine Learning algorithms—**Support Vector Machine (SVC)** and **Decision Tree Classifier**—using the `scikit-learn` Digits dataset. 

The script benchmarks both models on classification accuracy, exports precision/recall metrics to a text report, and renders side-by-side performance visualisations via confusion matrices.

---

##  Features

* **Dataset:** Utilises `scikit-learn`'s native 8x8 pixel handwritten digits dataset (`load_digits`).
* **Dual Benchmarking:** Trains and compares `sklearn.svm.SVC` and `sklearn.tree.DecisionTreeClassifier`.
* **Automated Artifacts:** Automatically exports structured classification metrics (`precision`, `recall`, `f1-score`) to an external `result.txt` file.
* **Visual Diagnostics:** Generates side-by-side confusion matrix heatmaps using `matplotlib` and `seaborn` to pin down precisely where misclassifications happen.

---

##  Installation & Setup

### 1. Clone the Repository
```bash
git clone https://github.com/your-username/digit-classification-svm-dt.git
cd digit-classification-svm-dt
```

### 2. Install Required Packages
Make sure you have Python 3.8+ ready. Install the mandatory data science stack:

```bash
pip install scikit-learn matplotlib seaborn
```

---

##  Usage

To execute the benchmark pipeline, run:

```bash
python SVM.py
```

### Expected Deliverables
1. **Console Acknowledgment:** Prints `Results saved to result.txt` upon successful execution.
2. **Text Artifact (`result.txt`):** A generated report containing strict multi-class evaluation scores for both classifiers.
3. **Interactive Graph UI:** An interactive window showcasing side-by-side confusion matrix heatmaps (`Blues` color scheme for SVM, `Greens` for Decision Tree).

---

##  Sample Metrics Format

The generated `result.txt` saves classification tables structured as follows:

```text
=== Support Vector Classifier ===
Accuracy: 0.9861
              precision    recall  f1-score   support
           0       1.00      1.00      1.00        33
...

=== Decision Tree Classifier ===
Accuracy: 0.8417
              precision    recall  f1-score   support
           0       0.97      0.88      0.92        33
...
```

---

##  Troubleshooting: PyCharm `tostring_rgb` Fix

If running this code in PyCharm triggers the following backend error:
`AttributeError: 'FigureCanvasInterAgg' object has no attribute 'tostring_rgb'`

This happens because newer versions of **Matplotlib (3.8+)** deprecated a rendering method that older PyCharm Scientific View installations look for. You can resolve this instantly using either of these solutions:

### Solution A: Override the Backend (Code Fix)
Inject an explicit GUI backend declaration at the **absolute top** of your `SVM.py` script before importing `pyplot`:

```python
import matplotlib
matplotlib.use('TkAgg')  # Redirects rendering to a standard external window
import matplotlib.pyplot as plt
```

### Solution B: Adjust PyCharm Settings (No Code Changes)
1. Go to **File** ➡️ **Settings** (or **PyCharm** ➡️ **Preferences** on macOS).
2. Expand **Tools** ➡️ select **Python Scientific**.
3. **Uncheck** the setting box labeled **"Show plots in tool window"**.
4. Click **Apply** and re-run your code.

---

##  License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
