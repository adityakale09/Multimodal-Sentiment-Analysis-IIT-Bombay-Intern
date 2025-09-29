# 🧠 Multimodal Sentiment Analysis

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange)](https://jupyter.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.0%2B-green)](https://scikit-learn.org/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> **Advanced multimodal sentiment classification using EEG, GSR, facial action units, and self-report data with late fusion approaches.**

## 🎯 Overview

This project implements a comprehensive multimodal sentiment analysis system that classifies emotions into three categories (Positive, Negative, Neutral) by fusing multiple physiological and behavioral modalities:

- **🧠 EEG**: Brainwave patterns and frequency band analysis
- **🫀 GSR**: Galvanic skin response and autonomic arousal
- **😊 Facial AU**: Action units from facial expression analysis  
- **📝 Self-Report**: Subjective valence and arousal ratings

## 🏆 Key Results

- **Best Performance**: 75.0% accuracy with F1-score of 0.758
- **Optimal Approach**: Weighted late fusion outperforms individual modalities
- **Dataset**: 38 participants, 59 multimodal features, balanced 3-class distribution
- **Cross-Validation**: Robust performance with comprehensive statistical validation

## 📁 Project Structure

```
multimodal-sentiment-analysis/
├── 📊 data/
│   ├── EEG.csv                    # EEG frequency bands and engagement metrics
│   ├── GSR.csv                    # Galvanic skin response measurements
│   ├── TIVA.csv                   # Facial Action Units (AU intensities)
│   ├── PSY.csv                    # Self-report psychological measures
│   ├── NSTLX.csv                  # Additional metadata
│   ├── multimodal_features.csv    # Integrated feature matrix
│   └── sentiment_labels.csv       # Target sentiment classifications
├── 📝 notebooks/
│   ├── 01_preprocessing.ipynb              # Data loading, sync, feature extraction
│   ├── 02_feature_engineering.ipynb       # Advanced feature engineering & selection
│   ├── 03_modeling_baseline_sentiment.ipynb # Baseline ML models comparison
│   └── 04_modeling_fusion.ipynb           # Late fusion multimodal approaches
├── 🤖 models/
│   ├── best_baseline_model.pkl           # Best performing Random Forest model
│   ├── feature_scaler.pkl               # Trained feature standardization
│   ├── fusion_results.json              # Comprehensive fusion results
│   └── unimodal_*.pkl                   # Individual modality classifiers
└── 📖 README.md
```

## 🚀 Quick Start

### Prerequisites

```bash
Python 3.8+
Jupyter Notebook
Required packages: pandas, numpy, scikit-learn, matplotlib, seaborn, xgboost
```

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/adityakale09/Multimodal-Sentiment-Analysis-IIT-Bombay-Intern.git
cd Multimodal-Sentiment-Analysis-IIT-Bombay-Intern
```

2. **Install dependencies**
```bash
pip install pandas numpy scikit-learn matplotlib seaborn xgboost jupyter
```

3. **Launch Jupyter Notebook**
```bash
jupyter notebook
```

4. **Run notebooks in order**
   - Start with `01_preprocessing.ipynb`
   - Follow the sequence through `04_modeling_fusion.ipynb`

## 📊 Methodology
- **Strategy**: Extract features within task blocks [routineStart, routineEnd]

### Step 2: Preprocessing Pipeline
- **Timestamp Synchronization**: Align signals across modalities
- **Feature Engineering**:
  - EEG: Bandpower (Delta, Theta, Alpha, Beta, Gamma), frontal asymmetry, engagement
  - GSR: Mean level, SCR peaks, recovery slope
  - Facial AUs: Emotion-relevant action units (AU12=smile, AU4=frown, etc.)
  - Self-reports: Valence/arousal ratings converted to sentiment categories

### Step 3: Modeling Approaches

#### 3.1 Baseline ML Models
- Concatenated feature vectors from all modalities
- Models: Logistic Regression, Random Forest, XGBoost
- Evaluation: Macro-F1, accuracy, confusion matrix

#### 3.2 Late Fusion
- Train separate unimodal classifiers for each modality
- Combine predictions via majority voting, stacking, or weighted averaging

#### 3.3 Multimodal Transformers
- Cross-modal attention mechanism
- Joint representation learning across modalities
- End-to-end sentiment classification

### Step 4: Evaluation Metrics
- **Primary**: Accuracy, Macro-F1 Score
- **Secondary**: Precision, Recall per class
- **Analysis**: Confusion matrix, Cohen's Kappa
- **Interpretability**: Feature importance, SHAP values, attention weights

### Step 5: Advanced Experiments
- Binary vs 3-class classification comparison
- Modality ablation studies
- Temporal sentiment trajectory analysis
- Multitask learning (sentiment + arousal/valence)

## Getting Started

1. **Setup Environment**:
   ```bash
   pip install pandas numpy scikit-learn matplotlib seaborn
   pip install tensorflow torch transformers
   pip install shap xgboost
   ```

2. **Data Preparation**:
   - Place your CSV files in the `data/` directory
   - Run `01_preprocessing.ipynb` to synchronize and clean data

3. **Feature Engineering**:
   - Execute `02_feature_engineering.ipynb` to extract multimodal features

4. **Modeling**:
   - Start with `03_modeling_baseline_sentiment.ipynb` for baseline results
   - Progress to fusion and transformer approaches

## Key Dependencies
- pandas, numpy: Data manipulation
- scikit-learn: ML models and evaluation
- tensorflow/pytorch: Deep learning models
- matplotlib, seaborn: Visualization
- xgboost: Gradient boosting
- shap: Model interpretability

## 📈 Results & Performance

### 🏆 Model Performance Summary

#### Baseline Models Comparison
| Model | Accuracy | F1-Score | Cohen's Kappa | Cross-Val Score |
|-------|----------|----------|---------------|-----------------|
| **Random Forest** ⭐ | **75.0%** | **0.758** | **0.625** | **0.638** |
| XGBoost | 66.7% | 0.675 | 0.500 | 0.556 |
| Logistic Regression | 66.7% | 0.653 | 0.500 | 0.522 |
| SVM (RBF) | 58.3% | 0.589 | 0.375 | 0.489 |

#### Late Fusion Analysis
| Fusion Strategy | Accuracy | F1-Score | Cohen's Kappa |
|----------------|----------|----------|---------------|
| **Weighted Average** ⭐ | **75.0%** | **0.758** | **0.625** |
| Early Fusion (Baseline) | 75.0% | 0.758 | 0.625 |
| Majority Voting | 58.3% | 0.601 | 0.375 |
| Best Single Modality (GSR) | 50.0% | 0.494 | 0.250 |

### 💡 Key Insights

- **🧠 GSR features** are most discriminative for sentiment classification
- **🔗 Late fusion** achieves competitive performance with early fusion approaches
- **⚖️ Weighted averaging** effectively leverages complementary modality information
- **📊 Balanced dataset** (38 participants) enables reliable model training
- **📈 Cross-validation** confirms model stability and generalization capabilities

### 📊 Dataset Statistics

#### Feature Distribution by Modality
| Modality | Feature Count | Key Features |
|----------|---------------|--------------|
| **EEG** | 21 | Delta, Theta, Alpha, Beta, Gamma bands |
| **GSR** | 24 | Conductance, resistance, peak analysis |
| **Facial AU** | 8 | Emotion composites from action units |
| **Self-Report** | 6 | Valence, arousal, category ratings |
| **Total** | **59** | **Multimodal feature vector** |

#### Sentiment Label Distribution
- **Negative (0)**: 36.8% (14 samples)
- **Neutral (1)**: 31.6% (12 samples)  
- **Positive (2)**: 31.6% (12 samples)

## 📋 Technical Requirements

### Core Dependencies
```bash
# Data Processing
pandas>=1.3.0
numpy>=1.20.0

# Machine Learning
scikit-learn>=1.0.0
xgboost>=1.5.0

# Visualization
matplotlib>=3.3.0
seaborn>=0.11.0

# Environment
jupyter>=1.0.0
```

### System Requirements
- **Python**: 3.8+ (tested on 3.12.4)
- **Memory**: 4GB+ RAM recommended
- **Storage**: 100MB+ for data and models
- **OS**: Windows/macOS/Linux compatible

## 🎯 Usage Examples

### Basic Classification Pipeline
```python
# Load preprocessed features
import pandas as pd
features = pd.read_csv('data/multimodal_features.csv')
labels = pd.read_csv('data/sentiment_labels.csv')

# Train Random Forest (best model)
from sklearn.ensemble import RandomForestClassifier
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(features, labels)

# Evaluate performance
from sklearn.metrics import accuracy_score, f1_score
accuracy = accuracy_score(y_test, predictions)
f1 = f1_score(y_test, predictions, average='macro')
```

### Late Fusion Implementation
```python
# Train unimodal classifiers
eeg_model = train_modality_classifier(eeg_features, labels)
gsr_model = train_modality_classifier(gsr_features, labels) 
au_model = train_modality_classifier(au_features, labels)
sr_model = train_modality_classifier(sr_features, labels)

# Weighted fusion
weights = [0.3, 0.4, 0.2, 0.1]  # EEG, GSR, AU, SR
fusion_predictions = weighted_fusion(predictions, weights)
```

## 🔬 Research Applications

### Potential Use Cases
- **Clinical Psychology**: Objective emotion assessment
- **Human-Computer Interaction**: Adaptive user interfaces
- **Mental Health**: Depression/anxiety screening
- **Market Research**: Product emotional response testing
- **Education**: Student engagement monitoring

### Future Extensions
- Real-time sentiment classification
- Transformer-based multimodal fusion
- Temporal sequence modeling
- Cross-cultural validation studies
- Mobile/wearable implementation

## 📚 Documentation & Support

### Notebook Descriptions
1. **01_preprocessing.ipynb**: Complete data loading, cleaning, and feature integration
2. **02_feature_engineering.ipynb**: Advanced feature creation, selection, and scaling
3. **03_modeling_baseline_sentiment.ipynb**: Comprehensive ML model comparison
4. **04_modeling_fusion.ipynb**: Late fusion strategies and performance analysis

### Visualization Dashboard
Each notebook includes comprehensive visualizations:
- 📊 Data distribution analysis
- 🎭 Modality-specific feature patterns
- 📈 Model performance comparisons
- 🔍 Confusion matrix analysis
- 📋 Cross-validation results

## 🤝 Contributing

We welcome contributions to improve this multimodal sentiment analysis framework!

### How to Contribute
1. **🍴 Fork** the repository
2. **🌿 Create** a feature branch (`git checkout -b feature/AmazingFeature`)
3. **💾 Commit** your changes (`git commit -m 'Add AmazingFeature'`)
4. **📤 Push** to the branch (`git push origin feature/AmazingFeature`)
5. **🔃 Open** a Pull Request

### Development Guidelines
- Follow PEP 8 style conventions
- Add comprehensive docstrings
- Include unit tests for new features
- Update documentation as needed

## 📄 License

This project is licensed under the **MIT License** - see the [LICENSE](LICENSE) file for details.

## 📚 Citation

If you use this work in your research, please cite:

```bibtex
@misc{multimodal_sentiment_analysis_2025,
  title={Multimodal Sentiment Analysis using EEG, GSR, Facial Action Units, and Self-Report Data},
  author={Aditya Kale},
  year={2025},
  url={https://github.com/adityakale09/Multimodal-Sentiment-Analysis-IIT-Bombay-Intern},
  note={Advanced late fusion approaches for physiological emotion classification}
}
```

## 📞 Contact & Support

- 👤 **Author**: Aditya Kale
- 📧 **Email**: kaleshrikrushna6@gmail.com  
- 🔗 **Repository**: [Multimodal Sentiment Analysis](https://github.com/adityakale09/Multimodal-Sentiment-Analysis-IIT-Bombay-Intern)
- 💬 **Issues**: [Report bugs or request features](https://github.com/adityakale09/Multimodal-Sentiment-Analysis-IIT-Bombay-Intern/issues)

## 🏆 Acknowledgments

- **Dataset**: IIT Multimodal Sentiment Analysis Dataset
- **Methodology**: Late fusion approaches in multimodal learning
- **Inspiration**: Advances in physiological emotion recognition

---

⭐ **If you found this project helpful, please give it a star!** ⭐