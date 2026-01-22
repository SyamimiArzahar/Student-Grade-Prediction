# Student-Grade-Prediction

# **Student Grade Prediction Model**

## **Problem**
Predicting student final grades based on academic and behavioral factors is challenging for educational institutions. Traditional methods rely on subjective assessment, while machine learning can provide data-driven, objective predictions to help identify at-risk students early and allocate resources effectively.

## **Dataset**
- **Size**: 1,000,000 student records
- **Features**:
  - `weekly_self_study_hours`: Hours of self-study per week (0-40)
  - `attendance_percentage`: Class attendance rate (50-100%)
  - `class_participation`: Participation score (0-10 scale)
  - `total_score`: Cumulative academic score (0-100)
- **Target**: `grade_numeric` (0=F, 1=D, 2=C, 3=B, 4=A)
- **Source**: Synthetic educational dataset for machine learning practice

## **Approach**

### **1. Data Preprocessing**
- Removed duplicate records
- Clipped unrealistic values to reasonable ranges
- Encoded categorical grades to numerical values (A=4, B=3, C=2, D=1, F=0)
- Split data: 80% training (800,000 samples), 20% testing (200,000 samples)

### **2. Model Development & Comparison**
- **Logistic Regression**: Baseline interpretable model with feature scaling
- **Decision Tree**: Alternative model with depth limitation to prevent overfitting
- **Evaluation**: Compared accuracy, confusion matrices, and feature importance

### **3. Implementation**
- Built using Python with scikit-learn
- Applied StandardScaler for feature normalization in Logistic Regression
- Used stratified sampling to preserve grade distribution
- Saved both trained models for comparison

## **Results**

### **Model Performance**
- **Logistic Regression**: 99.76% accuracy on test set
- **Decision Tree**: 99.80% accuracy on test set  
- **Best Model**: Decision Tree with 99.80% accuracy

### **Key Findings**
1. **Total score dominates predictions**: Both models heavily rely on total_score (Decision Tree gives it 100% importance)
2. **Near-perfect accuracy**: Both models achieve >99.7% accuracy due to strong correlation between total_score and grades
3. **Minimal overfitting**: Both models generalize well with train-test gaps <0.02%
4. **Adjacent grade confusion**: Most errors occur between neighboring grades (e.g., B↔C, C↔D)

### **Confusion Matrix Analysis (Logistic Regression)**
```
Actual vs Predicted Grades:
- Grade 0 (F): 1,208 correct, 34 misclassified as D
- Grade 1 (D): 8,925 correct, 74 misclassified (mostly as C)
- Grade 2 (C): 28,282 correct, 114 misclassified (mostly as B or D)
- Grade 3 (B): 51,474 correct, 161 misclassified (mostly as A or C)
- Grade 4 (A): 109,624 correct, 105 misclassified as B
```
**Total**: Only 488 misclassifications out of 200,000 test samples (99.76% accuracy)

### **Feature Importance Insights**
**Logistic Regression Coefficients**:
- **Grade A (4)**: Strongly predicted by high total_score (+92.9)
- **Grade B (3)**: Predicted by moderate total_score (+31.8) and study hours (+0.25)
- **Grade C (2)**: Slight positive from study hours (+0.04)
- **Grade D (1)**: Slight positive from participation (+0.01)
- **Grade F (0)**: Slight positive from attendance (+0.04), but strongly negative from total_score (-61.5)

**Decision Tree Feature Importance**:
- **total_score**: 100% importance (single deciding factor)
- **Other features**: 0% importance in this tree structure

## **Limitations**

### **Model Limitations**
1. **Total Score Dominance**: The models essentially learn grade cutoffs from total_score, making other features irrelevant
2. **Circular Logic Risk**: Using total_score to predict grades may be tautological if grades are directly derived from scores
3. **Decision Tree Simplicity**: Tree only uses total_score, missing potential nuances from other features
4. **Too Good Performance**: 99.8% accuracy suggests the problem may be too simple for meaningful ML insights

### **Practical Limitations**
1. **Real-World Usefulness**: In practice, if we already have total_score, we don't need ML to predict grades
2. **Missing Intervention Insights**: Model doesn't identify which factors to improve for struggling students
3. **Feature Redundancy**: Weekly study hours, attendance, and participation may already be reflected in total_score
4. **Categorical Boundaries**: Grade thresholds in total_score could be learned with simple rules, not ML

### **Technical Limitations**
1. **Interpretability Trade-off**: Logistic Regression provides coefficients but Decision Tree is essentially a rule-based system
2. **Scalability Question**: Simple decision rules could replace both models with equal performance
3. **Data Leakage Concern**: total_score may contain grade information directly

### **Ethical Limitations**
1. **Self-Fulfilling Prophecy**: Using scores to predict grades reinforces existing assessment methods
2. **No Value-Add**: Doesn't provide insights beyond what educators already know
3. **Resource Justification**: Hard to justify ML complexity when simple thresholds work equally well

## **Critical Insight**
The near-perfect accuracy reveals a fundamental issue: **This dataset presents a pseudo-ML problem**. The "prediction" is essentially a lookup table from total_score to grades. In educational contexts, the real challenge is predicting *future* performance from *current* behaviors, not mapping final scores to final grades.

## **Recommendations for Real Educational Applications**

1. **Reframe the Problem**: Predict *mid-term* grades from *early-term* behaviors
2. **Remove total_score**: Use it only for validation, not as a feature
3. **Time-Series Approach**: Track how study habits change over time
4. **Early Warning System**: Identify at-risk students before final scores are available
5. **Causal Analysis**: Investigate which behaviors actually improve learning outcomes

## **Files in Repository**
- `student_grade_prediction.ipynb` - Complete analysis notebook
- `train_test_split.pkl` - Serialized train/test data
- `logistic_regression_model.pkl` - Trained Logistic Regression model (99.76% accuracy)
- `decision_tree_model.pkl` - Trained Decision Tree model (99.80% accuracy)
- `requirements.txt` - Python dependencies

## **Usage**
```python
import joblib
# Load Decision Tree (better accuracy)
model_data = joblib.load('decision_tree_model.pkl')
predictions = model_data['model'].predict(new_data)
```

## **Conclusion**
While technically successful (99.8% accuracy), this project highlights the importance of problem framing in ML. The models learned that `total_score` perfectly predicts grades, which is mathematically correct but pedagogically trivial. For educational impact, future work should focus on predictive rather than correlational analysis.

**Note for Internship Context**: This project demonstrates ML pipeline proficiency (data cleaning, feature selection, model training, evaluation), but the results suggest the original problem formulation needs reconsideration for real-world educational applications.
