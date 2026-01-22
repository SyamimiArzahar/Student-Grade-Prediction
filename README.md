# **Student Grade Prediction Model**

## TL;DR
- Dataset: 1M student records
- Task: Multiclass grade prediction (F–A)
- Best Model: Logistic Regression (69.5% accuracy)
- Key Insight: Weekly self-study hours dominate grade prediction
- Use Case: Early identification of at-risk students

## **Problem**
Educational institutions need data-driven methods to predict student performance and identify at-risk students early. Traditional grade prediction relies on teacher intuition, while machine learning can provide objective, consistent predictions based on observable student behaviors.

## **Dataset**
- **Dataset sourced from Kaggle:** https://www.kaggle.com/datasets/nabeelqureshitiii/student-performance-dataset
- **Size**: 1,000,000 student records
- **Features Used**:
  - `weekly_self_study_hours`: Hours of self-study per week (0-40)
  - `attendance_percentage`: Class attendance rate (50-100%)
  - `class_participation`: Participation score (0-10 scale)
- **Target**: `grade_numeric` (0=F, 1=D, 2=C, 3=B, 4=A)
- **Note**: Intentionally excluded `total_score` to avoid circular reasoning
- **Source**: Synthetic educational dataset for machine learning practice

## **Approach**

### **1. Data Preprocessing**
- Removed duplicate records
- Encoded letter grades to numerical values: F=0, D=1, C=2, B=3, A=4
- Applied feature scaling (StandardScaler) for model compatibility
- Split data: 80% training (800,000 students), 20% testing (200,000 students)
- Used stratified sampling to preserve imbalanced grade distribution

### **2. Model Development**
- **Logistic Regression**: Baseline interpretable model with multinomial classification
- **Decision Tree**: Alternative model with depth limitation (max_depth=5) to prevent overfitting
- **Training**: Both models trained on scaled behavioral features only

### **3. Model Selection Criteria**
- Primary: Test set accuracy
- Secondary: Generalization (train-test gap)
- Tertiary: Interpretability for educational stakeholders

## **Results**

### **Model Performance**
- **Logistic Regression**: 69.54% test accuracy
- **Decision Tree**: 69.45% test accuracy  
- **Best Model**: **Logistic Regression** (slightly better accuracy and interpretability)
- **Generalization**: Excellent - minimal overfitting (0.05% train-test gap)

### **Confusion Matrix Analysis**
```
Grade Predictions (Logistic Regression):
• Grade F (0): 0/1,241 correct    (0.0% accuracy)
• Grade D (1): 3,115/8,999 correct (34.6% accuracy)  
• Grade C (2): 12,837/28,396 correct (45.2% accuracy)
• Grade B (3): 26,322/51,635 correct (51.0% accuracy)
• Grade A (4): 96,812/109,729 correct (88.2% accuracy)

Overall: 139,086/200,000 correct predictions (69.54%)
```

### **Key Findings**
1. **Study Hours Dominate**: Weekly self-study hours is the strongest predictor (Decision Tree gave it 100% importance)
2. **Grade-Specific Patterns**: 
   - Higher grades (A, B) strongly associated with more study hours
   - Lower grades (F, D) strongly associated with fewer study hours
   - Attendance and participation show minimal predictive power
3. **Prediction Difficulty Varies**:
   - Grade A: Easily predictable (88.2% accuracy)
   - Grades F/D: Hard to distinguish from each other
   - Most errors occur between adjacent grades
4. **Model Consistency**: Both models perform similarly, suggesting underlying patterns are robust

### **Feature Importance Insights**

**Logistic Regression Coefficients**:
- **Grade A**: Strongly predicted by high study hours (+4.73)
- **Grade B**: Moderately predicted by study hours (+2.07)
- **Grade C**: Very weak association with study hours (-0.08)
- **Grade D**: Strongly associated with low study hours (-2.25)
- **Grade F**: Very strongly associated with low study hours (-4.46)

**Decision Tree Insights**:
- Uses only `weekly_self_study_hours` (100% importance)
- Ignores attendance and participation entirely
- Creates simple decision boundaries based on study hour thresholds

## **Educational Implications**

### **What the Model Reveals**
1. **Study Hours Are Crucial**: The single most important behavioral factor for academic success
2. **Attendance Alone Isn't Enough**: High attendance doesn't guarantee good grades without sufficient study
3. **Participation Matters Less Than Expected**: Classroom engagement shows minimal predictive power
4. **Grade A Students Are Distinct**: They form a clearly separable group based on study habits

### **Practical Applications**
1. **Early Intervention**: Identify students with low study hours for targeted support
2. **Resource Allocation**: Focus tutoring resources on study skills development
3. **Expectation Setting**: Help students understand realistic grade expectations based on study habits
4. **Curriculum Design**: Consider mandatory study hour recommendations
5. Similar predictive approaches can be applied in workforce training, employee performance analytics, and early-risk identification in corporate learning environments.

## **Limitations**

### **Model Limitations**
1. **Moderate Accuracy**: 69.5% accuracy leaves room for improvement
2. **Feature Limitations**: Only three behavioral features considered
3. **Class Imbalance**: Severe under-representation of F/D grades affects their prediction
4. **Linear Assumptions**: Logistic Regression assumes linear relationships

### **Data Limitations**
1. **Synthetic Data**: May not capture real-world complexity
2. **Missing Factors**: No information on prior academic history, learning disabilities, or external stressors
3. **Self-Reported Bias**: Study hours likely self-reported and potentially inaccurate
4. **Temporal Aspects**: No progression tracking over time

### **Interpretation Challenges**
1. **Correlation vs Causation**: Cannot prove study hours cause better grades
2. **Threshold Effects**: Model may miss non-linear or interaction effects
3. **Context Missing**: Course difficulty, teaching quality, and peer effects not considered

## **Technical Implementation**

### **Files in Repository**
- `student_grade_prediction.ipynb` - Complete corrected analysis
- `grade_classification_models.pkl` - Trained Logistic Regression and Decision Tree models
- `requirements.txt` - Python dependencies

### **Model Usage**
```python
import joblib

# Load models
models = joblib.load('grade_classification_models.pkl')
lr_model = models['logistic_regression']['model']
scaler = models['logistic_regression']['scaler']

# Predict grade for new student
new_student = {
    'weekly_self_study_hours': 20,
    'attendance_percentage': 85,
    'class_participation': 7
}

# Scale and predict
new_data_scaled = scaler.transform([list(new_student.values())])
predicted_numeric = lr_model.predict(new_data_scaled)[0]
grade_mapping = {0: 'F', 1: 'D', 2: 'C', 3: 'B', 4: 'A'}
predicted_grade = grade_mapping[predicted_numeric]
print(f"Predicted grade: {predicted_grade}")
```

## **Future Improvements**

### **Immediate Next Steps**
1. **Address Class Imbalance**: Use SMOTE or class weights for better F/D grade prediction
2. **Feature Engineering**: Create interaction terms (study_hours × attendance)
3. **Model Ensemble**: Combine Logistic Regression with Decision Tree
4. **Threshold Optimization**: Adjust prediction thresholds for better recall on minority classes

### **Long-Term Enhancements**
1. **Additional Features**: Include homework completion, quiz scores, and peer interactions
2. **Time-Series Analysis**: Track behavior changes throughout semester
3. **Personalized Predictions**: Cluster students by learning styles
4. **Intervention Recommendations**: Suggest specific actions based on predictions

## **Conclusion**

This project demonstrates that **student grades can be predicted with 69.5% accuracy using only three behavioral features**, with weekly self-study hours emerging as the dominant predictor. The model successfully identifies clear patterns:

1. **Study hours strongly differentiate** high-performing (A/B) from low-performing (F/D) students
2. **Grade A students are highly predictable** from their study habits
3. **Attendance and participation alone are insufficient** predictors of academic success

While the accuracy is moderate, the model provides actionable insights for educational interventions and represents a significant improvement over intuitive prediction methods. The project also highlights the importance of **correct problem framing** in machine learning, as an earlier approach using total_score as a feature produced misleading 99.8% accuracy through circular reasoning.

**Key Takeaway**: For educational applications, focusing on measurable behaviors (like study hours) provides both predictive power and opportunities for meaningful intervention.
