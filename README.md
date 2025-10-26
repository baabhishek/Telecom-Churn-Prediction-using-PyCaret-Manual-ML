# Telecom Churn Prediction – Manual ML vs PyCaret Automation
![Telecom Project Banner](https://www.leti-cea.com/cea-tech/leti/PublishingImages/Recherche%20appliqu%C3%A9e/Plateformes/telecom/banner.jpg)


Predict telecom customer churn using machine learning by building multiple models manually and comparing them with PyCaret automation. Focused on feature engineering, hyperparameter tuning, and aligning results with business objectives.

---

## Project Overview

- Predict customer churn in the telecom domain using machine learning.
- Compare performance between manual model building and PyCaret’s automated ML pipeline.
- Highlight the importance of human intelligence in guiding AI models according to business needs.

---

## Models Built (manual)

| Model | Approach | Notes |
|-------|----------|-------|
| Logistic Regression | Manual | Baseline model for classification |
| Decision Tree | Manual | Simple interpretable tree-based model |
| Bagging | Manual | Ensemble method to reduce variance |
| Random Forest | Manual | Best accuracy (~80%) after fine-tuning |
| AdaBoost | Manual | Boosting ensemble method |
| Gradient Boosting | Manual | Sequential ensemble for performance |
| XGBoost | Manual | Optimized boosting algorithm |
| SVM | Manual | Classification with kernel tricks |
| KNN | Manual | Distance-based classification |
| GaussianNB | Manual | Probabilistic classification |
| BernoulliNB | Manual | Probabilistic classification for binary features |
| Voting Classifier | Manual | Combined predictions from multiple models |
| Multiple Models | PyCaret | Automated pipeline handling preprocessing, tuning, and selection |

---
## PyCaret automated ML pipeline internally handled:

- Preprocessing (missing values, encoding, scaling)  
- Model selection and training  
- Hyperparameter tuning  

## Key Features

- Feature engineering and domain-specific preprocessing for improved accuracy
- Hyperparameter tuning to optimize model performance
- Comparison of manual modeling vs automated ML (PyCaret)
- Emphasis on aligning predictions with business objectives
- Model evaluation using accuracy, recall, and relevant metrics

---

## Skills Used

- Python, PyCaret, Scikit-learn  
- Machine Learning (Supervised, Classification)  
- Feature Engineering & Hyperparameter Tuning  
- Data Preprocessing & Domain-Specific Transformations  
- Model Evaluation (Accuracy, Recall, F1-score, Confusion Matrix)  
- Business Context Understanding & AI Interpretation  

---

## Model Evaluation

### Manual Approach – Random Forest (after tuning)

**Classification Report:**

           precision    recall  f1-score   support
       0       0.88      0.81      0.84      1574
       1       0.82      0.89      0.85      1531
accuracy                           0.85      3105


**Confusion Matrix:**

[[1269 305]

[ 167 1364]]


---

### Original Model – Random Forest (manual)

**Classification Report:**

           precision    recall  f1-score   support
       0       0.87      0.82      0.84      1574
       1       0.82      0.87      0.85      1531
accuracy                           0.84      3105

## Final conclusion: - > Tuned Model : (rf2)

- the tuned random forest model provides balanced performance 
- with improved recall for churners, stable cross-validation accuracy, 
- and acceptable variance. 
- it captures customer churns more effectively, 
- making it a better choice for deployment over the original model.
---
### PyCaret Automation

- Achieved quick baseline models for all algorithms using PyCaret.  
- Automated hyperparameter tuning, preprocessing, and model selection internally.  
- The tuned K Neighbors Classifier achieved **higher accuracy (88.15%)** and precision (0.8905) than the original manual KNN, demonstrating the effectiveness of automated hyperparameter optimization.  
- Cross-validated results showed stable performance with mean accuracy of 78.32% across 10 folds and decent recall (0.4045).  
- Highlighted that automated ML pipelines can provide strong baseline or even superior performance for certain models, but domain-specific adjustments may still be needed for business objectives.

---
## Project Highlights

- Focused on real-world telecom business problem.  
- Explored multiple models, compared manual vs automated approaches.  
- Demonstrated the importance of domain knowledge and fine-tuning in achieving business-aligned ML outcomes.  
- Provided clear evaluation with metrics, confusion matrices, and final model recommendations. 
## Skills

Python, PyCaret, Scikit-learn, Machine Learning (Supervised, Classification), Feature Engineering, Hyperparameter Tuning, Data Preprocessing, Model Evaluation, Business Context Understanding  

---
### Cross-Validated Performance (10-fold CV, KNN)

| Fold | Accuracy | AUC    | Recall | Precision | F1    | Kappa  | MCC   |
|------|---------|--------|--------|-----------|-------|--------|-------|
| 0    | 0.8076  | 0.8683 | 0.4340 | 0.7419    | 0.5476| 0.4359 | 0.4612|
| 1    | 0.7873  | 0.8276 | 0.4340 | 0.6571    | 0.5227| 0.3932 | 0.4072|
| 2    | 0.7747  | 0.8057 | 0.3679 | 0.6393    | 0.4671| 0.3371 | 0.3578|
| 3    | 0.7995  | 0.8394 | 0.4571 | 0.6857    | 0.5486| 0.4262 | 0.4407|
| 4    | 0.7792  | 0.8279 | 0.3524 | 0.6607    | 0.4596| 0.3366 | 0.3629|
| 5    | 0.7970  | 0.8505 | 0.4623 | 0.6806    | 0.5506| 0.4255 | 0.4388|
| 6    | 0.7614  | 0.8033 | 0.4151 | 0.5789    | 0.4835| 0.3338 | 0.3416|
| 7    | 0.7437  | 0.8091 | 0.3208 | 0.5397    | 0.4024| 0.2524 | 0.2663|
| 8    | 0.8046  | 0.8378 | 0.4057 | 0.7544    | 0.5276| 0.4181 | 0.4501|
| 9    | 0.7766  | 0.8464 | 0.3962 | 0.6364    | 0.4884| 0.3553 | 0.3716|
| **Mean** | 0.7832 | 0.8316 | 0.4045 | 0.6575 | 0.4998 | 0.3714 | 0.3898 |
| **Std**  | 0.0192 | 0.0201 | 0.0437 | 0.0622 | 0.0460 | 0.0556 | 0.0580 |

---

### Unseen Test Set Performance (Tuned KNN)

| Model                   | Accuracy | Recall | Precision | F1    | Notes                                |
|-------------------------|---------|--------|-----------|-------|--------------------------------------|
| K Neighbors Classifier  | 0.8815  | 0.6404 | 0.8905    | 0.7450| PyCaret tuned model on unseen data   |

---

### Key Highlights

- PyCaret provides **fast automation** of preprocessing, model selection, and hyperparameter tuning.  
- Cross-validation gives **stable estimates** of performance across multiple folds.  
- Tuned KNN using PyCaret achieved **higher accuracy and precision** than manual KNN, demonstrating that automation can sometimes outperform manual tuning.  
- However, **manual modeling is still valuable** for incorporating domain knowledge, feature engineering, and business-specific adjustments.  
- **AI is a helper, not a replacement**: understanding the business context is crucial for achieving aligned objectives.  
- Combining **human intelligence with AI automation** ensures better predictions and informed decision-making.  

---

## Key Learnings

- Manual model building allows **feature engineering, domain-specific preprocessing, and hyperparameter tuning**, which is essential for aligning with specific business requirements.  
- PyCaret provides **fast automation, hyperparameter optimization, and baseline model comparisons**, sometimes achieving better metrics than manual models for certain algorithms.  
- **AI is a helper, not a replacement** for human intelligence—domain knowledge and business context are critical to achieve desired outcomes.  
- Combining **human insights with AI** ensures better predictions, proper evaluation, and informed business decisions.  

---

## Conclusion

- Tuned Random Forest (`rf2`) remains the best model for telecom churn prediction in terms of **recall for churners** and business alignment.  
- PyCaret’s automated KNN outperformed the original manual KNN in accuracy and precision, showing that automation can be highly effective for some models.  
- PyCaret is valuable for rapid prototyping, experimentation, and optimization, but **human guidance is crucial** to ensure models meet business objectives.  
- Effective ML projects rely on **synergy between human intelligence and AI**: humans define objectives and interpret results, while AI accelerates computations and provides strong suggestions.  

---

 
