# Heart_Disease_Prediction_Using_Machine_Learning
Heart Disease Prediction Using Machine Learning
This project uses supervised machine learning techniques to predict the presence of heart disease based on clinical parameters. The dataset consists of anonymized medical attributes from individuals, which are processed and fed into a classification model to determine the likelihood of heart disease.

# Features
Preprocessing of heart disease dataset with feature scaling

Data visualization using seaborn and matplotlib

Model training using:

Logistic Regression

Random Forest

Decision Tree

K-Nearest Neighbors (KNN)

Model evaluation using accuracy, confusion matrix, and ROC curves

Streamlined comparison of model performances

# Models Used
Logistic Regression

Random Forest Classifier

Decision Tree Classifier

K-Nearest Neighbors (KNN)

# Dataset
The dataset includes 14 features such as:

Age

Sex

Chest pain type

Resting blood pressure

Cholesterol

Fasting blood sugar

Resting ECG results

Maximum heart rate achieved

Exercise-induced angina

ST depression

Slope of the peak exercise ST segment

Number of major vessels

Thalassemia

Target (1 indicates heart disease)

Note: The dataset used appears to be a version of the Cleveland Heart Disease Dataset, available from UCI Machine Learning Repository.

# Libraries Required
numpy

pandas

matplotlib

seaborn

scikit-learn


# Results
Among the tested models, RandomForestClassifier showed the best performance on the dataset with the highest accuracy and reliable ROC-AUC score.

# Future Scope
Hyperparameter tuning using GridSearchCV

Implement cross-validation for better generalization

Integration with a Flask/Django web app for real-time prediction
