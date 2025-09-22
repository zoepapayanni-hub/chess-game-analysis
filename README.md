# === Α) EDA ===
# Στατιστική αποτύπωση των δεδομένων & οπτικοποίηση μεταβλητών ενδιαφέροντος

# Φόρτωση των βιβλιοθηκών
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler, PolynomialFeatures
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay, accuracy_score, roc_curve, auc, mean_squared_error
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from imblearn.over_sampling import SMOTE
from sklearn.pipeline import make_pipeline

# === Ανάγνωση δεδομένων ===
df = pd.read_csv('games_data.csv')
features = ['white_rating', 'black_rating', 'increment_code', 'opening_eco']
target = 'winner'
df = df[features + [target]].dropna()

# Επεξεργασία κατηγορικών μεταβλητών
le_increment = LabelEncoder()
le_opening = LabelEncoder()
le_target = LabelEncoder()

df['increment_code'] = le_increment.fit_transform(df['increment_code'])
df['opening_eco'] = le_opening.fit_transform(df['opening_eco'])
df['winner_encoded'] = le_target.fit_transform(df['winner'])

label_encoders = {}
for col in ['increment_code', 'opening_eco']:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

target_encoder = LabelEncoder()
df[target] = target_encoder.fit_transform(df[target])

# === Γραφήματα για ΕDA ===

# Κατανομή αξιολογήσεων παικτών
plt.figure(figsize=(12, 5))
sns.histplot(df['white_rating'], bins=50, color='blue', kde=True, label='White')
sns.histplot(df['black_rating'], bins=50, color='red', kde=True, label='Black')
plt.title('Distribution of Player Ratings')
plt.legend()
plt.show()

# Κατανομή νικητών
plt.figure(figsize=(6, 4))
sns.countplot(x=target, data=df)
plt.title('Winner Distribution')
plt.xlabel('Winner')
plt.ylabel('Count')
plt.xticks(ticks=[0, 1, 2], labels=target_encoder.classes_)
plt.show()

# Συχνότητα increment codes
plt.figure(figsize=(15, 4))
top_increment = df['increment_code'].value_counts().nlargest(15)
increment_labels = le_increment.inverse_transform(top_increment.index)
sns.barplot(x=increment_labels, y=top_increment.values)
plt.xticks(rotation=45)
plt.title('Top 15 Increment Codes')
plt.xlabel('Increment Code')
plt.ylabel('Frequency')
plt.show()

# Συχνότητα openings (ECO codes)
plt.figure(figsize=(15, 4))
top_opening = df['opening_eco'].value_counts().nlargest(15)
opening_labels = le_opening.inverse_transform(top_opening.index)
sns.barplot(x=opening_labels, y=top_opening.values)
plt.xticks(rotation=45)
plt.title('Top 15 Openings (ECO Codes)')
plt.xlabel('Opening Code')
plt.ylabel('Frequency')
plt.show()

# Top openings με τις περισσότερες νίκες για λευκά
top_opening_wins = df.groupby('opening_eco')['winner_encoded'].apply(lambda x: (x == 2).sum()).nlargest(15)
opening_labels = le_opening.inverse_transform(top_opening_wins.index)

plt.figure(figsize=(15, 4))
sns.barplot(x=opening_labels, y=top_opening_wins.values)
plt.xticks(rotation=45)
plt.title('Top 15 Openings with Most Wins')
plt.xlabel('Opening Code')
plt.ylabel('Number of Wins (White)')
plt.show()

# === Train-test split & SMOTE για όλα τα μοντέλα ===
# (χρησιμοποιείται σε όλες τις μεθόδους B1–B5)
X = df[['white_rating', 'black_rating', 'increment_code', 'opening_eco']]
y = df['winner_encoded']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=7200)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

smote = SMOTE(random_state=7195)
X_train_bal, y_train_bal = smote.fit_resample(X_train_scaled, y_train)

# === Συνάρτηση αξιολόγησης για όλα τα μοντέλα (B6) ===
def evaluate(model_name, y_true, y_pred):
    print(f"\n=== {model_name} ===")
    print(classification_report(y_true, y_pred, target_names=le_target.classes_))
    cm = confusion_matrix(y_true, y_pred)
    ConfusionMatrixDisplay(cm, display_labels=le_target.classes_).plot(cmap=plt.cm.Blues)
    plt.title(f"Confusion Matrix - {model_name}")
    plt.grid(False)
    plt.tight_layout()
    plt.show()
    return accuracy_score(y_true, y_pred)

# === B2: Λογιστική Παλινδρόμηση ===
logreg = LogisticRegression(max_iter=1000)
logreg.fit(X_train_bal, y_train_bal)
y_pred_logreg = logreg.predict(X_test_scaled)
y_proba_logreg = logreg.predict_proba(X_test_scaled)
acc_logreg = evaluate("Logistic Regression", y_test, y_pred_logreg)

# === B3: Linear Discriminant Analysis (LDA) ===
lda = LinearDiscriminantAnalysis()
lda.fit(X_train_bal, y_train_bal)
y_pred_lda = lda.predict(X_test_scaled)
y_proba_lda = lda.predict_proba(X_test_scaled)
acc_lda = evaluate("LDA", y_test, y_pred_lda)

# === B4: Quadratic Discriminant Analysis (QDA) ===
qda = QuadraticDiscriminantAnalysis()
qda.fit(X_train_bal, y_train_bal)
y_pred_qda = qda.predict(X_test_scaled)
y_proba_qda = qda.predict_proba(X_test_scaled)
acc_qda = evaluate("QDA", y_test, y_pred_qda)

# === B5: Naive Bayes ===
nb = GaussianNB()
nb.fit(X_train_bal, y_train_bal)
y_pred_nb = nb.predict(X_test_scaled)
y_proba_nb = nb.predict_proba(X_test_scaled)
acc_nb = evaluate("Naive Bayes", y_test, y_pred_nb)

# === B1: K-Nearest Neighbors με Cross-Validation ===
param_grid = {'n_neighbors': list(range(1, 21))}
grid_knn = GridSearchCV(KNeighborsClassifier(), param_grid, cv=5)
grid_knn.fit(X_train_bal, y_train_bal)
best_knn = grid_knn.best_estimator_
y_pred_knn = best_knn.predict(X_test_scaled)
y_proba_knn = best_knn.predict_proba(X_test_scaled)
acc_knn = evaluate("K-Nearest Neighbors", y_test, y_pred_knn)

# === B6: ROC Curve για class "white" ===
fpr_log, tpr_log, _ = roc_curve(y_test, y_proba_logreg[:, 2], pos_label=2)
fpr_lda, tpr_lda, _ = roc_curve(y_test, y_proba_lda[:, 2], pos_label=2)
fpr_qda, tpr_qda, _ = roc_curve(y_test, y_proba_qda[:, 2], pos_label=2)
fpr_nb, tpr_nb, _ = roc_curve(y_test, y_proba_nb[:, 2], pos_label=2)
fpr_knn, tpr_knn, _ = roc_curve(y_test, y_proba_knn[:, 2], pos_label=2)

plt.figure(figsize=(10, 6))
plt.plot(fpr_log, tpr_log, label=f'LogReg (AUC = {auc(fpr_log, tpr_log):.2f})')
plt.plot(fpr_lda, tpr_lda, label=f'LDA (AUC = {auc(fpr_lda, tpr_lda):.2f})')
plt.plot(fpr_qda, tpr_qda, label=f'QDA (AUC = {auc(fpr_qda, tpr_qda):.2f})')
plt.plot(fpr_nb, tpr_nb, label=f'Naive Bayes (AUC = {auc(fpr_nb, tpr_nb):.2f})')
plt.plot(fpr_knn, tpr_knn, label=f'KNN (AUC = {auc(fpr_knn, tpr_knn):.2f})')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve for "White" Wins')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# === B6: Συγκριτικός Πίνακας Ακριβείας ===
print("\n=== Model Accuracies ===")
print(f"Logistic Regression:     {acc_logreg:.4f}")
print(f"LDA:                     {acc_lda:.4f}")
print(f"QDA:                     {acc_qda:.4f}")
print(f"Naive Bayes:             {acc_nb:.4f}")
print(f"K-Nearest Neighbors:     {acc_knn:.4f}")

accuracies = {
    'Logistic Regression': acc_logreg,
    'LDA': acc_lda,
    'QDA': acc_qda,
    'Naive Bayes': acc_nb,
    'KNN': acc_knn
}

best_model = max(accuracies, key=accuracies.get)
print(f"\n Best performing model: {best_model} with accuracy {accuracies[best_model]:.4f}")

# === B6 (προαιρετικό bonus): Cross-validation 10-fold για επιπλέον σύγκριση ===
models = {
    "Logistic Regression": logreg,
    "LDA": lda,
    "QDA": qda,
    "Naive Bayes": nb,
    "KNN": best_knn
}

cv_scores = {}
mean_std_table = []

plt.figure(figsize=(10, 6))
for name, model in models.items():
    scores = cross_val_score(model, X_train_bal, y_train_bal, cv=10)
    cv_scores[name] = scores
    mean_score = scores.mean()
    std_score = scores.std()
    mean_std_table.append([name, mean_score, std_score])
    plt.plot(range(1, 11), scores, marker='o', label=f'{name} ({mean_score:.2f})')

plt.title('Cross-Validation Accuracy per Model (10 folds)')
plt.xlabel('Fold')
plt.ylabel('Accuracy')
plt.ylim(0, 1)
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

cv_df = pd.DataFrame(mean_std_table, columns=['Model', 'Mean Accuracy', 'Standard Deviation'])
cv_df_sorted = cv_df.sort_values(by='Mean Accuracy', ascending=False)
print("\n Cross-Validation Results:")
print(cv_df_sorted.to_string(index=False))

