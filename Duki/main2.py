import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
from sklearn.impute import SimpleImputer
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import plot_tree
from sklearn.ensemble import RandomForestClassifier

df = pd.read_csv('p3.csv')
protocoloDNS = df[df['Protocol'] == 'DNS'].copy()
features = protocoloDNS.groupby('Source').agg({
        'No.': 'count',              # Número de consultas DNS por IP
        'Length': ['mean', 'sum'],    # Tamaño promedio y total de bytes
        'delta_time': ['mean', 'std'] # Tiempo entre consultas (con espacio en el nombre)
    })
features.columns = ['_'.join(col).strip() for col in features.columns.values]
features = features.reset_index()
features['is_anomaly'] = features.apply(
    lambda row: -1 if (row['No._count'] > 100) and (row['delta_time_mean'] < 0.1) else 1,
    axis=1
)
X = features.drop(['Source', 'is_anomaly'], axis=1)
y = features['is_anomaly']

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.3,
    random_state=42,
    stratify=y
)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

imputer = SimpleImputer(strategy="mean")
X_train_scaled = imputer.fit_transform(X_train_scaled)

lr = LogisticRegression(random_state=42)
lr.fit(X_train_scaled, y_train)
y_pred_lr = lr.predict(X_test_scaled)
print("Regresión Logística:")
print(classification_report(y_test, y_pred_lr))
cm = confusion_matrix(y_test, y_pred_lr)

plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=set(y_test), yticklabels=set(y_test))
plt.xlabel('Predicción')
plt.ylabel('Real')
plt.title('Matriz de Confusión - Regresión Logística')
plt.show()


dt = DecisionTreeClassifier(max_depth=3, random_state=42)
dt.fit(X_train_scaled, y_train)
y_pred_dt = dt.predict(X_test_scaled)

print("\nÁrbol de Decisión:")
print(classification_report(y_test, y_pred_dt))

plt.figure(figsize=(12, 6))
plot_tree(dt, filled=True, feature_names=X_train.columns, class_names=[str(c) for c in set(y_train)])
plt.show()
rf = RandomForestClassifier(n_estimators=300,
                            random_state=42)
rf.fit(X_train_scaled, y_train)
y_pred_rf = rf.predict(X_test_scaled)

print("\nRandom Forest:")
print(classification_report(y_test, y_pred_rf))