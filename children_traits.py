# %%
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

data = pd.read_csv("data_csv.csv")

print(data.head())
print(data.columns)

# %%

# Data analisi

print(data.info())
print()

# %%

#  Ci sono valori nulli nel dataset?
print("Valori nulli")
print(data.isna().sum())
print()

# Soluzione.dropna()
print(data.dropna().isna().sum())
print()
data = data.dropna()
print(data.isna().sum())

#  Ci sono duplicati?
print("Numero di duplicati")
print(data.duplicated().sum())
data = data.drop_duplicates()
# Soluzione.drop_duplicates()
print(data.duplicated().sum())
print()

# %%

#  Analisi delle labels

data["Jaundice"] = data["Jaundice"].str.lower()

print(data.head())

# elimino una colonna non utile

data.drop(columns= ["CASE_NO_PATIENT'S"], axis=1, inplace=True)

# statistica del data set

print("Statistiche descrittive per colonne numeriche:")
print(data.describe())

# Preprocessing delle colonne

normalizer = MinMaxScaler()
label_encoder = LabelEncoder()


data['Social_Responsiveness_Scale'] = normalizer.fit_transform(data[['Social_Responsiveness_Scale']])
data['Qchat_10_Score'] = normalizer.fit_transform(data[['Qchat_10_Score']])
data["Childhood Autism Rating Scale"] = normalizer.fit_transform(data[["Childhood Autism Rating Scale"]])
data['Age_Years'] = normalizer.fit_transform(data[['Age_Years']])


for label in ["Speech Delay/Language Disorder", "Learning disorder", "Genetic_Disorders", "Depression", "Global developmental delay/intellectual disability", "Social/Behavioural Issues", "Anxiety_disorder", "Sex", "Ethnicity", "Jaundice", "Family_mem_with_ASD", "Who_completed_the_test", "ASD_traits"]:
    label_encoder.fit(data[[label]])
    data[label] = label_encoder.transform(data[[label]])

data['Ethnicity'] = normalizer.fit_transform(data[['Ethnicity']])

# %%
print(data.head(20))

# %%

# Modello di Machine Learning

# 1. Split dei dati
X = data.drop(columns= ["ASD_traits"], axis=1, inplace=False)
y = data['ASD_traits']

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=79, test_size = 0.2)

# 2. Uso un modello SVM per classificare
svm = SVC(kernel='linear')
svm.fit(X_train, y_train)

# 3. Faccio le previsioni sulle features del test set
y_pred = svm.predict(X_test)
print(X_test)

# 4. Valuta le performance del modello
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuratezza: {accuracy:.2f}')

# %%
correlation_matrix = data.corr()

# Set up the matplotlib figure
plt.figure(figsize=(8, 6))

# Create a heatmap
sns.heatmap(correlation_matrix, annot=False, cmap='coolwarm', square=True, cbar_kws={"shrink": .8})

# Set title
plt.title('Correlation Heatmap')
# %%
