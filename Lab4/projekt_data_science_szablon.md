# Projekt: od surowych danych do prostego modelu ML

## Cel projektu

W tym projekcie:
- poznasz dane poprzez EDA (Exploratory Data Analysis),
- przygotujesz dane (preprocessing: braki, kategorie, skalowanie),
- zbudujesz prosty model ML (regresja lub klasyfikacja),
- opiszesz swoje wnioski w Markdown.

Czas pracy: ok. 100 minut.

Na początek:
- opisz jednym akapitem, jaki problem chcesz rozwiązać (np. przewidywanie ceny, klasyfikacja klienta, itp.),
- napisz, jaka jest Twoja hipoteza lub intuicja dotycząca tych danych.

## 1. Importy i wczytanie danych

W tej sekcji:
- zaimportujemy potrzebne biblioteki,
- wczytamy dane do obiektu `DataFrame`,
- zrobimy pierwsze szybkie spojrzenie na dane.

Uzupełnij ścieżkę/URL do swojego pliku CSV.

```python
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, accuracy_score, confusion_matrix

import matplotlib.pyplot as plt

# TODO: Podaj nazwę pliku lub URL z danymi
# Przykład: df = pd.read_csv("dane.csv")
df = pd.read_csv("dane.csv")

df.head()
```

## 2. Pierwsze spojrzenie na dane

Sprawdzimy:
- kształt danych,
- typy kolumn,
- podstawowe statystyki,
- brakujące wartości.

Na końcu opisz w kilku zdaniach, co widzisz w danych.

```python
# Kształt danych (liczba wierszy i kolumn)
df.shape
```

```python
# Informacje o typach danych i brakach
df.info()
```

```python
# Statystyki opisowe
df.describe(include="all")
```

```python
# Liczba braków w każdej kolumnie
df.isna().sum()
```

### Komentarz: pierwsze wrażenia

Opisz:
- ile masz obserwacji i cech,
- które kolumny wyglądają na ważne,
- gdzie widać dużo braków,
- co może być dobrą zmienną docelową (targetem).

## 3. EDA: rozkłady, outliery, korelacje

W tej części:
- obejrzymy rozkłady zmiennych numerycznych (histogramy),
- sprawdzimy outliery (boxploty),
- policzymy macierz korelacji.

Na końcu opisz, jakie zależności i problemy zauważasz.

```python
numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns
numeric_cols
```

```python
for col in numeric_cols:
    plt.figure()
    df[col].hist(bins=30)
    plt.title(f"Histogram: {col}")
    plt.xlabel(col)
    plt.ylabel("Liczba obserwacji")
    plt.show()
```

```python
for col in numeric_cols:
    plt.figure()
    df.boxplot(column=col)
    plt.title(f"Boxplot: {col}")
    plt.show()
```

```python
corr = df.corr(numeric_only=True)
corr
```

## 4. Preprocessing danych

Tutaj:
- obsłużysz braki danych,
- zakodujesz zmienne kategoryczne,
- przygotujesz cechy i target,
- zastosujesz skalowanie cech.

Zapisz swoje decyzje i ich uzasadnienie.

```python
data = df.copy()

# Przykład: imputacja medianą dla kolumn numerycznych
for col in numeric_cols:
    if data[col].isna().sum() > 0:
        data[col] = data[col].fillna(data[col].median())

# Przykład: wypełnianie braków w kolumnach kategorycznych
cat_cols = data.select_dtypes(include=["object", "category"]).columns

for col in cat_cols:
    if data[col].isna().sum() > 0:
        data[col] = data[col].fillna("brak_danych")
```

```python
data_encoded = pd.get_dummies(data, drop_first=True)
data_encoded.head()
```

### Wybór zmiennej docelowej (targetu)

Wskaż, która kolumna będzie Twoim targetem (`y`) i uzasadnij krótko wybór.

```python
# TODO: podmień 'target' na nazwę wybranej kolumny docelowej
target_col = "target"

X = data_encoded.drop(target_col, axis=1)
y = data_encoded[target_col]

X.shape, y.shape
```

```python
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
```

## 5. Model ML: regresja lub klasyfikacja

Wybierz odpowiedni typ modelu.

```python
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

X_train.shape, X_test.shape, y_train.shape, y_test.shape
```

### 5.1. Model regresyjny (opcjonalnie)

```python
reg_model = LinearRegression()
reg_model.fit(X_train, y_train)

y_pred_reg = reg_model.predict(X_test)

mse = mean_squared_error(y_test, y_pred_reg)
mae = mean_absolute_error(y_test, y_pred_reg)

mse, mae
```

### 5.2. Model klasyfikacyjny (opcjonalnie)

```python
clf = LogisticRegression(max_iter=1000)
clf.fit(X_train, y_train)

y_pred_clf = clf.predict(X_test)

acc = accuracy_score(y_test, y_pred_clf)
cm = confusion_matrix(y_test, y_pred_clf)

acc, cm
```

## 6. Podsumowanie i wnioski

Napisz krótkie podsumowanie projektu, odpowiadając na pytania:
- czego dowiedziałeś/dowiedziałaś się o danych,
- jakie decyzje preprocessingowe podjąłeś/podjęłaś,
- jak poradził sobie model,
- co można by zrobić dalej.
