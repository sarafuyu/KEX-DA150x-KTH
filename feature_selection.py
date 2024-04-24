import pandas as pd

# %%
df_imputed = pd.read_csv('imputed_data.csv')
df_imputed.head()
# %%
from sklearn.model_selection import train_test_split

y = df_imputed['FT5']  # Vector for the target variable
X = df_imputed.iloc[:, 1:]  # Matrix with variable input

# Splitting the dataset into training and testing sets (80% - 20%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# %% md
## 1. Univariate Feature Selection
'''This method selects the best features based on univariate statistical tests. It can be seen as a preprocessing step to an estimator.'''
# %% md

# %%
from sklearn.feature_selection import SelectKBest, f_classif

# Select the top 100 features based on ANOVA F-value
select_k_best = SelectKBest(f_classif, k=100)
X_train_selected = select_k_best.fit_transform(X_train, y_train)
X_test_selected = select_k_best.transform(X_test)

print("Selected Features Shape:", X_train_selected.shape)
# %% md
## 2. Feature Selection Using Model
'''You can use a model to determine the importance of each feature and select the most important features accordingly. Here, we'll use ExtraTreesClassifier as an example for classification. For regression tasks, you could use ExtraTreesRegressor.'''
# %%
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import SelectFromModel

model = ExtraTreesClassifier(n_estimators=50)
model = model.fit(X_train, y_train)

# Model-based feature selection
model_select = SelectFromModel(model, prefit=True)
X_train_model = model_select.transform(X_train)
X_test_model = model_select.transform(X_test)

print("Model Selected Features Shape:", X_train_model.shape)
# %% md
## 3. Recursive Feature Elimination (RFE)
'''RFE works by recursively removing the least important feature and building a model on those features that remain.'''
# %%
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression

# Initialize the model to be used
model = LogisticRegression(max_iter=1000)

# Initialize RFE and select the top 100 features
rfe = RFE(estimator=model, n_features_to_select=100, step=1)
X_train_rfe = rfe.fit_transform(X_train, y_train)
X_test_rfe = rfe.transform(X_test)

print("RFE Selected Features Shape:", X_train_rfe.shape)
# %% md
# Feature selection
# %%
from sklearn.feature_selection import SelectKBest, f_classif

# Select the top 100 features based on ANOVA F-value
select_k_best = SelectKBest(f_classif, k=100)
X_train_selected = select_k_best.fit_transform(X_train, y_train)
X_test_selected = select_k_best.transform(X_test)

print("Selected Features Shape:", X_train_selected.shape)
# %%
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import SelectFromModel

model = ExtraTreesClassifier(n_estimators=50)
model = model.fit(X_train, y_train)

# Model-based feature selection
model_select = SelectFromModel(model, prefit=True)
X_train_model = model_select.transform(X_train)
X_test_model = model_select.transform(X_test)

print("Model Selected Features Shape:", X_train_model.shape)
# %%
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression

# Initialize the model to be used
model = LogisticRegression(max_iter=1000)

# Initialize RFE and select the top 100 features
rfe = RFE(estimator=model, n_features_to_select=100, step=1)
X_train_rfe = rfe.fit_transform(X_train, y_train)
X_test_rfe = rfe.transform(X_test)

print("RFE Selected Features Shape:", X_train_rfe.shape)
