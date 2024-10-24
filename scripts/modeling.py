from scipy.stats import chi2_contingency, ttest_ind
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, accuracy_score, classification_report
from imblearn.over_sampling import SMOTE
import joblib

def split_data(X, y, test_size=0.2, random_state=42):
    return train_test_split(X, y, test_size=test_size, random_state=random_state)

def resample(x_train, y_train, name):
    smote = SMOTE(random_state = 42)
    X_train_resample, y_train_resample = smote.fit_resample(x_train, y_train)
    joblib.dump(smote, f'../models/smote_{name}.pkl')
    return X_train_resample, y_train_resample

def train_model(x, y, name):
    if name == 'linear_regression':
        # Create and fit the linear regression model
        model = LinearRegression()
        model.fit(x, y)
        return  model