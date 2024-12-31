from scipy.stats import chi2_contingency, ttest_ind
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, accuracy_score, classification_report
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from imblearn.over_sampling import SMOTE
import joblib
import matplotlib.pyplot as plt 

def split_data(df, target, test_size=0.2, random_state=42):
    # Split the data
    X = df.drop(target, axis=1)
    y = df[target]
    return train_test_split(X, y, test_size=test_size, random_state=random_state)

def resample(x_train, y_train): # , name
    smote = SMOTE(random_state = 42)
    X_train_resample, y_train_resample = smote.fit_resample(x_train, y_train)
    # joblib.dump(smote, f'../models/smote_{name}.pkl')
    return X_train_resample, y_train_resample

def train_model(name, x_train, y_train):
    if name == 'linear_regression':
        # Create and fit the linear regression model
        linear_model = LinearRegression()
        linear_model.fit(x_train, y_train)

        dt_model = DecisionTreeClassifier(random_state=42)
        dt_model.fit(x_train, y_train)

        rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
        rf_model.fit(x_train, y_train)

        return model
    
def predict_model(model, x_test, y_test, pred_threshold=0.5):
    
    
    y_pred = model.predict(x_test)
    y_pred = (y_pred > pred_threshold).astype(int)

    accuracy = accuracy_score(y_test, y_pred)

    print('test accuracy', accuracy)
    print('classification report:\n',classification_report(y_test, y_pred) )


    # Create confusion matrix
    cm = confusion_matrix(y_test, y_pred)

    # Display the confusion matrix
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['normal', 'fraud'])
    disp.plot(cmap=plt.cm.Blues)  # You can choose a color map
    plt.title("Confusion Matrix")
    plt.show()

