from flask import Flask, request, render_template
from dash_app import create_dash_app
import logging
import joblib
import numpy as np
import pandas as pd 
import logging
pd.set_option('display.max_colwidth', None)
pd.set_option('display.max_columns', None)

app = Flask(__name__)

# Load model, scaler, and encoder
import joblib

try:
    model = joblib.load('../models/rf_model_20241030_110457.pkl')
    scaler = joblib.load('../models/standard_scaler_fraud_ecommerce.pkl')
    one_hot_encoder = joblib.load('../models/one_hot_encoder_fraud_ecommerce.pkl')
except Exception as e:
    print("Error loading:", e)


def preprocess_input(data, encoder, scaler):

    categorical_feature = ['source', 'browser', 'sex']
    numerical_feature = [ 'purchase_value', 'age',  'hour', 'day', 'transaction_count']

    print('START Preprocess')

    
    # Scale the numerical features
    data[numerical_feature] = scaler.transform(data[numerical_feature])
    logging.info('Scaled numerical features')


    # encoded_data = one_hot_encoder.transform(data[categorical_feature])

    # Combine processed features
    # processed_data = np.hstack((scaled_data, encoded_data))

    processed_data = pd.DataFrame(data, columns=[ 'purchase_value','age', 'hour', 'day', 'transaction_count']) #,'source_Direct',	'source_SEO',	'browser_FireFox', 'browser_IE',	'browser_Opera',	'browser_Safari', 'sex_M' ])

    logging.info('Encoded categorical features')
 

    return processed_data


@app.route("/",methods=["GET", "POST"])
def home():
    
    prediction = None
    if request.method == "POST":
        try:
            # Collect data from form fields
            source = request.form['source']
            browser = request.form['browser']
            sex = request.form['sex']
            # country = request.form['country']
            age = int(request.form['age'])
            purchase_value = float(request.form['purchase_value'])
            hour = int(request.form['hour'])
            day = int(request.form['day'])
            transaction_count = int(request.form['transaction_count'])

            # Create a DataFrame from the individual variables
            data_df = pd.DataFrame({
                'purchase_value': [purchase_value],
                'age': [age],
                'hour': [hour],
                'day': [day],
                'transaction_count': [transaction_count]
                
                })
            # 'source': [source],
            #     'browser': [browser],
            #     'sex': [sex]
            # input_df = pd.DataFrame([[purchase_value, age, hour, day, transaction_count, source, browser, sex]],
            #                     columns=[ 'purchase_value','age', 'hour', 'day', 'transaction_count', 'source', 'browser', 'sex'])
            # Display the DataFrame (for debugging purposes)
            print(data_df)

            # input_data = pd.DataFrame([data])
            # print(input_data.iloc[0])

            # Preprocess the input and make a prediction
            processed_data = preprocess_input(data_df, one_hot_encoder, scaler)
            print("Processed data shape:", processed_data.shape)
            # processed_data = np.array(processed_data).reshape(1, -1)
            processed_data = processed_data.to_numpy()
            print(processed_data)
            prediction = model.predict(processed_data)[0]
            print('prediction: ',prediction)
            # Render the prediction result in the HTML template
            # return render_template('index.html', prediction=prediction)
        
        except Exception as e:
            # Render the error in the HTML template if any exception occurs
            error = f"An error occurred: {str(e)}"
            # return render_template('index.html', error=str(e))
     #If the request method is GET
    return render_template("index.html",  prediction=prediction)

# Initialize the Dash app
create_dash_app(app)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)  













# from flask import Flask, request, jsonify, render_template
# # from routes import api
# import logging
# import pickle  # Use if your model is stored as a pickle file
# import joblib
# import numpy as np

# app = Flask(__name__)


# def preprocess_input(data, encoder, scaler):
#     # Example preprocessing function to prepare the data
#     categorical_data = encoder.transform([[data['categoricalFeature']]])
#     numerical_data = scaler.transform([[float(data['feature1']), float(data['feature2'])]])
#     return np.concatenate((numerical_data, categorical_data), axis=1)[0]



# model = joblib.load('../models/rf_model_20241029_104504.pkl')
# scaler = joblib.load('../models/standard_scaler_fraud_ecommerce.pkl')
# label_encoder = joblib.load('../models/label_encoder_fraud_ecommerce.pkl')


# @app.route("/")
# def home():
#     return render_template('index.html')


# @app.route('/predict', methods=['POST'])
# def predict():
#     try:
        
        
#         # Render the result back to the form page
#         return render_template('index.html', prediction=prediction)
#     except Exception as e:
#         return render_template('index.html', error=str(e))



# if __name__ == '__main__':
#     app.run(host='0.0.0.0', port=5000,debug=True)




# def prediction(data):

#     data_scaled = scaler.transform(data)

#     prediction = model.predict(data_scaled)
#     prediction = (prediction > 0.9).astype(int)

#     return prediction


# # Get data from the form
#         feature1 = request.form['feature1']
#         feature2 = request.form['feature2']
#         categoricalFeature = request.form['categoricalFeature']
        
#         # Process input and make prediction
#         data = {
#             'feature1': feature1,
#             'feature2': feature2,
#             'categoricalFeature': categoricalFeature
#         }
#         processed_data = preprocess_input(data, label_encoder, scaler)
#         prediction = model.predict([processed_data])[0]


# # Load your model (adjust the path and loading method to match your model file)
# with open('model/fraud_detection_model.pkl', 'rb') as f:
#     model = pickle.load(f)

# # Set up logging
# logging.basicConfig(level=logging.INFO)

# @app.route('/predict', methods=['POST'])
# def predict():
#     data = request.get_json(force=True)
#     logging.info(f"Received data: {data}")
#     try:
#         # Assuming 'data' is a JSON object with input features
#         features = [data['feature1'], data['feature2'], data['featureN']]
#         prediction = model.predict([features])[0]  # Adjust based on model input
#         logging.info(f"Prediction: {prediction}")
#         return jsonify({'prediction': prediction})
#     except Exception as e:
#         logging.error(f"Error in prediction: {e}")
#         return jsonify({'error': str(e)}), 500
# import pandas as pd  # Make sure to import pandas

# # Collect data from form fields
# source = request.form['source']
# browser = request.form['browser']
# sex = request.form['sex']
# country = request.form['country']
# age = float(request.form['age'])
# purchase_value = float(request.form['purchase_value'])
# hour = int(request.form['hour'])
# day = int(request.form['day'])
# transaction_count = int(request.form['transaction_count'])

    # # Encode categorical features
    # categorical_data = encoder.transform([[data['source'], data['browser'], data['sex'], data['country']]])[0]
    # print('pass')
    # # Scale numerical features
    # numerical_data = scaler.transform([[data['age'], data['purchase_value'], data['hour'], data['day'], data['transaction_count']]])
    # print('pass')
    # # Combine processed data
    # return (np.concatenate((numerical_data, categorical_data), axis=1)[0]).flatten()


#     from flask import Flask, request, render_template
# import logging
# import joblib
# import numpy as np
# import pandas as pd

# app = Flask(__name__)

# # Load model, scaler, and encoder
# model = joblib.load('../models/rf_model_20241029_104504.pkl')
# scaler = joblib.load('../models/standard_scaler_fraud_ecommerce.pkl')
# label_encoders = joblib.load('../models/label_encoder_fraud_ecommerce.pkl')  # Assuming this is a dictionary of encoders

# def preprocess_input(data, encoders, scaler):
#     categorical_features = ['source', 'browser', 'sex', 'country']
#     num_cat = ['age', 'purchase_value', 'hour', 'day', 'transaction_count']
    
#     logging.info('START Preprocess')
    
#     # Encode each categorical feature individually
#     encoded_data = []
#     for feature in categorical_features:
#         encoder = encoders[feature]
#         encoded_feature = encoder.transform(data[feature])
#         encoded_data.append(encoded_feature)
    
#     encoded_data = np.column_stack(encoded_data)
#     logging.info('Encoded categorical features')
    
#     # Scale the numerical features
#     scaled_data = scaler.transform(data[num_cat])
#     logging.info('Scaled numerical features')
    
#     # Combine processed features
#     processed_data = np.hstack((encoded_data, scaled_data))
#     return processed_data

# @app.route("/")
# def home():
#     return render_template('index.html')

# @app.route('/predict', methods=['POST'])
# def predict():
#     prediction = None
    
#     try:
#         # Collect data from form fields
#         source = request.form['source']
#         browser = request.form['browser']
#         sex = request.form['sex']
#         country = request.form['country']
#         age = int(request.form['age'])
#         purchase_value = float(request.form['purchase_value'])
#         hour = int(request.form['hour'])
#         day = int(request.form['day'])
#         transaction_count = int(request.form['transaction_count'])

#         # Create a DataFrame from the individual variables
#         data_df = pd.DataFrame({
#             'source': [source],
#             'browser': [browser],
#             'sex': [sex],
#             'country': [country],
#             'age': [age],
#             'purchase_value': [purchase_value],
#             'hour': [hour],
#             'day': [day],
#             'transaction_count': [transaction_count]
#         })

#         # Display the DataFrame (for debugging purposes)
#         print(data_df)

#         # Preprocess the input and make a prediction
#         processed_data = preprocess_input(data_df, label_encoders, scaler)
#         print("Processed data shape:", processed_data.shape)
#         prediction = model.predict(processed_data)[0]
        
#         # Render the prediction result in the HTML template
#         return render_template('index.html', prediction=prediction)
    
#     except Exception as e:
#         # Render the error in the HTML template if any exception occurs
#         return render_template('index.html', error=str(e))

# if __name__ == '__main__':
#     app.run(host='0.0.0.0', port=5000, debug=True)



    # categ = data[categorical_feature].values
    # print(categ)
    # Encode each categorical feature individually
    # encoded_data = label_encoder.transform(categ)
    # encoded_data = []
    # for feature in categorical_feature:
    #     encoded_feature = label_encoder.transform(data[feature])
    #     encoded_data.append(encoded_feature)