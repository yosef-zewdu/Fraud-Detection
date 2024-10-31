
def preprocess_input(data, scaler):

    # Scale numerical features
    processed_data = scaler.transform(data)

    return processed_data
