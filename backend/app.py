# Import necessary libraries
import numpy as np
import joblib  # For loading the serialized model
import pandas as pd  # For data manipulation
from flask import Flask, request, jsonify  # For creating the Flask API

# Initialize Flask app with a name
superkart_api = Flask("SuperKart")

# Load the trained model
try:
    model = joblib.load("superkart_model.joblib")
except Exception as e:
    model = None
    print(f"Error loading model: {e}")

# Helper function to categorize products into Perishables / Non Perishables
def categorize_product(product_type):
    perishables = ["Dairy", "Meat", "Fruits and Vegetables", "Breakfast", "Seafood", "Breads"]
    return "Perishables" if product_type in perishables else "Non Perishables"

# Define a route for the home page
@superkart_api.get('/')
def home():
    return "Welcome to the SuperKart System"

# Define an endpoint to predict sales for a single product
@superkart_api.post('/v1/predict')
def predict_sales():
    try:
        # Get JSON data from the request
        data = request.get_json()

        # Extract relevant features from the input data
        sample = {
            'Product_Weight': data.get('Product_Weight'),
            'Product_Sugar_Content': data.get('Product_Sugar_Content'),
            'Product_Allocated_Area': data.get('Product_Allocated_Area'),
            'Product_MRP': data.get('Product_MRP'),
            'Store_Size': data.get('Store_Size'),
            'Store_Location_City_Type': data.get('Store_Location_City_Type'),
            'Store_Type': data.get('Store_Type'),
            'Product_Id_char': data.get('Product_Id_char'),
            'Store_Age_Years': data.get('Store_Age_Years'),
            'Product_Type_Category': data.get('Product_Type_Category'),
            'Product_Type': data.get('Product_Type'),
            'Store_Id': data.get('Store_Id')
        }

        # Convert the extracted data into a DataFrame
        input_data = pd.DataFrame([sample])

        if model is None:
            raise ValueError("Machine learning model is not loaded properly.")

        # Make a prediction using the trained model
        prediction = model.predict(input_data).tolist()[0]

        # Return the prediction as a JSON response
        return jsonify({'Sales': prediction})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Define an endpoint to predict sales for a batch of products with robust preprocessing
@superkart_api.post('/v1/predictbatch')
def predict_sales_batch():
    try:
        # Get the uploaded CSV file from the request
        if 'file' not in request.files:
            raise ValueError("No file field found in the request.")

        file = request.files['file']

        # Read the file into a DataFrame
        raw_data = pd.read_csv(file)
        processed_data = raw_data.copy()

        # Robust Preprocessing: Only run feature engineering steps if the engineered columns are missing
        if "Product_Sugar_Content" in processed_data.columns:
            processed_data["Product_Sugar_Content"] = processed_data["Product_Sugar_Content"].replace("reg", "Regular")

        if "Product_Id_char" not in processed_data.columns:
            if "Product_Id" in processed_data.columns:
                processed_data["Product_Id_char"] = processed_data["Product_Id"].str[:2]
            else:
                raise ValueError("Either 'Product_Id_char' or raw 'Product_Id' must be present in the CSV.")

        if "Store_Age_Years" not in processed_data.columns:
            if "Store_Establishment_Year" in processed_data.columns:
                processed_data["Store_Age_Years"] = 2025 - processed_data["Store_Establishment_Year"]
            else:
                raise ValueError("Either 'Store_Age_Years' or raw 'Store_Establishment_Year' must be present in the CSV.")

        if "Product_Type_Category" not in processed_data.columns:
            if "Product_Type" in processed_data.columns:
                processed_data["Product_Type_Category"] = processed_data["Product_Type"].apply(categorize_product)
            else:
                raise ValueError("Either 'Product_Type_Category' or raw 'Product_Type' must be present in the CSV.")

        # Select only the expected feature columns
        features_to_keep = [
            "Product_Weight", "Product_Sugar_Content", "Product_Allocated_Area", "Product_MRP",
            "Store_Size", "Store_Location_City_Type", "Store_Type", "Product_Id_char",
            "Store_Age_Years", "Product_Type_Category", "Product_Type", "Store_Id"
        ]

        # Ensure all model target features are present
        missing_features = [col for col in features_to_keep if col not in processed_data.columns]
        if missing_features:
            raise ValueError(f"Missing required features for model prediction: {missing_features}")

        model_inputs = processed_data[features_to_keep]

        if model is None:
            raise ValueError("Machine learning model is not loaded properly.")

        # Make predictions for the batch data
        predictions = model.predict(model_inputs).tolist()

        # Create an output dictionary mapping row index to predicted sales
        output_dict = {str(i): round(pred, 2) for i, pred in enumerate(predictions)}

        return jsonify(output_dict)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Run the Flask app in debug mode
if __name__ == '__main__':
    superkart_api.run(debug=True)
