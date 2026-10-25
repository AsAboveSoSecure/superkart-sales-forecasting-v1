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

# Define an endpoint to predict sales for a batch of products with full preprocessing
@superkart_api.post('/v1/predictbatch')
def predict_sales_batch():
    try:
        # Get the uploaded CSV file from the request
        if 'file' not in request.files:
            raise ValueError("No file field found in the request.")
            
        file = request.files['file']

        # Read the file into a DataFrame
        raw_data = pd.read_csv(file)

        # 1. Apply exact same preprocessing steps as used during training
        processed_data = raw_data.copy()
        processed_data["Product_Sugar_Content"] = processed_data["Product_Sugar_Content"].replace("reg", "Regular")
        processed_data["Product_Id_char"] = processed_data["Product_Id"].str[:2]
        processed_data["Store_Age_Years"] = 2025 - processed_data["Store_Establishment_Year"]
        processed_data["Product_Type_Category"] = processed_data["Product_Type"].apply(categorize_product)

        # 2. Select only the expected feature columns
        features_to_keep = [
            "Product_Weight", "Product_Sugar_Content", "Product_Allocated_Area", "Product_MRP",
            "Store_Size", "Store_Location_City_Type", "Store_Type", "Product_Id_char",
            "Store_Age_Years", "Product_Type_Category", "Product_Type", "Store_Id"
        ]
        model_inputs = processed_data[features_to_keep]

        if model is None:
            raise ValueError("Machine learning model is not loaded properly.")

        # 3. Make predictions for the batch data
        predictions = model.predict(model_inputs).tolist()

        # 4. Create an output dictionary mapping row index to predicted sales
        output_dict = {str(i): round(pred, 2) for i, pred in enumerate(predictions)}

        return jsonify(output_dict)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Run the Flask app in debug mode
if __name__ == '__main__':
    superkart_api.run(debug=True)
