
# Imports
import streamlit as st
import pickle
import numpy as np

# Load model safely
def load_model():
    try:
        with open("Customer_Behaviour.pkl",'rb') as file:
            model = pickle.load(file)
    
        scaler = None 
        try:
            with open("scaler.pkl",'rb') as file:
                scaler = pickle.load(file)
        except:
            st.warning("Scaler not found or invalid")
    
        return model,scaler

    except FileNotFoundError as e:
        st.error(f"Model file not found: {e}")
        return None, None
    except Exception as e:
        st.error(f"Error loading artifacts: {e}")
        return None, None

model, scaler = load_model()

# Convert gender input to numeric
def genderInput(gender_input):
    gender_input = gender_input.lower().strip()
    if gender_input == 'male':
        return 0
    elif gender_input == 'female':
        return 1
    else:
        return None

# Convert prediction result to Yes / No (NO THRESHOLD - like Flask version)
def resultOutput(result):
    """
    Convert numeric prediction to meaningful output
    0 = Not Purchased, 1 = Purchased
    """
    if result == 1:
        return 'Yes'
    else:
        return 'No'

# Prediction function - CORRECTED to match Flask version
def customer_satisfaction_prediction(gender_input, age_input, salary_input, debug=False):
    try:
        if debug:
            st.write("DEBUG: Starting prediction function")
        
        gender_value = genderInput(gender_input)

        if gender_value is None:
            return "Error: Gender must be Male or Female", None

        age_value = float(age_input)
        salary_value = float(salary_input)

        # CORRECTED: Use same feature order as Flask version [Gender, Age, Salary]
        input_data = np.array([[gender_value, age_value, salary_value]])

        if debug:
            st.write(f"DEBUG: Raw input data - Gender: {gender_value}, Age: {age_value}, Salary: {salary_value}")

        if scaler is None or not hasattr(scaler, 'transform'):
            return "Error: Scaler not available or invalid. Please check scaler.pkl file.", None

        scaled_data = scaler.transform(input_data)
        
        if debug:
            st.write(f"DEBUG: Scaled data: {scaled_data}")

        prediction = model.predict(scaled_data)
        probabilities = model.predict_proba(scaled_data)
        predicted_purchase = int(prediction[0])  # numeric 0 or 1
        
        # CORRECTED: Use same confidence calculation as Flask version
        # Confidence for the predicted class (not just class 1)
        confidence = probabilities[0][predicted_purchase]

        if debug:
            st.write(f"DEBUG: Raw Prediction Output: {prediction}")
            st.write(f"DEBUG: Prediction Probabilities: {probabilities}")
            st.write(f"DEBUG: Predicted class: {predicted_purchase}")
            st.write(f"DEBUG: Confidence for predicted class: {confidence:.4f}")
        
        return predicted_purchase, confidence

    except Exception as e:
        return f"Prediction Error: {e}", None

# Main app
def main():
    st.title('Customer Behaviour Prediction Web App')
    
    # Add debug toggle
    debug_mode = st.checkbox('Enable Debug Mode', value=False)
    
    # Use better input methods
    gender_input = st.selectbox('Select Gender', ['Male', 'Female'])
    age_input = st.number_input('Enter Age', min_value=18, max_value=100, value=30)
    salary_input = st.number_input('Enter Estimated Salary', min_value=0, value=50000, step=1000)

    if st.button('Predict Customer Purchase'):
        if model is None:
            st.error("Model not loaded properly. Please check model files.")
            return
            
        result, confidence = customer_satisfaction_prediction(gender_input, age_input, salary_input, debug=debug_mode)

        # If result is error message (string), show error
        if isinstance(result, str) and (result.startswith("Error") or result.startswith("Prediction Error")):
            st.error(result)
        else:
            # CORRECTED: Use resultOutput without threshold (like Flask version)
            result_output = resultOutput(result)
            st.success(f"Will the customer purchase? : {result_output}")
            if confidence is not None:
                st.info(f"Confidence: {confidence:.2%}")

if __name__ == '__main__':
    main()
