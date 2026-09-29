import numpy as np
import shap
from tensorflow.keras.models import load_model



def explain_predictions(ticker):
    from step1_data import fetch_and_engineer_data
    from step2_preprocess import split_and_scale_data
    
    print(f"Loading data and model for {ticker}...")
    df = fetch_and_engineer_data(ticker)
    X_train, y_train, X_test, y_test, f_scaler, t_scaler = split_and_scale_data(df)
    
    model = load_model(f'models/{ticker}_lstm_model.keras')
    
    print("\nInitializing SHAP KernelExplainer (optimized for modern TF/Keras)...")
    # Use a small random sample of 50 from training data as background distribution to keep it fast
    np.random.seed(42)
    background_indices = np.random.choice(X_train.shape[0], 50, replace=False)
    background = X_train[background_indices]
    
    # SHAP requires 2D arrays, so we flatten our 3D data (samples, time_steps, features)
    background_flattened = background.reshape(background.shape[0], -1)
    
    # Create a wrapper function that reshapes the 2D flat array back to 3D for the LSTM model
    def predict_wrapper(flattened_x):
        reshaped_x = flattened_x.reshape(flattened_x.shape[0], X_train.shape[1], X_train.shape[2])
        return model.predict(reshaped_x, verbose=0).flatten()

    # Model-agnostic explainer
    explainer = shap.KernelExplainer(predict_wrapper, background_flattened)
    
    print("Calculating SHAP values for the first 5 test predictions...")
    test_samples = X_test[:5]
    test_flattened = test_samples.reshape(test_samples.shape[0], -1)
    
    shap_values = explainer.shap_values(test_flattened)
    
    print("\nSHAP explanation completed successfully!")
    print("The SHAP values array shape is:", np.array(shap_values).shape)
    print("These values represent the impact of each flattened feature (timestep x indicator) on the predicted price.")
    
if __name__ == "__main__":
    explain_predictions('BTC-USD')
