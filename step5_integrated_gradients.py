import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model

@tf.function
def compute_gradients(model, inputs):
    with tf.GradientTape() as tape:
        tape.watch(inputs)
        predictions = model(inputs)
    # Get the gradient of the prediction with respect to the input features
    return tape.gradient(predictions, inputs)

def integrated_gradients(model, inputs, baseline, steps=50):
    """
    Calculates the exact mathematical feature importance for a Deep Learning model
    using TensorFlow's native Integrated Gradients.
    """
    # 1. Create steps (alphas) between the baseline (all zeros) and the actual data
    alphas = tf.linspace(start=0.0, stop=1.0, num=steps+1)
    
    # Initialize a tensor to store the integrated gradients
    igrads = np.zeros_like(inputs)
    
    # 2. Calculate the difference between the input and the baseline
    delta = inputs - baseline
    
    # 3. For each sample in the dataset, calculate the path integral
    for i in range(len(inputs)):
        sample_input = inputs[i:i+1]
        sample_baseline = baseline[i:i+1]
        sample_delta = delta[i:i+1]
        
        # Generate the interpolated inputs along the path
        interpolated = [sample_baseline + alpha * sample_delta for alpha in alphas]
        interpolated = tf.concat(interpolated, axis=0)
        
        # Calculate gradients for all interpolated steps
        grads = compute_gradients(model, interpolated)
        
        # Approximate the integral using the trapezoidal rule
        grads = (grads[:-1] + grads[1:]) / 2.0
        avg_grads = tf.reduce_mean(grads, axis=0)
        
        # Multiply the average gradient by the delta to get the final Integrated Gradient
        igrads[i] = (sample_delta * avg_grads).numpy()
        
    return igrads

def explain_with_ig(ticker):
    from step1_data import fetch_and_engineer_data
    from step2_preprocess import split_and_scale_data
    
    print(f"Loading data and model for {ticker}...")
    df = fetch_and_engineer_data(ticker)
    X_train, y_train, X_test, y_test, f_scaler, t_scaler = split_and_scale_data(df)
    
    model = load_model(f'models/{ticker}_lstm_model.keras')
    
    # We use a baseline of zeros (representing "no signal")
    baseline = np.zeros_like(X_test)
    inputs = tf.convert_to_tensor(X_test, dtype=tf.float32)
    baseline = tf.convert_to_tensor(baseline, dtype=tf.float32)
    
    print("Calculating Integrated Gradients (Native TensorFlow)...")
    ig_values = integrated_gradients(model, inputs, baseline, steps=50)
    
    print("\nIntegrated Gradients calculation completed successfully!")
    print(f"IG array shape: {ig_values.shape}")
    
    # Let's aggregate the importance to show the top features
    feature_names = ['Close', 'Volume', 'SMA_20', 'SMA_50', 'EMA_20', 'RSI', 'MACD', 'ATR', 'BB_High', 'BB_Low', 'Nasdaq', 'DXY']
    
    # Sum the absolute importance across all test samples and all 60 timesteps
    global_importance = np.sum(np.abs(ig_values), axis=(0, 1))
    
    # Normalize to 100%
    global_importance = (global_importance / np.sum(global_importance)) * 100
    
    print("\n=== OVERALL FEATURE IMPORTANCE ===")
    importance_dict = {feature_names[i]: global_importance[i] for i in range(len(feature_names))}
    sorted_importance = dict(sorted(importance_dict.items(), key=lambda item: item[1], reverse=True))
    
    for feature, importance in sorted_importance.items():
        print(f"{feature.ljust(10)}: {importance:.2f}% impact")

if __name__ == "__main__":
    explain_with_ig('BTC-USD')
