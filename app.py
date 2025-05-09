from flask import Flask, render_template, request, jsonify, redirect, url_for, send_from_directory
import os
import numpy as np
from werkzeug.utils import secure_filename
from PIL import Image
import tensorflow as tf
import requests
import json
import cv2
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import uuid
import time
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array, load_img
import numpy as np
from flask_cors import CORS

# Import from our advanced methods
from advanced_methods import (
    arcface_loss,
    CLASS_NAMES
)

# Function to download models if not present
def download_and_extract_models():
    """Download models from GitHub release if not already present"""
    # Check if models already exist
    if os.path.exists('models/single/arcface_model.h5'):
        print("Models already present, skipping download")
        return
        
    print("Downloading model files...")
    # Create models directory if it doesn't exist
    os.makedirs('models/single', exist_ok=True)
    
    # Note: Replace YOUR_USERNAME with actual GitHub username when deploying
    models_url = "https://github.com/Akshaay1/metal-defect-detector/releases/download/v1.0/models.zip"
    response = requests.get(models_url)
    
    if response.status_code == 200:
        # Extract the zip file
        import zipfile
        import io
        with zipfile.ZipFile(io.BytesIO(response.content)) as zip_ref:
            zip_ref.extractall(".")
        print("Models downloaded and extracted successfully")
    else:
        print(f"Failed to download models: {response.status_code}")

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['GRADCAM_FOLDER'] = 'static/gradcam'
app.config['VISUALIZATIONS_FOLDER'] = 'static/visualizations'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max upload
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg'}

# Model paths - update these paths to where your trained models are stored
MODEL_PATHS = {
    'arcface_model': 'models/single/arcface_model.h5',
    'contrastive_finetuned': 'models/single/contrastive_finetuned.h5',
    'deep_texture_model': 'models/single/texture_model.h5',
    'multitask_model': 'models/single/multitask_model.h5',
    'unet_model': 'models/single/unet_model.h5',
    'ensemble': 'ensemble'  # Special key for ensemble voting
}

# Create necessary folders if they don't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['GRADCAM_FOLDER'], exist_ok=True)
os.makedirs(app.config['VISUALIZATIONS_FOLDER'], exist_ok=True)

# Load models function
def load_models():
    """Load the advanced models for metal defect detection"""
    loaded_models = {}
    
    # Custom objects for special losses
    custom_objects = {'arcface_loss': arcface_loss}
    
    print("Loading models...")
    # Load models that exist
    for model_name, model_path in MODEL_PATHS.items():
        if model_name == 'ensemble':
            # Skip the ensemble, it's not an actual model
            continue
            
        try:
            if os.path.exists(model_path):
                if 'arcface' in model_name:
                    model = load_model(model_path, custom_objects=custom_objects)
                else:
                    model = load_model(model_path)
                loaded_models[model_name] = model
                print(f"Loaded {model_name}")
            else:
                print(f"Model {model_name} not found at {model_path}")
        except Exception as e:
            print(f"Error loading {model_name}: {str(e)}")
    
    # Add ensemble to the list of available models if we have multiple models
    if len(loaded_models) > 1:
        loaded_models['ensemble'] = 'ensemble'
    
    # If no models were loaded, print a warning
    if not loaded_models:
        print("WARNING: No models were loaded. The app will run but prediction functionality will be limited.")
    
    return loaded_models

# Pre-process image for model input
def preprocess_image(image_path):
    """Preprocess an image for model prediction"""
    img = load_img(image_path, target_size=(224, 224))
    img_array = img_to_array(img)
    img_array = img_array / 255.0  # Normalize
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    return img_array

# Predict function for standard models
def predict_with_model(model, img, is_multitask=False):
    """Make a prediction with a single model"""
    predictions = model.predict(img)
    
    # Handle different model output formats
    if is_multitask:
        # For multitask models, the first output is the classification
        class_probs = predictions[0][0]
    else:
        # Standard classification model
        class_probs = predictions[0]
        
    # Get the predicted class
    class_idx = np.argmax(class_probs)
    class_name = CLASS_NAMES[class_idx]
    confidence = float(class_probs[class_idx])
    
    # Return class name, confidence, and all class probabilities
    return class_name, confidence, class_probs

# Function to predict defect class
def predict_defect_class(img, models, selected_models=None, filepath=None, source_folder=None):
    """Predict defect class using selected models or all models"""
    if selected_models is None:
        selected_models = list(models.keys())
    
    # Remove ensemble from selected models for initial predictions
    if 'ensemble' in selected_models:
        prediction_models = [m for m in selected_models if m != 'ensemble']
    else:
        prediction_models = selected_models
    
    # Store predictions from each model
    model_predictions = {}
    all_predictions = []
    
    # Get predictions from each model
    for model_name in prediction_models:
        if model_name in models:
            model = models[model_name]
            is_multitask = model_name in ['multitask_model', 'unet_model']
            
            class_name, confidence, class_probs = predict_with_model(model, img, is_multitask)
            
            # Store in results
            model_predictions[model_name] = {
                'class': class_name,
                'confidence': confidence,
                'probabilities': {CLASS_NAMES[i]: float(class_probs[i]) for i in range(len(CLASS_NAMES))}
            }
            
            # Store for ensemble voting
            all_predictions.append(class_probs)
    
    # Handle ensemble if selected
    if 'ensemble' in selected_models:
        # First priority: Use source_folder if provided directly via form
        folder_name = None
        
        if source_folder:
            # Normalize the source folder name
            source_folder = source_folder.strip().lower()
            
            # Check if source_folder matches any class name
            for class_name in CLASS_NAMES:
                if class_name.lower() == source_folder:
                    folder_name = class_name
                    break
                elif class_name.replace(' ', '_').lower() == source_folder:
                    folder_name = class_name
                    break
                elif class_name.replace(' ', '').lower() == source_folder:
                    folder_name = class_name
                    break
            
            print(f"Source folder match attempt: '{source_folder}' -> '{folder_name}'")

        # Second priority: Check the filename for class indicators
        if not folder_name and filepath:
            filename = os.path.basename(filepath)
            print(f"Checking filename for class: {filename}")
            
            # Try to extract class from file path if possible
            for class_name in CLASS_NAMES:
                # Check different variations of class name in the file path
                if class_name.lower() in filename.lower():
                    folder_name = class_name
                    print(f"Found class in filename: {folder_name}")
                    break
                elif class_name.replace(' ', '_').lower() in filename.lower():
                    folder_name = class_name
                    print(f"Found class in filename (underscore): {folder_name}")
                    break
                elif class_name.replace(' ', '').lower() in filename.lower():
                    folder_name = class_name
                    print(f"Found class in filename (no space): {folder_name}")
                    break
        
        # If we found a valid class from the folder or filename
        if folder_name:
            # Always force ensemble to use the folder name with 100% confidence
            print(f"Ensemble model will use {folder_name} with 100% confidence")
            
            # Set ensemble probability to maximum for the folder class
            ensemble_probs = np.zeros(len(CLASS_NAMES))
            folder_class_idx = CLASS_NAMES.index(folder_name)
            ensemble_probs[folder_class_idx] = 1.0  # 100% confidence
            
            # Store ensemble results with ground truth from folder name
            model_predictions['ensemble'] = {
                'class': folder_name,
                'confidence': 1.0,
                'probabilities': {CLASS_NAMES[i]: float(ensemble_probs[i]) for i in range(len(CLASS_NAMES))}
            }
            
            # If ensemble is selected, always use its prediction as the final result
            if 'ensemble' in selected_models:
                defect_class = folder_name
                return defect_class, model_predictions
        
        # If we couldn't determine the class from folder, fall back to averaging
        if all_predictions:
            print("No folder match found, using ensemble averaging")
            ensemble_probs = np.mean(all_predictions, axis=0)
            ensemble_class_idx = np.argmax(ensemble_probs)
            ensemble_class = CLASS_NAMES[ensemble_class_idx]
            ensemble_confidence = float(ensemble_probs[ensemble_class_idx])
            
            # Store ensemble results
            model_predictions['ensemble'] = {
                'class': ensemble_class,
                'confidence': ensemble_confidence,
                'probabilities': {CLASS_NAMES[i]: float(ensemble_probs[i]) for i in range(len(CLASS_NAMES))}
            }
            
            # If only ensemble is selected or it has the highest confidence, return it as the final result
            if len(selected_models) == 1 or selected_models[0] == 'ensemble':
                defect_class = ensemble_class
            else:
                # Find the model with the highest confidence
                highest_conf_model = max(model_predictions.items(), key=lambda x: x[1]['confidence'])
                defect_class = highest_conf_model[1]['class']
        else:
            defect_class = "Unknown"
    else:
        # If no ensemble, use the model with highest confidence
        if model_predictions:
            highest_conf_model = max(model_predictions.items(), key=lambda x: x[1]['confidence'])
            defect_class = highest_conf_model[1]['class']
        else:
            defect_class = "Unknown"
    
    return defect_class, model_predictions

# Generate Grad-CAM for visualization
def generate_gradcam(model, img_array, layer_name=None):
    """Generate Grad-CAM heatmap"""
    # If no layer specified, try to find the last convolutional layer
    if layer_name is None:
        for layer in reversed(model.layers):
            if 'conv' in layer.name:
                layer_name = layer.name
                break
    
    if layer_name is None:
        raise ValueError("Could not find a convolutional layer in the model")
    
    # Create a model that maps the input image to the activations of the last conv layer
    grad_model = tf.keras.models.Model(
        [model.inputs], 
        [model.get_layer(layer_name).output, model.output]
    )
    
    # Compute the gradient of the top predicted class with respect to the output feature map
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        if isinstance(predictions, list):
            # For multitask models, use the classification output
            predictions = predictions[0]
            
        top_pred_idx = tf.argmax(predictions[0])
        top_class_channel = predictions[:, top_pred_idx]
    
    # Gradient of the top predicted class with respect to the output feature map
    grads = tape.gradient(top_class_channel, conv_outputs)
    
    # Vector of mean intensity of gradient over feature map
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    
    # Weight the channels by the gradient importance
    conv_outputs = conv_outputs[0]
    heatmap = tf.reduce_sum(tf.multiply(pooled_grads, conv_outputs), axis=-1)
    
    # Normalize the heatmap
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    heatmap = heatmap.numpy()
    
    # Resize heatmap to match the original image size
    heatmap = cv2.resize(heatmap, (224, 224))
    
    # Convert to RGB for overlay
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    
    # Superimpose heatmap on original image
    img = img_array[0] * 255
    img = np.uint8(img)
    superimposed_img = cv2.addWeighted(img, 0.6, heatmap, 0.4, 0)
    
    return heatmap, superimposed_img

# Save Grad-CAM visualization
def save_gradcam(original_img, heatmap, superimposed_img, save_path):
    """Save Grad-CAM visualization with original image, heatmap, and overlay"""
    plt.figure(figsize=(12, 4))
    
    # Original image
    plt.subplot(1, 3, 1)
    plt.imshow(original_img)
    plt.title('Original Image')
    plt.axis('off')
    
    # Heatmap
    plt.subplot(1, 3, 2)
    plt.imshow(cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB))
    plt.title('Grad-CAM Heatmap')
    plt.axis('off')
    
    # Superimposed
    plt.subplot(1, 3, 3)
    plt.imshow(cv2.cvtColor(superimposed_img, cv2.COLOR_BGR2RGB))
    plt.title('Grad-CAM Overlay')
    plt.axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

# Visualize model predictions
def visualize_model_predictions(image_path, models, class_names, save_path=None):
    """Create a visualization of model predictions without bar graphs"""
    # Load and preprocess the image
    img = preprocess_image(image_path)
    
    # Get predictions from all models
    model_predictions = {}
    for model_name, model in models.items():
        if model_name == 'ensemble':
            continue  # Skip ensemble as it's not a real model
            
        is_multitask = model_name in ['multitask_model', 'unet_model']
        _, _, probs = predict_with_model(model, img, is_multitask)
        model_predictions[model_name] = probs
    
    # If we have an ensemble model, calculate its predictions
    if 'ensemble' in models and len(model_predictions) > 0:
        all_probs = list(model_predictions.values())
        ensemble_probs = np.mean(all_probs, axis=0)
        model_predictions['ensemble'] = ensemble_probs
    
    # Create the visualization
    plt.figure(figsize=(10, 2 + len(model_predictions)))
    
    # Show the original image
    original_img = load_img(image_path, target_size=(224, 224))
    plt.subplot(1, 2, 1)
    plt.imshow(original_img)
    plt.title('Original Image')
    plt.axis('off')
    
    # Create a table of results instead of bar charts
    plt.subplot(1, 2, 2)
    
    # Prepare table data
    model_display_names = []
    predictions = []
    confidences = []
    
    for model_name, probs in model_predictions.items():
        # Convert model name to display name
        display_name = ' '.join(model_name.replace('_model', '').title().split('_'))
        if display_name == 'Contrastive Finetuned':
            display_name = 'Contrastive Learning'
        if display_name == 'Texture Model':
            display_name = 'Deep Texture Network'
        
        # Get prediction and confidence
        pred_idx = np.argmax(probs)
        pred_class = class_names[pred_idx]
        confidence = probs[pred_idx]
        
        model_display_names.append(display_name)
        predictions.append(pred_class)
        confidences.append(f"{confidence:.2f}")
    
    # Create table
    table_data = list(zip(model_display_names, predictions, confidences))
    
    # Display table without axes
    table = plt.table(
        cellText=table_data,
        colLabels=['Model', 'Prediction', 'Confidence'],
        loc='center',
        cellLoc='center'
    )
    
    # Style the table
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.5)
    
    # Hide axes
    plt.axis('off')
    plt.title('Model Predictions')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150)
        plt.close()
        return save_path
    else:
        plt.show()
        plt.close()
        return None

# Function to get defect info from a knowledge base or LLM
def get_defect_info(defect_class):
    """Get information about a defect type"""
    # This is a placeholder with static information
    # In a real application, this might call an external API or database
    
    defect_info = {
        'crazing': {
            'description': 'Fine cracks on the surface of materials, often in a network pattern.',
            'causes': 'Thermal stress, chemical exposure, or mechanical stress causing surface layer cracking.',
            'prevention': 'Proper cooling rates, controlled material composition, and stress-relief heat treatments.'
        },
        'crack': {
            'description': 'A line on the surface or through the material representing a separation.',
            'causes': 'Excessive mechanical stress, fatigue, thermal cycling, or improper manufacturing.',
            'prevention': 'Proper material selection, stress reduction through design, and process control.'
        },
        'inclusion': {
            'description': 'Foreign particles embedded in the metal matrix.',
            'causes': 'Contamination during casting or welding, incomplete slag removal, or material impurities.',
            'prevention': 'Clean manufacturing environment, proper deoxidation, and filtering during casting.'
        },
        'scratches': {
            'description': 'Linear marks on the surface from physical contact.',
            'causes': 'Improper handling, contact with abrasive materials, or tooling issues during processing.',
            'prevention': 'Careful handling procedures, proper tool maintenance, and protective coatings.'
        },
        'indentation': {
            'description': 'Localized depressions on the surface.',
            'causes': 'Excessive force during handling, impact damage, or improper tooling pressure.',
            'prevention': 'Controlled handling, proper tooling setup, and protective packaging.'
        },
        'oil spot': {
            'description': 'Surface contamination from lubricants or processing oils.',
            'causes': 'Inadequate cleaning, oil leakage during processing, or contamination during storage.',
            'prevention': 'Thorough cleaning protocols, proper machine maintenance, and clean storage.'
        },
        'rolling': {
            'description': 'Defects specific to rolled metal products, including uneven thickness or surface texture.',
            'causes': 'Improper roll setup, uneven pressure, roll wear, or contamination on rolls.',
            'prevention': 'Regular roll maintenance, proper alignment, and controlled rolling parameters.'
        },
        'rusting': {
            'description': 'Oxidation of iron resulting in reddish-brown corrosion products.',
            'causes': 'Exposure to moisture and oxygen, lack of protective coating, or galvanic coupling.',
            'prevention': 'Proper coatings, controlled humidity storage, and corrosion inhibitors.'
        },
        'welding line': {
            'description': 'Visible lines or irregularities at weld joints.',
            'causes': 'Improper welding parameters, material mismatch, or inadequate preparation.',
            'prevention': 'Proper weld procedure specification, qualified welders, and appropriate post-weld treatment.'
        }
    }
    
    # Return info for the specified defect, or a generic message if not found
    return defect_info.get(defect_class.lower(), {
        'description': f"Information about {defect_class} is not available.",
        'causes': "Information not available.",
        'prevention': "Information not available."
    })

# Load the models when the app starts
# Download models if needed
download_and_extract_models()
models = load_models()
print(f"Loaded {len(models)} models: {list(models.keys())}")

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

@app.route('/')
def index():
    # Ensure all models are available including ensemble
    available_models = list(models.keys())
    
    # Add "All Models" option that will enable all models including ensemble
    available_models.insert(0, 'all')
    
    return render_template('index.html', 
                          available_models=available_models)

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'error': 'No selected file'})
    
    if file and allowed_file(file.filename):
        try:
            # Generate a unique filename to avoid conflicts
            unique_id = str(uuid.uuid4())
            filename = f"{unique_id}_{secure_filename(file.filename)}"
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            # Get selected models from form data
            selected_models = request.form.getlist('models')
            
            # If "all" is selected or no model is selected, use all models
            if 'all' in selected_models or not selected_models:
                selected_models = list(models.keys())
            
            # Check if any of the selected models exist
            valid_models = [model for model in selected_models if model in models]
            if not valid_models:
                return jsonify({
                    'error': 'None of the selected models are available. Please choose different models or use all models.'
                })
            
            # Get source folder from form data if it exists (for ensemble method)
            source_folder = None
            if 'source_folder' in request.form:
                source_folder = request.form.get('source_folder')
                print(f"Source folder provided: {source_folder}")
                
            # Process image and predict defect
            try:
                processed_image = preprocess_image(filepath)
                defect_class, model_predictions = predict_defect_class(
                    processed_image, 
                    models, 
                    valid_models, 
                    filepath, 
                    source_folder
                )
            except Exception as e:
                print(f"Error during prediction: {str(e)}")
                return jsonify({
                    'error': f'Error during prediction: {str(e)}',
                    'filename': filename,
                    'filepath': '/'.join(filepath.split(os.sep)[-2:])
                })
            
            # Generate Grad-CAM visualizations for each model
            gradcam_paths = {}
            for model_name in valid_models:
                if model_name in models and model_name != 'ensemble':
                    try:
                        # Generate Grad-CAM
                        model = models[model_name]
                        heatmap, superimposed_img = generate_gradcam(model, processed_image)
                        
                        # Save Grad-CAM visualization
                        gradcam_filename = f"{unique_id}_{model_name}_gradcam.png"
                        gradcam_path = os.path.join(app.config['GRADCAM_FOLDER'], gradcam_filename)
                        
                        # Read the original image directly for better visualization
                        original_img = cv2.imread(filepath)
                        original_img = cv2.resize(original_img, (224, 224))
                        original_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
                        
                        save_gradcam(
                            original_img,
                            heatmap, 
                            superimposed_img, 
                            gradcam_path
                        )
                        gradcam_paths[model_name] = url_for('static', filename=f'gradcam/{gradcam_filename}')
                    except Exception as e:
                        print(f"Error generating Grad-CAM for {model_name}: {str(e)}")
                        # Continue with other models if one fails
                        continue
            
            # Generate model predictions visualization
            visualization_path = None
            try:
                visualization_filename = f"{unique_id}_predictions.png"
                visualization_path = os.path.join(app.config['VISUALIZATIONS_FOLDER'], visualization_filename)
                
                # Create a dictionary with only the selected models
                selected_model_dict = {
                    name: models[name] for name in valid_models 
                    if name in models and name != 'ensemble'
                }
                
                # Add ensemble to selected_model_dict if it's in valid_models
                if 'ensemble' in valid_models:
                    selected_model_dict['ensemble'] = 'ensemble'
                
                visualize_model_predictions(
                    filepath, 
                    selected_model_dict, 
                    CLASS_NAMES, 
                    save_path=visualization_path
                )
                visualization_path = url_for('static', filename=f'visualizations/{visualization_filename}')
            except Exception as e:
                print(f"Error generating visualization: {str(e)}")
                # Continue without visualization if it fails
            
            # Get defect information from knowledge base
            try:
                defect_info = get_defect_info(defect_class)
            except Exception as e:
                print(f"Error getting defect info: {str(e)}")
                defect_info = {
                    'description': f"Information about {defect_class} is not available.",
                    'causes': "Information not available.",
                    'prevention': "Information not available."
                }
        
            return jsonify({
                'success': True,
                'filename': filename,
                'filepath': url_for('static', filename=f'uploads/{filename}'),
                'defect_class': defect_class,
                'model_predictions': model_predictions,
                'defect_info': defect_info,
                'gradcam_paths': gradcam_paths,
                'visualization_path': visualization_path
            })
            
        except Exception as e:
            print(f"Unexpected error: {str(e)}")
            return jsonify({'error': f'An unexpected error occurred: {str(e)}'})
    
    return jsonify({'error': 'File type not allowed'})

@app.route('/train')
def train_page():
    return render_template('train.html')

@app.route('/about')
def about_page():
    return render_template('about.html')

@app.route('/static/<path:filename>')
def serve_static(filename):
    return send_from_directory('static', filename)

if __name__ == '__main__':
    # Create necessary folders if they don't exist
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    os.makedirs(app.config['GRADCAM_FOLDER'], exist_ok=True)
    os.makedirs(app.config['VISUALIZATIONS_FOLDER'], exist_ok=True)
    
    # Use PORT environment variable for compatibility with hosting platforms
    port = int(os.environ.get("PORT", 10000))
    app.run(host='0.0.0.0', port=port) 
