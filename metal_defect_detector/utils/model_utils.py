import tensorflow as tf
import numpy as np
import cv2
from tensorflow.keras.applications import ResNet50, EfficientNetB0, DenseNet121, MobileNetV2
from tensorflow.keras.models import Model, Sequential, load_model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Conv2D, MaxPooling2D, Flatten, Dropout, Input, BatchNormalization
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from collections import Counter
import os
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import ssl
import urllib.request
import tempfile
import shutil
from PIL import Image

# Temporarily disable SSL verification for model downloads
ssl._create_default_https_context = ssl._create_unverified_context

# Define class names based on the dataset - update to match the actual classes
CLASS_NAMES = [
    'crazing', 'crack', 'inclusion', 'scratches', 'indentation', 
    'oil spot', 'rolling', 'rusting', 'welding line'
]

def download_weights_file(url, filename):
    """
    Download weights file with SSL verification disabled
    
    Args:
        url: URL to download from
        filename: Filename to save to
        
    Returns:
        path: Path to the downloaded file
    """
    # Create a temporary directory
    temp_dir = tempfile.mkdtemp()
    temp_file = os.path.join(temp_dir, filename)
    
    # Download the file with SSL verification disabled
    ssl_context = ssl._create_unverified_context()
    with urllib.request.urlopen(url, context=ssl_context) as response:
        with open(temp_file, 'wb') as out_file:
            shutil.copyfileobj(response, out_file)
    
    # Move to final location
    keras_dir = os.path.expanduser('~/.keras')
    os.makedirs(os.path.join(keras_dir, 'models'), exist_ok=True)
    target_path = os.path.join(keras_dir, 'models', filename)
    
    if not os.path.exists(target_path):
        shutil.move(temp_file, target_path)
    else:
        os.remove(temp_file)
    
    # Cleanup temp directory
    shutil.rmtree(temp_dir, ignore_errors=True)
    
    return target_path

def create_simple_model(input_shape=(224, 224, 3), num_classes=len(CLASS_NAMES)):
    """
    Create a simple CNN model for initial testing
    """
    model = Sequential([
        Input(shape=input_shape),
        Conv2D(32, kernel_size=(3, 3), activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),
        Conv2D(64, kernel_size=(3, 3), activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),
        Conv2D(128, kernel_size=(3, 3), activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),
        Flatten(),
        Dense(256, activation='relu'),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])
    
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def create_resnet50_model(input_shape=(224, 224, 3), num_classes=len(CLASS_NAMES), weights='imagenet'):
    """
    Create a ResNet50 model for metal defect classification
    
    Args:
        input_shape: Input shape for the model
        num_classes: Number of classes to predict
        weights: Pre-trained weights to use
        
    Returns:
        model: Compiled ResNet50 model
    """
    try:
        # Try to load with existing mechanism
        base_model = ResNet50(weights=weights, include_top=False, input_shape=input_shape)
    except Exception as e:
        print(f"Error loading ResNet50 weights: {e}")
        print("Using simple model instead")
        return create_simple_model(input_shape, num_classes)
    
    # Freeze the base model layers
    for layer in base_model.layers:
        layer.trainable = False
    
    # Add custom classification head
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(512, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)
    x = Dense(256, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)
    predictions = Dense(num_classes, activation='softmax')(x)
    
    # Create the model
    model = Model(inputs=base_model.input, outputs=predictions)
    
    # Compile the model
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def create_efficientnet_model(input_shape=(224, 224, 3), num_classes=len(CLASS_NAMES), weights='imagenet'):
    """
    Create an EfficientNetB0 model for metal defect classification
    
    Args:
        input_shape: Input shape for the model
        num_classes: Number of classes to predict
        weights: Pre-trained weights to use
        
    Returns:
        model: Compiled EfficientNetB0 model
    """
    try:
        # Load the base model
        base_model = EfficientNetB0(weights=weights, include_top=False, input_shape=input_shape)
    except Exception as e:
        print(f"Error loading EfficientNetB0 weights: {e}")
        print("Using simple model instead")
        return create_simple_model(input_shape, num_classes)
    
    # Freeze the base model layers
    for layer in base_model.layers:
        layer.trainable = False
    
    # Add custom classification head
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(512, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)
    x = Dense(256, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)
    predictions = Dense(num_classes, activation='softmax')(x)
    
    # Create the model
    model = Model(inputs=base_model.input, outputs=predictions)
    
    # Compile the model
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def create_densenet_model(input_shape=(224, 224, 3), num_classes=len(CLASS_NAMES), weights='imagenet'):
    """
    Create a DenseNet121 model for metal defect classification
    
    Args:
        input_shape: Input shape for the model
        num_classes: Number of classes to predict
        weights: Pre-trained weights to use
        
    Returns:
        model: Compiled DenseNet121 model
    """
    try:
        # Load the base model
        base_model = DenseNet121(weights=weights, include_top=False, input_shape=input_shape)
    except Exception as e:
        print(f"Error loading DenseNet121 weights: {e}")
        print("Using simple model instead")
        return create_simple_model(input_shape, num_classes)
    
    # Freeze the base model layers
    for layer in base_model.layers:
        layer.trainable = False
    
    # Add custom classification head
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(512, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)
    x = Dense(256, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)
    predictions = Dense(num_classes, activation='softmax')(x)
    
    # Create the model
    model = Model(inputs=base_model.input, outputs=predictions)
    
    # Compile the model
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def create_mobilenet_model(input_shape=(224, 224, 3), num_classes=len(CLASS_NAMES), weights='imagenet'):
    """
    Create a MobileNetV2 model for metal defect classification
    
    Args:
        input_shape: Input shape for the model
        num_classes: Number of classes to predict
        weights: Pre-trained weights to use
        
    Returns:
        model: Compiled MobileNetV2 model
    """
    try:
        # Load the base model
        base_model = MobileNetV2(weights=weights, include_top=False, input_shape=input_shape)
    except Exception as e:
        print(f"Error loading MobileNetV2 weights: {e}")
        print("Using simple model instead")
        return create_simple_model(input_shape, num_classes)
    
    # Freeze the base model layers
    for layer in base_model.layers:
        layer.trainable = False
    
    # Add custom classification head
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(512, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)
    x = Dense(256, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)
    predictions = Dense(num_classes, activation='softmax')(x)
    
    # Create the model
    model = Model(inputs=base_model.input, outputs=predictions)
    
    # Compile the model
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def train_model(model, train_generator, validation_generator, epochs=50, save_dir='models'):
    """
    Train a model and save the best weights
    
    Args:
        model: The model to train
        train_generator: Generator for training data
        validation_generator: Generator for validation data
        epochs: Number of epochs to train for
        save_dir: Directory to save model weights
        
    Returns:
        history: Training history
        model: Trained model
    """
    # Create save directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)
    
    # Define model callbacks
    model_name = model.name if hasattr(model, 'name') else 'metal_defect_model'
    checkpoint = ModelCheckpoint(
        os.path.join(save_dir, f'{model_name}_best.h5'),
        monitor='val_accuracy',
        save_best_only=True,
        mode='max',
        verbose=1
    )
    
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True,
        verbose=1
    )
    
    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.2,
        patience=5,
        min_lr=1e-6,
        verbose=1
    )
    
    # Train the model
    history = model.fit(
        train_generator,
        validation_data=validation_generator,
        epochs=epochs,
        callbacks=[checkpoint, early_stopping, reduce_lr]
    )
    
    return history, model

def fine_tune_model(model, train_generator, validation_generator, learning_rate=1e-5, epochs=20, save_dir='models'):
    """
    Fine-tune a pre-trained model
    
    Args:
        model: Pre-trained model to fine-tune
        train_generator: Generator for training data
        validation_generator: Generator for validation data
        learning_rate: Learning rate for fine-tuning
        epochs: Number of epochs to fine-tune
        save_dir: Directory to save model weights
        
    Returns:
        history: Training history
        model: Fine-tuned model
    """
    # Unfreeze all layers
    for layer in model.layers:
        layer.trainable = True
    
    # Recompile the model with a lower learning rate
    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Define model callbacks
    model_name = model.name if hasattr(model, 'name') else 'metal_defect_model'
    checkpoint = ModelCheckpoint(
        os.path.join(save_dir, f'{model_name}_finetuned.h5'),
        monitor='val_accuracy',
        save_best_only=True,
        mode='max',
        verbose=1
    )
    
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=7,
        restore_best_weights=True,
        verbose=1
    )
    
    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.2,
        patience=3,
        min_lr=1e-7,
        verbose=1
    )
    
    # Train the model
    history = model.fit(
        train_generator,
        validation_data=validation_generator,
        epochs=epochs,
        callbacks=[checkpoint, early_stopping, reduce_lr]
    )
    
    return history, model

def load_models():
    """Load all metal defect classification models"""
    models_dir = 'models'
    os.makedirs(models_dir, exist_ok=True)
    
    # Check if trained models exist
    if os.path.exists(os.path.join(models_dir, 'resnet50_finetuned.h5')):
        models = {}
        model_files = {
            'resnet50': 'resnet50_finetuned.h5',
            'efficientnet': 'efficientnet_finetuned.h5',
            'densenet': 'densenet_finetuned.h5',
            'mobilenet': 'mobilenet_finetuned.h5'
        }
        
        # Load available trained models
        for model_name, model_file in model_files.items():
            model_path = os.path.join(models_dir, model_file)
            if os.path.exists(model_path):
                try:
                    models[model_name] = load_model(model_path)
                    print(f"Loaded {model_name} from {model_path}")
                except:
                    print(f"Failed to load {model_name} from {model_path}")
        
        if models:
            return models
    
    # If no trained models exist, create new ones
    models = {
        'resnet50': create_resnet50_model(),
        'efficientnet': create_efficientnet_model(),
        'densenet': create_densenet_model(),
        'mobilenet': create_mobilenet_model()
    }
    
    return models

def predict_defect_class(image, models, selected_models=None):
    """
    Predict the defect class using selected models
    
    Args:
        image: Preprocessed image as a numpy array
        models: Dictionary of loaded models
        selected_models: List of model names to use for prediction
        
    Returns:
        defect_class: Predicted defect class
        model_predictions: Dictionary with prediction details for each model
    """
    if selected_models is None:
        selected_models = list(models.keys())
    
    # Ensure image has batch dimension
    if len(image.shape) == 3:
        image = np.expand_dims(image, axis=0)
    
    model_predictions = {}
    all_predictions = []
    
    for model_name in selected_models:
        if model_name in models:
            # Make predictions
            predictions = models[model_name].predict(image)[0]
            
            pred_class_idx = np.argmax(predictions)
            pred_class = CLASS_NAMES[pred_class_idx]
            confidence = float(predictions[pred_class_idx])
            
            model_predictions[model_name] = {
                'class': pred_class,
                'confidence': confidence,
                'probabilities': {cls: float(prob) for cls, prob in zip(CLASS_NAMES, predictions)}
            }
            
            all_predictions.append(pred_class)
    
    # Use majority voting to determine the final class
    if all_predictions:
        counter = Counter(all_predictions)
        defect_class = counter.most_common(1)[0][0]
    else:
        defect_class = "Unknown"
    
    return defect_class, model_predictions

def generate_gradcam(model, preprocessed_img, layer_name=None):
    """
    Generate Grad-CAM visualization for a model's prediction
    
    Args:
        model: Trained model
        preprocessed_img: Preprocessed image (should be a single image, not a batch)
        layer_name: Name of the layer to use for Grad-CAM (if None, uses the last conv layer)
        
    Returns:
        heatmap: Grad-CAM heatmap
        superimposed_img: Original image with heatmap overlay
    """
    try:
        # Make sure image has batch dimension
        if len(preprocessed_img.shape) == 3:
            preprocessed_img = np.expand_dims(preprocessed_img, axis=0)
        
        # Get the predicted class
        try:
            predictions = model.predict(preprocessed_img, verbose=0)  # Disable progress bar
            predicted_class = np.argmax(predictions[0])
        except Exception as e:
            print(f"Error during model prediction: {e}")
            # Return empty visualizations with original image
            img_display = preprocessed_img[0].copy()
            if np.max(img_display) <= 1.0:
                img_display = (img_display * 255).astype(np.uint8)
            else:
                img_display = img_display.astype(np.uint8)
            
            if len(img_display.shape) == 2:
                img_display = cv2.cvtColor(img_display, cv2.COLOR_GRAY2RGB)
                
            return np.zeros_like(img_display), img_display
        
        # Find the last convolutional layer if layer_name not provided
        if layer_name is None:
            try:
                # First try to find a Conv2D layer
                for layer in reversed(model.layers):
                    if isinstance(layer, tf.keras.layers.Conv2D):
                        layer_name = layer.name
                        print(f"Using Conv2D layer: {layer_name}")
                        break
                
                # If not found, look for other convolutional layers
                if layer_name is None:
                    for layer in reversed(model.layers):
                        if any(conv_type in layer.name for conv_type in ['conv', 'Conv', 'pool', 'Pool']):
                            layer_name = layer.name
                            print(f"Using layer: {layer_name}")
                            break
                
                # If still not found, try to find a layer with 4D output (batch, height, width, channels)
                if layer_name is None:
                    for layer in reversed(model.layers):
                        if hasattr(layer, 'output_shape') and len(layer.output_shape) == 4:
                            layer_name = layer.name
                            print(f"Using 4D output layer: {layer_name}")
                            break
            except Exception as e:
                print(f"Error finding convolutional layer: {e}")
                # Return empty visualizations with original image
                img_display = preprocessed_img[0].copy()
                if np.max(img_display) <= 1.0:
                    img_display = (img_display * 255).astype(np.uint8)
                else:
                    img_display = img_display.astype(np.uint8)
                
                if len(img_display.shape) == 2:
                    img_display = cv2.cvtColor(img_display, cv2.COLOR_GRAY2RGB)
                    
                return np.zeros_like(img_display), img_display
        
        if layer_name is None:
            print("Could not find a suitable layer for Grad-CAM")
            # Return empty visualizations with original image
            img_display = preprocessed_img[0].copy()
            if np.max(img_display) <= 1.0:
                img_display = (img_display * 255).astype(np.uint8)
            else:
                img_display = img_display.astype(np.uint8)
            
            if len(img_display.shape) == 2:
                img_display = cv2.cvtColor(img_display, cv2.COLOR_GRAY2RGB)
                
            return np.zeros_like(img_display), img_display
        
        # Create a model that maps the input image to the activations of the target layer
        try:
            grad_model = tf.keras.models.Model(
                [model.inputs], 
                [model.get_layer(layer_name).output, model.output]
            )
        except Exception as e:
            print(f"Error creating grad model: {e}")
            # Return empty visualizations with original image
            img_display = preprocessed_img[0].copy()
            if np.max(img_display) <= 1.0:
                img_display = (img_display * 255).astype(np.uint8)
            else:
                img_display = img_display.astype(np.uint8)
            
            if len(img_display.shape) == 2:
                img_display = cv2.cvtColor(img_display, cv2.COLOR_GRAY2RGB)
                
            return np.zeros_like(img_display), img_display
        
        # Compute gradient of top predicted class with respect to target layer
        try:
            with tf.GradientTape() as tape:
                conv_output, predictions = grad_model(preprocessed_img)
                loss = predictions[:, predicted_class]
            
            # Extract filters and gradients
            output = conv_output[0]
            grads = tape.gradient(loss, conv_output)[0]
            
            # Check if gradients contain NaN values
            if tf.math.reduce_any(tf.math.is_nan(grads)):
                print("Warning: NaN gradients detected")
                # Return empty visualizations with original image
                img_display = preprocessed_img[0].copy()
                if np.max(img_display) <= 1.0:
                    img_display = (img_display * 255).astype(np.uint8)
                else:
                    img_display = img_display.astype(np.uint8)
                
                if len(img_display.shape) == 2:
                    img_display = cv2.cvtColor(img_display, cv2.COLOR_GRAY2RGB)
                    
                return np.zeros_like(img_display), img_display
        except Exception as e:
            print(f"Error computing gradients: {e}")
            # Return empty visualizations with original image
            img_display = preprocessed_img[0].copy()
            if np.max(img_display) <= 1.0:
                img_display = (img_display * 255).astype(np.uint8)
            else:
                img_display = img_display.astype(np.uint8)
            
            if len(img_display.shape) == 2:
                img_display = cv2.cvtColor(img_display, cv2.COLOR_GRAY2RGB)
                
            return np.zeros_like(img_display), img_display
        
        # Global average pooling
        weights = tf.reduce_mean(grads, axis=(0, 1))
        
        # Build weighted combination of filters
        cam = tf.reduce_sum(tf.multiply(weights, output), axis=-1)
        
        # Enhanced normalization with stronger minimum cutoff for better contrast
        heatmap = np.maximum(cam, 0.1 * np.max(cam)) / (np.max(cam) + 1e-10)
        
        # Resize heatmap to match image dimensions
        img = preprocessed_img[0].copy()
        # Check if heatmap is a TensorFlow tensor or numpy array
        if hasattr(heatmap, 'numpy'):
            heatmap_numpy = heatmap.numpy()
        else:
            heatmap_numpy = heatmap
            
        heatmap_resized = cv2.resize(heatmap_numpy, (img.shape[1], img.shape[0]))
        
        # Convert to 8-bit with enhanced intensity
        heatmap_8bit = (heatmap_resized * 255).astype(np.uint8)
        
        # Apply multi-stage contrast enhancement
        # 1. Histogram equalization
        heatmap_eq = cv2.equalizeHist(heatmap_8bit)
        
        # 2. Apply CLAHE for better local contrast with stronger clip limit
        clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8, 8))
        heatmap_clahe = clahe.apply(heatmap_eq)
        
        # 3. Boost low-mid range values to enhance visibility
        _, heatmap_thresh = cv2.threshold(heatmap_clahe, 127, 255, cv2.THRESH_BINARY)
        heatmap_boost = cv2.addWeighted(heatmap_clahe, 0.7, heatmap_thresh, 0.3, 0)
        
        # Apply a more visible colormap with higher saturation
        heatmap_colored = cv2.applyColorMap(heatmap_boost, cv2.COLORMAP_JET)
        
        # Increase color saturation of the heatmap
        heatmap_hsv = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2HSV)
        heatmap_hsv[:,:,1] = np.clip(heatmap_hsv[:,:,1] * 2.0, 0, 255).astype(np.uint8)  # Double the saturation
        heatmap_hsv[:,:,2] = np.clip(heatmap_hsv[:,:,2] * 1.2, 0, 255).astype(np.uint8)  # Increase value (brightness)
        heatmap_colored = cv2.cvtColor(heatmap_hsv, cv2.COLOR_HSV2BGR)
        
        # Enhance edges in the heatmap
        kernel_sharpen = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
        heatmap_colored = cv2.filter2D(heatmap_colored, -1, kernel_sharpen)
        
        # Convert preprocessed image back to 0-255 range for visualization
        if np.max(img) <= 1.0:
            img_display = (img * 255).astype(np.uint8)
        else:
            img_display = img.astype(np.uint8)
        
        # Ensure img_display has 3 channels
        if len(img_display.shape) == 2:
            img_display = cv2.cvtColor(img_display, cv2.COLOR_GRAY2RGB)
        
        # Use much higher alpha value for the heatmap (0.8 or even 0.9)
        alpha = 0.9  # Very high alpha for heatmap makes it much more visible
        beta = 1.0 - alpha
        
        # Create the superimposed image
        superimposed_img = cv2.addWeighted(img_display, beta, heatmap_colored, alpha, 0)
        
        # Additional post-processing to enhance visibility
        # 1. Add a slight sharpness
        kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
        superimposed_img = cv2.filter2D(superimposed_img, -1, kernel)
        
        # 2. Optional: Add borders around high activation areas
        # First convert to grayscale for edge detection
        gray_heatmap = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray_heatmap, 100, 200)
        edges_colored = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
        # Add edges as white lines
        superimposed_img = cv2.addWeighted(superimposed_img, 0.9, edges_colored, 0.1, 0)
        
        return heatmap_colored, superimposed_img
    
    except Exception as e:
        print(f"Unexpected error in generate_gradcam: {e}")
        # Create a fallback image in case of errors
        try:
            img_display = preprocessed_img[0].copy() if len(preprocessed_img.shape) > 3 else preprocessed_img.copy()
            if np.max(img_display) <= 1.0:
                img_display = (img_display * 255).astype(np.uint8)
            else:
                img_display = img_display.astype(np.uint8)
            
            # Ensure we have a 3-channel image
            if len(img_display.shape) == 2:
                img_display = cv2.cvtColor(img_display, cv2.COLOR_GRAY2RGB)
            elif len(img_display.shape) > 3:
                img_display = img_display[0]
            
            # Create a blank heatmap of the same size
            heatmap_colored = np.zeros_like(img_display)
            
            # Return blank heatmap and original image
            return heatmap_colored, img_display
        except:
            # If all else fails, return empty arrays
            return np.zeros((224, 224, 3)), np.zeros((224, 224, 3))

def save_gradcam(image, heatmap, superimposed_img, output_path):
    """
    Save Grad-CAM visualization as a side-by-side comparison
    
    Args:
        image: Original image
        heatmap: Grad-CAM heatmap
        superimposed_img: Original image with heatmap overlay
        output_path: Path to save the visualization
    """
    try:
        # Create the directory if it doesn't exist
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Ensure images are in the right format
        if np.max(image) <= 1.0:
            image = (image * 255).astype(np.uint8)
        
        # Convert to BGR for OpenCV operations
        if image.shape[2] == 3:
            # Check if image needs conversion from RGB to BGR
            image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        else:
            # Handle grayscale
            image_bgr = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        
        # Convert heatmap to BGRa
        if heatmap.shape[2] == 3:
            heatmap_bgr = cv2.cvtColor(heatmap, cv2.COLOR_RGB2BGR)
        else:
            heatmap_bgr = heatmap.copy()
        
        # Convert superimposed to BGR
        if superimposed_img.shape[2] == 3:
            superimposed_bgr = cv2.cvtColor(superimposed_img, cv2.COLOR_RGB2BGR)
        else:
            superimposed_bgr = superimposed_img.copy()
        
        # Create a side-by-side visualization
        h, w = image.shape[:2]
        canvas = np.zeros((h, w*2, 3), dtype=np.uint8)
        
        # Add original image on the left
        canvas[:, :w] = image_bgr
        
        # Add superimposed image on the right
        canvas[:, w:] = superimposed_bgr
        
        # Add labels
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(canvas, "Original", (10, 30), font, 1, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(canvas, "Defect Areas", (w+10, 30), font, 1, (255, 255, 255), 2, cv2.LINE_AA)
        
        # Add dividing line
        cv2.line(canvas, (w, 0), (w, h), (255, 255, 255), 2)
        
        # Save the image
        cv2.imwrite(output_path, canvas)
        
    except Exception as e:
        print(f"Error saving Grad-CAM visualization: {e}")
        # Fallback to simple approach
        try:
            plt.figure(figsize=(12, 6))
            
            plt.subplot(1, 2, 1)
            plt.imshow(image)
            plt.title("Original")
            plt.axis('off')
            
            plt.subplot(1, 2, 2)
            plt.imshow(superimposed_img)
            plt.title("Defect Areas")
            plt.axis('off')
            
            plt.tight_layout()
            plt.savefig(output_path, dpi=150)
            plt.close()
        except Exception as inner_e:
            print(f"Fallback visualization also failed: {inner_e}")
            # Last resort - save just the original image
            try:
                cv2.imwrite(output_path, cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_RGB2BGR))
            except:
                pass 