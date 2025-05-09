import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Conv2D, Input, Multiply, Add
from tensorflow.keras.layers import MaxPooling2D, Flatten, Dropout, BatchNormalization, Reshape, Permute
from tensorflow.keras.layers import Activation, AveragePooling2D, Lambda, Concatenate, Layer
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, TensorBoard
from tensorflow.keras.losses import categorical_crossentropy
from sklearn.model_selection import train_test_split
import cv2
import matplotlib.pyplot as plt
from PIL import Image
import glob
from tqdm import tqdm
import pandas as pd

# Constants
CLASS_NAMES = [
    'crazing', 'crack', 'inclusion', 'scratches', 'indentation', 
    'oil spot', 'rolling', 'rusting', 'welding line'
]

# Create necessary directories
os.makedirs('models/advanced', exist_ok=True)
os.makedirs('results', exist_ok=True)
os.makedirs('logs', exist_ok=True)

# ==== METHOD 5: SPECIALIZED LOSS FUNCTIONS ====

def arcface_loss(y_true, y_pred, margin=0.5, scale=30):
    """
    ArcFace loss for better feature space separation between classes
    """
    cos_m = tf.math.cos(margin)
    sin_m = tf.math.sin(margin)
    th = tf.math.cos(np.pi - margin)
    mm = tf.math.sin(np.pi - margin) * margin

    # Extract the classes
    classes = tf.cast(tf.argmax(y_true, axis=1), dtype=tf.int32)
    
    # Extract features
    cosine = y_pred  # assuming cosine similarity is calculated
    
    # For numerical stability
    sine = tf.math.sqrt(1.0 - tf.math.pow(cosine, 2))
    
    # Get margin cosine(theta + m)
    cos_theta_m = cosine * cos_m - sine * sin_m
    
    # If cos(theta) > cos(pi - m), use cosine(theta + m)
    # Otherwise, use cosine(theta) - margin
    cond_v = cosine - th
    cond = tf.cast(tf.nn.relu(cond_v), dtype=tf.bool)
    keep_val = cosine - mm
    cos_theta_m = tf.where(cond, cos_theta_m, keep_val)
    
    # Use one-hot encoding to apply the margin to the correct class
    mask = tf.one_hot(classes, depth=y_pred.shape[-1])
    logits = tf.where(tf.cast(mask, dtype=tf.bool), cos_theta_m, cosine)
    
    # Scale the logits
    logits = logits * scale
    
    # Apply cross-entropy loss
    losses = tf.nn.softmax_cross_entropy_with_logits(labels=y_true, logits=logits)
    return tf.reduce_mean(losses)

def center_loss(y_true, y_pred, num_classes=9, feature_dim=512, alpha=0.5):
    """
    Center loss to minimize intra-class variations
    """
    # Initialize centers (global variable)
    if not hasattr(center_loss, 'centers'):
        center_loss.centers = tf.Variable(
            tf.zeros((num_classes, feature_dim)), trainable=False)
    
    # Get features and labels
    features = y_pred
    labels = tf.argmax(y_true, axis=1)
    labels = tf.cast(labels, tf.int32)
    
    # Get centers for the current batch
    centers_batch = tf.gather(center_loss.centers, labels)
    
    # Calculate the loss
    diff = centers_batch - features
    center_loss_val = tf.reduce_mean(tf.square(diff))
    
    # Update centers
    unique_label, unique_idx, unique_count = tf.unique_with_counts(labels)
    appear_times = tf.gather(unique_count, unique_idx)
    appear_times = tf.reshape(appear_times, [-1, 1])
    
    diff = diff / tf.cast(appear_times, tf.float32)
    diff = alpha * diff
    
    centers_update_op = tf.scatter_sub(
        center_loss.centers, labels, diff)
    
    # Ensure update happens
    with tf.control_dependencies([centers_update_op]):
        return center_loss_val

def triplet_loss(y_true, y_pred, alpha=0.2):
    """
    Triplet loss for better feature embedding
    """
    # Get batch size
    batch_size = tf.shape(y_pred)[0]
    
    # Reshape embeddings to rank 2
    embeddings = tf.reshape(y_pred, [batch_size, -1])
    
    # Get anchor, positive, and negative samples
    # In this implementation, we assume that each triplet is
    # arranged sequentially in the batch (anchor, positive, negative)
    anchors = embeddings[0::3]
    positives = embeddings[1::3]
    negatives = embeddings[2::3]
    
    # Calculate distances
    pos_dist = tf.reduce_sum(tf.square(anchors - positives), axis=1)
    neg_dist = tf.reduce_sum(tf.square(anchors - negatives), axis=1)
    
    # Calculate triplet loss
    basic_loss = pos_dist - neg_dist + alpha
    loss = tf.maximum(basic_loss, 0.0)
    
    return tf.reduce_mean(loss)

def create_model_with_specialized_loss(input_shape=(224, 224, 3), num_classes=9, loss_type='arcface'):
    """
    Create a model with specialized loss function
    
    Args:
        input_shape: Input shape for the model
        num_classes: Number of classes to predict
        loss_type: Type of loss function ('arcface', 'center', 'triplet')
        
    Returns:
        model: Compiled model
    """
    # Base model (ResNet50)
    base_model = ResNet50(include_top=False, weights='imagenet', input_shape=input_shape)
    
    # Add custom layers
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(512, activation='relu')(x)
    
    if loss_type == 'arcface':
        # For ArcFace, we need to use cosine similarity
        x = BatchNormalization()(x)
        x = Lambda(lambda x: K.l2_normalize(x, axis=1))(x)
        outputs = Dense(num_classes, activation='linear', use_bias=False)(x)
        
        # Custom loss
        model = Model(inputs=base_model.input, outputs=outputs)
        model.compile(
            optimizer=Adam(learning_rate=0.0001),
            loss=arcface_loss,
            metrics=['accuracy']
        )
    
    elif loss_type == 'center':
        # For Center Loss, combine with softmax loss
        feature_layer = x
        outputs = Dense(num_classes, activation='softmax')(feature_layer)
        
        model = Model(inputs=base_model.input, outputs=[outputs, feature_layer])
        
        # Custom loss combining softmax and center loss
        def combined_loss(y_true, y_pred):
            # Split the predictions
            y_pred_softmax, features = y_pred
            
            # Calculate softmax loss
            softmax_loss_val = categorical_crossentropy(y_true, y_pred_softmax)
            
            # Calculate center loss
            center_loss_val = center_loss(y_true, features, num_classes=num_classes)
            
            # Combine losses
            return softmax_loss_val + 0.1 * center_loss_val
        
        model.compile(
            optimizer=Adam(learning_rate=0.0001),
            loss=combined_loss,
            metrics=['accuracy']
        )
    
    elif loss_type == 'triplet':
        # For Triplet Loss, remove classification layer
        outputs = Dense(128, activation=None)(x)  # Embedding space
        outputs = Lambda(lambda x: K.l2_normalize(x, axis=1))(outputs)
        
        model = Model(inputs=base_model.input, outputs=outputs)
        model.compile(
            optimizer=Adam(learning_rate=0.0001),
            loss=triplet_loss
        )
    
    else:
        # Standard softmax
        outputs = Dense(num_classes, activation='softmax')(x)
        model = Model(inputs=base_model.input, outputs=outputs)
        model.compile(
            optimizer=Adam(learning_rate=0.0001),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
    
    return model

# ==== METHOD 6: MULTI-TASK LEARNING ====

def create_multitask_model(input_shape=(224, 224, 3), num_classes=9):
    """
    Create a multi-task learning model for classification and segmentation
    
    Args:
        input_shape: Input shape for the model
        num_classes: Number of classes to predict
        
    Returns:
        model: Compiled model
    """
    # Create an encoder (based on ResNet50)
    inputs = Input(shape=input_shape)
    base_model = ResNet50(include_top=False, weights='imagenet', input_tensor=inputs)
    
    # Add classification branch
    x_class = GlobalAveragePooling2D()(base_model.output)
    x_class = Dense(512, activation='relu')(x_class)
    x_class = BatchNormalization()(x_class)
    x_class = Dropout(0.5)(x_class)
    classification = Dense(num_classes, activation='softmax', name='classification')(x_class)
    
    # Add segmentation branch (simplified)
    x_seg = Conv2D(256, (3, 3), padding='same', activation='relu')(base_model.output)
    x_seg = Conv2D(256, (3, 3), padding='same', activation='relu')(x_seg)
    x_seg = Conv2D(128, (3, 3), padding='same', activation='relu')(x_seg)
    x_seg = Conv2D(64, (3, 3), padding='same', activation='relu')(x_seg)
    
    # Upsampling to get a segmentation map
    x_seg = tf.keras.layers.UpSampling2D(size=(8, 8))(x_seg)
    segmentation = Conv2D(1, (1, 1), activation='sigmoid', name='segmentation')(x_seg)
    
    # Binary defect detection branch
    x_detect = GlobalAveragePooling2D()(base_model.output)
    x_detect = Dense(128, activation='relu')(x_detect)
    x_detect = BatchNormalization()(x_detect)
    detection = Dense(1, activation='sigmoid', name='detection')(x_detect)
    
    # Create model with multiple outputs
    model = Model(inputs=inputs, outputs=[classification, segmentation, detection])
    
    # Compile with different loss functions for each task
    model.compile(
        optimizer=Adam(learning_rate=0.0001),
        loss={
            'classification': 'categorical_crossentropy',
            'segmentation': 'binary_crossentropy',
            'detection': 'binary_crossentropy'
        },
        loss_weights={
            'classification': 1.0,
            'segmentation': 0.5,
            'detection': 0.2
        },
        metrics={
            'classification': 'accuracy',
            'segmentation': tf.keras.metrics.MeanIoU(num_classes=2),
            'detection': 'accuracy'
        }
    )
    
    return model

# ==== METHOD 7: CONTRASTIVE/SELF-SUPERVISED LEARNING ====

class SimCLR(Model):
    """
    SimCLR model for contrastive learning
    """
    def __init__(self, encoder, projection_dim=128, temperature=0.1):
        super(SimCLR, self).__init__()
        self.encoder = encoder
        self.projection_head = Sequential([
            Dense(512, activation='relu'),
            Dense(projection_dim)
        ])
        self.temperature = temperature
        
    def compile(self, optimizer, **kwargs):
        super(SimCLR, self).compile(**kwargs)
        self.optimizer = optimizer
        
    def train_step(self, data):
        # Unpack the data
        view1, view2 = data
        
        with tf.GradientTape() as tape:
            # Get the representations
            z1 = self.projection_head(self.encoder(view1))
            z2 = self.projection_head(self.encoder(view2))
            
            # Normalize the representations
            z1 = tf.math.l2_normalize(z1, axis=1)
            z2 = tf.math.l2_normalize(z2, axis=1)
            
            # Get the similarity matrix
            similarity_matrix = tf.matmul(z1, z2, transpose_b=True) / self.temperature
            
            # Get the positive samples (diagonal elements)
            batch_size = tf.shape(z1)[0]
            pos_sim = tf.linalg.diag_part(similarity_matrix)
            
            # Get the negative samples (off-diagonal elements)
            neg_sim = similarity_matrix - tf.eye(batch_size) * similarity_matrix
            
            # The contrastive loss
            loss = -tf.reduce_mean(
                tf.math.log(
                    tf.exp(pos_sim) / (tf.exp(pos_sim) + tf.reduce_sum(tf.exp(neg_sim), axis=1))
                )
            )
        
        # Compute gradients and update weights
        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        
        return {"loss": loss}

def create_contrastive_model(input_shape=(224, 224, 3)):
    """
    Create a contrastive learning model (SimCLR)
    
    Args:
        input_shape: Input shape for the model
        
    Returns:
        model: Compiled SimCLR model
    """
    # Create encoder (ResNet50)
    encoder = ResNet50(include_top=False, weights='imagenet', input_shape=input_shape)
    encoder.trainable = True
    
    # Add pooling layer
    encoder_model = Sequential([
        encoder,
        GlobalAveragePooling2D(),
    ])
    
    # Create SimCLR model
    simclr_model = SimCLR(encoder_model, projection_dim=128, temperature=0.1)
    simclr_model.compile(optimizer=Adam(learning_rate=0.0001))
    
    return simclr_model

def fine_tune_from_contrastive(pretrained_encoder, input_shape=(224, 224, 3), num_classes=9):
    """
    Fine-tune a model from pre-trained contrastive encoder
    
    Args:
        pretrained_encoder: Pre-trained encoder from contrastive learning
        input_shape: Input shape for the model
        num_classes: Number of classes to predict
        
    Returns:
        model: Compiled model for fine-tuning
    """
    # Freeze encoder for initial training
    pretrained_encoder.trainable = False
    
    # Create a new model with the pre-trained encoder
    inputs = Input(shape=input_shape)
    x = pretrained_encoder(inputs)
    x = GlobalAveragePooling2D()(x)
    x = Dense(512, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)
    outputs = Dense(num_classes, activation='softmax')(x)
    
    model = Model(inputs=inputs, outputs=outputs)
    
    # Compile the model
    model.compile(
        optimizer=Adam(learning_rate=0.0001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

# ==== METHOD 8: DEFECT-AWARE DATA AUGMENTATION ====

def get_defect_regions(image, threshold=0.1):
    """
    Detect potential defect regions using edge detection and thresholding
    
    Args:
        image: Input image
        threshold: Threshold for defect detection
        
    Returns:
        regions: List of (x, y, w, h) bounding boxes
    """
    # Convert to grayscale
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = image
    
    # Apply gaussian blur
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Canny edge detection
    edges = cv2.Canny(blur, 50, 150)
    
    # Dilate edges to connect nearby edges
    kernel = np.ones((5, 5), np.uint8)
    dilated = cv2.dilate(edges, kernel, iterations=1)
    
    # Find contours
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Filter contours by area and get bounding boxes
    regions = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > threshold * image.shape[0] * image.shape[1]:
            x, y, w, h = cv2.boundingRect(contour)
            regions.append((x, y, w, h))
    
    return regions

def defect_aware_augmentation(image, regions):
    """
    Apply augmentation focused on defect regions
    
    Args:
        image: Input image
        regions: List of (x, y, w, h) bounding boxes
        
    Returns:
        augmented_image: Augmented image
    """
    # If no regions found, apply regular augmentation
    if not regions:
        return regular_augmentation(image)
    
    # Randomly select a region
    region = random.choice(regions)
    x, y, w, h = region
    
    # Ensure minimum size
    w = max(w, 32)
    h = max(h, 32)
    
    # Ensure region is within image bounds
    x = min(x, image.shape[1] - w)
    y = min(y, image.shape[0] - h)
    
    # Extract the region
    region_img = image[y:y+h, x:x+w].copy()
    
    # Apply augmentation to the region
    aug = A.Compose([
        A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.2, rotate_limit=45, p=0.7),
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.7),
        A.GaussNoise(var_limit=(10, 50), p=0.5),
    ])
    
    region_img = aug(image=region_img)["image"]
    
    # Insert augmented region back into image
    result = image.copy()
    result[y:y+h, x:x+w] = region_img
    
    return result

def regular_augmentation(image):
    """
    Apply regular augmentation to the entire image
    
    Args:
        image: Input image
        
    Returns:
        augmented_image: Augmented image
    """
    aug = A.Compose([
        A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.2, rotate_limit=30, p=0.7),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.7),
    ])
    
    return aug(image=image)["image"]

# ==== METHOD 9: DOMAIN-SPECIFIC ARCHITECTURES ====

def create_unet_model(input_shape=(224, 224, 3), num_classes=9):
    """
    Create a U-Net model for combined segmentation and classification
    
    Args:
        input_shape: Input shape for the model
        num_classes: Number of classes to predict
        
    Returns:
        model: Compiled U-Net model
    """
    # Encoder
    inputs = Input(shape=input_shape)
    
    # Encoder path
    conv1 = Conv2D(64, 3, activation='relu', padding='same')(inputs)
    conv1 = Conv2D(64, 3, activation='relu', padding='same')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    
    conv2 = Conv2D(128, 3, activation='relu', padding='same')(pool1)
    conv2 = Conv2D(128, 3, activation='relu', padding='same')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    
    conv3 = Conv2D(256, 3, activation='relu', padding='same')(pool2)
    conv3 = Conv2D(256, 3, activation='relu', padding='same')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    
    conv4 = Conv2D(512, 3, activation='relu', padding='same')(pool3)
    conv4 = Conv2D(512, 3, activation='relu', padding='same')(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)
    
    # Bridge
    conv5 = Conv2D(1024, 3, activation='relu', padding='same')(pool4)
    conv5 = Conv2D(1024, 3, activation='relu', padding='same')(conv5)
    
    # Decoder path
    up6 = Conv2D(512, 2, activation='relu', padding='same')(tf.keras.layers.UpSampling2D(size=(2, 2))(conv5))
    merge6 = Concatenate(axis=3)([conv4, up6])
    conv6 = Conv2D(512, 3, activation='relu', padding='same')(merge6)
    conv6 = Conv2D(512, 3, activation='relu', padding='same')(conv6)
    
    up7 = Conv2D(256, 2, activation='relu', padding='same')(tf.keras.layers.UpSampling2D(size=(2, 2))(conv6))
    merge7 = Concatenate(axis=3)([conv3, up7])
    conv7 = Conv2D(256, 3, activation='relu', padding='same')(merge7)
    conv7 = Conv2D(256, 3, activation='relu', padding='same')(conv7)
    
    up8 = Conv2D(128, 2, activation='relu', padding='same')(tf.keras.layers.UpSampling2D(size=(2, 2))(conv7))
    merge8 = Concatenate(axis=3)([conv2, up8])
    conv8 = Conv2D(128, 3, activation='relu', padding='same')(merge8)
    conv8 = Conv2D(128, 3, activation='relu', padding='same')(conv8)
    
    up9 = Conv2D(64, 2, activation='relu', padding='same')(tf.keras.layers.UpSampling2D(size=(2, 2))(conv8))
    merge9 = Concatenate(axis=3)([conv1, up9])
    conv9 = Conv2D(64, 3, activation='relu', padding='same')(merge9)
    conv9 = Conv2D(64, 3, activation='relu', padding='same')(conv9)
    
    # Segmentation output
    segmentation = Conv2D(1, 1, activation='sigmoid', name='segmentation')(conv9)
    
    # Classification branch from bottleneck features
    class_features = GlobalAveragePooling2D()(conv5)
    class_features = Dense(512, activation='relu')(class_features)
    class_features = BatchNormalization()(class_features)
    class_features = Dropout(0.5)(class_features)
    classification = Dense(num_classes, activation='softmax', name='classification')(class_features)
    
    # Create model with multiple outputs
    model = Model(inputs=inputs, outputs=[segmentation, classification])
    
    # Compile with different loss functions
    model.compile(
        optimizer=Adam(learning_rate=0.0001),
        loss={
            'segmentation': 'binary_crossentropy',
            'classification': 'categorical_crossentropy'
        },
        loss_weights={
            'segmentation': 0.5,
            'classification': 1.0
        },
        metrics={
            'segmentation': 'accuracy',
            'classification': 'accuracy'
        }
    )
    
    return model

def create_deep_texture_model(input_shape=(224, 224, 3), num_classes=9):
    """
    Create a DeepTEN (Deep Texture Encoding Network) model for texture classification
    
    Args:
        input_shape: Input shape for the model
        num_classes: Number of classes to predict
        
    Returns:
        model: Compiled DeepTEN model
    """
    # Use ResNet50 as the base model
    base_model = ResNet50(include_top=False, weights='imagenet', input_shape=input_shape)
    
    # Add texture encoding layer
    x = base_model.output
    
    # Global orderless representation
    # 1. Global average pooling
    x1 = GlobalAveragePooling2D()(x)
    
    # 2. Global max pooling
    x2 = Lambda(lambda x: tf.reduce_max(x, axis=[1, 2]))(x)
    
    # 3. Spatial pyramid pooling (simplified)
    # Level 1: 1x1
    x3_1 = GlobalAveragePooling2D()(x)
    
    # Level 2: 2x2
    x3_2 = Lambda(lambda x: tf.image.resize(x, [2, 2]))(x)
    x3_2 = Reshape((4, -1))(x3_2)
    x3_2 = Lambda(lambda x: tf.reduce_mean(x, axis=2))(x3_2)
    x3_2 = Flatten()(x3_2)
    
    # Level 3: 4x4
    x3_3 = Lambda(lambda x: tf.image.resize(x, [4, 4]))(x)
    x3_3 = Reshape((16, -1))(x3_3)
    x3_3 = Lambda(lambda x: tf.reduce_mean(x, axis=2))(x3_3)
    x3_3 = Flatten()(x3_3)
    
    # Concatenate all features
    x = Concatenate()([x1, x2, x3_1, x3_2, x3_3])
    
    # FC layers
    x = Dense(1024, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)
    x = Dense(512, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)
    
    # Classification layer
    outputs = Dense(num_classes, activation='softmax')(x)
    
    # Create and compile model
    model = Model(inputs=base_model.input, outputs=outputs)
    model.compile(
        optimizer=Adam(learning_rate=0.0001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

# ==== EXPERIMENT TRACKING ====

def create_result_entry(method_name, accuracy, f1_score, parameters, architecture, epochs, loss):
    """
    Create a dictionary entry for the results CSV
    
    Args:
        method_name: Name of the method
        accuracy: Test accuracy
        f1_score: Macro F1 score
        parameters: Dictionary of parameters
        architecture: Description of architecture
        epochs: Number of epochs trained
        loss: Final validation loss
        
    Returns:
        entry: Dictionary with result data
    """
    return {
        'Method': method_name,
        'Accuracy': accuracy,
        'F1_Score': f1_score,
        'Parameters': str(parameters),
        'Architecture': architecture,
        'Epochs': epochs,
        'Val_Loss': loss
    }

def save_results_to_csv(results, filename='results/methods_comparison.csv'):
    """
    Save experiment results to CSV
    
    Args:
        results: List of result dictionaries
        filename: Output CSV filename
    """
    df = pd.DataFrame(results)
    df.to_csv(filename, index=False)
    print(f"Results saved to {filename}")

# Main function will be implemented in the comprehensive evaluation script 