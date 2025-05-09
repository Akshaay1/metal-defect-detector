import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os
import cv2
import matplotlib.pyplot as plt
import albumentations as A
from sklearn.model_selection import train_test_split
import random
import shutil
from tqdm import tqdm

def preprocess_image(image_path, target_size=(224, 224), grayscale=False):
    """
    Preprocess an image for model inference with enhanced contrast
    
    Args:
        image_path: Path to the image file
        target_size: Target size for the image (default: 224x224)
        grayscale: Whether to convert to grayscale (True) or keep RGB (False)
        
    Returns:
        preprocessed_image: Preprocessed image ready for model inference
    """
    # Load image with OpenCV
    try:
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Could not load image at {image_path}")
        
        # Determine if the image is grayscale
        is_grayscale = len(img.shape) == 2 or (len(img.shape) == 3 and img.shape[2] == 1)
        
        if grayscale or is_grayscale:
            # Convert to grayscale if not already
            if not is_grayscale:
                img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            else:
                img_gray = img if len(img.shape) == 2 else img[:,:,0]
            
            # Apply CLAHE for better contrast in grayscale images
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            img_enhanced = clahe.apply(img_gray)
            
            # Convert back to RGB for model compatibility
            img = cv2.cvtColor(img_enhanced, cv2.COLOR_GRAY2RGB)
        else:
            # For color images
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # Split channels and apply CLAHE to luminance
            img_lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
            l, a, b = cv2.split(img_lab)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            l_enhanced = clahe.apply(l)
            img_lab = cv2.merge((l_enhanced, a, b))
            img = cv2.cvtColor(img_lab, cv2.COLOR_LAB2RGB)
        
        # Resize image
        img = cv2.resize(img, target_size)
        img_array = img
    
    except Exception as e:
        print(f"Error preprocessing image: {e}")
        # Create a blank image if loading fails
        img_array = np.zeros((target_size[0], target_size[1], 3), dtype=np.uint8)
    
    # Normalize to [0, 1] range
    img_array = img_array.astype(np.float32) / 255.0
    
    # Apply model-specific preprocessing
    img_array = tf.keras.applications.imagenet_utils.preprocess_input(
        img_array, mode='tf'
    )
    
    return img_array

def apply_basic_augmentation(image_path, output_dir, num_augmentations=5, target_size=(224, 224)):
    """
    Apply basic data augmentation to an image and save the augmented images
    
    Args:
        image_path: Path to the original image
        output_dir: Directory to save the augmented images
        num_augmentations: Number of augmented images to generate
        target_size: Target size for the images
        
    Returns:
        augmented_paths: List of paths to the augmented images
    """
    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Load the image
    img = Image.open(image_path).convert('RGB')
    img = img.resize(target_size)
    img_array = np.array(img)
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    
    # Create an augmentation generator
    datagen = ImageDataGenerator(
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        vertical_flip=True,
        brightness_range=[0.8, 1.2],
        fill_mode='reflect'
    )
    
    # Generate and save augmented images
    augmented_paths = []
    base_filename = os.path.splitext(os.path.basename(image_path))[0]
    
    i = 0
    for batch in datagen.flow(img_array, batch_size=1):
        augmented_img = batch[0].astype(np.uint8)
        
        # Save the augmented image
        output_path = os.path.join(output_dir, f"{base_filename}_aug_{i}.jpg")
        Image.fromarray(augmented_img).save(output_path)
        augmented_paths.append(output_path)
        
        i += 1
        if i >= num_augmentations:
            break
    
    return augmented_paths

def apply_advanced_augmentation(image_path, output_dir, num_augmentations=10, target_size=(224, 224)):
    """
    Apply advanced data augmentation to a metal defect image using Albumentations
    
    Args:
        image_path: Path to the original image
        output_dir: Directory to save the augmented images
        num_augmentations: Number of augmented images to generate
        target_size: Target size for the images
        
    Returns:
        augmented_paths: List of paths to the augmented images
    """
    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Load the image
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, target_size)
    
    # Define augmentation pipeline specifically for metal defect images
    augmentation = A.Compose([
        # Spatial augmentations
        A.RandomRotate90(p=0.5),
        A.Flip(p=0.5),
        A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.2, rotate_limit=45, p=0.7),
        A.ElasticTransform(alpha=1, sigma=50, alpha_affine=50, p=0.3),
        
        # Color and brightness augmentations
        A.OneOf([
            A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3),
            A.RandomGamma(gamma_limit=(80, 120)),
            A.CLAHE(clip_limit=4.0, tile_grid_size=(8, 8), p=0.5),
        ], p=0.8),
        
        # Noise augmentations
        A.OneOf([
            A.GaussNoise(var_limit=(10, 50)),
            A.MultiplicativeNoise(multiplier=(0.9, 1.1)),
            A.ISONoise(color_shift=(0.01, 0.05), intensity=(0.1, 0.5)),
        ], p=0.5),
        
        # Blur and sharpness
        A.OneOf([
            A.GaussianBlur(blur_limit=(3, 7)),
            A.MotionBlur(blur_limit=(3, 7)),
            A.Sharpen(alpha=(0.2, 0.5), lightness=(0.5, 1.0)),
        ], p=0.3),
        
        # Texture and edge enhancements for defect detection
        A.OneOf([
            A.Emboss(alpha=(0.2, 0.5), strength=(0.5, 1.0)),
            A.ChannelShuffle(p=0.2),
            A.ToGray(p=0.2),  # Sometimes convert to grayscale
        ], p=0.3),
        
        # Grid distortions and elastic transforms
        A.OneOf([
            A.GridDistortion(num_steps=5, distort_limit=0.3),
            A.OpticalDistortion(distort_limit=0.2, shift_limit=0.15),
        ], p=0.3),
    ])
    
    # Generate and save augmented images
    augmented_paths = []
    base_filename = os.path.splitext(os.path.basename(image_path))[0]
    
    for i in range(num_augmentations):
        # Apply augmentation
        augmented = augmentation(image=img)
        augmented_img = augmented['image']
        
        # Save the augmented image
        output_path = os.path.join(output_dir, f"{base_filename}_adv_aug_{i}.jpg")
        cv2.imwrite(output_path, cv2.cvtColor(augmented_img, cv2.COLOR_RGB2BGR))
        augmented_paths.append(output_path)
    
    return augmented_paths

def create_augmented_dataset(src_dir, dest_dir, augmentations_per_image=20, train_size=0.8, val_size=0.1, test_size=0.1):
    """
    Create an augmented dataset from source images
    
    Args:
        src_dir: Source directory containing original images in class subdirectories
        dest_dir: Destination directory for the augmented dataset
        augmentations_per_image: Number of augmentations to create per image
        train_size: Proportion of data for training
        val_size: Proportion of data for validation
        test_size: Proportion of data for testing
        
    Returns:
        dataset_stats: Dictionary with dataset statistics
    """
    # Create destination directories
    train_dir = os.path.join(dest_dir, 'train')
    val_dir = os.path.join(dest_dir, 'val')
    test_dir = os.path.join(dest_dir, 'test')
    
    os.makedirs(dest_dir, exist_ok=True)
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)
    
    # Get all classes (subdirectories in source directory)
    classes = [d for d in os.listdir(src_dir) if os.path.isdir(os.path.join(src_dir, d))]
    
    dataset_stats = {
        'original_images': 0,
        'augmented_images': 0,
        'class_distribution': {}
    }
    
    for class_name in classes:
        print(f"Processing class: {class_name}")
        
        # Create class directories in train/val/test
        os.makedirs(os.path.join(train_dir, class_name), exist_ok=True)
        os.makedirs(os.path.join(val_dir, class_name), exist_ok=True)
        os.makedirs(os.path.join(test_dir, class_name), exist_ok=True)
        
        # Get all images in this class
        class_dir = os.path.join(src_dir, class_name)
        image_files = [f for f in os.listdir(class_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        
        # Update stats
        dataset_stats['original_images'] += len(image_files)
        dataset_stats['class_distribution'][class_name] = {
            'original': len(image_files),
            'augmented': 0,
            'train': 0,
            'val': 0,
            'test': 0
        }
        
        # Split images into train, val, test
        train_val_files, test_files = train_test_split(image_files, test_size=test_size, random_state=42)
        train_files, val_files = train_test_split(train_val_files, test_size=val_size/(train_size+val_size), random_state=42)
        
        # Copy original train images and create augmentations
        for img_file in tqdm(train_files, desc=f"{class_name} - Train"):
            src_path = os.path.join(class_dir, img_file)
            dest_path = os.path.join(train_dir, class_name, img_file)
            shutil.copy2(src_path, dest_path)
            
            # Apply augmentations
            aug_paths = apply_advanced_augmentation(
                src_path, 
                os.path.join(train_dir, class_name),
                num_augmentations=augmentations_per_image
            )
            
            dataset_stats['augmented_images'] += len(aug_paths)
            dataset_stats['class_distribution'][class_name]['augmented'] += len(aug_paths)
            dataset_stats['class_distribution'][class_name]['train'] += 1 + len(aug_paths)
        
        # Copy original validation images (no augmentation for validation)
        for img_file in tqdm(val_files, desc=f"{class_name} - Validation"):
            src_path = os.path.join(class_dir, img_file)
            dest_path = os.path.join(val_dir, class_name, img_file)
            shutil.copy2(src_path, dest_path)
            dataset_stats['class_distribution'][class_name]['val'] += 1
        
        # Copy original test images (no augmentation for test)
        for img_file in tqdm(test_files, desc=f"{class_name} - Test"):
            src_path = os.path.join(class_dir, img_file)
            dest_path = os.path.join(test_dir, class_name, img_file)
            shutil.copy2(src_path, dest_path)
            dataset_stats['class_distribution'][class_name]['test'] += 1
    
    return dataset_stats

def load_and_preprocess_dataset(dataset_dir, target_size=(224, 224), batch_size=32, augmentation=True):
    """
    Load and preprocess a dataset of images for training
    
    Args:
        dataset_dir: Directory containing the dataset (with subdirectories for each class)
        target_size: Target size for the images
        batch_size: Batch size for training
        augmentation: Whether to apply augmentation to the training data
        
    Returns:
        train_generator: Generator for training data
        validation_generator: Generator for validation data
        test_generator: Generator for test data (if available)
    """
    # Define augmentation parameters for training
    if augmentation:
        train_datagen = ImageDataGenerator(
            rescale=1./255,
            rotation_range=40,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
            vertical_flip=True,
            brightness_range=[0.8, 1.2],
            fill_mode='reflect'
        )
    else:
        train_datagen = ImageDataGenerator(rescale=1./255)
    
    # No augmentation for validation/test
    val_datagen = ImageDataGenerator(rescale=1./255)
    
    # Check if we have a train/val/test split structure
    if os.path.exists(os.path.join(dataset_dir, 'train')):
        # We have a train/val/test split
        train_dir = os.path.join(dataset_dir, 'train')
        val_dir = os.path.join(dataset_dir, 'val')
        test_dir = os.path.join(dataset_dir, 'test')
        
        # Create generators
        train_generator = train_datagen.flow_from_directory(
            train_dir,
            target_size=target_size,
            batch_size=batch_size,
            class_mode='categorical',
            shuffle=True
        )
        
        validation_generator = val_datagen.flow_from_directory(
            val_dir,
            target_size=target_size,
            batch_size=batch_size,
            class_mode='categorical',
            shuffle=False
        )
        
        # Check if test directory exists and has data
        if os.path.exists(test_dir) and len(os.listdir(test_dir)) > 0:
            test_generator = val_datagen.flow_from_directory(
                test_dir,
                target_size=target_size,
                batch_size=batch_size,
                class_mode='categorical',
                shuffle=False
            )
            return train_generator, validation_generator, test_generator
        
        return train_generator, validation_generator
    
    else:
        # Use the same directory with train/val split
        train_datagen.validation_split = 0.2
        
        # Create training generator
        train_generator = train_datagen.flow_from_directory(
            dataset_dir,
            target_size=target_size,
            batch_size=batch_size,
            class_mode='categorical',
            subset='training',
            shuffle=True
        )
        
        # Create validation generator
        validation_generator = train_datagen.flow_from_directory(
            dataset_dir,
            target_size=target_size,
            batch_size=batch_size,
            class_mode='categorical',
            subset='validation',
            shuffle=False
        )
        
        return train_generator, validation_generator

def visualize_augmentations(image_path, num_samples=5, save_path=None):
    """
    Visualize different augmentations of an image
    
    Args:
        image_path: Path to the image
        num_samples: Number of augmentation samples to show
        save_path: Path to save the visualization (if None, displays instead)
    """
    # Load the image
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Define augmentation pipeline
    augmentation = A.Compose([
        # Spatial augmentations
        A.RandomRotate90(p=0.5),
        A.Flip(p=0.5),
        A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.2, rotate_limit=45, p=0.7),
        
        # Color and brightness augmentations
        A.OneOf([
            A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3),
            A.RandomGamma(gamma_limit=(80, 120)),
            A.CLAHE(clip_limit=4.0, tile_grid_size=(8, 8), p=0.5),
        ], p=0.8),
        
        # Noise augmentations
        A.OneOf([
            A.GaussNoise(var_limit=(10, 50)),
            A.MultiplicativeNoise(multiplier=(0.9, 1.1)),
        ], p=0.5),
    ])
    
    # Create a figure to display the augmentations
    plt.figure(figsize=(15, 10))
    
    # Show original image
    plt.subplot(2, 3, 1)
    plt.title('Original Image')
    plt.imshow(img)
    plt.axis('off')
    
    # Generate and show augmented images
    for i in range(num_samples):
        augmented = augmentation(image=img)
        augmented_img = augmented['image']
        
        plt.subplot(2, 3, i + 2)
        plt.title(f'Augmentation {i+1}')
        plt.imshow(augmented_img)
        plt.axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()

def visualize_model_predictions(image_path, models, class_names, target_size=(224, 224), save_path=None):
    """
    Visualize model predictions for an image - UPDATED to remove bar graph

    Args:
        image_path: Path to the image file
        models: Dictionary of models to use for prediction
        class_names: List of class names
        target_size: Target size for the image
        save_path: Path to save the visualization (if None, displays the plot)
        
    Returns:
        predicted_class: The predicted defect class
    """
    # Preprocess the image
    preprocessed_img = preprocess_image(image_path, target_size)
    
    # Create a batch of one image
    if len(preprocessed_img.shape) == 3:
        preprocessed_img = np.expand_dims(preprocessed_img, axis=0)
    
    # Get predictions from all models
    all_predictions = []
    model_confidences = {}
    
    for model_name, model in models.items():
        # Make prediction
        try:
            prediction = model.predict(preprocessed_img, verbose=0)[0]
            
            # Get the predicted class and confidence
            pred_class_idx = np.argmax(prediction)
            pred_class = class_names[pred_class_idx].upper() if pred_class_idx < len(class_names) else "UNKNOWN"
            confidence = prediction[pred_class_idx]
            
            all_predictions.append(pred_class)
            model_confidences[model_name] = (pred_class, confidence)
        except Exception as e:
            print(f"Error making prediction with {model_name}: {e}")
            model_confidences[model_name] = ("ERROR", 0.0)
    
    # Determine final prediction (majority vote)
    if all_predictions:
        from collections import Counter
        counter = Counter(all_predictions)
        final_prediction = counter.most_common(1)[0][0]
    else:
        final_prediction = "UNKNOWN"
    
    # Read the original image for visualization
    try:
        original_img = cv2.imread(image_path)
        original_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
        original_img = cv2.resize(original_img, target_size)
    except Exception as e:
        print(f"Error reading image for visualization: {e}")
        # Create a placeholder if image cannot be read
        original_img = np.ones((target_size[0], target_size[1], 3), dtype=np.uint8) * 200
    
    # Create visualization with just the image and text annotations
    plt.figure(figsize=(12, 8))
    
    # Set a high DPI for better resolution
    plt.rcParams['figure.dpi'] = 150
    
    # Display the original image
    plt.imshow(original_img)
    plt.title(f"Final Prediction: {final_prediction}", fontsize=16, fontweight='bold')
    plt.axis('off')
    
    # Add model predictions as text annotations
    text_y_position = 0.05
    for i, (model_name, (pred_class, confidence)) in enumerate(model_confidences.items()):
        # Color code the text - green for matching the final prediction, red otherwise
        text_color = 'green' if pred_class == final_prediction else 'red'
        
        # Add text annotation
        plt.text(
            0.02, text_y_position, 
            f"{model_name}: {pred_class} ({confidence:.2f})",
            transform=plt.gca().transAxes,
            fontsize=14,
            verticalalignment='bottom',
            bbox=dict(facecolor='white', alpha=0.7, edgecolor=text_color, pad=10),
            color=text_color
        )
        text_y_position += 0.08
    
    plt.tight_layout()
    
    # Save or display the figure
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()
    else:
        plt.show()
    
    return final_prediction

def visualize_dataset_distribution(dataset_dir, save_path=None):
    """
    Visualize the distribution of classes in a dataset
    
    Args:
        dataset_dir: Directory containing the dataset (with subdirectories for each class)
        save_path: Path to save the visualization (if None, displays instead)
    """
    # Check if we have a train/val/test split structure
    if os.path.exists(os.path.join(dataset_dir, 'train')):
        # We have a train/val/test split
        split_structure = True
        splits = ['train', 'val', 'test']
        class_counts = {split: {} for split in splits}
        
        for split in splits:
            split_dir = os.path.join(dataset_dir, split)
            if os.path.exists(split_dir):
                for class_name in os.listdir(split_dir):
                    class_dir = os.path.join(split_dir, class_name)
                    if os.path.isdir(class_dir):
                        img_count = len([f for f in os.listdir(class_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
                        class_counts[split][class_name] = img_count
    else:
        # Single directory structure
        split_structure = False
        class_counts = {}
        
        for class_name in os.listdir(dataset_dir):
            class_dir = os.path.join(dataset_dir, class_name)
            if os.path.isdir(class_dir):
                img_count = len([f for f in os.listdir(class_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
                class_counts[class_name] = img_count
    
    # Create plots
    if split_structure:
        # Plot for each split
        fig, axes = plt.subplots(len(splits), 1, figsize=(12, 5 * len(splits)))
        
        for i, split in enumerate(splits):
            if os.path.exists(os.path.join(dataset_dir, split)):
                # Sort classes by counts
                classes = sorted(class_counts[split].keys(), key=lambda x: class_counts[split][x], reverse=True)
                counts = [class_counts[split][c] for c in classes]
                
                axes[i].bar(classes, counts)
                axes[i].set_title(f'{split.capitalize()} Set Distribution')
                axes[i].set_ylabel('Number of Images')
                axes[i].set_xlabel('Class')
                axes[i].tick_params(axis='x', rotation=45)
                axes[i].grid(True, linestyle='--', alpha=0.7)
    else:
        # Single plot
        plt.figure(figsize=(12, 6))
        
        # Sort classes by counts
        classes = sorted(class_counts.keys(), key=lambda x: class_counts[x], reverse=True)
        counts = [class_counts[c] for c in classes]
        
        plt.bar(classes, counts)
        plt.title('Dataset Distribution')
        plt.ylabel('Number of Images')
        plt.xlabel('Class')
        plt.tick_params(axis='x', rotation=45)
        plt.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show() 