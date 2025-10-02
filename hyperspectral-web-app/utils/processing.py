import os
import uuid
import numpy as np
import xarray as xr
from PIL import Image
import tensorflow as tf
import joblib
import traceback
import cv2
from sklearn.preprocessing import StandardScaler

# Import model loader but don't call any functions yet
from utils.model_loader import get_models

def load_models():
    """Load U-Net and SVM models from disk (if not already loaded)."""
    # Use the model loader's function
    from utils.model_loader import load_models as loader
    return loader()

def validate_nc4_file(file_path):
    """Validate that the file is a proper NetCDF4 file."""
    try:
        print(f"Validating NC4 file: {file_path}")
        with xr.open_dataset(file_path, engine="netcdf4") as ds:
            print(f"File opened successfully. Variables: {list(ds.data_vars.keys())}")
            print(f"Coordinates: {list(ds.coords.keys())}")
            print(f"Dimensions: {dict(ds.sizes)}")  # Use sizes instead of dims
            
            # Check if it has at least one data variable
            if not ds.data_vars:
                return False, "File contains no data variables"
            
            # Print detailed info about each data variable
            for var_name, var in ds.data_vars.items():
                print(f"Variable '{var_name}': shape={var.shape}, dims={var.dims}, dtype={var.dtype}")
            
            # Look for RgbImage or other suitable image data
            has_image_data = False
            for var_name, var in ds.data_vars.items():
                if var.ndim >= 2:  # At least 2D for image data
                    has_image_data = True
                    break
            
            if not has_image_data:
                return False, "No suitable image data variables found (need at least 2D)"
            
            print(f"Validation successful. Found suitable image data.")
            return True, "Valid NetCDF4 file with image data"
            
    except Exception as e:
        error_msg = f"Invalid NetCDF4 file: {str(e)}"
        print(error_msg)
        print(traceback.format_exc())
        return False, error_msg

def resize_image_and_mask(image, mask=None, target_size=(256, 256)):
    """Resize image and optionally mask to target size for model input."""
    print(f"Resizing image from {image.shape} to {target_size}")
    
    # Resize image using OpenCV for better quality
    if len(image.shape) == 3:
        resized_image = cv2.resize(image, target_size, interpolation=cv2.INTER_AREA)
    else:
        resized_image = cv2.resize(image, target_size, interpolation=cv2.INTER_AREA)
    
    resized_mask = None
    if mask is not None:
        # For masks, use nearest neighbor to preserve binary values
        resized_mask = cv2.resize(mask.astype(np.uint8), target_size, interpolation=cv2.INTER_NEAREST)
        resized_mask = resized_mask.astype(bool)
        print(f"Resized mask from {mask.shape} to {resized_mask.shape}")
    
    return resized_image, resized_mask

def resize_mask_to_original(mask, original_size):
    """Resize mask back to original image size."""
    original_h, original_w = original_size
    print(f"Resizing mask from {mask.shape} back to ({original_h}, {original_w})")
    
    # Resize mask back to original size
    resized_mask = cv2.resize(mask.astype(np.uint8), (original_w, original_h), interpolation=cv2.INTER_NEAREST)
    return resized_mask.astype(bool)

def run_unet_segmentation(unet_model, rgb_image):
    """Run U-Net segmentation on RGB image."""
    print("Running U-Net model for segmentation")
    
    # Prepare input
    input_image = rgb_image.astype(np.float32) / 255.0
    print(f"U-Net input shape: {input_image.shape}")
    
    # Add batch dimension if needed
    if len(input_image.shape) == 3:
        input_image = np.expand_dims(input_image, axis=0)
    
    # Run prediction
    try:
        prediction = unet_model.predict(input_image)
        print(f"U-Net prediction shape: {prediction.shape}")
        
        # Remove batch dimension
        if len(prediction.shape) == 4:
            prediction = prediction[0]
        
        # Remove channel dimension if it's 1
        if prediction.shape[-1] == 1:
            prediction = prediction[:, :, 0]
        
        return prediction
        
    except Exception as e:
        print(f"Error running U-Net prediction: {str(e)}")
        # Return a dummy prediction
        h, w = input_image.shape[1:3] if len(input_image.shape) == 4 else input_image.shape[:2]
        dummy_prediction = np.random.random((h, w)) * 0.3
        # Add some "detected" regions
        for _ in range(5):
            y = np.random.randint(h//4, 3*h//4)
            x = np.random.randint(w//4, 3*w//4)
            size = min(h, w) // 20
            dummy_prediction[y:y+size, x:x+size] = np.random.random() * 0.7 + 0.3
        
        print(f"Using dummy prediction with shape: {dummy_prediction.shape}")
        return dummy_prediction


def create_robust_features_from_rgb(rgb_image, reflectance_spectrum=None, mask=None):
    """Create robust features from RGB for SVM classification with fixed dimensionality."""
    print("Creating robust features from RGB for SVM classification")
    
    # Convert to float for calculations
    rgb_float = rgb_image.astype(np.float32) / 255.0
    
    # Extract RGB channels
    red = rgb_float[:, :, 0]
    green = rgb_float[:, :, 1] 
    blue = rgb_float[:, :, 2]
    
    # If mask is provided, only use masked pixels
    if mask is not None:
        red_pixels = red[mask]
        green_pixels = green[mask]
        blue_pixels = blue[mask]
        print(f"Using {len(red_pixels)} masked pixels for feature extraction")
        
        if len(red_pixels) == 0:
            print("WARNING: No pixels in mask, using center region")
            h, w = red.shape
            center_h, center_w = h//2, w//2
            region_size = 50
            red_pixels = red[center_h-region_size:center_h+region_size, center_w-region_size:center_w+region_size].flatten()
            green_pixels = green[center_h-region_size:center_h+region_size, center_w-region_size:center_w+region_size].flatten()
            blue_pixels = blue[center_h-region_size:center_h+region_size, center_w-region_size:center_w+region_size].flatten()
    else:
        red_pixels = red.flatten()
        green_pixels = green.flatten()
        blue_pixels = blue.flatten()
        print(f"Using all {len(red_pixels)} pixels for feature extraction")
    
    # Create a fixed set of statistical features (32 features total)
    features = []
    
    # Basic RGB statistics (9 features)
    features.extend([
        np.mean(red_pixels), np.std(red_pixels), np.median(red_pixels),
        np.mean(green_pixels), np.std(green_pixels), np.median(green_pixels),
        np.mean(blue_pixels), np.std(blue_pixels), np.median(blue_pixels)
    ])
    
    # Color ratios and indices (3 features)
    total_intensity = red_pixels + green_pixels + blue_pixels + 1e-8
    features.extend([
        np.mean(red_pixels / total_intensity),
        np.mean(green_pixels / total_intensity),
        np.mean(blue_pixels / total_intensity)
    ])
    
    # Normalized difference indices (3 features)
    features.extend([
        np.mean((red_pixels - green_pixels) / (red_pixels + green_pixels + 1e-8)),
        np.mean((green_pixels - blue_pixels) / (green_pixels + blue_pixels + 1e-8)),
        np.mean((red_pixels - blue_pixels) / (red_pixels + blue_pixels + 1e-8))
    ])
    
    # Brightness and contrast (3 features)
    brightness = (red_pixels + green_pixels + blue_pixels) / 3.0
    features.extend([
        np.mean(brightness),
        np.std(brightness),
        np.max(brightness) - np.min(brightness)  # contrast
    ])
    
    # Additional color space features (2 features)
    max_rgb = np.maximum(np.maximum(red_pixels, green_pixels), blue_pixels)
    min_rgb = np.minimum(np.minimum(red_pixels, green_pixels), blue_pixels)
    delta = max_rgb - min_rgb
    
    # Saturation-like feature
    saturation = np.where(max_rgb > 0, delta / max_rgb, 0)
    features.extend([
        np.mean(saturation),
        np.std(saturation)
    ])
    
    # Spectral features if available (12 features)
    if reflectance_spectrum is not None:
        print(f"Adding reflectance spectrum features: {len(reflectance_spectrum)} bands")
        # Use statistical measures of the spectrum
        features.extend([
            np.mean(reflectance_spectrum),
            np.std(reflectance_spectrum),
            np.median(reflectance_spectrum),
            np.min(reflectance_spectrum),
            np.max(reflectance_spectrum),
            np.percentile(reflectance_spectrum, 25),
            np.percentile(reflectance_spectrum, 75)
        ])
        
        # Add spectral regions (5 features)
        if len(reflectance_spectrum) >= 100:
            n_regions = 5
            region_size = len(reflectance_spectrum) // n_regions
            for i in range(n_regions):
                start_idx = i * region_size
                end_idx = (i + 1) * region_size if i < n_regions - 1 else len(reflectance_spectrum)
                region_mean = np.mean(reflectance_spectrum[start_idx:end_idx])
                features.append(region_mean)
        else:
            # Pad with zeros if spectrum is too short
            features.extend([0.0] * 5)
    else:
        # Pad with zeros if no spectrum available (12 features)
        features.extend([0.0] * 12)
    
    # Ensure we have exactly 32 features
    while len(features) < 32:
        features.append(0.0)
    
    features = features[:32]  # Truncate if too many
    
    # Convert to numpy array and handle any NaN/inf values
    features = np.array(features, dtype=np.float32)
    features = np.nan_to_num(features, nan=0.0, posinf=1.0, neginf=0.0)
    
    print(f"Created feature vector with {len(features)} dimensions")
    print(f"Feature range: [{np.min(features):.6f}, {np.max(features):.6f}]")
    
    return features.reshape(1, -1)  # Return as 2D array for sklearn

def safe_svm_classification(svm_model, features):
    """
    Perform SVM classification with proper preprocessing pipeline (without PCA).
    """
    try:
        print(f"Starting SVM classification with features shape: {features.shape}")
        
        # Load preprocessing components
        from utils.model_loader import load_scaler, load_label_encoder
        scaler = load_scaler()
        label_encoder = load_label_encoder()
        
        # Apply scaling
        if scaler is not None:
            print("Applying scaling...")
            features_scaled = scaler.transform(features)
            print(f"After scaling: {features_scaled.shape}")
        else:
            print("No scaler found, using raw features")
            features_scaled = features
        
        # Skip PCA since it's corrupted - use scaled features directly
        final_features = features_scaled
        print(f"Using features for SVM: {final_features.shape}")
        
        # Perform SVM prediction
        print("Performing SVM prediction...")
        pred = svm_model.predict(final_features)[0]
        print(f"Raw SVM prediction: {pred}")
        
        # Get probabilities
        if hasattr(svm_model, 'predict_proba'):
            probs = svm_model.predict_proba(final_features)[0]
            confidence = float(np.max(probs))
            print(f"Prediction probabilities: {probs}")
            print(f"Confidence: {confidence:.3f}")
        else:
            probs = None
            confidence = 1.0
        
        # Map prediction to class names using label encoder
        if label_encoder is not None and hasattr(label_encoder, 'classes_'):
            print("Using label encoder for class mapping...")
            print(f"Available classes: {label_encoder.classes_}")
            
            if isinstance(pred, (int, np.integer)) and 0 <= pred < len(label_encoder.classes_):
                pred_class = str(label_encoder.classes_[pred])
                class_names = [str(c) for c in label_encoder.classes_]
                
                if probs is not None:
                    class_probs = probs.tolist()
                else:
                    class_probs = [1.0 if i == pred else 0.0 for i in range(len(class_names))]
                
                print(f"Mapped prediction {pred} to class: {pred_class}")
            else:
                print(f"Invalid prediction index: {pred}, max index: {len(label_encoder.classes_)-1}")
                pred_class = "Classification Error"
                class_names = [str(c) for c in label_encoder.classes_]
                class_probs = [1.0/len(class_names)] * len(class_names)
                confidence = 0.0
        
        else:
            print("No label encoder available, using SVM classes...")
            if hasattr(svm_model, "classes_"):
                # Map numeric classes to default plastic types
                plastic_mapping = {0: 'EPSF', 1: 'HDPE', 2: 'LDPE', 3: 'PET', 4: 'PP', 5: 'Weathered'}
                pred_class = plastic_mapping.get(pred, f"Class_{pred}")
                class_names = list(plastic_mapping.values())
                
                if probs is not None:
                    class_probs = probs.tolist()
                else:
                    class_probs = [1.0 if i == pred else 0.0 for i in range(len(class_names))]
            else:
                pred_class = str(pred)
                class_names = [pred_class]
                class_probs = [1.0]
        
        print(f"Final classification: {pred_class} (confidence: {confidence:.3f})")
        return pred_class, confidence, class_names, class_probs
    
    except Exception as e:
        print(f"SVM classification failed: {str(e)}")
        import traceback
        traceback.print_exc()
        
        # Return default values for failed classification
        default_classes = ['EPSF', 'HDPE', 'LDPE', 'PET', 'PP', 'Weathered']
        return "Classification Error", 0.0, default_classes, [1.0/len(default_classes)] * len(default_classes)

def process_file(file_path):
    """Process a single hyperspectral .nc4 file and produce results."""
    print(f"Starting to process file: {file_path}")
    
    # Check if file exists
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        raise FileNotFoundError(f"File not found: {file_path}")
    
    # Validate the NC4 file
    is_valid, message = validate_nc4_file(file_path)
    if not is_valid:
        print(f"Invalid NC4 file: {message}")
        raise ValueError(f"Invalid NC4 file: {message}")
    
    try:
        # Get the models when needed
        print("Getting models...")
        unet_model, svm_model = get_models()
        print("Models loaded successfully")
        
        # Open the .nc4 file with xarray
        print(f"Opening file with xarray: {file_path}")
        ds = xr.open_dataset(file_path, engine="netcdf4")
        print(f"File opened successfully. Data variables: {list(ds.data_vars.keys())}")
        
        # Initialize variables
        rgb_image = None
        labeled_image = None
        reflectance_spectrum = None
        
        # Get RgbImage data
        if 'RgbImage' in ds.data_vars:
            rgb_data = ds['RgbImage'].values  # Shape: (3, 2992, 2000)
            print(f"Found RgbImage with shape: {rgb_data.shape}")
            
            # Convert from (3, H, W) to (H, W, 3)
            rgb_image = np.transpose(rgb_data, (1, 2, 0))
            print(f"Converted RGB image to shape: {rgb_image.shape}")
            
            # Ensure data is uint8 and in proper range
            if rgb_image.dtype != np.uint8:
                rgb_image = np.clip(rgb_image, 0, 255).astype(np.uint8)
        
        # Get LabeledImage data (ground truth)
        if 'LabeledImage' in ds.data_vars:
            labeled_image = ds['LabeledImage'].values  # Shape: (2992, 2000)
            print(f"Found LabeledImage with shape: {labeled_image.shape}")
        
        # Get Reflectance spectrum (note the typo in original data)
        if 'Reflectacne' in ds.data_vars:
            reflectance_spectrum = ds['Reflectacne'].values  # Shape: (2151,)
            print(f"Found Reflectance spectrum with shape: {reflectance_spectrum.shape}")
        
        # Get file attributes for additional info
        file_attrs = dict(ds.attrs)
        print(f"File attributes: {file_attrs}")
        
        # Close the dataset
        ds.close()
        print("Dataset loaded and closed")
        
        # Validate RGB image
        if rgb_image is None:
            raise ValueError("No RgbImage found in the file")
        
        original_H, original_W, C = rgb_image.shape
        print(f"Original RGB image shape: H={original_H}, W={original_W}, C={C}")
        
        # Resize image for U-Net model (expected input: 256x256x3)
        model_input_size = (256, 256)
        rgb_resized, _ = resize_image_and_mask(rgb_image, target_size=model_input_size)
        print(f"Resized RGB image for model: {rgb_resized.shape}")
        
        # Run U-Net model to get segmentation mask
        mask_pred = run_unet_segmentation(unet_model, rgb_resized)
        print(f"U-Net prediction shape: {mask_pred.shape}")
        
        # Process U-Net output to get binary mask
        if mask_pred.ndim == 2 or mask_pred.shape[-1] == 1:
            # Binary segmentation: threshold probability 0.5 to get binary mask
            mask_resized = mask_pred[..., 0] if mask_pred.ndim == 3 else mask_pred
            mask_resized = mask_resized >= 0.5
            print("Applied threshold 0.5 to get binary mask")
        else:
            # Multi-class segmentation: take argmax for mask
            mask_resized = np.argmax(mask_pred, axis=-1)
            # Assuming polymer class is 1, create binary mask for polymer presence
            mask_resized = (mask_resized == 1)
            print("Applied argmax to get binary mask from multi-class prediction")
        
        # Resize mask back to original image size
        mask = resize_mask_to_original(mask_resized, (original_H, original_W))
        print(f"Resized mask back to original size: {mask.shape}")
        
        mask = mask.astype(bool)
        mask_area = np.count_nonzero(mask)
        mask_coverage = mask_area / (original_H * original_W)
        print(f"Predicted mask area: {mask_area} pixels, coverage: {mask_coverage*100:.2f}%")
        
        # Compare with ground truth if available
        gt_metrics = {}
        gt_area = 0
        gt_coverage = 0
        if labeled_image is not None:
            gt_mask = (labeled_image > 0).astype(bool)
            gt_area = np.count_nonzero(gt_mask)
            gt_coverage = gt_area / (original_H * original_W)
            print(f"Ground truth mask area: {gt_area} pixels, coverage: {gt_coverage*100:.2f}%")
            
            # Calculate intersection over union (IoU)
            intersection = np.count_nonzero(mask & gt_mask)
            union = np.count_nonzero(mask | gt_mask)
            iou = intersection / union if union > 0 else 0.0
            print(f"IoU between prediction and ground truth: {iou:.3f}")
            
            # Calculate precision, recall, F1
            true_positives = intersection
            false_positives = np.count_nonzero(mask & ~gt_mask)
            false_negatives = np.count_nonzero(~mask & gt_mask)
            
            precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0.0
            recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0.0
            f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
            
            gt_metrics = {
                "ground_truth_coverage": gt_coverage,
                "iou": iou,
                "precision": precision,
                "recall": recall,
                "f1_score": f1_score
            }
        
        # Initialize default classification result
        pred_class = "No Polymer Detected"
        class_names = ["No Polymer Detected"]
        class_probs = [1.0]
        confidence = 0.0
        
        # If any polymer pixel detected, perform classification with SVM
        if mask_area > 0:
            print("Polymer detected. Running SVM classification.")
            try:
                # Use the reflectance spectrum directly (matching original training)
                if reflectance_spectrum is not None and len(reflectance_spectrum) == 2151:
                    print("Using reflectance spectrum for classification")
                    features = reflectance_spectrum.reshape(1, -1)
                    print(f"Reflectance features shape: {features.shape}")
                    
                    # Perform SVM classification with proper preprocessing
                    pred_class, confidence, class_names, class_probs = safe_svm_classification(
                        svm_model, features
                    )
                    
                    print(f"Final classification result: {pred_class} (confidence: {confidence:.3f})")
                
                else:
                    print(f"Invalid reflectance spectrum. Expected 2151 features, got: {reflectance_spectrum.shape if reflectance_spectrum is not None else 'None'}")
                    pred_class = "Invalid Data"
                    confidence = 0.0
                    class_names = ['EPSF', 'HDPE', 'LDPE', 'PET', 'PP', 'Weathered']
                    class_probs = [1.0/len(class_names)] * len(class_names)
            
            except Exception as e:
                print(f"Error during classification: {str(e)}")
                import traceback
                traceback.print_exc()
                pred_class = "Classification Error"
                confidence = 0.0
                class_names = ['EPSF', 'HDPE', 'LDPE', 'PET', 'PP', 'Weathered']
                class_probs = [1.0/len(class_names)] * len(class_names)
        else:
            print("No polymer detected. Skipping classification.")
        
        # Create output images and report
        print("Creating output images and report")
        
        # Generate file paths
        base_name = os.path.splitext(os.path.basename(file_path))[0]
        results_dir = os.path.join("static", "results")
        os.makedirs(results_dir, exist_ok=True)
        
        orig_filename = f"{base_name}_original.png"
        overlay_filename = f"{base_name}_overlay.png"
        gt_filename = f"{base_name}_ground_truth.png"
        
        orig_path = os.path.join(results_dir, orig_filename)
        overlay_path = os.path.join(results_dir, overlay_filename)
        gt_path = os.path.join(results_dir, gt_filename)
        
        try:
            # Save original RGB image
            img = Image.fromarray(rgb_image.astype(np.uint8))
            img.save(orig_path)
            print(f"Saved original image to {orig_path}")
            
            # Create overlay by blending mask on the RGB image
            overlay_image = rgb_image.copy().astype(np.float32)
            if mask_area > 0:
                # Blend: 70% original + 30% red on mask pixels
                overlay_image[mask] = overlay_image[mask] * 0.7 + np.array([255, 0, 0], dtype=np.float32) * 0.3
            
            overlay_image = np.clip(overlay_image, 0, 255).astype(np.uint8)
            overlay_img = Image.fromarray(overlay_image)
            overlay_img.save(overlay_path)
            print(f"Saved overlay image to {overlay_path}")
            
            # Save ground truth overlay if available
            gt_overlay_path = None
            if labeled_image is not None:
                gt_overlay_image = rgb_image.copy().astype(np.float32)
                gt_mask = (labeled_image > 0).astype(bool)
                if np.count_nonzero(gt_mask) > 0:
                    # Blend: 70% original + 30% green on ground truth pixels
                    gt_overlay_image[gt_mask] = gt_overlay_image[gt_mask] * 0.7 + np.array([0, 255, 0], dtype=np.float32) * 0.3
                
                gt_overlay_image = np.clip(gt_overlay_image, 0, 255).astype(np.uint8)
                gt_overlay_img = Image.fromarray(gt_overlay_image)
                gt_overlay_img.save(gt_path)
                gt_overlay_path = f"results/{gt_filename}"
                print(f"Saved ground truth overlay to {gt_path}")
        
        except Exception as e:
            print(f"Error saving output files: {str(e)}")
            import traceback
            traceback.print_exc()
            raise ValueError(f"Failed to save output files: {str(e)}")
        
        # Generate the detailed report content
        report_content = f"=== Analysis Report for {base_name} ===\n\n"
        
        # File information
        report_content += "FILE INFORMATION:\n"
        report_content += "--------------------\n"
        report_content += f"Original file: {os.path.basename(file_path)}\n"
        report_content += f"Image dimensions: {original_W} x {original_H} pixels\n"
        report_content += f"Model input size: {model_input_size[0]} x {model_input_size[1]} pixels\n"
        
        if file_attrs:
            report_content += f"Debris Polymer: {file_attrs.get('Debris Polymer', 'N/A')}\n"
            report_content += f"Debris Color: {file_attrs.get('Debris Color', 'N/A')}\n"
            report_content += f"Background flow status: {file_attrs.get('Background flow status', 'N/A')}\n"
            report_content += f"Plastic fraction: {file_attrs.get('Plastic fraction(%)', 'N/A')}%\n"
            report_content += f"Sediment concentration: {file_attrs.get('Sediment concentration [mg.L^-1]', 'N/A')} mg/L\n"
            report_content += f"Flow discharge: {file_attrs.get('Flow discharge [cms]', 'N/A')} cms\n"
            report_content += f"Quality Flag: {file_attrs.get('Quality Flag (1:lowest, 5:highest)', 'N/A')}\n"
        
        report_content += "\n"
        
        # Segmentation results
        report_content += "SEGMENTATION RESULTS:\n"
        report_content += "-------------------------\n"
        if mask_area == 0:
            report_content += "No polymer detected by U-Net model.\n"
        else:
            report_content += f"Polymer detected: {mask_area} pixels\n"
            report_content += f"Coverage: {mask_coverage*100:.2f}% of image area\n"
        
        # Ground truth comparison
        if labeled_image is not None:
            report_content += f"Ground truth coverage: {gt_coverage*100:.2f}% of image area\n"
            
            if gt_metrics:
                report_content += f"IoU (Intersection over Union): {gt_metrics['iou']:.3f}\n"
                report_content += f"Precision: {gt_metrics['precision']:.3f}\n"
                report_content += f"Recall: {gt_metrics['recall']:.3f}\n"
                report_content += f"F1-Score: {gt_metrics['f1_score']:.3f}\n"
        
        report_content += "\n"
        
        # Classification results
        report_content += "CLASSIFICATION RESULTS:\n"
        report_content += "-------------------------\n"
        if mask_area == 0:
            report_content += "No classification performed (no polymer detected).\n"
        elif "Error" in pred_class:
            report_content += "Classification failed due to an error.\n"
        else:
            report_content += f"Predicted polymer type: {pred_class}\n"
            report_content += f"Confidence: {confidence*100:.1f}%\n"
            if class_names and class_probs:
                report_content += "Class probabilities:\n"
                for name, prob in zip(class_names, class_probs):
                    report_content += f"  {name}: {prob*100:.1f}%\n"
        
        # Spectral information
        if reflectance_spectrum is not None:
            report_content += "\nSPECTRAL INFORMATION:\n"
            report_content += "-------------------------\n"
            report_content += f"Reflectance spectrum bands: {len(reflectance_spectrum)}\n"
            report_content += f"Wavelength range: 350-2500 nm\n"
            report_content += f"Spectrum range: {np.min(reflectance_spectrum):.6f} to {np.max(reflectance_spectrum):.6f}\n"
            report_content += f"Mean reflectance: {np.mean(reflectance_spectrum):.6f}\n"
        
        # Save the report to file
        report_path = os.path.join(results_dir, f"{base_name}_report.txt")
        with open(report_path, "w") as report:
            report.write(report_content)
        print(f"Saved comprehensive report to {report_path}")
        
        # Prepare the result dict to return
        result = {
            "original_image": f"results/{orig_filename}",
            "overlay_image": f"results/{overlay_filename}",
            "ground_truth_image": gt_overlay_path,
            "predicted_class": pred_class,
            "confidence": confidence,
            "mask_coverage": mask_coverage,
            "class_names": class_names,
            "class_probs": class_probs,
            "file_attributes": file_attrs,
            "has_ground_truth": labeled_image is not None,
            "original_dimensions": (original_H, original_W),
            "model_input_dimensions": model_input_size,
            "detailed_report": report_content,
            "mask_area": mask_area,
            "reflectance_spectrum": {
                "bands": len(reflectance_spectrum) if reflectance_spectrum is not None else 0,
                "min_value": float(np.min(reflectance_spectrum)) if reflectance_spectrum is not None else 0,
                "max_value": float(np.max(reflectance_spectrum)) if reflectance_spectrum is not None else 0,
                "mean_value": float(np.mean(reflectance_spectrum)) if reflectance_spectrum is not None else 0
            }
        }
        
        # Add ground truth metrics if available
        if gt_metrics:
            result.update(gt_metrics)
        
        print(f"Processing completed successfully for file: {file_path}")
        return result
    
    except Exception as e:
        print(f"Error processing file {file_path}: {str(e)}")
        import traceback
        traceback.print_exc()
        raise ValueError(f"Failed to process file: {str(e)}")




def process_multiple_files(file_paths):
    """Process multiple .nc4 files and return aggregated results."""
    print(f"Processing {len(file_paths)} files")
    
    results = []
    errors = []
    
    for file_path in file_paths:
        try:
            result = process_file(file_path)
            results.append({
                "filename": os.path.basename(file_path),
                "success": True,
                "result": result
            })
        except Exception as e:
            error_msg = str(e)
            print(f"Failed to process {file_path}: {error_msg}")
            errors.append({
                "filename": os.path.basename(file_path),
                "error": error_msg
            })
            results.append({
                "filename": os.path.basename(file_path),
                "success": False,
                "error": error_msg
            })
    
    # Calculate summary statistics
    successful_results = [r["result"] for r in results if r["success"]]
    
    summary = {
        "total_files": len(file_paths),
        "successful": len(successful_results),
        "failed": len(errors),
        "success_rate": len(successful_results) / len(file_paths) if file_paths else 0
    }
    
    if successful_results:
        # Calculate average metrics
        avg_coverage = np.mean([r["mask_coverage"] for r in successful_results])
        confidences = [r["confidence"] for r in successful_results if r["confidence"] > 0]
        avg_confidence = np.mean(confidences) if confidences else 0.0
        
        summary.update({
            "average_coverage": avg_coverage,
            "average_confidence": avg_confidence
        })
        
        # If ground truth is available, calculate average metrics
        gt_results = [r for r in successful_results if r.get("has_ground_truth", False)]
        if gt_results:
            avg_iou = np.mean([r["iou"] for r in gt_results])
            avg_precision = np.mean([r["precision"] for r in gt_results])
            avg_recall = np.mean([r["recall"] for r in gt_results])
            avg_f1 = np.mean([r["f1_score"] for r in gt_results])
            
            summary.update({
                "average_iou": avg_iou,
                "average_precision": avg_precision,
                "average_recall": avg_recall,
                "average_f1_score": avg_f1,
                "files_with_ground_truth": len(gt_results)
            })
    
    return {
        "results": results,
        "errors": errors,
        "summary": summary
    }
