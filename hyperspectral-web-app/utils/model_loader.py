import json
import h5py
import os
import tensorflow as tf
import joblib
import numpy as np

# Global variables to hold loaded models
_unet_model = None
_svm_model = None
_scaler = None
_pca = None
_label_encoder = None

def load_unet_model(h5_path):
    """Load U-Net model with custom handling for compatibility issues."""
    try:
        # First try the standard loading method
        model = tf.keras.models.load_model(h5_path, compile=False)
        return model, "Standard loading method"
    except Exception as e:
        print(f"Standard model loading failed: {e}")
        print("Trying alternative loading method...")
        
        # Try custom loading approach
        try:
            # Load model directly with custom_objects
            model = tf.keras.models.load_model(h5_path, compile=False,
                                              custom_objects={'tf': tf})
            return model, "Custom objects loading method"
        except Exception as nested_e:
            # Last resort: try to load with h5py and reconstruct
            try:
                with h5py.File(h5_path, 'r') as f:
                    # Get model config
                    if 'model_config' in f.attrs:
                        model_config = json.loads(f.attrs['model_config'].decode('utf-8'))
                        # Create model from config
                        model = tf.keras.models.model_from_json(json.dumps(model_config))
                        # Load weights
                        model.load_weights(h5_path)
                        return model, "H5py reconstruction method"
            except Exception as final_e:
                raise Exception(f"Failed to load model: {str(e)} -> {str(nested_e)} -> {str(final_e)}")

def get_models():
    """Get the loaded models or load them if not already loaded."""
    global _unet_model, _svm_model
    
    if _unet_model is None or _svm_model is None:
        load_models()
    return _unet_model, _svm_model

def load_scaler():
    """Load and return the scaler."""
    global _scaler
    if _scaler is None:
        load_models()
    return _scaler

def load_pca():
    """Load PCA model - currently skipped due to corruption."""
    print("Skipping PCA loading (model corrupted)")
    return None

def load_label_encoder():
    """Load and return the label encoder."""
    global _label_encoder
    if _label_encoder is None:
        load_models()
    return _label_encoder

def safe_serialize(obj):
    """Safely serialize objects, converting numpy arrays to lists."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif hasattr(obj, '__array__'):
        return np.array(obj).tolist()
    else:
        return str(obj) if obj is not None else 'Not available'

def load_models():
    """Load U-Net and SVM models from disk along with preprocessing objects."""
    global _unet_model, _svm_model, _scaler, _pca, _label_encoder
    
    loading_details = {}
    model_dir = "models"  # Update this path as needed
    
    # Load U-Net model
    if _unet_model is None:
        unet_path = os.path.join(model_dir, "unet_plastic_segmentation_LAST.h5")
        try:
            _unet_model, method = load_unet_model(unet_path)
            loading_details["unet"] = {
                "status": "success",
                "path": unet_path,
                "method": method
            }
            print(f"✓ U-Net model loaded successfully using: {method}")
        except Exception as e:
            loading_details["unet"] = {
                "status": "failed",
                "path": unet_path,
                "error": str(e)
            }
            raise Exception(f"U-Net model loading failed: {str(e)}")
    else:
        # U-Net model is already loaded
        loading_details["unet"] = {
            "status": "already_loaded",
            "path": os.path.join(model_dir, "unet_plastic_segmentation_LAST.h5")
        }
        print("✓ U-Net model already loaded")
    
    # Load SVM model
    if _svm_model is None:
        svm_path = os.path.join(model_dir, "polymer_svm_classifier.pkl")
        try:
            _svm_model = joblib.load(svm_path)
            loading_details["svm"] = {
                "status": "success",
                "path": svm_path,
                "classes": safe_serialize(getattr(_svm_model, 'classes_', 'Not available')),
                "n_features": safe_serialize(getattr(_svm_model, 'n_features_in_', 'Not available'))
            }
            print(f"✓ SVM model loaded successfully")
            print(f"  - Classes: {getattr(_svm_model, 'classes_', 'Not available')}")
            print(f"  - Features: {getattr(_svm_model, 'n_features_in_', 'Not available')}")
        except Exception as e:
            loading_details["svm"] = {
                "status": "failed",
                "path": svm_path,
                "error": str(e)
            }
            raise Exception(f"SVM model loading failed: {str(e)}")
    else:
        # SVM model is already loaded
        loading_details["svm"] = {
            "status": "already_loaded",
            "path": os.path.join(model_dir, "polymer_svm_classifier.pkl"),
            "classes": safe_serialize(getattr(_svm_model, 'classes_', 'Not available')),
            "n_features": safe_serialize(getattr(_svm_model, 'n_features_in_', 'Not available'))
        }
        print("✓ SVM model already loaded")
    
    # Load scaler
    if _scaler is None:
        scaler_path = os.path.join(model_dir, "scaler.pkl")
        try:
            _scaler = joblib.load(scaler_path)
            loading_details["scaler"] = {
                "status": "success",
                "path": scaler_path,
                "n_features": safe_serialize(getattr(_scaler, 'n_features_in_', 'Not available'))
            }
            print(f"✓ Scaler loaded successfully")
            print(f"  - Features: {getattr(_scaler, 'n_features_in_', 'Not available')}")
        except Exception as e:
            print(f"⚠ Warning: Could not load scaler: {e}")
            loading_details["scaler"] = {
                "status": "failed",
                "path": scaler_path,
                "error": str(e)
            }
            _scaler = None
    else:
        # Scaler is already loaded
        loading_details["scaler"] = {
            "status": "already_loaded",
            "path": os.path.join(model_dir, "scaler.pkl"),
            "n_features": safe_serialize(getattr(_scaler, 'n_features_in_', 'Not available'))
        }
        print("✓ Scaler already loaded")
    
    # Load PCA (skip due to corruption)
    loading_details["pca"] = {
        "status": "skipped",
        "path": os.path.join(model_dir, "pca.pkl"),
        "reason": "Model corrupted - using direct features"
    }
    print("⚠ PCA loading skipped (model corrupted)")
    _pca = None
    
    # Load label encoder
    if _label_encoder is None:
        le_path = os.path.join(model_dir, "label_encoder.pkl")
        try:
            _label_encoder = joblib.load(le_path)
            loading_details["label_encoder"] = {
                "status": "success",
                "path": le_path,
                "classes": safe_serialize(getattr(_label_encoder, 'classes_', 'Not available'))
            }
            print(f"✓ Label encoder loaded successfully")
            print(f"  - Classes: {getattr(_label_encoder, 'classes_', 'Not available')}")
        except Exception as e:
            print(f"⚠ Warning: Could not load label encoder: {e}")
            loading_details["label_encoder"] = {
                "status": "failed",
                "path": le_path,
                "error": str(e)
            }
            _label_encoder = None
    else:
        # Label encoder is already loaded
        loading_details["label_encoder"] = {
            "status": "already_loaded",
            "path": os.path.join(model_dir, "label_encoder.pkl"),
            "classes": safe_serialize(getattr(_label_encoder, 'classes_', 'Not available'))
        }
        print("✓ Label encoder already loaded")
    
    print("\n=== MODEL LOADING SUMMARY ===")
    for model_name, details in loading_details.items():
        status = details["status"]
        status_symbol = "✓" if status in ["success", "already_loaded"] else "✗" if status == "failed" else "⚠"
        print(f"{status_symbol} {model_name.upper()}: {status}")
    
    return _unet_model, _svm_model, loading_details

def get_model_info():
    """Get information about loaded models without loading them."""
    info = {
        "unet_loaded": _unet_model is not None,
        "svm_loaded": _svm_model is not None,
        "scaler_loaded": _scaler is not None,
        "pca_loaded": _pca is not None,
        "label_encoder_loaded": _label_encoder is not None
    }
    
    if _svm_model is not None:
        info["svm_classes"] = safe_serialize(getattr(_svm_model, 'classes_', None))
        info["svm_features"] = safe_serialize(getattr(_svm_model, 'n_features_in_', None))
    
    if _label_encoder is not None:
        info["label_classes"] = safe_serialize(getattr(_label_encoder, 'classes_', None))
    
    return info

def reset_models():
    """Reset all loaded models (useful for testing)."""
    global _unet_model, _svm_model, _scaler, _pca, _label_encoder
    _unet_model = None
    _svm_model = None
    _scaler = None
    _pca = None
    _label_encoder = None
    print("All models reset")
