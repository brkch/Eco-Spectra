import numpy as np
# Add compatibility for NumPy 2.0
if not hasattr(np, 'unicode_'):
    np.unicode_ = np.str_

import os
import uuid
import time
import re
import glob
import netCDF4 as nc
from sklearn.metrics import confusion_matrix
from flask import Flask, request, jsonify, render_template, url_for, send_file, abort
from werkzeug.utils import secure_filename
from io import BytesIO
import json
import zipfile
import xarray as xr
import base64
from PIL import Image
from io import BytesIO
# Import processing module
from utils import processing

app = Flask(__name__, static_folder='static', template_folder='templates')

# Configuration
UPLOAD_FOLDER = "uploads"
RESULT_FOLDER = os.path.join("static", "results")
ALLOWED_EXTENSIONS = {'.nc4'}

# File size limits (in bytes)
MAX_FILE_SIZE = 1 * 1024 * 1024 * 1024  # 100 MB per file
MAX_TOTAL_SIZE = 5 * 1024 * 1024 * 1024  # 500 MB total for all files

session_results = []

# Ensure upload and result directories exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)

def allowed_file(filename):
    """Check if file has allowed extension."""
    return any(filename.lower().endswith(ext) for ext in ALLOWED_EXTENSIONS)

def validate_file_size(file_size):
    """Check if individual file size is within limits."""
    return file_size <= MAX_FILE_SIZE

def validate_total_size(total_size):
    """Check if total size of all files is within limits."""
    return total_size <= MAX_TOTAL_SIZE

def format_file_size(bytes_size):
    """Format file size for display."""
    if bytes_size == 0:
        return '0 Bytes'
    k = 1024
    sizes = ['Bytes', 'KB', 'MB', 'GB']
    i = int(np.floor(np.log(bytes_size) / np.log(k)))
    return f"{round(bytes_size / (k ** i), 2)} {sizes[i]}"



def extract_value(pattern, text, default='N/A'):
    """Extract value from text using regex pattern."""
    match = re.search(pattern, text, re.IGNORECASE)
    return match.group(1).strip() if match else default

def extract_reflectance_from_nc4(nc4_file_path):
    """Extract the full 2151 reflectance data directly from the NetCDF4 file."""
    try:
        print(f"Attempting to extract reflectance from: {nc4_file_path}")
        
        # Open the NetCDF4 file
        dataset = nc.Dataset(nc4_file_path, 'r')
        
        # Extract the reflectance array (note the typo in variable name 'Reflectacne')
        if 'Reflectacne' in dataset.variables:
            reflectance = dataset.variables['Reflectacne'][:]
            print(f"Successfully extracted reflectance data with shape: {reflectance.shape}")
            
            # Create wavelength array from 300 nm to 2500 nm over the same number of points
            wavelengths = np.linspace(300, 2500, num=reflectance.size)
            
            # Close the dataset
            dataset.close()
            
            # Convert to Python lists for JSON serialization
            return reflectance.tolist(), wavelengths.tolist()
        else:
            print(f"Available variables in NC4 file: {list(dataset.variables.keys())}")
            dataset.close()
            return None, None
        
    except Exception as e:
        print(f"Error extracting reflectance from NC4 file: {e}")
        import traceback
        traceback.print_exc()
        return None, None

def parse_analysis_content(content):
    """Parse the analysis text file content into structured data."""
    analysis = {}
    
    # File Information
    analysis['original_file'] = extract_value(r'Original file:\s*(.+)', content)
    analysis['image_dimensions'] = extract_value(r'Image dimensions:\s*(.+)', content)
    analysis['model_input_size'] = extract_value(r'Model input size:\s*(.+)', content)
    analysis['debris_polymer'] = extract_value(r'Debris Polymer:\s*(.+)', content)
    analysis['debris_color'] = extract_value(r'Debris Color:\s*(.+)', content)
    analysis['background_flow_status'] = extract_value(r'Background flow status:\s*(.+)', content)
    analysis['plastic_fraction'] = extract_value(r'Plastic fraction:\s*(.+)', content)
    analysis['sediment_concentration'] = extract_value(r'Sediment concentration:\s*(.+)', content)
    analysis['flow_discharge'] = extract_value(r'Flow discharge:\s*(.+)', content)
    analysis['quality_flag'] = extract_value(r'Quality Flag:\s*(.+)', content)
    
    # Segmentation Results
    analysis['polymer_detected_pixels'] = extract_value(r'Polymer detected:\s*(.+)', content)
    analysis['coverage'] = extract_value(r'Coverage:\s*(.+)', content)
    analysis['ground_truth_coverage'] = extract_value(r'Ground truth coverage:\s*(.+)', content)
    analysis['iou'] = extract_value(r'IoU.*?:\s*(.+)', content)
    analysis['precision'] = extract_value(r'Precision:\s*(.+)', content)
    analysis['recall'] = extract_value(r'Recall:\s*(.+)', content)
    analysis['f1_score'] = extract_value(r'F1-Score:\s*(.+)', content)
    
    # Classification Results
    analysis['predicted_polymer_type'] = extract_value(r'Predicted polymer type:\s*(.+)', content)
    analysis['confidence'] = extract_value(r'Confidence:\s*(.+)', content)
    
    # Parse class probabilities
    class_probs = {}
    prob_section = re.search(r'Class probabilities:(.*?)(?=\n\n|\nSPECTRAL|$)', content, re.DOTALL)
    if prob_section:
        prob_lines = prob_section.group(1).strip().split('\n')
        for line in prob_lines:
            line = line.strip()
            if ':' in line:
                parts = line.split(':')
                if len(parts) == 2:
                    polymer = parts[0].strip()
                    prob = parts[1].strip()
                    class_probs[polymer] = prob
    
    analysis['class_probabilities'] = class_probs if class_probs else None
    
    # Spectral Information
    analysis['reflectance_spectrum_bands'] = extract_value(r'Reflectance spectrum bands:\s*(.+)', content)
    analysis['wavelength_range'] = extract_value(r'Wavelength range:\s*(.+)', content)
    analysis['spectrum_range'] = extract_value(r'Spectrum range:\s*(.+)', content)
    analysis['mean_reflectance'] = extract_value(r'Mean reflectance:\s*(.+)', content)
    
    return analysis

@app.route("/")
def index():
    """Render the file upload interface."""
    return render_template("index.html")

@app.route("/load_models", methods=["GET"])
def load_models_route():
    """Load ML models (U-Net and SVM) into memory on user request."""
    try:
        # Load models and get details
        unet_model, svm_model = processing.get_models()
        
        # Create detailed success message
        message = "Models loaded successfully:\n"
        message += "- U-Net model: Loaded successfully\n"
        message += "- SVM model: Loaded successfully\n"
        
        loading_details = {
            "unet": {"status": "success"},
            "svm": {"status": "success"}
        }
        
        return jsonify({
            "message": message,
            "details": loading_details,
            "already_loaded": False
        })
        
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        print(error_details)
        return jsonify({"message": f"Model loading failed: {str(e)}"}), 500

@app.route("/check_models", methods=["GET"])
def check_models_route():
    """Check if models are already loaded."""
    try:
        # Try to get models to see if they're loaded
        unet_model, svm_model = processing.get_models()
        return jsonify({
            "unet_loaded": unet_model is not None,
            "svm_loaded": svm_model is not None
        })
    except Exception as e:
        print(f"Error checking models: {e}")
        return jsonify({
            "unet_loaded": False,
            "svm_loaded": False,
            "error": str(e)
        })

@app.route('/get_detailed_analysis/<filename>')
def get_detailed_analysis(filename):
    """Get detailed analysis for a specific file."""
    try:
        # Remove file extension to get base filename
        base_filename = os.path.splitext(filename)[0]
        
        # Search for report files that end with this pattern
        search_pattern = os.path.join('static', 'results', f'*{base_filename}_report.txt')
        matching_files = glob.glob(search_pattern)
        
        if not matching_files:
            return jsonify({
                'success': False,
                'error': f'Analysis file not found for: {filename}'
            })
        
        # Use the first matching file (should be unique)
        txt_path = matching_files[0]
        
        with open(txt_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Parse the analysis content
        analysis = parse_analysis_content(content)
        
        # Extract the original NC4 file path to get the full reflectance data
        original_filename = analysis.get('original_file', filename)
        
        # Try to find the original NC4 file in uploads folder
        nc4_file_path = None
        
        # Search for the original NC4 file in uploads folder
        for uploaded_file in os.listdir(UPLOAD_FOLDER):
            if uploaded_file.endswith('.nc4') and (
                original_filename in uploaded_file or
                base_filename in uploaded_file or
                uploaded_file.endswith(f'{base_filename}.nc4')
            ):
                nc4_file_path = os.path.join(UPLOAD_FOLDER, uploaded_file)
                break
        
        # If not found in uploads, try to construct the path
        if not nc4_file_path:
            # Try direct path construction
            potential_paths = [
                os.path.join(UPLOAD_FOLDER, original_filename),
                os.path.join(UPLOAD_FOLDER, f"{base_filename}.nc4"),
                original_filename  # In case it's an absolute path
            ]
            
            for path in potential_paths:
                if os.path.exists(path):
                    nc4_file_path = path
                    break
        
        # Extract reflectance data from the NC4 file
        if nc4_file_path and os.path.exists(nc4_file_path):
            print(f"Extracting reflectance data from: {nc4_file_path}")
            reflectance_data, wavelengths = extract_reflectance_from_nc4(nc4_file_path)
            
            if reflectance_data is not None:
                analysis['spectrum_data'] = reflectance_data
                analysis['wavelengths'] = wavelengths
                print(f"Successfully extracted {len(reflectance_data)} reflectance points")
            else:
                print("Failed to extract reflectance data from NC4 file")
                analysis['spectrum_data'] = None
                analysis['wavelengths'] = None
        else:
            print(f"NC4 file not found. Searched for: {original_filename}")
            print(f"Available files in uploads: {os.listdir(UPLOAD_FOLDER) if os.path.exists(UPLOAD_FOLDER) else 'Upload folder not found'}")
            analysis['spectrum_data'] = None
            analysis['wavelengths'] = None
        
        return jsonify({'success': True, 'analysis': analysis})
        
    except Exception as e:
        print(f"Error getting detailed analysis: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)})

def save_report_to_file(result, filename):
    """Save analysis data to a report text file."""
    try:
        base_filename = os.path.splitext(filename)[0]
        
        # Check if result has a consistent naming pattern from the processing module
        # Look for existing image files to match the naming pattern
        existing_images = []
        for ext in ['_overlay.png', '_mask.png', '_original.png']:
            pattern = os.path.join(RESULT_FOLDER, f"*{base_filename}{ext}")
            matches = glob.glob(pattern)
            if matches:
                # Extract the prefix from the first match
                image_path = matches[0]
                image_filename = os.path.basename(image_path)
                # Remove the extension part to get the base name
                prefix = image_filename.replace(ext, '')
                existing_images.append(prefix)
        
        # Use the same prefix as the images if found
        if existing_images:
            # Use the most common prefix (should be the same for all images)
            report_prefix = existing_images[0]
            report_filename = f"{report_prefix}_report.txt"
        else:
            # Fallback to timestamp-based naming if no images found
            timestamp = time.strftime('%Y%m%d_%H%M%S')
            report_filename = f"{timestamp}_{base_filename}_report.txt"
        
        report_path = os.path.join(RESULT_FOLDER, report_filename)
        
        # Check if report already exists to avoid duplicates
        if os.path.exists(report_path):
            print(f"Report already exists, skipping: {report_path}")
            result['report_file'] = report_filename
            return
        
        with open(report_path, 'w', encoding='utf-8') as f:
            # Use the detailed_report from the result if available
            if 'detailed_report' in result:
                f.write(result['detailed_report'])
            else:
                # Create comprehensive report
                f.write(f"ANALYSIS REPORT\n")
                f.write(f"=" * 50 + "\n\n")
                
                # File Information
                f.write(f"FILE INFORMATION:\n")
                f.write(f"-" * 20 + "\n")
                f.write(f"Original file: {filename}\n")
                f.write(f"Processing timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"Image dimensions: {result.get('image_dimensions', 'N/A')}\n")
                f.write(f"Model input size: {result.get('model_input_size', 'N/A')}\n")
                
                # Add all available metadata
                metadata_fields = [
                    'debris_polymer', 'debris_color', 'background_flow_status',
                    'plastic_fraction', 'sediment_concentration', 'flow_discharge',
                    'quality_flag'
                ]
                
                for field in metadata_fields:
                    if field in result:
                        f.write(f"{field.replace('_', ' ').title()}: {result[field]}\n")
                
                f.write("\n")
                
                # Classification Results
                f.write(f"CLASSIFICATION RESULTS:\n")
                f.write(f"-" * 25 + "\n")
                f.write(f"Predicted polymer type: {result.get('predicted_class', 'N/A')}\n")
                f.write(f"Confidence: {result.get('confidence', 0)*100:.2f}%\n\n")
                
                # Add class probabilities if available
                if 'class_probabilities' in result and result['class_probabilities']:
                    f.write("Class probabilities:\n")
                    for polymer, prob in result['class_probabilities'].items():
                        f.write(f"  {polymer}: {prob}\n")
                    f.write("\n")
                
                # Segmentation Results
                f.write(f"SEGMENTATION RESULTS:\n")
                f.write(f"-" * 25 + "\n")
                f.write(f"Polymer detected: {result.get('polymer_pixels', 'N/A')} pixels\n")
                f.write(f"Coverage: {result.get('mask_coverage', 0)*100:.2f}%\n")
                
                if result.get('has_ground_truth', False):
                    f.write(f"Ground truth coverage: {result.get('ground_truth_coverage', 'N/A')}\n")
                    f.write(f"IoU (Intersection over Union): {result.get('iou', 'N/A')}\n")
                    f.write(f"Precision: {result.get('precision', 'N/A')}\n")
                    f.write(f"Recall: {result.get('recall', 'N/A')}\n")
                    f.write(f"F1-Score: {result.get('f1_score', 'N/A')}\n")
                f.write("\n")
                
                # Spectral Information
                f.write(f"SPECTRAL INFORMATION:\n")
                f.write(f"-" * 25 + "\n")
                f.write(f"Reflectance spectrum bands: {result.get('spectrum_bands', 'N/A')}\n")
                f.write(f"Wavelength range: {result.get('wavelength_range', 'N/A')}\n")
                f.write(f"Spectrum range: {result.get('spectrum_range', 'N/A')}\n")
                f.write(f"Mean reflectance: {result.get('mean_reflectance', 'N/A')}\n")
        
        print(f"Report saved to: {report_path}")
        
        # Store the report filename in the result for later reference
        result['report_file'] = report_filename
        
    except Exception as e:
        print(f"Error saving report for {filename}: {e}")


def process_single_file(file, filename):
    start_time = time.time()  # Start timing for this file

    try:
        # Validate file extension
        if not allowed_file(filename):
            return None, f"Invalid file extension: {filename}"

        # Create unique filename to avoid collisions
        unique_id = uuid.uuid4().hex
        save_path = os.path.join(UPLOAD_FOLDER, f"{unique_id}_{filename}")

        # Save the uploaded file
        file.save(save_path)
        print(f"Saved file to: {save_path}")

        # Validate file was saved properly
        if not os.path.exists(save_path):
            return None, f"File was not saved properly: {filename}"

        file_size = os.path.getsize(save_path)
        if file_size == 0:
            return None, f"File is empty: {filename}"

        print(f"File size: {file_size} bytes")

        # Process the file
        print(f"Processing file: {filename}")
        result = processing.process_file(save_path)
        print(f"Successfully processed file: {filename}")

        # Add original filename for display
        result["filename"] = filename

        # Generate screenshot
        result['screenshot'] = generate_screenshot(result)

        # Generate and save report file
        try:
            save_report_to_file(result, filename)
            print(f"Report file generated for: {filename}")
        except Exception as e:
            print(f"Error generating report file for {filename}: {e}")

        end_time = time.time()
        result['elapsed_time'] = round(end_time - start_time, 2)  # Store processing time

        return result, None

    except Exception as e:
        error_msg = f"Error processing {filename}: {str(e)}"
        print(error_msg)
        import traceback
        traceback.print_exc()
        return None, error_msg



@app.route("/process", methods=["POST"])
def process_files():
    """Handle file upload and processing with size validation."""
    global session_results
    start_time = time.time()
    print("=== PROCESS FILES REQUEST START ===")
    print(f"Request method: {request.method}")
    print(f"Request files: {list(request.files.keys())}")
    print(f"Request form: {dict(request.form)}")

    # Get list of files from the upload form
    files = request.files.getlist("files")
    print(f"Number of files received: {len(files)}")

    if not files or len(files) == 0 or files[0].filename == "":
        print("No files uploaded.")
        return jsonify({
            "success": False,
            "error": "No files uploaded."
        }), 400

    # Log received files
    print(f"Received files: {[f.filename for f in files]}")

    # Validate file sizes and extensions
    validation_errors = []
    total_size = 0

    for file in files:
        filename = secure_filename(file.filename)
        if not filename:
            continue

        # Check file extension
        if not allowed_file(filename):
            validation_errors.append(f"{filename}: Invalid file type. Only .nc4 files are allowed.")
            continue

        # Get file size by seeking to end
        file.seek(0, 2)  # Seek to end
        file_size = file.tell()
        file.seek(0)  # Reset to beginning

        # Check individual file size
        if not validate_file_size(file_size):
            validation_errors.append(
                f"{filename}: File too large ({format_file_size(file_size)}). "
                f"Maximum allowed: {format_file_size(MAX_FILE_SIZE)}"
            )
            continue

        total_size += file_size

    # Check total size
    if not validate_total_size(total_size):
        validation_errors.append(
            f"Total file size ({format_file_size(total_size)}) exceeds the maximum limit "
            f"of {format_file_size(MAX_TOTAL_SIZE)}"
        )

    # If there are validation errors, return them
    if validation_errors:
        error_message = "File validation failed:\n" + "\n".join(validation_errors)
        print(f"Validation errors: {validation_errors}")
        return jsonify({
            "success": False,
            "error": error_message,
            "validation_errors": validation_errors
        }), 400

    # Ensure models are loaded
    try:
        processing.get_models()
        print("Models loaded successfully")
    except Exception as e:
        error_msg = f"Failed to load models: {str(e)}"
        print(error_msg)
        return jsonify({
            "success": False,
            "error": error_msg
        }), 500

    # Create upload directory if it doesn't exist
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)

    results = []
    processing_errors = []

    # Process each file
    for file in files:
        filename = secure_filename(file.filename)
        if not filename:
            print("Skipping file with empty filename")
            continue

        print(f"Processing file: {filename}")

        result, error = process_single_file(file, filename)

        if result:

            results.append(result)
            print(f"Successfully processed: {filename}")
        else:
            processing_errors.append(f"{filename}: {error}")
            print(f"Failed to process: {filename} - {error}")

    # Check if any files were processed successfully
    if not results:
        error_details = "\n".join(processing_errors) if processing_errors else "Unknown processing errors"
        error_message = f"No valid .nc4 files were processed successfully.\n\nDetailed errors:\n{error_details}"
        print("No valid results were generated.")
        print(f"Processing errors: {processing_errors}")
        return jsonify({
            "success": False,
            "error": error_message,
            "processing_errors": processing_errors
        }), 400

    # Log processing summary
    end_time = time.time()
    elapsed = end_time - start_time
    print(f"=== PROCESS FILES REQUEST END: elapsed={elapsed:.2f} seconds ===")
    print(f"Successfully processed {len(results)} out of {len(files)} files")
    session_results = results
    if processing_errors:
        print(f"Errors encountered: {processing_errors}")

    # Render results page
    print(f"Rendering results for {len(results)} files")
    return render_template("results.html", results=results, errors=processing_errors, elapsed_time=round(elapsed, 2))


@app.route('/debug_files')
def debug_files():
    """Debug endpoint to see what files are available."""
    try:
        results_dir = os.path.join('static', 'results')
        upload_dir = UPLOAD_FOLDER
        
        result_files = []
        upload_files = []
        
        if os.path.exists(results_dir):
            result_files = os.listdir(results_dir)
        
        if os.path.exists(upload_dir):
            upload_files = os.listdir(upload_dir)
        
        return jsonify({
            "results_directory": results_dir,
            "results_directory_exists": os.path.exists(results_dir),
            "upload_directory": upload_dir,
            "upload_directory_exists": os.path.exists(upload_dir),
            "result_files": result_files,
            "upload_files": upload_files,
            "session_results_count": len(session_results),
            "session_filenames": [r.get('filename', 'N/A') for r in session_results],
            "current_working_directory": os.getcwd()
        })
    except Exception as e:
        return jsonify({"error": str(e)})

@app.route('/clear_session')
def clear_session():
    """Clear session results and temporary files."""
    global session_results
    session_results = []
    
    # Optionally clean up old files
    try:
        # Clean files older than 1 hour
        current_time = time.time()
        for folder in [UPLOAD_FOLDER, RESULT_FOLDER]:
            if os.path.exists(folder):
                for filename in os.listdir(folder):
                    file_path = os.path.join(folder, filename)
                    if os.path.isfile(file_path):
                        file_age = current_time - os.path.getctime(file_path)
                        if file_age > 3600:  # 1 hour
                            os.remove(file_path)
    except Exception as e:
        print(f"Error during cleanup: {e}")
    
    return jsonify({"success": True, "message": "Session cleared"})

@app.errorhandler(413)
def too_large(error):
    """Handle file too large errors."""
    return jsonify({
        "success": False,
        "error": f"File is too large. Maximum allowed size is {format_file_size(MAX_FILE_SIZE)} per file "
                f"and {format_file_size(MAX_TOTAL_SIZE)} total."
    }), 413

@app.route("/file_limits", methods=["GET"])
def get_file_limits():
    """Return file size limits for frontend validation."""
    return jsonify({
        "max_file_size": MAX_FILE_SIZE,
        "max_total_size": MAX_TOTAL_SIZE,
        "max_file_size_formatted": format_file_size(MAX_FILE_SIZE),
        "max_total_size_formatted": format_file_size(MAX_TOTAL_SIZE)
    })

@app.route('/download_report/<filename>')
def download_report(filename):
    """Download individual report file."""
    try:
        print(f"Download request for filename: {filename}")
        
        # Remove file extension and get base name
        base_name = os.path.splitext(filename)[0]
        print(f"Base name: {base_name}")
        
        # Look for the report file in the results directory
        results_dir = os.path.join('static', 'results')
        print(f"Looking in directory: {results_dir}")
        
        if not os.path.exists(results_dir):
            print(f"Results directory does not exist: {results_dir}")
            return jsonify({"error": "Results directory not found"}), 404
        
        # List all files in results directory
        all_files = os.listdir(results_dir)
        print(f"All files in results directory: {all_files}")
        
        # Look for report files that match this filename
        matching_files = []
        for file in all_files:
            if base_name in file and file.endswith('_report.txt'):
                matching_files.append(file)
        
        print(f"Matching files: {matching_files}")
        
        if not matching_files:
            # Try alternative matching patterns
            for file in all_files:
                if file.endswith('_report.txt'):
                    matching_files.append(file)
            
            if matching_files:
                print(f"Using first available report file: {matching_files[0]}")
                report_file = matching_files[0]
            else:
                print("No report files found at all")
                return jsonify({"error": f"No report file found for {filename}"}), 404
        else:
            report_file = matching_files[0]
        
        report_path = os.path.join(results_dir, report_file)
        print(f"Attempting to send file: {report_path}")
        
        if os.path.exists(report_path):
            return send_file(report_path, as_attachment=True, download_name=report_file)
        else:
            print(f"Report file does not exist: {report_path}")
            return jsonify({"error": "Report file not found"}), 404
        
    except Exception as e:
        print(f"Error downloading report: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500



@app.route('/download_all_results')
def download_all_results():
    """Download all results from current session."""
    global session_results
    
    try:
        print(f"Download all results requested. Session results count: {len(session_results)}")
        
        if not session_results:
            return jsonify({
                "success": False,
                "error": "No results available in current session. Please process some files first."
            }), 400

        # Create a BytesIO object to store the zip file
        zip_buffer = BytesIO()
        
        # Create a zip file
        with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
            results_dir = os.path.join('static', 'results')
            print(f"Looking for files in: {results_dir}")
            
            if not os.path.exists(results_dir):
                print(f"Results directory does not exist: {results_dir}")
                return jsonify({
                    "success": False,
                    "error": "Results directory not found"
                }), 404
            
            # Get all files in results directory
            all_files = os.listdir(results_dir)
            print(f"All files in results directory: {all_files}")
            
            # Get list of base filenames from current session
            session_filenames = set()
            for result in session_results:
                if 'filename' in result:
                    base_name = os.path.splitext(result['filename'])[0]
                    session_filenames.add(base_name)
            
            print(f"Session base filenames: {session_filenames}")
            
            added_files = []
            
            # Add all files (for now, to debug)
            for filename in all_files:
                file_path = os.path.join(results_dir, filename)
                
                if os.path.isfile(file_path):
                    # Add all text and image files
                    if (filename.endswith('.txt') or 
                        filename.lower().endswith(('.png', '.jpg', '.jpeg'))):
                        
                        zip_file.write(file_path, filename)
                        added_files.append(filename)
                        print(f"Added file to zip: {filename}")
            
            print(f"Total files added to zip: {len(added_files)}")
            
            if not added_files:
                return jsonify({
                    "success": False,
                    "error": f"No files found in results directory. Available files: {all_files}"
                }), 404

        zip_buffer.seek(0)
        
        # Generate timestamp for unique filename
        timestamp = time.strftime('%Y%m%d_%H%M%S')
        zip_filename = f'all_results_{timestamp}.zip'
        
        return send_file(
            zip_buffer,
            mimetype='application/zip',
            as_attachment=True,
            download_name=zip_filename
        )
        
    except Exception as e:
        print(f"Error creating zip file: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({
            "success": False,
            "error": f"Failed to create zip file: {str(e)}"
        }), 500



def generate_screenshot(result):
    """Generate a screenshot of the result visualization."""
    try:
        # Get the visualization image path
        img_path = os.path.join('static', result['overlay_image'])
        img = Image.open(img_path)
        
        # Create a new image with analysis information
        from PIL import ImageDraw, ImageFont
        font = ImageFont.load_default()
        
        # Create a new image with text overlay
        new_img = Image.new('RGB', (img.width, img.height + 50), color=(255, 255, 255))
        new_img.paste(img, (0, 0))
        
        # Add text information
        draw = ImageDraw.Draw(new_img)
        text = f"Polymer: {result.get('predicted_class', 'N/A')} | Confidence: {result.get('confidence', 0)*100:.1f}%"
        draw.text((10, img.height + 10), text, fill=(0, 0, 0), font=font)
        
        # Convert to bytes
        buffered = BytesIO()
        new_img.save(buffered, format="PNG")
        return base64.b64encode(buffered.getvalue()).decode('utf-8')
        
    except Exception as e:
        print(f"Error generating screenshot: {e}")
        return None

@app.route('/get_session_results')
def get_session_results():
    """Get current session results for display."""
    global session_results
    try:
        return jsonify({
            "success": True,
            "results": session_results,
            "count": len(session_results)
        })
    except Exception as e:
        print(f"Error getting session results: {e}")
        return jsonify({
            "success": False,
            "error": str(e),
            "results": [],
            "count": 0
        })

@app.route('/health')
def health_check():
    """Health check endpoint."""
    try:
        # Check if models can be loaded
        unet_model, svm_model = processing.get_models()
        models_status = "loaded" if (unet_model is not None and svm_model is not None) else "not_loaded"
        
        return jsonify({
            "status": "healthy",
            "models": models_status,
            "upload_folder": os.path.exists(UPLOAD_FOLDER),
            "result_folder": os.path.exists(RESULT_FOLDER),
            "timestamp": time.strftime('%Y-%m-%d %H:%M:%S')
        })
    except Exception as e:
        return jsonify({
            "status": "unhealthy",
            "error": str(e),
            "timestamp": time.strftime('%Y-%m-%d %H:%M:%S')
        }), 500

@app.route('/api/process_status/<task_id>')
def get_process_status(task_id):
    """Get processing status for a specific task (placeholder for future async processing)."""
    # This is a placeholder for future implementation of async processing
    return jsonify({
        "task_id": task_id,
        "status": "completed",
        "message": "Synchronous processing completed"
    })

@app.route('/cleanup_old_files')
def cleanup_old_files():
    """Manual cleanup of old files."""
    try:
        current_time = time.time()
        cleaned_files = []
        
        for folder in [UPLOAD_FOLDER, RESULT_FOLDER]:
            if os.path.exists(folder):
                for filename in os.listdir(folder):
                    file_path = os.path.join(folder, filename)
                    if os.path.isfile(file_path):
                        file_age = current_time - os.path.getctime(file_path)
                        if file_age > 3600:  # 1 hour
                            os.remove(file_path)
                            cleaned_files.append(filename)
        
        return jsonify({
            "success": True,
            "message": f"Cleaned up {len(cleaned_files)} old files",
            "cleaned_files": cleaned_files
        })
        
    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

if __name__ == "__main__":
    # Configuration for development
    app.config['MAX_CONTENT_LENGTH'] = MAX_TOTAL_SIZE  # Use the same limit as our validation
    app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
    
    print("Starting Flask application...")
    print(f"Upload folder: {UPLOAD_FOLDER}")
    print(f"Result folder: {RESULT_FOLDER}")
    print(f"Max file size: {format_file_size(MAX_FILE_SIZE)}")
    print(f"Max total size: {format_file_size(MAX_TOTAL_SIZE)}")
    
    # Run the Flask development server
    app.run(debug=True, host='0.0.0.0', port=5000)

