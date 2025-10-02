document.addEventListener('DOMContentLoaded', function () {
    const loadModelsBtn = document.getElementById('load-models-btn');
    const statusDiv      = document.getElementById('status');
    const fileInput      = document.getElementById('file-input');
    const dropzone       = document.getElementById('dropzone');
    const fileList       = document.getElementById('file-list');
    const uploadForm     = document.getElementById('upload-form');
    
    // File size limits in bytes
    const MAX_FILE_SIZE = 1 * 1024 * 1024 * 1024; // 100 MB per file
    const MAX_TOTAL_SIZE = 5 * 1024 * 1024 * 1024; // 500 MB total
    
    // Supported file extensions
    const SUPPORTED_EXTENSIONS = ['.nc4'];
    
    // Store selected files array
    let selectedFiles = [];
    
    // Bootstrap color map: [background, text]
    const bsColors = {
        primary:   ['#0d6efd', '#ffffff'],
        secondary: ['#6c757d', '#ffffff'],
        success:   ['#198754', '#ffffff'],
        danger:    ['#dc3545', '#ffffff'],
        warning:   ['#ffc107', '#000000'],
        info:      ['#0dcaf0', '#000000'],
        light:     ['#f8f9fa', '#000000'],
        dark:      ['#212529', '#ffffff']
    };
    
    // Spinner setup
    const spinnerFrames = ['⠋','⠙','⠹','⠸','⠼','⠴','⠦','⠧','⠇','⠏'];
    let spinnerInterval = null;
    let currentFrame    = 0;
    
    function startSpinner(baseMessage, bg, fg) {
        stopSpinner();
        spinnerInterval = setInterval(() => {
            const frame = spinnerFrames[currentFrame = (currentFrame + 1) % spinnerFrames.length];
            statusDiv.innerText = `${frame} ${baseMessage}`;
        }, 100);
        statusDiv.style.backgroundColor = bg;
        statusDiv.style.color           = fg;
        statusDiv.style.padding         = '0.5em';
        statusDiv.style.borderRadius    = '0.25em';
    }
    
    function stopSpinner() {
        if (spinnerInterval) {
            clearInterval(spinnerInterval);
            spinnerInterval = null;
        }
    }
    
    const showStatus = (message, type = 'info', showSpinner = false) => {
        const [bg, fg] = bsColors[type] || bsColors.info;
        if (showSpinner) {
            startSpinner(message, bg, fg);
        } else {
            stopSpinner();
            statusDiv.innerHTML       = message;
            statusDiv.style.backgroundColor = bg;
            statusDiv.style.color           = fg;
            statusDiv.style.padding         = '0.5em';
            statusDiv.style.borderRadius    = '0.25em';
        }
    };
    
    // Format file size for display
    function formatFileSize(bytes) {
        if (bytes === 0) return '0 Bytes';
        const k = 1024;
        const sizes = ['Bytes', 'KB', 'MB', 'GB'];
        const i = Math.floor(Math.log(bytes) / Math.log(k));
        return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
    }
    
    // Check if file has supported extension
    function isSupportedFile(filename) {
        const lowerName = filename.toLowerCase();
        return SUPPORTED_EXTENSIONS.some(ext => lowerName.endsWith(ext));
    }
    
    // Get file type from extension
    function getFileType(filename) {
        const lowerName = filename.toLowerCase();
        if (lowerName.endsWith('.nc4')) return 'NC4';
        return 'Unknown';
    }
    
    // Calculate total size of selected files
    function getTotalSize() {
        return selectedFiles.reduce((total, file) => total + file.size, 0);
    }
    
    // Validate file size
    function validateFileSize(file) {
        return file.size <= MAX_FILE_SIZE;
    }
    
    // Validate total size
    function validateTotalSize(newFiles = []) {
        const currentTotal = getTotalSize();
        const newFilesSize = newFiles.reduce((total, file) => total + file.size, 0);
        return (currentTotal + newFilesSize) <= MAX_TOTAL_SIZE;
    }
    
    // Initial check
    fetch('/check_models')
        .then(res => res.json())
        .then(data => {
            if (data.unet_loaded && data.svm_loaded) {
                loadModelsBtn.style.display = 'none';
                showStatus('Models are already loaded and ready for processing.', 'info');
            } else {
                showStatus('Models need to be loaded before processing files.', 'warning');
            }
        })
        .catch(err => {
            console.error('Error checking models:', err);
            showStatus('Unable to check model status. Please try loading models.', 'warning');
        });
    
    // Load models click
    loadModelsBtn.addEventListener('click', function () {
        showStatus('Loading AI models (U-Net + SVM)...', 'primary', true);
        fetch('/load_models')
            .then(res => res.json())
            .then(data => {
                stopSpinner();
                const formatted = data.message.replace(/\n/g, '<br>');
                const ok = data.details?.unet?.status === 'success'
                        && data.details?.svm?.status === 'success';
                showStatus(formatted, ok ? 'success' : 'danger');
                if (ok) {
                    loadModelsBtn.style.display = 'none';
                }
            })
            .catch(err => {
                stopSpinner();
                showStatus('Error loading models: ' + err.message, 'danger');
            });
    });
    
    // File input change
    fileInput.addEventListener('change', () => {
        if (fileInput.files.length) {
            const files = Array.from(fileInput.files);
            handleFiles(files);
        }
    });
    
    // Form submit
    uploadForm.addEventListener('submit', function (e) {
        if (!selectedFiles.length) {
            e.preventDefault();
            showStatus('Error: No files selected for processing.', 'danger');
            return;
        }
        
        // Check file size validation
        const oversizedFiles = selectedFiles.filter(file => !validateFileSize(file));
        if (oversizedFiles.length > 0) {
            e.preventDefault();
            showStatus(`Error: Some files exceed the maximum size limit of ${formatFileSize(MAX_FILE_SIZE)}.`, 'danger');
            return;
        }
        
        // Check total size validation
        const totalSize = getTotalSize();
        if (totalSize > MAX_TOTAL_SIZE) {
            e.preventDefault();
            showStatus(`Error: Total file size (${formatFileSize(totalSize)}) exceeds the maximum limit of ${formatFileSize(MAX_TOTAL_SIZE)}.`, 'danger');
            return;
        }
        
        // Check if all files are supported
        const unsupportedFiles = selectedFiles.filter(f => !isSupportedFile(f.name));
        if (unsupportedFiles.length > 0) {
            e.preventDefault();
            const fileList = unsupportedFiles.map(f => f.name).join(', ');
            showStatus(`Error: Unsupported file types detected: ${fileList}<br>Only .nc4 files are supported.`, 'danger');
            return;
        }
        
        // Show processing status
        const fileTypes = [...new Set(selectedFiles.map(f => getFileType(f.name)))];
        const typeText = fileTypes.join(' and ');
        showStatus(`Processing ${selectedFiles.length} ${typeText} file(s), please wait...`, 'warning', true);
    });
    
    // Drag & drop events
    dropzone.addEventListener('dragover', e => {
        e.preventDefault();
        dropzone.style.border     = '2px dashed #0d6efd';
        dropzone.style.background = '#e7f1ff';
    });
    
    dropzone.addEventListener('dragleave', () => {
        dropzone.style.border     = '';
        dropzone.style.background = '';
    });
    
    dropzone.addEventListener('drop', function (e) {
        e.preventDefault();
        dropzone.style.border     = '';
        dropzone.style.background = '';
        if (e.dataTransfer.files.length) {
            const files = Array.from(e.dataTransfer.files);
            handleFiles(files);
        }
    });
    
    dropzone.addEventListener('click', e => {
        if (e.target === dropzone) fileInput.click();
    });
    
    // Handle selected files with size validation
    function handleFiles(files) {
        const validFiles = [];
        const invalidFiles = [];
        
        // Filter out duplicates first
        const newFiles = files.filter(file => 
            !selectedFiles.find(f => f.name === file.name && f.size === file.size)
        );
        
        newFiles.forEach(file => {
            if (isSupportedFile(file.name)) {
                if (validateFileSize(file)) {
                    validFiles.push(file);
                } else {
                    invalidFiles.push({
                        file: file,
                        error: `File too large (${formatFileSize(file.size)}). Maximum allowed: ${formatFileSize(MAX_FILE_SIZE)}`
                    });
                }
            } else {
                invalidFiles.push({
                    file: file,
                    error: `Invalid file type. Only ${SUPPORTED_EXTENSIONS.join(', ')} files are supported.`
                });
            }
        });
        
        // Check if adding valid files would exceed total size limit
        if (validFiles.length > 0 && !validateTotalSize(validFiles)) {
            const currentTotal = getTotalSize();
            const newFilesSize = validFiles.reduce((total, file) => total + file.size, 0);
            const wouldBeTotal = currentTotal + newFilesSize;
            
            showStatus(
                `Error: Adding these files would exceed the total size limit.<br>` +
                `Current total: ${formatFileSize(currentTotal)}<br>` +
                `New files: ${formatFileSize(newFilesSize)}<br>` +
                `Would be: ${formatFileSize(wouldBeTotal)}<br>` +
                `Maximum allowed: ${formatFileSize(MAX_TOTAL_SIZE)}`,
                'danger'
            );
            return;
        }
        
        // Add valid files to selection
        selectedFiles = selectedFiles.concat(validFiles);
        
        // Show errors for invalid files
        if (invalidFiles.length > 0) {
            const errorMessages = invalidFiles.map(item => `${item.file.name}: ${item.error}`).join('<br>');
            showStatus(`File validation errors:<br>${errorMessages}`, 'danger');
        } else if (validFiles.length > 0) {
            const addedTypes = [...new Set(validFiles.map(f => getFileType(f.name)))];
            const typeText = addedTypes.join(' and ');
            showStatus(`Added ${validFiles.length} ${typeText} file(s) successfully.`, 'success');
            
            // Clear status after 3 seconds if no errors
            setTimeout(() => {
                if (selectedFiles.length > 0) {
                    stopSpinner();
                    statusDiv.innerHTML = '';
                    statusDiv.style.backgroundColor = '';
                    statusDiv.style.color = '';
                }
            }, 3000);
        }
        
        updateFileList();
        updateFormFiles();
    }
    
    // Remove file from selection
    function removeFile(index) {
        const removedFile = selectedFiles[index];
        selectedFiles.splice(index, 1);
        updateFileList();
        updateFormFiles();
        
        if (selectedFiles.length === 0) {
            stopSpinner();
            statusDiv.innerHTML = '';
            statusDiv.style.backgroundColor = '';
            statusDiv.style.color = '';
        } else {
            showStatus(`Removed ${removedFile.name} from selection.`, 'info');
            setTimeout(() => {
                stopSpinner();
                statusDiv.innerHTML = '';
                statusDiv.style.backgroundColor = '';
                statusDiv.style.color = '';
            }, 2000);
        }
    }
    
    // Update form with selected files
    function updateFormFiles() {
        const dt = new DataTransfer();
        selectedFiles.forEach(file => {
            dt.items.add(file);
        });
        fileInput.files = dt.files;
    }
    
    // Update file list display with cancel buttons and total size
    function updateFileList() {
        fileList.innerHTML = '';
        
        if (!selectedFiles.length) return;
        
        const container = document.createElement('div');
        container.style.marginTop = '1rem';
        
        // Header with file count, types, and total size
        const header = document.createElement('div');
        header.style.cssText = `
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 1rem;
            padding: 0.75rem;
            background: #f1f5f9;
            border: 1px solid #cbd5e1;
            border-radius: 0.5rem;
        `;
        
        const totalSize = getTotalSize();
        const totalSizeColor = totalSize > MAX_TOTAL_SIZE ? '#dc2626' : '#059669';
        const totalSizeWeight = totalSize > MAX_TOTAL_SIZE ? 'bold' : 'normal';
        
        // Get file type summary
        const fileTypes = selectedFiles.reduce((acc, file) => {
            const type = getFileType(file.name);
            acc[type] = (acc[type] || 0) + 1;
            return acc;
        }, {});
        
        const typeSummary = Object.entries(fileTypes)
            .map(([type, count]) => `${count} ${type}`)
            .join(', ');
        
        header.innerHTML = `
            <div>
                <h4 style="margin: 0; color: #1e293b;">Selected Files (${selectedFiles.length})</h4>
                <div style="font-size: 0.85rem; color: #6b7280; margin-top: 0.25rem;">${typeSummary}</div>
            </div>
            <div style="text-align: right;">
                <div style="font-size: 0.9rem; color: #6b7280; margin-bottom: 0.25rem;">Total Size:</div>
                <div style="font-size: 1.1rem; font-weight: ${totalSizeWeight}; color: ${totalSizeColor};">
                    ${formatFileSize(totalSize)} / ${formatFileSize(MAX_TOTAL_SIZE)}
                </div>
            </div>
        `;
        container.appendChild(header);
        
        // Individual file items
        selectedFiles.forEach((file, index) => {
            const fileItem = document.createElement('div');
            fileItem.style.cssText = `
                display: flex;
                justify-content: space-between;
                align-items: center;
                padding: 0.75rem;
                background: #f8fafc;
                border: 1px solid #e2e8f0;
                border-radius: 0.375rem;
                margin-bottom: 0.5rem;
                transition: background-color 0.2s;
            `;
            
            fileItem.addEventListener('mouseenter', () => {
                fileItem.style.backgroundColor = '#f1f5f9';
            });
            
            fileItem.addEventListener('mouseleave', () => {
                fileItem.style.backgroundColor = '#f8fafc';
            });
            
            const fileInfo = document.createElement('div');
            const fileType = getFileType(file.name);
            const typeColor = fileType === 'NC4' ? '#059669' : ''; 
            
            fileInfo.innerHTML = `
                <div style="display: flex; align-items: center; margin-bottom: 0.25rem;">
                    <span style="font-weight: 500; color: #1e293b; margin-right: 0.5rem;">${file.name}</span>
                    <span style="
                        background: ${typeColor}; 
                        color: white; 
                        padding: 0.125rem 0.375rem; 
                        border-radius: 0.25rem; 
                        font-size: 0.7rem; 
                        font-weight: 500;
                    ">${fileType}</span>
                </div>
                <div style="font-size: 0.8rem; color: #6b7280;">${formatFileSize(file.size)}</div>
            `;
            
            const removeBtn = document.createElement('button');
            removeBtn.type = 'button';
            removeBtn.innerHTML = '✕';
            removeBtn.title = `Remove ${file.name}`;
            removeBtn.style.cssText = `
                background: #dc2626;
                color: white;
                border: none;
                width: 28px;
                height: 28px;
                border-radius: 50%;
                cursor: pointer;
                font-size: 0.9rem;
                display: flex;
                align-items: center;
                justify-content: center;
                transition: all 0.2s;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            `;
            
            removeBtn.addEventListener('mouseenter', () => {
                removeBtn.style.backgroundColor = '#b91c1c';
                removeBtn.style.transform = 'scale(1.1)';
            });
            
            removeBtn.addEventListener('mouseleave', () => {
                removeBtn.style.backgroundColor = '#dc2626';
                removeBtn.style.transform = 'scale(1)';
            });
            
            removeBtn.addEventListener('click', () => removeFile(index));
            
            fileItem.appendChild(fileInfo);
            fileItem.appendChild(removeBtn);
            container.appendChild(fileItem);
        });
        
        // Add file format info footer
        const footer = document.createElement('div');
        footer.style.cssText = `
            margin-top: 1rem;
            padding: 0.5rem;
            background: #f8fafc;
            border: 1px solid #e2e8f0;
            border-radius: 0.375rem;
            font-size: 0.8rem;
            color: #6b7280;
            text-align: center;
        `;
        footer.innerHTML = `
            <strong>Supported formats:</strong> 
            <span style="color: #059669;">NC4</span> (NetCDF4 hyperspectral data)  
        `;
        container.appendChild(footer);
        
        fileList.appendChild(container);
    }
    
    // Update dropzone text to reflect supported formats
    const dropzoneText = dropzone.querySelector('p');
    if (dropzoneText) {
        dropzoneText.innerHTML = `
            <strong>Drop your hyperspectral files here</strong><br>
            <span style="font-size: 0.9rem; color: #6b7280;">
                Supported formats: .nc4 (NetCDF4)<br>
                Maximum file size: ${formatFileSize(MAX_FILE_SIZE)} per file<br>
                Maximum total size: ${formatFileSize(MAX_TOTAL_SIZE)}
            </span>
        `;
    }
    
    // Update file input accept attribute
    fileInput.setAttribute('accept', '.nc4');
    
    // Add keyboard support for file removal
    document.addEventListener('keydown', function(e) {
        if (e.key === 'Escape' && selectedFiles.length > 0) {
            if (confirm(`Remove all ${selectedFiles.length} selected files?`)) {
                selectedFiles = [];
                updateFileList();
                updateFormFiles();
                stopSpinner();
                statusDiv.innerHTML = '';
                statusDiv.style.backgroundColor = '';
                statusDiv.style.color = '';
                showStatus('All files removed.', 'info');
                setTimeout(() => {
                    stopSpinner();
                    statusDiv.innerHTML = '';
                    statusDiv.style.backgroundColor = '';
                    statusDiv.style.color = '';
                }, 2000);
            }
        }
    });
    
    // Add file statistics display
    function updateFileStats() {
        const statsDiv = document.getElementById('file-stats');
        if (statsDiv && selectedFiles.length > 0) {
            const stats = selectedFiles.reduce((acc, file) => {
                const type = getFileType(file.name);
                acc[type] = acc[type] || { count: 0, size: 0 };
                acc[type].count++;
                acc[type].size += file.size;
                return acc;
            }, {});
            
            const statsHtml = Object.entries(stats)
                .map(([type, data]) => 
                    `${type}: ${data.count} files (${formatFileSize(data.size)})`
                )
                .join(' • ');
            
            statsDiv.innerHTML = `<small class="text-muted">${statsHtml}</small>`;
        }
    }
    
    // Enhanced error handling for network issues
    function handleNetworkError(error, operation) {
        console.error(`Network error during ${operation}:`, error);
        if (error.name === 'TypeError' && error.message.includes('fetch')) {
            showStatus(`Network error: Unable to connect to server. Please check your connection and try again.`, 'danger');
        } else {
            showStatus(`Error during ${operation}: ${error.message}`, 'danger');
        }
    }
    
    // Update all fetch calls to use enhanced error handling
    const originalFetch = window.fetch;
    window.fetch = function(...args) {
        return originalFetch.apply(this, args)
            .catch(error => {
                if (error.name === 'TypeError') {
                    throw new Error('Network connection failed. Please check your internet connection.');
                }
                throw error;
            });
    };
    
    // Make removeFile function globally accessible
    window.removeFile = removeFile;
    
    // Add file validation summary
    window.getFileValidationSummary = function() {
        return {
            totalFiles: selectedFiles.length,
            totalSize: getTotalSize(),
            maxFileSize: MAX_FILE_SIZE,
            maxTotalSize: MAX_TOTAL_SIZE,
            supportedExtensions: SUPPORTED_EXTENSIONS,
            fileTypes: selectedFiles.reduce((acc, file) => {
                const type = getFileType(file.name);
                acc[type] = (acc[type] || 0) + 1;
                return acc;
            }, {}),
            isValid: selectedFiles.length > 0 && 
                     selectedFiles.every(f => validateFileSize(f) && isSupportedFile(f.name)) &&
                     getTotalSize() <= MAX_TOTAL_SIZE
        };
    };
    
    // Add progress tracking for large file uploads
    let uploadProgress = 0;
    
    function updateUploadProgress(percent) {
        uploadProgress = percent;
        if (percent < 100) {
            showStatus(`Uploading files... ${Math.round(percent)}%`, 'primary', true);
        }
    }
    
    // Enhanced form submission with progress tracking
    const originalSubmit = uploadForm.addEventListener;
    uploadForm.addEventListener('submit', function(e) {
        // ... existing validation code ...
        
        if (selectedFiles.length > 0) {
            // Add upload progress simulation for large files
            const totalSize = getTotalSize();
            if (totalSize > 10 * 1024 * 1024) { // 10MB+
                let progress = 0;
                const progressInterval = setInterval(() => {
                    progress += Math.random() * 15;
                    if (progress >= 95) {
                        clearInterval(progressInterval);
                        showStatus('Processing files on server...', 'warning', true);
                    } else {
                        updateUploadProgress(progress);
                    }
                }, 500);
            }
        }
    });
    
    console.log('Hyperspectral file processor initialized');
    console.log(`Supported file types: ${SUPPORTED_EXTENSIONS.join(', ')}`);
    console.log(`Max file size: ${formatFileSize(MAX_FILE_SIZE)}`);
    console.log(`Max total size: ${formatFileSize(MAX_TOTAL_SIZE)}`);
});

