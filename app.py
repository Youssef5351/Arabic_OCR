import streamlit as st
import numpy as np
from PIL import Image
import time
import utlis
import easyocr
import cv2
import os

# Updated caching decorators
@st.cache_data  # For processed images
def get_image(img):
    """Convert uploaded image to numpy array"""
    try:
        image = Image.open(img)
        return np.asarray(image)
    except Exception as e:
        st.error(f"Error loading image: {e}")
        return None

@st.cache_resource  # For expensive resources like EasyOCR reader
def get_reader_lang(lang='ar'):
    """Initialize EasyOCR reader with caching"""
    try:
        reader = easyocr.Reader([lang])
        st.success(f"✓ EasyOCR initialized for {lang}")
        return reader
    except Exception as e:
        st.error(f"Failed to initialize EasyOCR: {e}")
        return None

@st.cache_data  # For OCR results
def get_result(image, lang='ar'):
    """Perform OCR on image with caching"""
    if image is None:
        return []
    
    reader = get_reader_lang(lang)
    if reader is None:
        return []
    
    try:
        result = reader.readtext(image)
        return result
    except Exception as e:
        st.error(f"OCR processing error: {e}")
        return []

def validate_ocr_result(result):
    """Validate and fix OCR result structure"""
    validated = []
    for item in result:
        if len(item) >= 3:  # Should have (bbox, text, confidence)
            bbox, text, confidence = item[0], item[1], item[2]
            
            # Ensure bbox has proper format
            if isinstance(bbox, list) and len(bbox) >= 4:
                # Convert to list of tuples if needed
                bbox = [(int(point[0]), int(point[1])) for point in bbox]
                validated.append((bbox, text, confidence))
    return validated

def main():
    # Page configuration
    st.set_page_config(
        page_title="Arabic OCR System",
        page_icon="📜",
        layout="wide"
    )
    
    # Title with better styling
    st.markdown("""
        <h1 style='text-align: center; color: #1E3A8A; font-size: 2.5rem; 
                   margin-bottom: 2rem;'>
            📜 Arabic OCR System
        </h1>
    """, unsafe_allow_html=True)
    
    # Sidebar
    side_bar = st.sidebar
    side_bar.title("Settings")
    
    # Language selection
    lang = side_bar.selectbox(
        "Select Language",
        ["ar", "en", "ar,en"],
        help="Select language for OCR (Arabic, English, or both)"
    )
    
    # Confidence threshold
    confidence_threshold = side_bar.slider(
        "Confidence Threshold",
        min_value=0.0,
        max_value=1.0,
        value=0.5,
        step=0.05,
        help="Filter results by confidence score"
    )
    
    # File uploader
    side_bar.markdown("---")
    side_bar.subheader("Upload Image")
    image_file = side_bar.file_uploader(
        'Choose an image file',
        type=['png', 'jpg', 'jpeg', 'bmp'],
        help="Supported formats: PNG, JPG, JPEG, BMP"
    )
    
    # Main content columns
    col1, col2 = st.columns(2)
    
    if image_file is not None:
        # Load and display original image
        with st.spinner("Loading image..."):
            image = get_image(image_file)
        
        if image is not None:
            col1.subheader("Original Image")
            col1.image(image, use_column_width=True)
            
            # Show image info
            with st.expander("Image Details"):
                st.write(f"**Dimensions:** {image.shape[1]} × {image.shape[0]} pixels")
                st.write(f"**Channels:** {image.shape[2] if len(image.shape) > 2 else 1}")
                st.write(f"**Data type:** {image.dtype}")
            
            # OCR button
            start_btn = side_bar.button('🚀 Start OCR', type="primary")
            
            if start_btn:
                try:
                    # Show processing animation
                    with st.spinner("Performing OCR..."):
                        progress_bar = st.progress(0)
                        
                        # Get OCR results
                        t1 = time.time()
                        result = get_result(image, lang)
                        progress_bar.progress(50)
                        
                        # Validate and filter results
                        validated_result = validate_ocr_result(result)
                        
                        # Apply confidence threshold
                        filtered_result = [
                            item for item in validated_result 
                            if item[2] >= confidence_threshold
                        ]
                        
                        progress_bar.progress(75)
                        
                        # Annotate image
                        annoted_image = utlis.annotate_image(image.copy(), filtered_result)
                        progress_bar.progress(90)
                        
                        # Extract text
                        extracted_text = utlis.get_raw_text(filtered_result)
                        
                        progress_bar.progress(100)
                        t2 = time.time()
                    
                    # Display results
                    col2.subheader("OCR Results")
                    
                    if filtered_result:
                        # Show annotated image
                        col2.image(annoted_image, use_column_width=True, 
                                 caption="Text Detection Results")
                        
                        # Show extracted text
                        with st.expander("📝 Extracted Text", expanded=True):
                            st.text_area("", extracted_text, height=200)
                        
                        # Show statistics
                        st.info(f"""
                        **Statistics:**
                        - Detected text regions: {len(filtered_result)}
                        - Average confidence: {sum([r[2] for r in filtered_result])/len(filtered_result):.2%}
                        - Processing time: {t2-t1:.2f} seconds
                        """)
                        
                        # Optional: Show detailed results in a table
                        with st.expander("🔍 Detailed Results"):
                            result_data = []
                            for i, (bbox, text, conf) in enumerate(filtered_result):
                                result_data.append({
                                    "ID": i+1,
                                    "Text": text,
                                    "Confidence": f"{conf:.2%}",
                                    "Position": f"{bbox[0][0]},{bbox[0][1]}"
                                })
                            st.table(result_data)
                    else:
                        col2.warning("No text detected with current confidence threshold.")
                        if validated_result:
                            col2.write(f"Found {len(validated_result)} regions, but all below {confidence_threshold:.0%} confidence")
                
                except Exception as e:
                    st.error(f"OCR Processing Error: {str(e)}")
                    st.code(f"Error details: {repr(e)}")
                    
                    # Debug information
                    with st.expander("Debug Information"):
                        st.write(f"Image shape: {image.shape}")
                        st.write(f"Image type: {type(image)}")
                        if 'result' in locals():
                            st.write(f"OCR result type: {type(result)}")
                            st.write(f"OCR result length: {len(result) if result else 0}")
    
    else:
        # Welcome screen when no image is uploaded
        col1.markdown("""
        ### Welcome to Arabic OCR System
        
        **Instructions:**
        1. Upload an image containing Arabic text
        2. Adjust language and confidence settings
        3. Click 'Start OCR' to extract text
        
        **Supported languages:** Arabic (ar), English (en), or both
        """)
        
        col2.markdown("""
        ### Example Images
        
        For best results:
        - Use clear, high-resolution images
        - Ensure text is horizontally aligned
        - Good lighting, no shadows/glare
        - Text should occupy significant portion of image
        
        **Sample Arabic text:**
        ```
        النص العربي للتعرف البصري على الحروف
        يمكن للنظام قراءة النصوص العربية
        من الصور والوثائق الرقمية
        ```
        """)
        
        # Demo with sample image
        if st.checkbox("Load sample image for testing"):
            # You can add a sample image path here
            sample_path = "sample_arabic.png"  # Update with your sample path
            if os.path.exists(sample_path):
                sample_image = Image.open(sample_path)
                col2.image(sample_image, caption="Sample Arabic Text", use_column_width=True)

if __name__ == "__main__":
    main()
