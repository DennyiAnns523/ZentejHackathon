# Deepfake-Proof eKYC System

This system is designed to perform identity verification and forgery detection in real-time. The system leverages SCRFD for preprocessing, fine-tuned Vision Transformer (ViT) for deepfake detection, and InsightFace for identity verification through face embedding similarity. The solution is deployed as a user-friendly Streamlit web application supporting both static image uploads and live webcam capture.

# For testing it on your device:
1. Clone the repository
2. Install all the required packages
3. run app.py

# **Technical Implementation Details**
Dependencies and Libraries:
   
  a) Deep Learning: PyTorch, Transformers (HuggingFace)
  
  b) Face Analysis: InsightFace (buffalo_l model)
  
  c) Image Processing: OpenCV (cv2), PIL
  
  d) UI Framework: Streamlit
  
  e) Computer Vision: NumPy, Pandas
  
  f) Utilities: TQDM for progress tracking

Model Loading and Caching

  a) ViT model weights loaded from fine-tuned checkpoint

  b) Models cached using Streamlit @st.cache_resource decorator
  
  c) GPU/CPU device selection based on availability
  
  d) Models loaded in evaluation mode for inference

Inference Pipeline

  ViT Inference:
  
  a) Input image resized to 224×224
  
  b) Normalize using ImageNet statistics
  
  c) Convert to tensor and move to device
  
  d) Forward pass through model
  
  e) Apply softmax for probability distribution
  
  f) Extract real and fake probabilities

  InsightFace Inference:
  
  a) Convert PIL image to OpenCV format (BGR)
  
  b) Detect faces using buffalo_l detector
  
  c) Extract embeddings for detected faces
  
  d) Compute L2-normalized cosine similarity
  
  e) Return similarity score



