

import streamlit as st
import torch
import torch.nn.functional as F
import timm
from torchvision import transforms
from PIL import Image
import numpy as np
import cv2
from insightface.app import FaceAnalysis
import matplotlib.cm as cm


@st.cache_resource
def load_xception_model(ckpt_path):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Create model
    model = timm.create_model("xception", pretrained=False, num_classes=2)

    # Load checkpoint
    ckpt = torch.load(ckpt_path, map_location=device)
    if "state_dict" in ckpt:
        ckpt = ckpt["state_dict"]

    # Fix any 2D conv weights (reshape to 4D)
    fixed_ckpt = {}
    for k, v in ckpt.items():
        if isinstance(v, torch.Tensor):
            if v.ndim == 2:
                v = v.unsqueeze(-1).unsqueeze(-1)
            fixed_ckpt[k] = v

    # Remove classifier mismatches
    for key in list(fixed_ckpt.keys()):
        if "fc" in key or "classifier" in key:
            del fixed_ckpt[key]

    # Load weights safely
    missing, unexpected = model.load_state_dict(fixed_ckpt, strict=False)
    print(" Missing keys:", missing)
    print("⚠️ Unexpected keys:", unexpected)
    print(" Model loaded successfully!\n")

    model.to(device)
    model.eval()

    transform = transforms.Compose([
        transforms.Resize((299, 299)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5],
                             std=[0.5, 0.5, 0.5])
    ])

    return model, transform, device


# ===============================
# 2️⃣ Deepfake Prediction (Xception)
# ===============================
def predict_deepfake_xception(img_pil, model, transform, device):
    img_tensor = transform(img_pil).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(img_tensor)
        probs = F.softmax(logits, dim=1)
        real_prob = probs[0, 0].item()
        fake_prob = probs[0, 1].item()
    return {"real_prob": real_prob, "fake_prob": fake_prob}


# ===============================
# 3️⃣ Simple Attention-like Heatmap (using activations)
# ===============================
def generate_heatmap(model, img_pil, transform, device):
    """
    Generates an activation heatmap using the last convolutional feature map.
    """
    model.eval()
    img_tensor = transform(img_pil).unsqueeze(0).to(device)

    # Hook to capture last conv activations
    activation = {}
    def hook_fn(module, input, output):
        activation["feat"] = output.detach()

    last_conv = None
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Conv2d):
            last_conv = module
    hook_handle = last_conv.register_forward_hook(hook_fn)

    with torch.no_grad():
        _ = model(img_tensor)

    hook_handle.remove()

    feat = activation["feat"].squeeze(0).mean(0).cpu().numpy()
    feat = (feat - feat.min()) / (feat.max() - feat.min() + 1e-8)
    heatmap = cv2.resize(feat, img_pil.size)
    cmap = cm.get_cmap("jet")
    colored_map = (cmap(heatmap)[:, :, :3] * 255).astype(np.uint8)

    img_np = np.array(img_pil).astype(np.float32)
    overlay = (0.6 * img_np + 0.4 * colored_map).astype(np.uint8)
    return Image.fromarray(overlay)


# ===============================
# 4️⃣ Face Similarity using InsightFace
# ===============================
@st.cache_resource
def init_insightface():
    app = FaceAnalysis(name='buffalo_l')
    app.prepare(ctx_id=0, det_size=(224, 224))
    return app

def face_similarity(img1, img2, app):
    img1 = cv2.cvtColor(np.array(img1), cv2.COLOR_RGB2BGR)
    img2 = cv2.cvtColor(np.array(img2), cv2.COLOR_RGB2BGR)

    faces1 = app.get(img1)
    faces2 = app.get(img2)

    if not faces1 or not faces2:
        return None

    emb1 = faces1[0].embedding
    emb2 = faces2[0].embedding

    sim = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
    return float(sim)


# ===============================
# 5️⃣ Streamlit App
# ===============================
st.title("Deepfake Detection + Face Verification ")

# Load model and face engine
MODEL_PATH = "/Users/vikram/Downloads/xception-b5690688.pth"  # ✅ Update this path
model, val_transform, device = load_xception_model(MODEL_PATH)
app = init_insightface()

# Step 1: Upload original image
st.header("Step 1: Deepfake Detection")
orig_file = st.file_uploader("Upload Original Image", type=["jpg", "png"])
if orig_file:
    orig_img = Image.open(orig_file).convert("RGB")
    st.image(orig_img, caption="Original Image", use_container_width=True)

    # Deepfake prediction
    probs = predict_deepfake_xception(orig_img, model, val_transform, device)
    st.write(f"Real probability: {probs['real_prob']:.3f}")
    st.write(f"Fake probability: {probs['fake_prob']:.3f}")

    # Visualization
    st.subheader("Feature Activation Heatmap")
    try:
        heatmap_img = generate_heatmap(model, orig_img, val_transform, device)
        st.image(heatmap_img, caption="Activation Heatmap", use_container_width=True)
    except Exception as e:
        st.warning(f"⚠️ Heatmap generation failed: {e}")

    if probs["fake_prob"] > 0.5:
        st.error(" Image is likely Fake! Pipeline stopped.")
    else:
        st.success("✔️ Image is Real! Proceed to Step 2.")

        # Step 2: Face Verification
        st.header("Step 2: Face Verification")
        option = st.radio("Select method for second image:", ("Upload Image", "Capture from Webcam"))

        if option == "Upload Image":
            second_file = st.file_uploader("Upload Second Image", type=["jpg", "png"], key="second")
            if second_file:
                second_img = Image.open(second_file).convert("RGB")
                st.image(second_img, caption="Second Image", use_container_width=True)

                sim_score = face_similarity(orig_img, second_img, app)
                if sim_score is not None:
                    threshold = 0.7
                    st.write(f"🔹 Face similarity score: {sim_score:.3f}")
                    if sim_score > threshold:
                        st.success("✔️ Faces match! Verification success.")
                    else:
                        st.warning("⚠️ Faces do NOT match! Verification failed.")

        else:
            st.info("⚙️ Click 'Capture from Webcam' to take a picture.")
            if st.button("Capture from Webcam"):
                cap = cv2.VideoCapture(0)
                ret, frame = cap.read()
                cap.release()
                if ret:
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    webcam_img = Image.fromarray(frame_rgb)
                    st.image(webcam_img, caption="Captured Image", use_container_width=True)

                    sim_score = face_similarity(orig_img, webcam_img, app)
                    if sim_score is not None:
                        threshold = 0.7
                        st.write(f"🔹 Face similarity score: {sim_score:.3f}")
                        if sim_score > threshold:
                            st.success("✔️ Faces match! Verification success.")
                        else:
                            st.warning("⚠️ Faces do NOT match! Verification failed.")
                else:
                    st.error(" Unable to capture from webcam.")
