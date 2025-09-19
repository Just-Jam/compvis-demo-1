from pathlib import Path

import streamlit as st
import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader
from torchvision.models import resnet18
from torchvision import transforms
import random
from utils import denormalize, CsvImageDataset

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# For display & mapping
CLASS_NAMES = ["no pedestrian", "pedestrian"]   # index 0 -> no pedestrian, 1 -> pedestrian
CLASS_TO_IDX = {name: i for i, name in enumerate(CLASS_NAMES)}
IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp"}
VAL_DIR   = Path("data/validation")
VAL_CSV   = VAL_DIR / "labels.csv"

DATALOADER_KW = dict(
    num_workers=0,          # <— main fix: avoid background workers in notebooks
    pin_memory=False,       # safe default; you can set True if using CUDA
    persistent_workers=False
)

def show_class_examples_native(model, val_loader, class_names, samples_per_class=3):
    """
    Streamlit-native version without matplotlib dependencies
    """
    model.eval()
    collected = {cls: [] for cls in range(len(class_names))}

    # Collect examples
    all_examples = []
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            preds = outputs.argmax(dim=1)
            for i in range(len(labels)):
                all_examples.append((images[i], labels[i], preds[i]))

    random.shuffle(all_examples)

    for img, label, pred in all_examples:
        cls = label.item()
        if len(collected[cls]) < samples_per_class:
            collected[cls].append((img, label, pred))
        if all(len(v) >= samples_per_class for v in collected.values()):
            break

    # Streamlit display
    st.subheader("🧪 Model Performance Examples")

    for cls, examples in collected.items():
        class_name = class_names[cls] if cls < len(class_names) else f"Class {cls}"

        with st.expander(f"Class: {class_name}", expanded=True):
            cols = st.columns(samples_per_class)

            for col_idx, (img, label, pred) in enumerate(examples):
                with cols[col_idx]:
                    pil_img = denormalize(img)

                    # Display image
                    st.image(pil_img, use_container_width=True)

                    # Prediction info
                    true_label = class_names[label.item()] if label.item() < len(
                        class_names) else f"Class {label.item()}"
                    pred_label = class_names[pred.item()] if pred.item() < len(class_names) else f"Class {pred.item()}"

                    is_correct = pred == label
                    status = "✅ Correct" if is_correct else "❌ Wrong"

                    st.caption(status)
                    st.write(f"**True:** {true_label}")
                    st.write(f"**Predicted:** {pred_label}")

                    # Show confidence
                    with torch.no_grad():
                        output = model(img.unsqueeze(0))
                        confidence = torch.nn.functional.softmax(output[0], dim=0)[pred].item()

                    st.write(f"**Confidence:** {confidence:.2%}")

def main():
    # Set page configuration
    st.set_page_config(
        page_title="Image Classification Demo",
        page_icon="🖼️",
        layout="wide"
    )

    # Title and description
    st.title("🖼️ Image Classification Demo")

    # Load pre-trained model (using ResNet as default)
    @st.cache_resource
    def load_model(path):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        checkpoint = torch.load(
            path,
            map_location=device  # load to CPU first, we'll move to GPU if available
        )

        # Recreate same model architecture
        model = resnet18(weights=None)  # no pretrained weights here, we only load your trained weights
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, 2)
        model.load_state_dict(checkpoint["model"])
        model = model.to(DEVICE)
        return model, checkpoint

    resnet_model, checkpoint = load_model('models/resnet18_pedestrian.pt')

    # 2. Reuse the class names (stored in checkpoint or from your earlier variable)
    CLASS_NAMES = checkpoint.get("class_names", ["no pedestrian", "pedestrian"])

    # Load validation data
    val_tfms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406),
                             std=(0.229, 0.224, 0.225)),
    ])
    val_ds = CsvImageDataset(VAL_DIR, VAL_CSV, transform=val_tfms)
    BATCH_SIZE = 32
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, **DATALOADER_KW)

    # 3. Show predictions
    # show_class_examples_grid(resnet_model, val_loader, CLASS_NAMES, samples_per_class=4)
    show_class_examples_native(resnet_model, val_loader, CLASS_NAMES, samples_per_class=4)
    st.markdown("---")

    # Display results
    col1, col2, col3 = st.columns(3)
    with col1:
        # File uploader
        st.write("Upload an image and let our AI model classify it!")
        uploaded_file = st.file_uploader(
            "Choose an image...",
            type=IMG_EXT
        )
        predict_btn = None
        image = None

    if uploaded_file:
        predict_btn = st.button("Predict Image")
        if predict_btn:
            # Load and preprocess image
            image = Image.open(uploaded_file).convert("RGB")
            preprocess = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.485, 0.456, 0.406),
                                     std=(0.229, 0.224, 0.225)),
            ])
            input_tensor = preprocess(image).unsqueeze(0).to(DEVICE)

            # Load model if not already loaded
            if 'resnet_model' not in locals():
                resnet_model, _ = load_model('models/resnet18_pedestrian.pt')

            # Make prediction
            with torch.no_grad():
                output = resnet_model(input_tensor)
                probabilities = torch.nn.functional.softmax(output[0], dim=0)
                predicted_class = probabilities.argmax().item()
                confidence = probabilities[predicted_class].item()

    if uploaded_file and predict_btn:
        with col2:
                st.subheader("Uploaded Image")
                st.image(image, caption='Uploaded Image', use_container_width=True)
        with col3:
            st.subheader("Prediction")
            st.write(f"**Predicted Class:** {CLASS_NAMES[predicted_class]}")
            st.write(f"**Confidence:** {confidence:.2%}")




if __name__ == '__main__':
    main()