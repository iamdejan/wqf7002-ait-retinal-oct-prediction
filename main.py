import streamlit as st
import torch
from torch import nn
from torch.autograd import Variable
from torchvision import models, transforms
from PIL import Image


def get_device():
    if torch.cuda.is_available():
        device = torch.device("cuda")  # NVIDIA GPU
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")  # Apple Silicon (Metal)
    else:
        device = torch.device("cpu")  # Fallback
    return device


@st.cache_resource
def load_model():
    vgg16 = models.vgg16_bn()
    vgg16.classifier[-1] = nn.Linear(4096, 4)
    model_path = 'model/VGG16_OCT_Retina_trained_model.pt'
    vgg16.load_state_dict(torch.load(model_path))
    vgg16.eval()
    vgg16 = vgg16.to(get_device())
    return vgg16


def predict(uploaded_file):
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="Uploaded image")

    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
    ])
    img = transform(img)
    img = Variable(img.unsqueeze(0)).to(get_device())

    vgg16 = load_model()
    with torch.no_grad():
        preds_new_model = vgg16(img)

    _, predicted_class_new_model = torch.max(preds_new_model, 1)
    result_new_model = int(predicted_class_new_model)
    st.write(result_new_model)


def main():
    st.title("OCT Retina Program")
    st.text("This is a simple program to upload OCT Retina image, then detect the disease")
    uploaded_file = st.file_uploader("Choose a file")
    if uploaded_file is not None:
        predict(uploaded_file)


if __name__ == "__main__":
    main()
