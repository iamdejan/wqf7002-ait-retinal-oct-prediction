import streamlit as st
import torch
from torch import nn
from torch.autograd import Variable
from torchvision import models, transforms
from PIL import Image


def get_device():
    # Run everything in CPU, since GPU is not supported by Streamlit.
    return torch.device("cpu")


@st.cache_resource
def load_model():
    vgg16 = models.vgg16_bn().to(get_device())
    vgg16.classifier[-1] = nn.Linear(4096, 4)
    model_path = 'model/VGG16_OCT_Retina_trained_model.pt'
    vgg16.load_state_dict(torch.load(model_path, map_location=get_device()))
    vgg16.eval()
    return vgg16


def load_image(uploaded_file) -> Image:
    return Image.open(uploaded_file).convert("RGB")


def predict(img: Image) -> int:
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
    return int(predicted_class_new_model)


def main():
    st.title("OCT Retina Program")
    st.text("This is a simple program to upload OCT Retina image, then detect the disease")
    uploaded_file = st.file_uploader("Choose a file")
    if uploaded_file is not None:
        img = load_image(uploaded_file)
        st.image(img, caption="Uploaded image")
        result = predict(img)
        st.write(result)


if __name__ == "__main__":
    main()
