import streamlit as st
import torch
from torchvision import transforms
from PIL import Image
import pickle
import io
import torch.nn as nn
from torchvision import models


class FashionLabelEncoder:
    def __init__(self):
        self.encoders = {
            "color": LabelEncoder(),
            "type": LabelEncoder(),
            "season": LabelEncoder(),
            "gender": LabelEncoder()
        }
        self.fitted = False
        
    def fit(self, dataset):
        
        if self.fitted:
            return
        
        label_collections = {
            "color": [],
            "type": [],
            "season": [],
            "gender": []
        }
        
        
        print("Fitting label encoders...")
        for i in tqdm(range(len(dataset))):
            _, labels = dataset[i]
            for category in label_collections:
                label_collections[category].append(labels[category])
        
        
        for category, values in label_collections.items():
            self.encoders[category].fit(values)
            
        self.fitted = True
        
        
        for category, encoder in self.encoders.items():
            print(f"{category} classes: {len(encoder.classes_)}")
            print(f"First 10 {category} mappings: {dict(zip(encoder.classes_[:10], range(10)))}")
    
    def encode(self, labels):
        
        if not self.fitted:
            raise ValueError("Encoder must be fitted before encoding")
            
        encoded = {}
        for category, value in labels.items():
            try:
                encoded[category] = self.encoders[category].transform([value])[0]
            except ValueError:
                
                encoded[category] = -1
                
        return encoded
    
    def decode(self, encoded_labels):
        
        if not self.fitted:
            raise ValueError("Encoder must be fitted before decoding")
            
        decoded = {}
        for category, value in encoded_labels.items():
            if value == -1:
                decoded[category] = "unknown"
            else:
                decoded[category] = self.encoders[category].inverse_transform([value])[0]
                
        return decoded
    
    def get_num_classes(self):
        
        return {category: len(encoder.classes_) for category, encoder in self.encoders.items()}
    
class FashionClassifier(nn.Module):
    def __init__(self, num_classes_dict, backbone="resnet50", pretrained=True):
        super(FashionClassifier, self).__init__()
        
        
        if backbone == "resnet18":
            base_model = models.resnet18(pretrained=pretrained)
        elif backbone == "resnet34":
            base_model = models.resnet34(pretrained=pretrained)
        elif backbone == "resnet50":
            base_model = models.resnet50(pretrained=pretrained)
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")
        
        
        self.features = nn.Sequential(*list(base_model.children())[:-1])
        
        
        self.feature_dim = base_model.fc.in_features
        
        
        self.color_classifier = nn.Linear(self.feature_dim, num_classes_dict["color"])
        self.type_classifier = nn.Linear(self.feature_dim, num_classes_dict["type"])
        self.season_classifier = nn.Linear(self.feature_dim, num_classes_dict["season"])
        self.gender_classifier = nn.Linear(self.feature_dim, num_classes_dict["gender"])
        
    def forward(self, x):
        
        features = self.features(x)
        features = torch.flatten(features, 1)
        
        
        color_output = self.color_classifier(features)
        type_output = self.type_classifier(features)
        season_output = self.season_classifier(features)
        gender_output = self.gender_classifier(features)
        
        return {
            "color": color_output,
            "type": type_output,
            "season": season_output,
            "gender": gender_output
        }

def predict_fashion_attributes(model, label_encoder, image_file, device=None):
    if device is None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    model.eval()
    model = model.to(device)
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    image = Image.open(image_file).convert('RGB')
    image_tensor = transform(image).unsqueeze(0).to(device)
    
    with torch.no_grad():
        outputs = model(image_tensor)
    
    predicted_indices = {}
    for category in ["color", "type", "season", "gender"]:
        _, pred_idx = torch.max(outputs[category], 1)
        predicted_indices[category] = pred_idx.item()
    
    predictions = {}
    for category, idx in predicted_indices.items():
        try:
            predictions[category] = label_encoder.encoders[category].classes_[idx]
        except IndexError:
            predictions[category] = "unknown"
    
    return predictions


def load_model():
    checkpoint = torch.load('./fashion_classifier_model_continued.pth', map_location=torch.device('cpu'))
    num_classes = checkpoint['num_classes']
    
    model = FashionClassifier(num_classes)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    with open('./label_encoder.pkl', 'rb') as f:
        label_encoder = pickle.load(f)
    
    return model, label_encoder


st.title("Fashion Attribute Predictor")
st.write("Drag and drop an image file below to predict fashion attributes.")

uploaded_file = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    st.image(uploaded_file, caption='Uploaded Image', use_column_width=True)
    
    if 'model' not in st.session_state:
        with st.spinner("Loading model..."):
            model, label_encoder = load_model()
            st.session_state['model'] = model
            st.session_state['label_encoder'] = label_encoder
    
    if st.button("Predict Fashion Attributes"):
        with st.spinner("Predicting..."):
            predictions = predict_fashion_attributes(
                st.session_state['model'],
                st.session_state['label_encoder'],
                uploaded_file
            )
        
        st.subheader("Prediction Results:")
        st.write(f"**Color:** {predictions['color']}")
        st.write(f"**Type:** {predictions['type']}")
        st.write(f"**Season:** {predictions['season']}")
        st.write(f"**Gender:** {predictions['gender']}")
