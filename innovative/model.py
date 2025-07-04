import numpy as np
from PIL import Image
from sklearn.preprocessing import LabelEncoder
import joblib
import warnings
from sklearn.exceptions import InconsistentVersionWarning

warnings.filterwarnings("ignore", category=InconsistentVersionWarning)

def process_uploaded_image(uploaded_file):
    image = Image.open(uploaded_file).convert("L")  # Convert to grayscale
    image = image.resize((128, 128))  
    img_array = np.array(image).flatten().reshape(1, -1)
    return img_array/255.0

def analysis(X_test):
    le = LabelEncoder() 
    lkl=['Cyst','Normal','Stone','Tumor']
    y_encoded = le.fit_transform(lkl) 
    loaded_model = joblib.load("stacking_model.pkl")
    probas = loaded_model.predict_proba(X_test)
    top2_indices = np.argsort(probas, axis=1)[:, -2:][:, ::-1]  
    top2_labels =[[str(le.inverse_transform([i])[0]) for i in row] for row in top2_indices]
    return top2_labels[0]
