from flask import Flask, request, render_template
import tensorflow as tf
import numpy as np
from PIL import Image
import os
import uuid

app = Flask(__name__)

# Load trained model
MODEL_PATH = 'skin_cancer_model_finetuned.keras'
model = tf.keras.models.load_model(MODEL_PATH)

# Define image size and class labels
IMG_SIZE = (224, 224)
CLASS_NAMES = ['akiec', 'bcc', 'bkl', 'df', 'mel', 'nv', 'vasc']  # should match train/test folder names

# Static folder for image uploads
UPLOAD_FOLDER = 'static/uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def prepare_image(img_path):
    img = Image.open(img_path).convert('RGB')
    img = img.resize(IMG_SIZE)
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        image_file = request.files['image']
        if image_file:
            # Save the uploaded file
            unique_filename = str(uuid.uuid4()) + "_" + image_file.filename
            save_path = os.path.join(UPLOAD_FOLDER, unique_filename)
            image_file.save(save_path)

            try:
                # Preprocess and predict
                processed_image = prepare_image(save_path)
                prediction = model.predict(processed_image)[0]
                predicted_class = CLASS_NAMES[np.argmax(prediction)]
                confidence = round(100 * np.max(prediction), 2)

                # Render result
                result_html = render_template('result.html',
                                              prediction=predicted_class.upper(),
                                              confidence=confidence,
                                              img_path=save_path)

            finally:
                # Always try to delete the uploaded image file
                try:
                    os.remove(save_path)
                except Exception as e:
                    print(f"⚠️ Could not delete image file: {e}")

            return result_html

    return render_template('index.html')

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 10000))
    app.run(host='0.0.0.0', port=port, debug=False)
