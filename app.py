from flask import Flask, request, render_template
import numpy as np
from PIL import Image
import tensorflow as tf
import os
import uuid

app = Flask(__name__)

# Load TFLite model
TFLITE_MODEL_PATH = "skin_cancer_model.tflite"
interpreter = tf.lite.Interpreter(model_path=TFLITE_MODEL_PATH)
interpreter.allocate_tensors()

# Get input/output details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Constants
IMG_SIZE = (224, 224)
CLASS_NAMES = ['akiec', 'bcc', 'bkl', 'df', 'mel', 'nv', 'vasc']
UPLOAD_FOLDER = 'static/uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def prepare_image(image_path):
    img = Image.open(image_path).convert('RGB').resize(IMG_SIZE)
    img_array = np.array(img, dtype=np.float32) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

def predict(image_array):
    interpreter.set_tensor(input_details[0]['index'], image_array)
    interpreter.invoke()
    prediction = interpreter.get_tensor(output_details[0]['index'])[0]
    return prediction

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        image_file = request.files['image']
        if image_file:
            filename = str(uuid.uuid4()) + "_" + image_file.filename
            save_path = os.path.join(UPLOAD_FOLDER, filename)
            image_file.save(save_path)

            try:
                img = prepare_image(save_path)
                preds = predict(img)
                predicted_class = CLASS_NAMES[np.argmax(preds)]
                confidence = round(100 * np.max(preds), 2)

                result_html = render_template("result.html",
                                              prediction=predicted_class.upper(),
                                              confidence=confidence,
                                              img_path=save_path)
            finally:
                try:
                    os.remove(save_path)
                except Exception as e:
                    print(f"⚠️ Could not delete image: {e}")

            return result_html

    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
