from flask import Flask, request, jsonify
import base64
import numpy as np
from PIL import Image
from io import BytesIO
import tflite_runtime.interpreter as tflite

app = Flask(__name__)

# Load model
interpreter = tflite.Interpreter(model_path="model/mobilenetv5.tflite")
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

def preprocess_image(image_base64):
    img_bytes = base64.b64decode(image_base64)
    img = Image.open(BytesIO(img_bytes)).resize((224,224))
    img_array = np.array(img, dtype=np.float32)/255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        image_base64 = data['image']
        img = preprocess_image(image_base64)
        interpreter.set_tensor(input_details[0]['index'], img)
        interpreter.invoke()
        output = interpreter.get_tensor(output_details[0]['index'])
        result = float(output[0][0])
        confidence = float(output[0][1])
        return jsonify({'result': result, 'confidence': confidence})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
