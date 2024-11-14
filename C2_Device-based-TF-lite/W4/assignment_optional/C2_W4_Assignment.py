# Copyright 2019 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from tflite_runtime.interpreter import Interpreter
from PIL import Image
import numpy as np
import argparse


parser = argparse.ArgumentParser(description='Rock, paper and scissors')
parser.add_argument('--filename', type=str, help='Specify the filename', required=True)
parser.add_argument('--model_path', type=str, help='Specify the model path', required=True)

args = parser.parse_args()

filename = args.filename
model_path = args.model_path 

labels = ['Rock', 'Paper', 'Scissors']

# Load TFLite model and allocate tensors
interpreter = Interpreter(model_path=model_path)
#Allocate tensors to the interpreter
interpreter.allocate_tensors()

# Get input and output tensors.
input_details =  interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Read image with Pillow
img = Image.open(filename).convert('RGB')

# Get input size
input_shape = input_details[0]['shape']
size = input_shape[:2] if len(input_shape) == 3 else input_shape[1:3]

# Preprocess image
# Resize the image
img = mg.resize(size)
# Convert to Numpy with float32 as the datatype
img = np.array(img, dtype=np.float32)
# Normalize the image
img = img / 255.0

# Add a batch dimension
input_data =np.expand_dims(img, axis=0)

# Point the data to be used for testing and run the interpreter
interpreter.set_tensor(input_details[0]['index'], input_data)
interpreter.invoke()

# Obtain results and print the predicted category
predictions = interpreter.get_tensor(output_details[0]['index'])[0]
# Get the label with highest probability
predicted_label = labels[np.argmax(predictions)]
# Print the predicted category
print("Predicted category:", predicted_label)
