
import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_datasets as tfds
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import argparse
import json


def process_image(image): 
   
    image = tf.cast(image, tf.float32)
    image= tf.image.resize(image, (224, 224))
    image /= 255
    
    return image.numpy()
    

def predict(image_path, model, top_k=5):
    
    image = Image.open(image_path)
    image = np.asarray(image)
    image = np.expand_dims(image,  axis=0)
    image_pp = process_image(image)
    
    prediction = model.predict(image_pp)
    top_k_pred, top_k_predind = tf.nn.top_k(prediction, k=top_k)
    
    return top_k_pred.numpy(), top_k_predind.numpy()

# Parsing input arguments 
parser = argparse.ArgumentParser()
parser.add_argument('path')
parser.add_argument('model')
parser.add_argument('--top_k')
parser.add_argument('--category_names') 

args = parser.parse_args()
path = args.path

top_k = args.top_k
if top_k is None: 
    top_k = 1

# Loading the Keras Model
model = tf.keras.models.load_model(args.model ,custom_objects={'KerasLayer':hub.KerasLayer} )

# Performing predictions
pred_prob, pred_ind = predict(path, model, top_k)

# Retrieving Class names
category_names = args.category_names

if category_names is not None:
    with open(args.category_names, 'r') as f:
        class_names = json.load(f)
    
    flower_class = []
    for n in pred_ind[0]:
        flower_class.append(class_names[str(n+1)])
    print(pred_prob)
    print(flower_class)
else:  
    print(pred_prob)
    print(pred_ind)
