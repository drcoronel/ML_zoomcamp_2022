from tensorflow import keras
import tensorflow.lite as tflite 

model = keras.models.load_model("dino_dragon_10_0.899.h5")

converter = tflite.TFLiteConverter.from_keras_model(model)

tflite_model = converter.convert()

with open('dino_dragon.tflite', 'wb') as f_out:
    f_out.write(tflite_model)

