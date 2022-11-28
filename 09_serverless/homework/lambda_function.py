import tflite_runtime.interpreter as tflite
from utils import process_from_url


interpreter = tflite.Interpreter(model_path='dino-vs-dragon-v2.tflite')
interpreter.allocate_tensors()

input_index = interpreter.get_input_details()[0]['index']
output_index = interpreter.get_output_details()[0]['index']

def predict(url):
    X = process_from_url(url)
    interpreter.set_tensor(input_index, X)
    interpreter.invoke()
    preds = interpreter.get_tensor(output_index)
    results = float(preds[0])
    return results

def lambda_handler(event,context):
    url = event['url']
    result = predict(url)
    return result
