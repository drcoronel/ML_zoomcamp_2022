import tflite_runtime.interpreter as tflite
from keras_image_helper import create_preprocessor


classes = [
    'dress',
    'hat',
    'longsleeve',
    'outwear',
    'pants',
    'shirt',
    'shoes',
    'shorts',
    'skirt',
    't-shirt'
]


interpreter = tflite.Interpreter(model_path='clothing-model.tflite')
interpreter.allocate_tensors()

input_index = interpreter.get_input_details()[0]['index']
output_index = interpreter.get_output_details()[0]['index']


preprocessor = create_preprocessor('xception', target_size=(299, 299))


def predict(url):
    X = preprocessor.from_url(url)
    interpreter.set_tensor(input_index, X)
    interpreter.invoke()
    preds = interpreter.get_tensor(output_index)
    results = dict(zip(classes, preds[0].tolist())) 
    results = dict(sorted(results.items(),key=lambda item:item[1],reverse=True))
    return results

def lambda_handler(event,context):
    url = event['url']
    result = predict(url)
    return result


ACCOUNT=283335960283
REGION=eu-west-1
REGISTRY=clothing-tflite-images
REGISTRY_PREFIX=${ACCOUNT}.dkr.ecr.${REGION}.amazonaws.com/${REGISTRY}
TAG=clothing-model-xception-v4-001
REMOTE_URI=${REGISTRY_PREFIX}:${TAG}