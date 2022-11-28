#bin/bash! 

source envfile.env 

aws ecr create-repository --repository-name ${REGISTRY}
aws ecr get-login-password --region ${REGION} | docker login --username AWS --password-stdin ${REGISTRY_PREFIX} &&
docker tag ${IMAGE_NAME}:latest ${REMOTE_URI} &&
docker push ${REMOTE_URI} 
#docker build -t ${MODEL_NAME} . 