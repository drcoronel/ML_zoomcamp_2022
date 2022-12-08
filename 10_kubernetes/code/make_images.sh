#!bin/sh


docker build -t zoomcamp-10-model:xception-v4-001 -f image-model.Dockerfile .

docker run -it --rm \
    -p 8500:8500 \
    zoomcamp-10-model:xception-v4-001 


docker build -t zoomcamp-10-gateway:002 -f image-gateway.dockerfile  .

docker run -it --rm \
    -p 9696:9696 \
    zoomcamp-10-gateway:001