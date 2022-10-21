#!/bin/bash 
q1=0
q2=0
source "$WORKON_HOME/week7/bin/activate"
echo "Answer question 1:"
echo 'Q1:'"$(bentoml -v)" > questions.txt
echo "Answer question 2:"
echo 'Q2:'"$(bentoml models list | grep "credit_risk_model:7thh5rcqowstv3jy")" >> questions.txt
echo "Answer question 4:"
echo 'Q4: Version of'"$(bentoml models get mlzoomcamp_homework:qtzdz3slg6mwwdu5 | grep "scikit-learn" | sed 's/^&//g')" >> questions.txt 
echo "Anser question 5"
echo 'Q5: Response :'"$(curl -X 'POST' \
  'http://0.0.0.0:3000/predict' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '[
  [6.4,3.5,4.5,1.2]
]')" >> questions.txt