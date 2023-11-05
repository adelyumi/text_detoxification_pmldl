# Text Detoxification

This repository contains a solution for the assignment 1 of PMLDL course.

Author: Adelina Kildeeva 

Email: a.kildeeva@innoppolis.university

Group: B21-DS-02

## Task
Text Detoxification Task is a process of transforming the text with toxic style into the text with the same meaning but with neutral style.

A more complete description can be found in task_description.md

## Solution

Pretrained BERT model that replaces toxic words with their neutral synonyms.

## How to use
1. Clone the repository

2. Install the required packages
```
pip install -r requirements.txt
```

3. Preprocess data
```
python src/data/make_dataset.py
```

4. Use the algorithm 
```
python src/models/predict.py "Some text example"
```
