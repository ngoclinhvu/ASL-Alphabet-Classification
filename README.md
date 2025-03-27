# ASL-Alphabet-Classification
## Problem statement:
### 1. Context:
In daily life, most people communicate using their ears and vocal languages. Hearing-impaired people can only communicate via sign language.
### 2. Problem definition:
People who are deaf or hard of hearing find it hard to communicate with people without deafness, the majority of the population.
### 3. Impact of the problem:
Deaf people and people with hearing impairment cannot engage fully with society. Therefore, hearing impairment gives them fewer opportunities in life, and society cannot fully utilize their potential.
### 4. Objectives:
The project develops the deep learning model translating the American Sign Language Alphabet from images, taking a first few steps in the way of helping people with hearing impairment.
## Idea:
### 1. Dataset definition:
The data set is a collection of images of alphabets from the American Sign Language, separated in 29 folders which represent the various classes.
### 2. Dataset description:
The training data set contains 87,000 images which are 200x200 pixels. There are 29 classes, of which 26 are for the letters A-Z and 3 classes for SPACE, DELETE, and NOTHING.

These 3 classes are very helpful in real-time applications and classification.

The test data set contains a mere 29 images to encourage the use of real-world test images.
### 3. Architecture design:
Customized EfficientNet B0

Performed hyperparameter tuning to improve sentiment polarity accuracy by analyzing time and accuracy across different network architectures and batch sizes
