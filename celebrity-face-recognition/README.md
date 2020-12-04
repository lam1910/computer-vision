# Celebrity Face Recognition
Face detection and recognition of Vietnamese celebrities

## Data Crawling
The dataset is collected by crawling celebrities faces from Google Images.
The crawler file can be found in crawler module and in the colab notebook.

## Model Pipeline
We use MTCNN to crop face regions from the images, then finetune the pretrained VGG model to classify the cropped images.
Firstly, run the crop_face.ipynb colab notebooks to crop the face regions from the dataset.
After that, run the following code to start training process:
```
python -m face_recognition.main
```
Configuration for the training process can be found in config/config.yml
After finish the training, the model will be saved in pretrained folder.

## Web Demo
To start the web demo, change the saved model name in file demo/inference.py, then run:
```
python -m demo.server
```