# Path-Prediction-of-Pedestrians-using-Bi-Directional-LSTM---A-complete-pipeline
Codes for my Bachelor's thesis with BITS Pilani, Pilani campus in the topic - Path Prediction of Pedestrians using Deep Learning. This is my first time creating a complete coding project and using Github, so kindly excuse me for not living upto any standards :)

Complete modeling, coding and simulation of the model has been done in Google Colab in order to avoid dependency issues and to promote better reproducibility for future works. Google drive will be used for managing everything, hence high speed internet connection is a must.

Create a working folder inside your google drive. Upload the given two folders going by the name “Keras-LSTM-Trajectory-Prediction” and “Yolov3_DeepSort_Pytorch” to your google drive.

In order to simplify and organize the implementation of codes, all calls to python files have been done via python notebooks which are herewith attached. There are three notebooks for various functionalities. Instructions to use the notebooks are attached within the respective notebooks itself. One needs to upload these files to their google drive and open these with Google Collaboratory.

Notebooks are:

1. Tracking.ipynb : This notebook is specifically used to track people in a video and store their tracks frame wise in the form of a .csv file which can be used for training the prediction model.

2. Prediction_training.ipynb : This notebook contains all functionalities of the base model used for training/updating the model. (In order to make changes to the model itself, one has to go to the “Keras-LSTM-Trajectory-Prediction” folder and make changes to the python file itself.)

3. Tracknpred.ipynb : This notebook is used for processing and visualizing predictions of path, for a given video file. (The input media file should be in mp4 format). It takes a mp4 video as its input and gives a mp4 video as its output which has visualization of the predictions.
