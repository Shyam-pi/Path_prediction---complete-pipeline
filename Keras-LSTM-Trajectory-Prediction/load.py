import numpy as np
import os
import matplotlib.pyplot as plt
import json
import argparse
from keras.utils import HDF5Matrix
from keras import backend
from keras.models import load_model
from keras_radam import RAdam
from keras_lookahead import Lookahead


def rmse(y_true, y_pred):
	return backend.sqrt(backend.mean(backend.square(y_pred - y_true), axis=-1))
    
def compute_color_for_labels(label):
    """
    Simple function that adds fixed color depending on the class
    """
    color = [int((p * (label ** 2 - label + 1)) % 255) for p in palette]
    return tuple(color)

def draw_predicted_trajectories(img, predicted_coordinates, identity):
    overlay = img.copy()
    color = compute_color_for_labels(identity)
    cv2.polylines(overlay, predicted_coordinates, isClosed = False,color = color,thickness = 3, linetype = cv2.LINE_AA)
    alpha = 0.4
    img = cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0)
    return img
    

def main():
    lstm_model = load_model("/content/gdrive/My Drive/Colab Notebooks/BE Thesis/Prediction/Keras-LSTM-Trajectory-Prediction/model_output/ARCH--15i_30o_10sld_norm3_vill_cascade2-_Data-_final_concat512___bs-512_lr-0.01_loss-mae_opt-Ranger_BD-True_BDmrg-concat_amsG-False_DP-False_sw-0.9_sync-1_act-selu_minLR-1e-05_ptc-10_ep-100/lstm_model.h5", custom_objects={'rmse': rmse, 'Lookahead': Lookahead, 'RAdam': RAdam})
    source_traj = np.array([[570,570,570,570,570,570,570,570,570,570,570,570,570,570,566], [130,132,130,132,132,132,132,130,135,137,137,137,137,137,137], [33,33,30,30,33,30,30,30,30,33,33,33,33,33,33], [63,65,67,65,65,65,65,67,67,65,65,65,65,63,63]])
    src_trajectory_batch = np.expand_dims(source_traj, axis=0)
    predicted_trajectory = lstm_model.predict(src_trajectory_batch)
    predicted_trajectory = np.squeeze(predicted_trajectory)
    predicted_trajectory = predicted_trajectory.tolist()
    
    print(predicted_trajectory)

    pred_coordinates = []
    
    for i in range(30):
        x_center = predicted_trajectory[0][i] + (predicted_trajectory[2][i] / 2)
        y_center = predicted_trajectory[1][i] + (predicted_trajectory[3][i] / 2)
        coord = (x_center,y_center)
        pred_coordinates.append(coord)
    print(pred_coordinates)
    
if __name__ == "__main__":
	main()