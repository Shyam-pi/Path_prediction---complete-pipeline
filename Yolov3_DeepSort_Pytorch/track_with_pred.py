import argparse
import csv
from sys import platform
import pandas as pd

import numpy as np
import os
import matplotlib.pyplot as plt
import json
from keras.utils import HDF5Matrix
from keras import backend
from keras.models import load_model
from keras_radam import RAdam
from keras_lookahead import Lookahead

from yolov3.models import *  # set ONNX_EXPORT in models.py
from yolov3.utils.datasets import *
from yolov3.utils.utils import *
from deep_sort import DeepSort

deepsort = DeepSort("deep_sort/deep/checkpoint/ckpt.t7")
palette = (2 ** 11 - 1, 2 ** 15 - 1, 2 ** 20 - 1)


def bbox_rel(image_width, image_height, bbox_left, bbox_top, bbox_w, bbox_h):
    """" Calculates the relative bounding box from absolute pixel values. """
    x_c = (bbox_left + bbox_w / 2)
    y_c = (bbox_top + bbox_h / 2)
    w = bbox_w
    h = bbox_h
    return x_c, y_c, w, h


def compute_color_for_labels(label):
    """
    Simple function that adds fixed color depending on the class
    """
    color = [int((p * (label ** 2 - label + 1)) % 255) for p in palette]
    return tuple(color)

def load_prediction_model(pred_model_path):
    lstm_model_path = os.path.join(pred_model_path, "lstm_model.h5")
    lstm_model = load_model(lstm_model_path, custom_objects={'rmse': rmse, 'Lookahead': Lookahead, 'RAdam': RAdam})
    return lstm_model
    
def predict_sequence(lstm_model, source_trajectory):
	"""
    Predicts the next n_steps_out bounding boxes using pre-trained models
    """
	src_trajectory_batch = np.expand_dims(source_trajectory, axis=0)
	predicted_trajectory = lstm_model.predict(src_trajectory_batch)
	return np.array(predicted_trajectory)
    
def coordinates_from_predictions(predicted_trajectory):
    """
    Coordinates of predicted box centers from the box values
    """
    predicted_trajectory = np.squeeze(predicted_trajectory)
    predicted_trajectory = predicted_trajectory.tolist()
    
    x_center_start = round(predicted_trajectory[0][0] + (predicted_trajectory[2][0] / 2))
    y_center_start = round(predicted_trajectory[1][0] + (predicted_trajectory[3][0] / 2))
    
    x_center_end = round(predicted_trajectory[0][29] + (predicted_trajectory[2][29] / 2))
    y_center_end = round(predicted_trajectory[1][29] + (predicted_trajectory[3][29] / 2))
    
    start = (x_center_start, y_center_start)
    end = (x_center_end, y_center_end)
    
    return start, end

def rmse(y_true, y_pred):
	"""
    Calculate Root Mean Squared Error custom metric
    """
	return backend.sqrt(backend.mean(backend.square(y_pred - y_true), axis=-1))
    
def draw_predicted_trajectories(img, start, end, identity):
    print("Person predicted : ",identity)
    color = compute_color_for_labels(identity)
    thickness = 3
    cv2.arrowedLine(img, start, end, color, thickness)
    return img

def draw_boxes(img, bbox, identities=None, offset=(0,0)):
    for i, box in enumerate(bbox):
        x1, y1, x2, y2 = [int(i) for i in box]
        x1 += offset[0]
        x2 += offset[0]
        y1 += offset[1]
        y2 += offset[1]
        # box text and bar
        id = int(identities[i]) if identities is not None else 0    
        color = compute_color_for_labels(id)
        label = '{}{:d}'.format("", id)
        t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, 2 , 2)[0]
        cv2.rectangle(img, (x1, y1),(x2,y2), color, 3)
        cv2.rectangle(img, (x1, y1), (x1 + t_size[0] + 3, y1 + t_size[1] + 4), color, -1)
        cv2.putText(img, label, (x1, y1 + t_size[1] + 4), cv2.FONT_HERSHEY_PLAIN, 2, [255, 255, 255], 2)
    return img



def detect(save_img=True):
    img_size = (320, 192) if ONNX_EXPORT else opt.img_size  # (320, 192) or (416, 256) or (608, 352) for (height, width)
    out, source, weights, half, view_img, save_txt = opt.output, opt.source, opt.weights, opt.half, opt.view_img, opt.save_txt
    webcam = source == '0' or source.startswith('rtsp') or source.startswith('http') or source.endswith('.txt')

    # Initialize
    device = torch_utils.select_device(device='cpu' if ONNX_EXPORT else opt.device)
    if os.path.exists(out):
        shutil.rmtree(out)  # delete output folder
    os.makedirs(out)  # make new output folder

    # Initialize detection model
    model = Darknet(opt.cfg, img_size)
    
    #Initialize prediction model
    lstm_model = load_prediction_model(opt.pred_model)

    # Load weights
    attempt_download(weights)
    if weights.endswith('.pt'):  # pytorch format
        model.load_state_dict(torch.load(weights, map_location=device)['model'])
    else:  # darknet format
        load_darknet_weights(model, weights)

    # Eval mode
    model.to(device).eval()

    # Half precision
    half = half and device.type != 'cpu'  # half precision only supported on CUDA
    if half:
        model.half()

    # Set Dataloader
    vid_path, vid_writer = None, None
    if webcam:
        save_img = False
        view_img = True
        torch.backends.cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(source, img_size=img_size, half=half)
    else:
        save_img = True
        dataset = LoadImages(source, img_size=img_size, half=half)
        

    # Get names and colors
    names = load_classes(opt.names)
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(names))]

    # Run inference
    t0 = time.time()
    for path, img, im0s, vid_cap in dataset:
        t = time.time()

        # Get detections
        img = torch.from_numpy(img).to(device)
        if img.ndimension() == 3:
            img = img.unsqueeze(0)
        pred = model(img)[0]

        if opt.half:
            pred = pred.float()

        # Apply NMS
        pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)

        # Process detections
        for i, det in enumerate(pred):  # detections per image
            if webcam:  # batch_size >= 1
                p, s, im0 = path[i], '%g: ' % i, im0s[i]
            else:
                p, s, im0 = path, '', im0s

            save_path = str(Path(out) / Path(p).name)
            s += '%gx%g ' % img.shape[2:]  # print string
            if det is not None and len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()
                # print(det[:, :5])
                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += '%g %ss, ' % (n, names[int(c)])  # add to string

                bbox_xywh = []
                confs = []

                # Write results
                for *xyxy, conf, cls in det:
                    img_h, img_w, _ = im0.shape  # get image shape
                    bbox_left = min([xyxy[0].item(), xyxy[2].item()])
                    bbox_top = min([xyxy[1].item(), xyxy[3].item()])
                    bbox_w = abs(xyxy[0].item() - xyxy[2].item())
                    bbox_h = abs(xyxy[1].item() - xyxy[3].item())
                    x_c, y_c, bbox_w, bbox_h = bbox_rel(img_w, img_h, bbox_left, bbox_top, bbox_w, bbox_h)
                    #print(x_c, y_c, bbox_w, bbox_h)
                    obj = [x_c, y_c, bbox_w, bbox_h]
                    bbox_xywh.append(obj)
                    confs.append([conf.item()])
                    label = '%s %.2f' % (names[int(cls)], conf)
                    #
                    #print('bboxes')
                    #print(torch.Tensor(bbox_xywh))
                    #print('confs')
                    #print(torch.Tensor(confs))
                    outputs = deepsort.update((torch.Tensor(bbox_xywh)), (torch.Tensor(confs)) , im0)
                    if len(outputs) > 0:
                        bbox_xyxy = outputs[:, :4]
                        identities = outputs[:, -1]
                        draw_boxes(im0, bbox_xyxy, identities)
                        with open(source.split("/")[-1][:-4]+'.csv', 'a+', newline='') as file:
                            writer = csv.writer(file)
                            for i, box in enumerate(bbox_xyxy):
                                df = pd.read_csv("tracks.csv")
                                x1, y1, x2, y2 = [int(i) for i in box]
                                start = (round((x1+x2)/2) , round((y1+y2)/2))
                                id = int(identities[i]) if identities is not None else 0
                                df1 = df.loc[df['Frame'] == dataset.frame]

                                if df1.empty:
                                  writer.writerow([dataset.frame, id, "PEDESTRIAN", x1, y1, abs(x2-x1), abs(y2-y1), "90"])

                                else:
                                  df2 = df1.loc[df1['Track'] == id]
                                  if df2.empty:
                                    writer.writerow([dataset.frame, id, "PEDESTRIAN", x1, y1, abs(x2-x1), abs(y2-y1), "90"])
                                
                                df = pd.read_csv("tracks.csv")                                
                                df3 = df.loc[df['Track'] == id]
                                prev_traj = df3.tail(15)
                                
                                if(prev_traj.shape[0] == 15):
                                    left = prev_traj['Left'].to_list()
                                    top = prev_traj['Top'].to_list()
                                    width = prev_traj['Width'].to_list()
                                    height = prev_traj['Height'].to_list()
                                    
                                    source_traj = [left,top,width,height]
                                    source_traj = np.array(source_traj)
                                    pred_traj = predict_sequence(lstm_model,source_traj)
                                    pred_start, end = coordinates_from_predictions(pred_traj)
                                    draw_predicted_trajectories(im0, start, end, id)
                                            
                    #print('\n\n\t\ttracked objects')
                    #print(outputs)

            # Print time (inference + NMS)
            print('%sDone. (%.3fs)' % (s, time.time() - t))

            # Stream results
            if view_img:
                cv2.imshow(p, im0)
                if cv2.waitKey(1) == ord('q'):  # q to quit
                    raise StopIteration

            # Save results (image with detections)
            if save_img:
                if dataset.mode == 'images':
                    cv2.imwrite(save_path, im0)
                else:
                    if vid_path != save_path:  # new video
                        vid_path = save_path
                        if isinstance(vid_writer, cv2.VideoWriter):
                            vid_writer.release()  # release previous video writer

                        fps = vid_cap.get(cv2.CAP_PROP_FPS)
                        w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                        h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*opt.fourcc), fps, (w, h))
                    vid_writer.write(im0)

    if save_txt or save_img:
        print('Results saved to %s' % os.getcwd() + os.sep + out)
        if platform == 'darwin':  # MacOS
            os.system('open ' + out + ' ' + save_path)

    print('Done. (%.3fs)' % (time.time() - t0))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default='yolov3/cfg/yolov3-spp.cfg', help='*.cfg path')
    parser.add_argument('--names', type=str, default='yolov3/data/coco.names', help='*.names path')
    parser.add_argument('--weights', type=str, default='yolov3/weights/yolov3-spp-ultralytics.pt', help='path to weights file')
    parser.add_argument('--source', type=str, default='0', help='source')  # input file/folder, 0 for webcam
    parser.add_argument('--output', type=str, default='output', help='output folder')  # output folder
    parser.add_argument('--img-size', type=int, default=608, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.3, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.5, help='IOU threshold for NMS')
    parser.add_argument('--fourcc', type=str, default='mp4v', help='output video codec (verify ffmpeg support)')
    parser.add_argument('--half', action='store_true', help='half precision FP16 inference')
    parser.add_argument('--device', default='', help='device id (i.e. 0 or 0,1) or cpu')
    parser.add_argument('--view-img', action='store_true', help='display results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--classes', nargs='+', type=int, default=[0], help='filter by class')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--pred_model', type=str, help='prediction model folder')
    opt = parser.parse_args()
    print(opt)
    with open(source.split("/")[-1][:-4]+'.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Frame", "Track", "Class", "Left", "Top", "Width", "Height", "DetectionProbability"])
    
    with torch.no_grad():
        detect()
