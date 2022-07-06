import os
import argparse
import torch
import cv2
from test import GOTURN
from torchvision import models
import torch.nn as nn

args = None
parser = argparse.ArgumentParser(description='GOTURN Testing')
parser.add_argument('-w', '--model-weights',
                    type=str, help='path to pretrained model')
parser.add_argument('-d', '--data-directory',
                    default='../data/imagedata++/01-Light/01-Light_video00001', type=str,
                    help='path to video frames')
# parser.add_argument('-d', '--data-directory',
#                     default='../data/OTB/Man', type=str,
#                     help='path to video frames')
parser.add_argument('-s', '--save-directory',
                    default='../test_iou/run',
                    type=str, help='path to save directory')

def get_iou(box1, box2):
    assert(box1[0] <= box1[2])
    assert(box1[1] <= box1[3])
    assert(box2[0] <= box2[2])
    assert(box2[1] <= box2[3])

    x1_int, y1_int = max(box1[0], box2[0]), max(box1[1], box2[1])
    x2_int, y2_int = min(box1[2], box2[2]), min(box1[3], box2[3])
    int_area = max(0, x2_int - x1_int + 1) * max(0, y2_int - y1_int + 1)

    box1_area = (box1[2] - box1[0] + 1) * (box1[3] - box1[1] + 1)
    box2_area = (box2[2] - box2[0] + 1) * (box2[3] - box2[1] + 1)
    return int_area / float(box1_area + box2_area - int_area)

def save_annotations(_im, _bbox, _gt_bbox, idx):
    bbox, gt_bbox = list(map(int, _bbox)), list(map(int, _gt_bbox))
    print(_im.shape)
    im = cv2.rectangle(_im, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 0, 255), 2)
    im = cv2.rectangle(im, (gt_bbox[0], gt_bbox[1]), (gt_bbox[2], gt_bbox[3]), (255, 255, 255), 2)
    print(os.path.join(args.save_directory, str(idx)+'.jpg'))
    cv2.imwrite(os.path.join(args.save_directory, str(idx)+'.jpg'), im)
    
def save_annotations_with_serc(_im, _bbox, _gt_bbox, idx, search_reg):
    serc_bbox = search_reg.get_bb_list()
    print('sercccccccccccccc', serc_bbox)
    bbox, gt_bbox = list(map(int, _bbox)), list(map(int, _gt_bbox))
    print(_im.shape)
    print('predicted bbox',bbox)
    print('groundtruth',gt_bbox)
    im = cv2.rectangle(_im, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 0, 255), 2)
    im = cv2.rectangle(im, (gt_bbox[0], gt_bbox[1]), (gt_bbox[2], gt_bbox[3]), (255, 255, 255), 2)
    im = cv2.rectangle(im, (int(serc_bbox[0]), int(serc_bbox[1])), (int(serc_bbox[2]), int(serc_bbox[3])), (255, 0, 0), 2)
    print(os.path.join(args.save_directory, str(idx)+'.jpg'))
    cv2.imwrite(os.path.join(args.save_directory, str(idx)+'.jpg'), im)
    
def initiate_tester(tester):
    for i in range(tester.leng):
        bbox_inp, raw_img, search_reg = tester[i]
        bbox = tester.get_rect(bbox_inp, raw_img)
        tester.prev_rect = bbox
        # save current image with predicted rectangle and gt box
        save_annotations_with_serc(cv2.cvtColor(tester.img[i][1], cv2.COLOR_RGB2BGR), bbox, \
                            tester.gt[i], i+2, search_reg)
        print(f'frame: {i+2}, IoU = {get_iou(tester.gt[i], bbox)}')
        test_file.write(str(get_iou(tester.gt[i], bbox)) + '\n')

if __name__ == "__main__":
   
    args = parser.parse_args()
    test_file = open(args.save_directory + '/iou.txt', 'w+')
#     device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    device = 'cpu'
    tester = GOTURN(args.data_directory, args.model_weights, device)
    if os.path.exists(args.save_directory):
        print(f'Save directory {args.save_directory} already exists')
    else:
        os.makedirs(args.save_directory)
    # save initial frame with bounding box
    save_annotations(cv2.cvtColor(tester.img[0][0], cv2.COLOR_RGB2BGR), tester.prev_rect, tester.prev_rect, 1)
#     tester.model.eval()
    initiate_tester(tester)
    test_file.close()
