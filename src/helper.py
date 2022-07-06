import math
import warnings

import numpy as np
import torch
import cv2
import math
from torchvision import transforms
from boundingbox import BoundingBox

warnings.filterwarnings("ignore")

def scale(sample,opts,new_h,new_w):
    image, bb = sample['image'], sample['bb']
    new_h, new_w = int(new_h), int(new_w)
    img = cv2.resize(image, (new_h, new_w), interpolation=cv2.INTER_CUBIC)
    bbox = BoundingBox(bb[0], bb[1], bb[2], bb[3])
#     print("bef sc", bbox.get_bb_list())
    bbox.scale(opts['search_region'])
#     print("aft sc", bbox.get_bb_list())
    return {'image': img, 'bb': bbox.get_bb_list()}

def bgr2rgb(image):
    if len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    else:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image

def rgb2bgr(image):
    if len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    else:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return image

class NormalizeToTensor(object):
    """Returns torch tensor normalized images."""

    def __call__(self, sample):
        prev_img, curr_img = sample['previmg'], sample['currimg']
        self.transform = transforms.Compose([transforms.ToTensor(),
                                            transforms.Normalize(
                                            mean=[0.485, 0.456, 0.406],
                                            std=[0.229, 0.224, 0.225])
                                        ])
        prev_img = self.transform(prev_img)
        curr_img = self.transform(curr_img)
        if 'currbb' in sample:
            currbb = np.array(sample['currbb'])
            return {'previmg': prev_img,
                    'currimg': curr_img,
                    'currbb': torch.from_numpy(currbb).float()}
        else:
            return {'previmg': prev_img,
                    'currimg': curr_img}




def shift_crop_training_sample(sample, bb_params):
    """
    Given an image with bounding box, this method randomly shifts the box and
    generates a training example. It returns current image crop with shifted
    box (with respect to current image).
    """
    output_sample = {}
    opts = {}
    currimg = sample['image']
    currbb = sample['bb']
    bbox_curr_gt = BoundingBox(currbb[0], currbb[1], currbb[2], currbb[3])
    bbox_curr_shift = BoundingBox(0, 0, 0, 0)
#     bbox_curr_shift = BoundingBox(currbb[0], currbb[1], currbb[2], currbb[3])
    
    bbox_curr_shift = bbox_curr_gt.shift(currimg,
                                         bb_params['lambda_scale_frac'],
                                         bb_params['lambda_shift_frac'],
                                         bb_params['min_scale'],
                                         bb_params['max_scale'], True,
                                         bbox_curr_shift)
    (rand_search_region, rand_search_location,
        edge_spacing_x, edge_spacing_y) = search_region(bbox_curr_shift,
                                                       currimg)
    bbox_curr_gt = BoundingBox(currbb[0], currbb[1], currbb[2], currbb[3])
    bbox_gt_recentered = BoundingBox(0, 0, 0, 0)
#     bbox_gt_recentered = BoundingBox(currbb[0], currbb[1], currbb[2], currbb[3])
    bbox_gt_recentered = bbox_curr_gt.recenter(rand_search_location,
                                               edge_spacing_x,
                                               edge_spacing_y,
                                               bbox_gt_recentered)
#     bbox_gt_recentered.scale(rand_search_region)
    output_sample['image'] = rand_search_region
    output_sample['bb'] = bbox_gt_recentered.get_bb_list()

    # additional options for visualization

    opts['edge_spacing_x'] = edge_spacing_x
    opts['edge_spacing_y'] = edge_spacing_y
    opts['search_location'] = rand_search_location
    opts['search_region'] = rand_search_region
    return output_sample, opts


def crop_sample(sample):
    """
    Given a sample image with bounding box, this method returns the image crop
    at the bounding box location with twice the width and height for context.
    """
    output_sample = {}
    opts = {}
    image, bb = sample['image'], sample['bb']
    orig_bbox = BoundingBox(bb[0], bb[1], bb[2], bb[3])
    (output_image, pad_image_location,
        edge_spacing_x, edge_spacing_y) = search_region(orig_bbox, image)
    new_bbox = BoundingBox(0, 0, 0, 0)
#     new_bbox = BoundingBox(bb[0], bb[1], bb[2], bb[3])
    new_bbox = orig_bbox.recenter(pad_image_location,
                                 edge_spacing_x,
                                 edge_spacing_y,
                                 new_bbox)
#     new_bbox.scale(output_image)
#     new_bbox.uncenter(image, pad_image_location, edge_spacing_x,
#                       edge_spacing_y)
    output_sample['image'] = output_image
    output_sample['bb'] = new_bbox.get_bb_list()

    # additional options for visualization
    opts['edge_spacing_x'] = edge_spacing_x
    opts['edge_spacing_y'] = edge_spacing_y
    opts['search_location'] = pad_image_location
    opts['search_region'] = output_image
    return output_sample, opts

def search_region(bb,image):
    # ../data/ILSVRC2013_DET_val/ILSVRC2012_val_00041559.JPEG
    # [45.0, 81.0, 249.0, 612.0]
    search_region_width=bb.compute_search_width()
    search_region_height=bb.compute_search_height()
    bb_centre_x,bb_centre_y=bb.get_center()
    x1 = min(image.shape[1]-1, max(bb_centre_x-search_region_width/2.0, 0.0))
    y1 = min(image.shape[0]-1, max(bb_centre_y-search_region_height/2.0, 0.0))
    
    rw = max(1.0, min(search_region_width / 2., bb_centre_x) + \
                    min(search_region_width / 2., image.shape[1] - bb_centre_x))
    rh = max(1.0, min(search_region_height / 2., bb_centre_y) + \
                min(search_region_height / 2., image.shape[0] - bb_centre_y))

    startx = min(image.shape[1]-1, max(0.0,search_region_width/2.-bb_centre_x))
    starty = min(image.shape[0]-1, max(0.0,search_region_height/2.-bb_centre_y))  
    rw = min(image.shape[1], max(1.0, math.ceil(rw)))
    rh = min(image.shape[0], max(1.0, math.ceil(rh)))
    x2, y2 = x1 + rw, y1 + rh
    
    err = 0.000000001
    crop_image = image[int(y1+err):int(y2), int(x1+err):int(x2)]
    endx=int(startx)+crop_image.shape[1]
    endy=int(starty)+crop_image.shape[0]
    output_width = max(math.ceil(search_region_width), rw)
    output_height = max(math.ceil(search_region_height), rh)
 
    if len(image.shape)==3:
        output_image=np.zeros((int(output_height),int(output_width),3),dtype=image.dtype)
    else:
        output_image=np.zeros((int(output_height),int(output_width)),dtype=image.dtype)

    output_image[int(starty):endy,int(startx):endx] = crop_image
    image_loc=BoundingBox(x1,y1,x2,y2)
    return output_image,image_loc,startx,starty