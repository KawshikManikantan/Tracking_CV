import os
import time
import argparse
import re

import torch
import numpy as np
import cv2
import pytorch_lightning as pl
from helper import NormalizeToTensor, scale, crop_sample, bgr2rgb
from boundingbox import BoundingBox
from torchvision import models
import torch.nn as nn

args = None
parser = argparse.ArgumentParser(description='GOTURN Testing')
parser.add_argument('-w', '--model-weights',
                    type=str, help='path to pretrained model')
parser.add_argument('-d', '--data-directory',
                    default='../data/imagedata++/01-Light/01-Light_video00001', type=str,
                    help='path to video frames')

class Tracker2(pl.LightningModule):
    
  def __init__(self,lr=1e-5,momentum = 0.9,weight_decay = 0.0005,lr_decay= 100000,gamma= 0.1):
    super().__init__()
    self.lr = lr
    self.momentum = momentum
    self.weight_decay = weight_decay
    self.lr_decay = lr_decay
    self.gamma = gamma
    caffenet = models.alexnet(pretrained=True)
    self.convnet = nn.Sequential(*list(caffenet.children())[:-1])
    for param in self.convnet.parameters():
        param.requires_grad = False
    self.classifier = nn.Sequential(
            nn.Linear(256*6*6*2, 4096, bias=True),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4),
#             nn.ReLU(inplace=True),
            )
#     self.init_dict = torch.load('../pytorch_goturn.pth.tar', map_location='cpu')['state_dict']
    self.init_dict = torch.load('../pytorch_goturn.pth.tar')['state_dict']
    new = list(self.init_dict.items())
#     print(new)
#     print(len(new))
    my_model_kvpair = self.classifier.state_dict()
    count = -8
    for key,value in my_model_kvpair.items():
        if count == 0:
            break
        layer_name, weights = new[count]  
        my_model_kvpair[key] = weights
        count += 1
    print(my_model_kvpair.keys())
    self.classifier.load_state_dict(my_model_kvpair)
    
#     
#     print(self.model)
#     self.weight_init()
    
  def weight_init(self):
    for m in self.classifier.modules():
        if isinstance(m, nn.Linear):
            m.bias.data.fill_(1)
            m.weight.data.normal_(0, 0.005)
    

  def forward(self, x1,x2):
    batch_size, channels,width, height = x1.size()
    x1 = self.convnet(x1)
    x1 = x1.view(batch_size, 256*6*6)
    x2 = self.convnet(x2)
    x2 = x2.view(batch_size, 256*6*6)
    x = torch.cat((x1, x2), 1)
    z = self.classifier(x)
#     print("z", z)
    return z

  def training_step(self, train_batch, batch_idx):
    x1,x2,y = train_batch
#     print("Train gt", y)
#     print('-'*120)
    logits = self.forward(x1, x2)
#     print("Logits", logits)
#     print('#'*120)
    loss_fn = torch.nn.L1Loss(size_average=False)
    loss = loss_fn(logits,y)
    return loss

  def validation_step(self, val_batch, batch_idx):
    x1,x2,y = val_batch
    logits = self.forward(x1,x2)
    loss_fn = torch.nn.L1Loss(size_average=False)
    loss = loss_fn(logits,y)
#     print("y", y)
    return loss

  def test_step(self, test_batch, batch_idx):
    x1,x2,y = test_batch
    logits = self.forward(x1,x2)
    loss_fn = torch.nn.L2Loss(size_average=False)
    loss = loss_fn(logits,y)
    return loss
      
  def configure_optimizers(self):
    optimizer =  torch.optim.SGD(self.parameters(),
                          lr=self.lr,
                          momentum=self.momentum,
                          weight_decay=self.weight_decay)
    scheduler =  torch.optim.lr_scheduler.StepLR(optimizer,
                                          step_size=self.lr_decay,
                                          gamma=self.gamma)
    return [optimizer],[scheduler]

class GOTURN:
    """Tester for OTB formatted sequences"""
    def __init__(self, root_dir, model_path, device):
        self.device = device
        self.gt = []
        self.opts = None
        self.curr_img = None
        self.model = Tracker2()
        
#         checkpoint = torch.load(model_path, map_location=lambda storage, loc: storage)

#         checkpoint = torch.load('../pytorch_goturn.pth.tar',map_location='cpu')
# #         print(checkpoint)
#         self.model.load_state_dict(checkpoint['state_dict'])
#         self.model.to(device)
        
# #         self.model = Tracker2.load_from_checkpoint('./lightning_logs/version_667104/checkpoints/epoch=1-step=6076.ckpt')
#         self.model = Tracker2.load_from_checkpoint('./train_ablation_alov.ckpt')
#         self.model = Tracker2()

        frames = np.array(sorted([root_dir + "/" + frame for frame in os.listdir(root_dir + '/') if 'txt' not in frame]))
        self.leng = len(frames) - 1
        self.x = []
        
#         with open(root_dir + '/groundtruth_rect.txt', 'r') as _f:
#             lines = _f.readlines()
        with open('../data/groundtruth_rect_light.txt', 'r') as _f:
            lines = _f.readlines()
            
        initial_bbox = re.sub(' +', ',', re.sub('\t', ',', lines[0])).strip().split(',')
        initial_bbox = list(map(float, initial_bbox))
#         self.prev_rect = np.array([initial_bbox[0], initial_bbox[1], 
#                                 initial_bbox[0] + initial_bbox[2],
#                                 initial_bbox[1] + initial_bbox[3]])
        self.prev_rect = np.array([initial_bbox[3], initial_bbox[2], 
                                initial_bbox[1],
                                initial_bbox[-1]])
        self.img = []
        for i in range(self.leng):
            self.x.append([frames[i], frames[i+1]])
            img_prev = bgr2rgb(cv2.imread(frames[i]))
            img_curr = bgr2rgb(cv2.imread(frames[i+1]))
            self.img.append([img_prev, img_curr])
            bbox = re.sub(' +', ',', re.sub('\t', ',', lines[i+1])).strip().split(',')
            bbox = list(map(float, bbox))
#             self.gt.append([bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]])
            self.gt.append([bbox[3], bbox[2], bbox[1], bbox[-1]])
        self.x = np.array(self.x)

    def __getitem__(self, idx):
        """
        Returns transformed torch tensor which is passed to the network
        after cropping and scaling to (224,224,3).
        """
        print("Input bb", self.gt[idx])
        #print("Prev-")
        prev_sample, opts_prev = crop_sample({'image': self.img[idx][0], 'bb': self.prev_rect})
        #print("Curr-")
        curr_sample, opts_curr = crop_sample({'image': self.img[idx][1], 'bb': self.prev_rect})
        search_reg = opts_curr['search_location']
        #print(curr_sample['bb'])
        #print("Curr scale-")
        curr_img = scale(curr_sample, opts_curr,224,224)['image']
        #print("Prev scale-")
        prev_img = scale(prev_sample, opts_prev,224,224)['image']
        bb_test = scale(prev_sample, opts_prev,224,224)['bb']
        self.curr_img, self.opts = curr_img, opts_curr
        return NormalizeToTensor()({'previmg': prev_img, 'currimg': curr_img, 'currbb': bb_test}), self.img[idx][1], search_reg

    def get_rect(self, sample, raw_img):
        """
        Regresses the bounding box coordinates in the original image dimensions
        for an input sample.
        """
        x1, x2 = sample['previmg'].unsqueeze(0).to(self.device), \
                sample['currimg'].unsqueeze(0).to(self.device)
        y = self.model(x1, x2)
        print("Logitssssss", y)
        bbox = y.data.cpu().numpy().transpose((1, 0))[:, 0]
#         bbox = y
        bbox = BoundingBox(bbox[0], bbox[1], bbox[2], bbox[3])
        # inplace conversion
        bbox.unscale(self.opts['search_region'])
        #print("After unsc", bbox.get_bb_list())
        bbox.uncenter(raw_img, self.opts['search_location'],
                      self.opts['edge_spacing_x'], self.opts['edge_spacing_y'])
        #print("After uncent", bbox.get_bb_list())
#         print(bbox.get_bb_list())
        return bbox.get_bb_list()

    def test(self):
        """
        Loops through all the frames of test sequence and tracks the target.
        Prints predicted box location on console with frame ID.
        """
#         self.model.eval()
        st = time.time()
        for i in range(self.leng):
            self.prev_rect = self.get_rect(self[i])
            print("frame: {}".format(i+1), self.prev_rect)
        end = time.time()
        print("Frames per second: {:.3f}".format(self.leng/(end-st)))

if __name__ == "__main__":
    args = parser.parse_args()
    print(args)
#     device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    device = 'cpu'
    tester = GOTURN(args.data_directory, args.model_weights, device)
    tester.test()