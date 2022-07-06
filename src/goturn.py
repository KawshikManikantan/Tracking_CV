import torch
import numpy as np
import cv2
from torchvision import transforms
import pytorch_lightning as pl
from got10k.trackers import Tracker
from helper import NormalizeToTensor, scale, BoundingBox, crop_sample
from torchvision import models
import torch.nn as nn

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
            nn.Linear(256*6*6*2, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4),
            )
    self.weight_init()
    
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
    return z

  def training_step(self, train_batch, batch_idx):
    x1,x2,y = train_batch
    logits = self.forward(x1,x2)
    loss_fn = torch.nn.L1Loss(size_average=False)
    loss = loss_fn(logits,y)
    return loss

  def validation_step(self, val_batch, batch_idx):
    x1,x2,y = val_batch
    logits = self.forward(x1,x2)
    loss_fn = torch.nn.L1Loss(size_average=False)
    loss = loss_fn(logits,y)
    return loss

  def test_step(self, test_batch, batch_idx):
    x1,x2,y = test_batch
    logits = self.forward(x1,x2)
    loss_fn = torch.nn.L1Loss(size_average=False)
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

class TrackerGOTURN(Tracker):
    """GOTURN got10k class for benchmark and evaluation on tracking datasets.

    This class overrides the default got10k 'Tracker' class methods namely,
    'init' and 'update'.

    Attributes:
        device: device on which GOTURN is evaluated ('cuda:0', 'cpu')
        net: GOTURN pytorch model
        prev_box: previous bounding box
        prev_img: previous tracking image
        opts: bounding box config to unscale and uncenter network output.
    """
    def __init__(self, net_path=None, **kargs):
        super(TrackerGOTURN, self).__init__(name='PyTorchGOTURN', is_deterministic=True)
        # setup GPU device if available
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        # setup model
#         self.net = GoNet()
#         if net_path is not None:
#             checkpoint = torch.load(net_path, map_location=lambda storage, loc: storage)
#             self.net.load_state_dict(checkpoint['state_dict'])
#             self.net.eval()
#         self.net = self.net.to(self.device)
        self.net = Tracker2.load_from_checkpoint('./lightning_logs/version_663640/checkpoints/epoch=1-step=4558.ckpt')
        self.net.eval()
        # setup transforms
        self.prev_img = None  # previous image in numpy format
        self.prev_box = None  # follows format: [xmin, ymin, xmax, ymax]
        self.opts = None

    def init(self, image, box):
        """
        Initiates the tracker at given box location.
        Aassumes that the initial box has format: [xmin, ymin, width, height]
        """
        image = np.array(image)
        if image.ndim == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        # goturn helper functions expect box in [xmin, ymin, xmax, ymax] format
        box[2], box[3] = box[0] + box[2], box[1] + box[3]
        self.prev_box, self.prev_img = box, image

    def _get_sample(self, image):
        """
        Returns cropped previous and current frame at the previous predicted
        location. Note that the images are scaled to (224,224,3).
        """
        prev_sample, opts_prev = crop_sample({'image': self.prev_img, 'bb': self.prev_box})
        curr_sample, opts_curr = crop_sample({'image': image, 'bb': self.prev_box})
        curr_img = scale(curr_sample, opts_curr,224,224)['image']
        prev_img = scale(prev_sample, opts_prev,224,224)['image']
        self.curr_img, self.opts = image, opts_curr
        return {'previmg': prev_img, 'currimg': curr_img, 'bb': curr_sample['bb']}
    
    def update(self, image):
        """
        Given current image, returns target box.
        """
        image = np.array(image)
        if image.ndim == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        # crop current and previous image at previous box location
        sample = _get_sample(self, image)
        sample = transforms.Compose([NormalizeToTensor()])(sample)
        # do forward pass to get box
        bbox = np.array(self._get_rect(sample))
        # update previous bbox and image
        self.prev_img, self.prev_bbox = image, np.copy(bbox)
        # convert [xmin, ymin, xmax, ymax] bbox to [xmin, ymin, width, height]
        # for correct evaluation by got10k toolkit
        bbox[2], bbox[3] = bbox[2] - bbox[0], bbox[3]-bbox[1]
        return bbox
    
    def _get_rect(self, sample):
        """
        Regresses the bounding box coordinates in the original image dimensions
        for an input sample.
        """
        x1, x2 = sample['previmg'].unsqueeze(0).to(self.device), \
                sample['currimg'].unsqueeze(0).to(self.device)
#         y = self.net(x1, x2)
        y = sample['bb']
        bbox = y.data.cpu().numpy().transpose((1, 0))[:, 0]
        bbox = BoundingBox(bbox[0], bbox[1], bbox[2], bbox[3])
        # inplace conversion
        bbox.unscale(self.opts['search_region'])
        bbox.uncenter(self.curr_img, self.opts['search_location'],
                      self.opts['edge_spacing_x'], self.opts['edge_spacing_y'])
        return bbox.get_bb_list()