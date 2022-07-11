from cProfile import label
import torch
import os
import glob
import numpy as np
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from PIL import Image

from xml.dom.minidom import parse

import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

import utils
import transforms as T
from engine import train_one_epoch, evaluate

# tensorflow csv formart
# def read_data_ego(data_dir, is_train):
#     csv_fname = os.path.join(data_dir, 'train' if is_train
#                             else 'valid', '_annotations.csv')
    
#     print(csv_fname)
#     csv_data = pd.read_csv(csv_fname)
#     csv_data = csv_data.set_index('filename')
#     images, targets = [], []
#     for img_name, target in csv_data.iterrows():
#         images.append(img_name)
#         targets.append(list(target))
        
#     return images, targets


# Here i use VOC format dataset
def read_data_ego(data_dir, is_train):
    images = sorted(glob.glob(os.path.join(data_dir, 'train' if is_train else 'valid', '*.jpg')))

    targets = sorted(glob.glob(os.path.join(data_dir, 'train' if is_train else 'valid', '*.xml')))

    print(len(images))
    print(len(targets))
        
    return images, targets

class HandDataset(torch.utils.data.Dataset):
    def __init__(self, data_dir, is_train, transforms=None):
        self.data_dir = data_dir
        self.is_train = is_train
        self.transforms = transforms
        self.imgs, self.bbox_xml = read_data_ego(data_dir, is_train)
        
    def __getitem__(self, idx):
        img_path = self.imgs[idx]
        img = Image.open(img_path).convert('RGB')
        
        bbox_xml_path = self.bbox_xml[idx]
        
        dom = parse(bbox_xml_path)
        data = dom.documentElement

        objects = data.getElementsByTagName('object')   

        # get bounding box coordinates
        boxes = []
      
        for object_ in objects:
            bndbox = object_.getElementsByTagName('bndbox')[0]
            xmin = np.float32(bndbox.getElementsByTagName('xmin')[0].childNodes[0].nodeValue)
            ymin = np.float32(bndbox.getElementsByTagName('ymin')[0].childNodes[0].nodeValue)
            xmax = np.float32(bndbox.getElementsByTagName('xmax')[0].childNodes[0].nodeValue)
            ymax = np.float32(bndbox.getElementsByTagName('ymax')[0].childNodes[0].nodeValue)
            boxes.append([xmin, ymin, xmax, ymax])     
 
        

        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.ones((len(objects),), dtype=torch.int64)
        image_id = torch.tensor([idx])
        iscrowd = torch.zeros((len(objects),), dtype=torch.int64)

        # boxes = torch.as_tensor(boxes, dtype=torch.float32)
   
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd

        if self.transforms is not None:
            img, target = self.transforms(img, target)


        return img, target 

    def __len__(self):
        return len(self.imgs)

data = HandDataset('./dataset/EgoHands Public.v1-specific.voc', True)
print(data[0])

def get_object_detection_model(num_classes):
    # load an object detection model pre-trained on COCO
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    
    # replace the classifier with a new one, that has num_classes which is user-defined
    #num_classes = 2  # 2 class hand + background
 
    # get the number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
 
    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
 
    return model


 
def get_transform(train):
    transforms = []
    # converts the image, a PIL image, into a PyTorch Tensor
    transforms.append(T.ToTensor())
    if train:
        # during training, randomly flip the training images
        # and ground-truth for data augmentation
        # RandomHorizontalFlip with 50% 
        transforms.append(T.RandomHorizontalFlip(0.5))
 
    return T.Compose(transforms)


from engine import train_one_epoch, evaluate
import utils


root = './dataset/EgoHands Public.v1-specific.voc'

# train on the GPU or on the CPU, if a GPU is not available
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# 2 classes, hands，background
num_classes = 2
# use our dataset and defined transformations
dataset = HandDataset(root, True, get_transform(train=True))
# print(dataset[0])
dataset_test = HandDataset(root, False, get_transform(train=False))


# randomly choose data from the dataset and the dataset_test
indices = torch.randperm(len(dataset)).tolist()
dataset = torch.utils.data.Subset(dataset, indices[:400])

indices_test = torch.randperm(len(dataset_test)).tolist()
dataset_test = torch.utils.data.Subset(dataset_test, indices_test[-100:])

# define training and validation data loaders
# 在colab里训练模型时num_workers参数只能为2
# if you want to train the model in colab, the max value of `num_workers` is 2, otherwise you can set it bigger.
data_loader = torch.utils.data.DataLoader(
    dataset, batch_size=2, shuffle=True, num_workers=2,
    collate_fn=utils.collate_fn)

data_loader_test = torch.utils.data.DataLoader(
    dataset_test, batch_size=2, shuffle=False,  num_workers=2,
    collate_fn=utils.collate_fn)

# get the model using our helper function
model = get_object_detection_model(num_classes)

# move model to the right device
model.to(device)

# construct an optimizer
params = [p for p in model.parameters() if p.requires_grad]

# SGD
optimizer = torch.optim.SGD(params, lr=0.0003,
                            momentum=0.9, weight_decay=0.0005)

# and a learning rate scheduler
# 学习率(learning rate)
lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=1, T_mult=2)

# let's train it for   epochs
num_epochs = 5

for epoch in range(num_epochs):
    # train for one epoch, printing every 10 iterations
    train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq=50)

    # update the learning rate
    lr_scheduler.step()

    # evaluate on the test dataset    
    evaluate(model, data_loader_test, device=device)    
    
    print('')
    print('==================================================')
    print('')

print("That's it!")

torch.save(model.state_dict(), "./mask_rcnn_pedestrian_model.pt")


# test code, just ignpre it
# # pick one image from the test set
# img, _ = dataset_test[0]
# # put the model in evaluation mode
# model.eval()
# with torch.no_grad():
#     prediction = model([img.to(device)])


# from PIL import ImageDraw, Image
# boxes = prediction[0]['boxes'].cpu().detach().numpy()
# #print(boxes)
# x1, y1, x2, y2 = boxes[0]
# image = Image.fromarray(img.mul(255).permute(1, 2, 0).byte().numpy())
# draw = ImageDraw.Draw(image)
# draw.rectangle([x1, y1, x2, y2])
# image.show()
# image