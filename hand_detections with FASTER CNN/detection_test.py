import torch
import numpy as np

import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

from PIL import ImageDraw, Image

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

model = get_object_detection_model(2)
model.load_state_dict(torch.load('./mask_rcnn_pedestrian_model.pt'))
model.eval()

transform = torchvision.transforms.Compose(
    [torchvision.transforms.ToTensor()]
)

img_path = "/home/readlnh/workspace/pytorch-learn/detection/dataset/EgoHands Public.v1-specific.tensorflow/test/CARDS_COURTYARD_B_T_frame_1024_jpg.rf.9e96aa293d2f73e690259e37130a8c1f.jpg"
img = Image.open(img_path).convert('RGB')
img = transform(img)

# with torch.no_grad:
predictor = model([img])
boxes = predictor[0]['boxes'].cpu().detach().numpy()
scores = predictor[0]['scores'].cpu().detach().numpy()
image = Image.fromarray(img.mul(255).permute(1, 2, 0).byte().numpy())

index = 0
print(boxes)
draw = ImageDraw.Draw(image)
for x1, y1, x2, y2 in boxes:
    if scores[index] > 0.9:
        draw.rectangle([x1, y1, x2, y2])
    index += 1

image.show()





