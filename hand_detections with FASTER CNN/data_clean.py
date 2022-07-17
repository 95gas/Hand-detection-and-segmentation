import os
import glob
from xml.dom.minidom import parse

def read_data_ego(data_dir, is_train):
    # print(os.path.join(data_dir, 'train' if is_train else 'valid', '*.jpg'))
    images = sorted(glob.glob(os.path.join(data_dir, 'train' if is_train else 'valid', '*.jpg')))

    targets = sorted(glob.glob(os.path.join(data_dir, 'train' if is_train else 'valid', '*.xml')))

    print(len(images))
    print(len(targets))
        
    return images, targets

def clean_dataset(data_dir, is_train):
    img_paths, xml_paths = read_data_ego(data_dir, is_train)

    for i in range(len(xml_paths)):
        img_path = img_paths[i]
        xml_path = xml_paths[i]
        dom = parse(xml_path)
        data = dom.documentElement
        objects = data.getElementsByTagName('object')
        if len(objects) == 0:
            print(img_path)
            os.remove(img_path)
            os.remove(xml_path)


clean_dataset('./dataset/EgoHands Public.v1-specific.voc', True)
clean_dataset('./dataset/EgoHands Public.v1-specific.voc', False)