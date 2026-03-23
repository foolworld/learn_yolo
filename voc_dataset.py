import xmltodict
from PIL import Image
from torch.utils.data import Dataset
import torch
import os

from torchvision import transforms


class VOCDataset(Dataset):
    def __init__(self, image_folder, label_folder, transform,label_transform):
        self.image_folder = image_folder
        self.label_folder = label_folder
        self.transform = transform
        self.label_transform = label_transform
        self.img_names = os.listdir(self.image_folder)
        self.classes_list = ["no helmet","motor","number","with helmet"]

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, idx):
        img_name = self.img_names[idx]
        img_path = os.path.join(self.image_folder, img_name)
        image = Image.open(img_path).convert('RGB')
        # new1,png -> new1.xml
        # new1.png -> [new1,png] -> new1 + '.xml"
        label_name = img_name.split('.')[0] + ".xml"
        label_path = os.path.join(self.label_folder, label_name)
        with open(label_path,"r",encoding="utf-8") as f:
            label_content = f.read()
        label_dict = xmltodict.parse(label_content)
        objects = label_dict['annotation']['object']
        target = []
        for object in objects:
            object_name = object['name']
            object_class_id = self.classes_list.index(object_name)
            object_xmax = float(object["bndbox"]['xmax'])
            object_ymax = float(object["bndbox"]['ymax'])
            object_xmin = float(object["bndbox"]['xmin'])
            object_ymin = float(object["bndbox"]['ymin'])

            target.extend([object_class_id,object_ymin,object_xmax,object_ymax])

        target = torch.tensor(target)
        if self.transform is  not  None:
            image =self.transform(image)
        return image, label_content


if __name__ == '__main__':
    train_dataset = VOCDataset(r"C:\Users\xs\Desktop\learn_yolo\YOLO\HelmetDataset-VOC\train\images",r"C:\Users\xs\Desktop\learn_yolo\YOLO\HelmetDataset-VOC\train\labels",transforms.ToTensor(),None)
    print(len(train_dataset))
    print(train_dataset[11])