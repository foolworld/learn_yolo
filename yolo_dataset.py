import xmltodict
from PIL import Image
from torch.utils.data import Dataset
import torch
import os

from torchvision import transforms


class YOLODataset(Dataset):
    def __init__(self, image_folder, label_folder, transform,label_transform):
        self.image_folder = image_folder
        self.label_folder = label_folder
        self.transform = transform
        self.label_transform = label_transform
        self.img_names = os.listdir(self.image_folder)
        # self.classes_list = ["no helmet","motor","number","with helmet"]

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, idx):
        img_name = self.img_names[idx]
        img_path = os.path.join(self.image_folder, img_name)
        image = Image.open(img_path).convert('RGB')
        # new1,png -> new1.txt
        # new1.png -> [new1,png] -> new1 + '.xml"
        label_name = img_name.split('.')[0] + ".txt"
        label_path = os.path.join(self.label_folder, label_name)
        with open(label_path,"r",encoding="utf-8") as f:
            label_content = f.read()

        object_infos = label_content.strip().split("\n") #strip去除头尾的空字符串再按照换行进行分割

        target=[]
        for object_info in object_infos:
            info_list = object_info.strip().split(" ")
            class_id = float(info_list[0])
            center_x = float(info_list[1])
            center_y = float(info_list[2])
            width = float(info_list[3])
            height = float(info_list[4])
            target.extend([class_id,center_x,center_y,width,height])
        # label_dict = xmltodict.parse(label_content)
        # objects = label_dict['annotation']['object']
        # target = []
        # for object in objects:
        #     object_name = object['name']
        #     object_class_id = self.classes_list.index(object_name)
        #     object_xmax = float(object["bndbox"]['xmax'])
        #     object_ymax = float(object["bndbox"]['ymax'])
        #     object_xmin = float(object["bndbox"]['xmin'])
        #     object_ymin = float(object["bndbox"]['ymin'])
        #
        #     target.extend([object_class_id,object_ymin,object_xmax,object_ymax])
        #
        target = torch.tensor(target)
        if self.transform is  not  None:
            image =self.transform(image)
        return image, target


if __name__ == '__main__':
    train_dataset = YOLODataset(r"C:\Users\xs\Desktop\learn_yolo\YOLO\HelmetDataset-YOLO\train\images",r"C:\Users\xs\Desktop\learn_yolo\YOLO\HelmetDataset-YOLO\train\labels",transforms.ToTensor(),None)
    print(len(train_dataset))
    print(train_dataset[11])