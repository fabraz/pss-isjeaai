import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch.utils.data as data
import cv2
import base64
import numpy as np

class ResizeBinarizeVGG(object):
    def __call__(self, image):        
        img_type = type(image)
        if img_type == type(None):
            image = np.zeros((224,224),dtype='uint8')
        if img_type == str:
            if image[-4:] in ['.jpg', '.png', 'jpeg']:
                image = cv2.imread(image, cv2.IMREAD_GRAYSCALE)
            else:
                image = self.base64_to_cv2(image, mode=cv2.IMREAD_GRAYSCALE)
        resized = cv2.resize(image, (224,224), interpolation = cv2.INTER_AREA)
        retval,bin = cv2.threshold(resized,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        return bin
    
    def base64_to_cv2(self, img, mode=cv2.IMREAD_COLOR):
        img = base64.b64decode(img)
        img = np.frombuffer(img, np.uint8)
        img = cv2.imdecode(img, mode)
        return img
    
class ResizeBinarizeEFF(object):
    def __call__(self, image):        
        img_type = type(image)
        if img_type == type(None):
            image = np.zeros((224,224),dtype='uint8')
        if img_type == str:
            if image[-4:] in ['.jpg', '.png', 'jpeg']:
                image = cv2.imread(image, cv2.IMREAD_GRAYSCALE)
            else:
                image = self.base64_to_cv2(image, mode=cv2.IMREAD_GRAYSCALE)
        resized = cv2.resize(image, (224,224), interpolation = cv2.INTER_CUBIC)
        retval,bin = cv2.threshold(resized,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        return bin
    
    def base64_to_cv2(self, img, mode=cv2.IMREAD_COLOR):
        img = base64.b64decode(img)
        img = np.frombuffer(img, np.uint8)
        img = cv2.imdecode(img, mode)
        return img

class AILabDatasetOne(data.Dataset):
    
    def __init__(self, dataframe, class_column, transform=None):
        self.dataframe = dataframe
        self.transform = transform
        self.class_column = class_column

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, index):
        label2Idx = {'FirstPage' : 1, 'NextPage' : 0}
        if type(index) != int:
            i = index.item()
        else: i = index
        row = self.dataframe.iloc[i]
        file_name = self.split_name(row["page_name"],14)
        img = row["img"]
        
        
        if self.transform:
            img = self.transform(img)
        
        return (
            img,
            label2Idx[row[self.class_column]],
            file_name
        )
    def split_name(self, name, max_size):
        new_name = ''
        while name:
            new_name += (name[:max_size]) + '\n'
            name = name[max_size:]
        return new_name
    
class AILabDatasetThree(data.Dataset):
    'Characterizes a dataset for PyTorch'
    
    def __init__(self, dataframe, label2Idx, class_column, transform=None):
        self.dataframe = dataframe
        self.transform = transform
        self.label2Idx = label2Idx
        self.class_column = class_column

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, index):
        if type(index) != int:
            i = index.item()
        else: i = index
        row = self.dataframe.iloc[i]
        
        file_name = self.split_name(row["page_name"],14)
        
        current_page = row["img"]
        
        prev_row = self.dataframe.loc[self.dataframe['page_name'] == row["prev_page_name"]]
        
        if len(prev_row)>0:
            prev_page = prev_row.img.values[0]
        else:
            prev_page = None
            
        if i < len(self.dataframe) - 1:
            next_row = self.dataframe.iloc[i+1]
            next_page = next_row["img"]
        else:
            next_page = None
                
        if self.transform:
            current_page = self.transform(current_page)
            prev_page = self.transform(prev_page)
            next_page = self.transform(next_page)
        
        return (
            [prev_page, current_page, next_page],
            self.label2Idx[row[self.class_column]],
            file_name
        )
    
    def split_name(self, name, max_size):
        new_name = ''
        while name:
            new_name += (name[:max_size]) + '\n'
            name = name[max_size:]
        return new_name

    
    
class AILabDatasetTwo(data.Dataset):
    'Characterizes a dataset for PyTorch'
    
    def __init__(self, dataframe, transform=None):
        self.dataframe = dataframe
        self.transform = transform

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, index):
        label2Idx = {'FirstPage' : 1, 'NextPage' : 0}
        if type(index) != int:
            i = index.item()
        else: i = index
        row = self.dataframe.iloc[i]
        
        file_name = self.split_name(row["page_name"],14)
        
        current_page = row["img"]
        
        prev_row = self.dataframe.loc[self.dataframe['page_name'] == row["prev_page_name"]]
        
        if len(prev_row)>0:
            prev_page = prev_row.img.values[0]
        else:
            prev_page = None
                
        if self.transform:
            current_page = self.transform(current_page)
            prev_page = self.transform(prev_page)
        
        return (
            [prev_page, current_page],
            label2Idx[row["class"]],
            file_name
        )
    
    def split_name(self, name, max_size):
        new_name = ''
        while name:
            new_name += (name[:max_size]) + '\n'
            name = name[max_size:]
        return new_name