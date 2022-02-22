import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch.utils.data as data
import cv2
import base64
import numpy as np
from PIL import Image
from skimage import io as skio

class ResizeBinarizeVGG16(object):
    def __call__(self, image):        
        img_type = type(image)
        #print(type)
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
    
class ResizeBinarizeEff(object):
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
    
    
class AILabDataset(data.Dataset):
    def __init__(self, 
                 dataframe, 
                 label2Idx, 
                 class_column, 
                 transform=None):
        self.dataframe = dataframe
        self.class_column = class_column
        self.root_dir=root_dir
        self.transform = transform
        self.label2Idx = label2Idx        

    def __len__(self):
        return len(self.dataframe)
    

class AILabDataset1(AILabDataset):
    def __init__(self, dataframe, label2Idx, class_column, transform=None):
        super(AILabDataset1, self).__init__(dataframe, 
                                              label2Idx,
                                              class_column, 
                                              transform)
    
    def __getitem__(self, index):
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
    
class AILabDataset3(AILabDataset):
    def __init__(self, dataframe, label2Idx, class_column, transform=None):
        super(AILabDataset3, self).__init__(dataframe, 
                                              label2Idx,
                                              class_column, 
                                              transform)

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
    
    
class AILabDataset2(AILabDataset):
    def __init__(self, dataframe, label2Idx, class_column, transform=None):
        super(AILabDataset2, self).__init__(dataframe, 
                                              label2Idx,
                                              class_column, 
                                              transform)

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
                
        if self.transform:
            current_page = self.transform(current_page)
            prev_page = self.transform(prev_page)
        
        return (
            [prev_page, current_page],
            label2Idx[row[class_column]],
            file_name
        )
    
    
class TobaccoDataset(data.Dataset):
    def __init__(self, 
                 dataframe, 
                 label2Idx,
                 class_column,
                 root_dir, 
                 transform=None):
        self.dataframe = dataframe
        self.class_column = class_column
        self.root_dir=root_dir
        self.transform = transform
        self.label2Idx = label2Idx        

    def __len__(self):
        return len(self.dataframe)
    
    
class TobaccoDataset1(TobaccoDataset):
    def __init__(self, 
                 dataframe, 
                 label2Idx,
                 class_column,
                 root_dir, 
                 transform=None):
        super(TobaccoDataset1, self).__init__(dataframe, 
                                                label2Idx, 
                                                class_column,
                                                root_dir, 
                                                transform)
    
    def __getitem__(self, index):
        if type(index) != int:
            i = index.item()
        else: i = index
        row = self.dataframe.iloc[i]
        file_name = row["docid"]+'.tif'
        img = skio.imread(self.root_dir+file_name,plugin='pil')
        
        if self.transform:
            img = self.transform(img)
        
        return (
            img,
            self.label2Idx[row[self.class_column]],
            file_name
        )
    
class TobaccoDataset2(TobaccoDataset):
    def __init__(self, 
                 dataframe, 
                 label2Idx, 
                 class_column,
                 root_dir, 
                 transform=None):
        super(TobaccoDataset2, self).__init__(dataframe, 
                                                label2Idx,
                                                class_column,
                                                root_dir, 
                                                transform)

    def __getitem__(self, index):
        if type(index) != int:
            i = index.item()
        else: i = index
        row = self.dataframe.iloc[i]
        file_name = row["docid"]+'.tif'
        
        current_page = skio.imread(self.root_dir+file_name,plugin='pil')
        prev_name = 'UNKNOWN'
        
        if i == 0:
            prev_page = np.full_like(current_page, 255)
        else:
            prev_row = self.dataframe.iloc[i-1]
            prev_name = prev_row["docid"]+'.tif'
            prev_page = skio.imread(self.root_dir+prev_name,plugin='pil')
        
        if self.transform:
            #print(file_name)
            current_page = self.transform(current_page)
            #print(prev_name)
            prev_page = self.transform(prev_page)
        
        
        return (
            [prev_page, current_page],
            self.label2Idx[row[self.class_column]],
            file_name
        )
    
class TobaccoDataset3(TobaccoDataset):
    def __init__(self, 
                 dataframe, 
                 label2Idx, 
                 class_column,
                 root_dir, 
                 transform=None):
        super(TobaccoDataset3, self).__init__(dataframe, 
                                                label2Idx, 
                                                class_column,  
                                                root_dir, 
                                                transform)

    def __getitem__(self, index):
        if type(index) != int:
            i = index.item()
        else: i = index
            
        row = self.dataframe.iloc[index]
        file_name = row["docid"]+'.tif'
        target_page = skio.imread(self.root_dir+file_name,plugin='pil')
        label = row[self.class_column]        
        
        if index < 1:
            next_row = self.dataframe.iloc[index+1]
            prev_page = np.full_like(target_page, 255)
            next_name = next_row["docid"]+'.tif'
            next_page = skio.imread(self.root_dir+next_name,plugin='pil')
        elif index >= len(self.dataframe) - 1:
            prev_row = self.dataframe.iloc[index-1]
            next_page = np.full_like(target_page, 255)
            prev_name = prev_row["docid"]+'.tif'
            prev_page = skio.imread(self.root_dir+prev_name,plugin='pil')
        else:
            prev_row = self.dataframe.iloc[index-1]
            next_row = self.dataframe.iloc[index+1]
            prev_name = prev_row["docid"]+'.tif'
            prev_page = skio.imread(self.root_dir+prev_name,plugin='pil')
            next_name = next_row["docid"]+'.tif'
            next_page = skio.imread(self.root_dir+next_name,plugin='pil')            
                    
        if self.transform:
            target_page = self.transform(target_page)
            prev_page = self.transform(prev_page)
            next_page = self.transform(next_page)
            
        return (
            [prev_page, target_page, next_page],
            self.label2Idx[label],
            file_name
        )    