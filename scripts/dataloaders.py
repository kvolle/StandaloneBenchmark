import torch
import numpy as np

##from utils import plot_images
from torchvision import datasets
from torchvision.datasets import ImageFolder

from torchvision import transforms
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import DataLoader, Subset, Dataset

import cv2
from PIL import Image, ImageFile

OUT_SIZE = 224


def get_CelebA_loader(data_dir,
                    batch_size,
                    shape = (3, 64, 64),
                    augment = False,
                    shuffle=True,
                    num_workers=4,
                    pin_memory=False):
    """
    Utility function for loading and returning a multi-process
    test iterator over the CIFAR-10 dataset.
    If using CUDA, num_workers should be set to 1 and pin_memory to True.
    Params
    ------
    - data_dir: path directory to the dataset.
    - batch_size: how many samples per batch to load.
    - shape: size of images to return 
    - augment: whether to apply the data augmentation scheme
      mentioned in the paper. Only applied on the train split.
    - shuffle: whether to shuffle the dataset after every epoch.
    - num_workers: number of subprocesses to use when loading the dataset.
    - pin_memory: whether to copy tensors into CUDA pinned memory. Set it to
      True if using GPU.
    Returns
    -------
    - data_loader: test set iterator.
    """
    if (shape[0] == 1):
        #normalize = transforms.Normalize(mean=0.1307, std=0.3081)
        normalize = transforms.Normalize(mean=0., std=1.)
    elif (shape[0] == 3):
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    else:
        assert False, "Expected number of channels to be either 1 or 3"

    # define transform
    if augment:
        transform = transforms.Compose([
            #transforms.ColorJitter(),
            #transforms.GaussianBlur(1),
            transforms.ToTensor(),
            #transforms.Grayscale(),
            #normalize,
            transforms.RandomHorizontalFlip(),
        ])
    else:
        transform = transforms.Compose([
            transforms.ToTensor(),
            #normalize,
        ])
    
    dataset = CelebADataset(names_file = data_dir, channels = shape[0], transforms = transform)

    data_loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle,
        num_workers=num_workers, pin_memory=pin_memory,
    )

    return data_loader

class CelebADataset(Dataset):
    def __init__(self, names_file, channels = 3, transforms=None):
        
        self.transforms = transforms
        f = open(names_file, 'r')
        self.names = f.read().splitlines()
        f.close()
        self.num_samples = len(self.names)
        print(self.num_samples)
        if channels == 1:
            self.read_mode = 0
        else:
            self.read_mode = 1


    def __getitem__(self, index):
        filename = self.names[index]

        filename = './data/celeba/50k/' + filename + '.jpg'
        image = cv2.imread(filename, self.read_mode)
            
        if image is None:
            print(f"Error loading image: {filename}")
            image = np.zeros((64, 64), dtype=np.float32)#uint8)
            
        else:
            image = image/255.
            image = image.astype(np.float32)
        if len(image.shape) == 2:
            image = np.expand_dims(image, 0)
        else:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        #image = torch.as_tensor(image)

        # Add transforms
        if self.transforms is not None:
            image = self.transforms(image)
        else:
            pass
        label = self.names[index]
        return {'data': image, 'target': label}

    def __len__(self):
        return self.num_samples
    

class SquareCropAndResize(torch.nn.Module):
    def __init__(self, size=None):
        super().__init__()
        self.size = size

    def __call__(self, img):
        """
        Args:
            img (PIL Image or Tensor): Image to be cropped.

        Returns:
            PIL Image or Tensor: Cropped image.
        """
        
        h, w = img.shape[-2:]
        

        #print("W: %d H: %d" % (w, h))
        min_dim = min(h, w)
        img = transforms.functional.center_crop(img, min_dim)

        if self.size is None:
            return img
        else:
            return transforms.functional.resize(img, self.size, antialias=True)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(size={self.size})"

def get_AltPet_loader(data_dir,
                    batch_size,
                    shape = (3, OUT_SIZE, OUT_SIZE),
                    augment = False,
                    shuffle=True,
                    num_workers=4,
                    pin_memory=False):
    """
    Utility function for loading and returning a multi-process
    test iterator over the CIFAR-10 dataset.
    If using CUDA, num_workers should be set to 1 and pin_memory to True.
    Params
    ------
    - data_dir: path directory to the dataset.
    - batch_size: how many samples per batch to load.
    - shape: size of images to return 
    - augment: whether to apply the data augmentation scheme
      mentioned in the paper. Only applied on the train split.
    - shuffle: whether to shuffle the dataset after every epoch.
    - num_workers: number of subprocesses to use when loading the dataset.
    - pin_memory: whether to copy tensors into CUDA pinned memory. Set it to
      True if using GPU.
    Returns
    -------
    - data_loader: test set iterator.
    """
    if (shape[0] == 1):
        #normalize = transforms.Normalize(mean=0.1307, std=0.3081)
        normalize = transforms.Normalize(mean=0., std=1.)
    elif (shape[0] == 3):
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    else:
        assert False, "Expected number of channels to be either 1 or 3"
    # TODO make this more general so it doesn't have to be square
    squarecrop = SquareCropAndResize(shape[1])

    # define transform
    if augment:
        transform = transforms.Compose([
            #transforms.ColorJitter(),
            #transforms.GaussianBlur(1),
            transforms.ToTensor(),
            squarecrop,
            #transforms.Grayscale(),
            #normalize,
            transforms.RandomHorizontalFlip(),
        ])
    else:
        transform = transforms.Compose([
            transforms.ToTensor(),
            squarecrop,
            #normalize,
        ])
    
    dataset = AltPetDataset(names_file = data_dir, channels = shape[0], transforms = transform)

    data_loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle,
        num_workers=num_workers, pin_memory=pin_memory,
    )

    return data_loader

class AltPetDataset(Dataset):
    def __init__(self, names_file, channels = 3, transforms=None):
        
        self.transforms = transforms
        f = open(names_file, 'r')
        self.names = f.read().splitlines()
        f.close()
        self.num_samples = len(self.names)
        #ImageFile.LOAD_TRUNCATED_IMAGES=True
        f = open('./data/AltPets/folder_names.txt', 'r')
        breeds = f.read().splitlines()
        f.close()
        self.label_from_breed = {}
        for i, b in enumerate(breeds):
            self.label_from_breed[b] = i
        if channels == 1:
            self.read_mode = 0
        else:
            self.read_mode = 1


    def __getitem__(self, index):
        filename = self.names[index]
        split_strings = filename.split('/')
        breed = split_strings[1]
        filename = './data/AltPets/' + breed + '/' + split_strings[-1]
        image = cv2.imread(filename, self.read_mode)
        
        #parts = filename.split("/")
        #result = "/" + parts[-2] + "/" + parts[-1]
        #image=cv2.imread("../../data/AltPets" + result,0)
        
        #try:
        #    image = cv2.imread(filename,0)
        #    if image is None:
        #        raise Exception(f"Error loading image: {filename}")
        #except Exception as e:
        #    print(e)
            
        if image is None:
            print(f"Error loading image: {filename}")
            image = np.zeros((224, 224), dtype=np.float32)#uint8)
            
        else:
            image = image/255.
            image = image.astype(np.float32)
        if len(image.shape) == 2:
            image = np.expand_dims(image, 0)
        else:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        #image = torch.as_tensor(image)

        # Add transforms
        if self.transforms is not None:
            image = self.transforms(image)
        else:
            pass
        label = self.label_from_breed[breed]

        return {'data': image, 'target': label}

    def __len__(self):
        return self.num_samples
    
def get_PetExpression_loader(data_dir,
                             batch_size,
                             shape = (3, 64, 64),
                             augment = False,
                             shuffle=True,
                             num_workers=4,
                             pin_memory=False):
    C, IMG_H, IMG_W = shape
    IMG_MEAN = [0.0, 0.0, 0.0] # [0.485, 0.456, 0.406] # 
    IMG_STD = [1.0, 1.0, 1.0] # [0.229, 0.224, 0.225] #
    normalize = transforms.Normalize(mean=IMG_MEAN,
                                 std=IMG_STD)
    
    if augment:
        transform =     transforms.Compose([
            transforms.Resize((IMG_H, IMG_W)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize
        ])
    else:
        transform = transforms.Compose([
            transforms.Resize((IMG_H, IMG_W)),
            transforms.ToTensor(),
            normalize
        ])

    dataset = PetDataset(data_dir, transform=transform)

    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=pin_memory)


class PetDataset(Dataset):
    def __init__(self,root_dir,transform=None):
        self.dataset = ImageFolder(root = root_dir,transform=transform)
        self.transform = transform
        
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self,index):
        image , label  = self.dataset[index]
        samples = {'data':image,'target':label}
        return samples