import os 
import torch 
import numpy as np 
import torch.nn.functional as F

from torchvision import transforms
from augmentations import RandomFlipPair 


class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, root, train, transform=None):
        self.root = os.path.abspath(root)  
        self.data_root = os.path.join(root, 'data') 
        self.label_root = os.path.join(root, 'label') 

        self.train = train 
        self.transform = transform 
        self.flip_transform = RandomFlipPair(p_h=0.5, p_v=0.5)

        data_files = sorted(os.listdir(self.data_root)) 
        if train == 'train': 
            self.data_files = data_files[100:] 
            print('initializing train dataset: ', len(self.data_files))
        elif train == 'val': 
            self.data_files = data_files[:100] 
            print('initializing test dataset: ', len(self.data_files)) 
        elif train == 'test': 
            self.data_files = os.listdir('/workspace/neural-nets-route-planner/test_dataset/data') 
            self.data_root = '/workspace/neural-nets-route-planner/test_dataset/data'
            self.label_root = '/workspace/neural-nets-route-planner/test_dataset/label'
            print('initializing test dataset: ', len(self.data_files)) 

    def __len__(self):
        return len(self.data_files)

    def __getitem__(self, index):
        label_fname = self.data_files[index].replace('a_star', 'a_star_gt')
 
        data_path = os.path.join(self.data_root, self.data_files[index]) 
        label_path = os.path.join(self.label_root, label_fname) 

        data = np.load(data_path) 
        label = np.array([np.load(label_path)]) 

        if self.transform: 
            data = data.astype(np.float32) 
            label = label.astype(np.float32)
            data = self.transform(np.transpose(data, (1, 2, 0))) 
            label = self.transform(np.transpose(label, (1, 2, 0))) 
            if self.train == 'train':
                data, label = self.flip_transform(data, label)

        return data, label 
    

class TestDataset(torch.utils.data.Dataset):
    def __init__(self, root, start, goal, transform=None):
        self.root = os.path.abspath(root) 
        self.data_root = os.path.join(root, 'data') 
        self.label_root = os.path.join(root, 'label') 
        self.start = start 
        self.goal = goal 

        self.transform = transform 

        self.data_files = os.listdir(self.data_root)[0] 

    def __len__(self):
        return len(self.data_files)

    def __getitem__(self, index):
 
        data_path = os.path.join(self.data_root, self.data_files) 

        data = np.load(data_path) 

        data[:2, :, :] = 0.0 

        data[0, 64-self.start[1]-1, self.start[0]] = 1.0 
        data[1, 64-self.goal[1]-1, self.goal[0]] = 1.0 

        c1 = data[0] + data[1] 
        c2 = data[2]
        c3 = np.zeros_like(c1)
        grid_map = np.stack([c1, c2, c3])  

        if self.transform: 
            data = data.astype(np.float32) 
            data = self.transform(np.transpose(data, (1, 2, 0))) 
            grid_map = self.transform(np.transpose(grid_map, (1, 2, 0)))
        

        return data, grid_map 
    

if __name__ == '__main__': 

    import numpy as np
    import matplotlib.pyplot as plt  

    from augmentations import ChannelwisePad 
    from utils import vis_numpy_array


    data_transform = transforms.Compose([
        transforms.ToTensor(),  
        ChannelwisePad(target_height=64, target_width=96, padding_values=[0.0, 0.0, 1.0]),
    ])

    train_dataset = CustomDataset(root='./dataset', train=True, transform=data_transform) 

    data_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=10, shuffle=True)

    # data, label = train_dataset[7] 
    datas, labels = next(iter(data_loader)) 

    data = datas[0] 
    label = labels[0] 

    print(labels.shape)
    print(data.shape) 
    print(label.shape) 
    print(data.dtype) 
    print(label.dtype) 
    print(data.max())
    print(data.min())
    print('channel 1 sum : ', data[0].sum()) 
    print('channel 2 sum: ', data[1].sum())
    print('channel 3 sum: ', data[2].sum())

    # vis_numpy_array(data[0].numpy()) 
    # vis_numpy_array(data[1].numpy()) 
    # vis_numpy_array(data[2].numpy()) 
    # vis_numpy_array(label[0].numpy()) 
    # vis_numpy_array(data.numpy())
    c1 = torch.tensor(data[0] + data[1]).unsqueeze(0)
    c2 = torch.tensor(data[2]).unsqueeze(0) 
    torch_array = torch.cat([c1, c2, label], axis=0) 
    print(torch_array.shape)
    vis_numpy_array(torch_array.cpu().numpy())
    
