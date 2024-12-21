import torch 
import torch.nn as nn 
from torchvision import transforms
import segmentation_models_pytorch as smp 

import math 
import numpy as np 
from importlib import import_module 
import sys 
sys.path.append('..')
from augmentations import ChannelwisePad 


class RNN_D(nn.Module): 
    def __init__(self, input_size, hidden_size, num_layers): 
        super(RNN_D, self).__init__() 
        self.rnn = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, bias=True, bidirectional=False) 

    def forward(self, x, hidden=None): 
        output, hidden = self.rnn(x) if hidden is None else self.rnn(x, hidden) 
        return output, hidden 
    
class RNN_U(nn.Module): 
    def __init__(self, input_size, hidden_size, num_layers): 
        super(RNN_U, self).__init__() 
        self.rnn = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, bias=True, bidirectional=False) 

    def forward(self, x, hidden=None): 
        x = torch.flip(x, dims=[1])
        output, hidden = self.rnn(x) if hidden is None else self.rnn(x, hidden) 
        output = torch.flip(output, dims=[1])
        return output, hidden 
    
class RNN_R(nn.Module): 
    def __init__(self, input_size, hidden_size, num_layers): 
        super(RNN_R, self).__init__() 
        self.rnn = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, bias=True, bidirectional=False) 

    def forward(self, x, hidden=None): 
        x = x.permute(0, 2, 1)
        output, hidden = self.rnn(x) if hidden is None else self.rnn(x, hidden)
        output = output.permute(0, 2, 1) 
        return output, hidden 
    
class RNN_L(nn.Module): 
    def __init__(self, input_size, hidden_size, num_layers): 
        super(RNN_L, self).__init__() 
        self.rnn = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, bias=True, bidirectional=False) 

    def forward(self, x, hidden=None): 
        x = x.permute(0, 2, 1)
        x = torch.flip(x, dims=[1])
        output, hidden = self.rnn(x) if hidden is None else self.rnn(x, hidden) 
        output = torch.flip(output, dims=[1])
        output = output.permute(0, 2, 1) 
        return output, hidden 
    

class RNNModule(nn.Module): 
    def __init__(self, vertical_dim, horizontal_dim, num_layers=1): 
        super(RNNModule, self).__init__() 
        
        self.rnn_d = RNN_D(horizontal_dim, horizontal_dim, num_layers) 
        self.rnn_u = RNN_U(horizontal_dim, horizontal_dim, num_layers)  
        self.rnn_r = RNN_R(vertical_dim, vertical_dim, num_layers) 
        self.rnn_l = RNN_L(vertical_dim, vertical_dim, num_layers)

    # def forward(self, x):  
    #     output_d, hidden_d = self.rnn_d(x) 
    #     output_u, hidden_u = self.rnn_u(x) 
    #     output_r, hidden_r = self.rnn_r(x) 
    #     output_l, hidden_l = self.rnn_l(x) 

    #     output = output_r + output_l + output_d + output_u 
    #     return output  
    
    def forward(self, x): 
        output_d, hidden_d = self.rnn_d(x) 
        output_u, hidden_u = self.rnn_u(x) 
        output_r, hidden_r = self.rnn_r(x) 
        output_l, hidden_l = self.rnn_l(x)  # b x 64 x 96 

        # b x 64 x 96 -> b x 4 x 64 x 96 
        output = torch.stack([output_d, output_u, output_r, output_l], dim=1) 
        return output


class ConvNet(nn.Module):
    def __init__(self, num_classes=1):
        super(ConvNet, self).__init__()
        self.model = smp.Unet(
            encoder_name="resnet34", 
            encoder_weights="imagenet", 
            in_channels=3,                
            classes=num_classes,                  
        ) 

    def forward(self, x):
        return self.model(x) 
    

class ConvModule(nn.Module): 
    def __init__(self, num_classes=1): 
        super(ConvModule, self).__init__() 
        self.conv_module = nn.Sequential(
                                nn.Conv2d(in_channels=4, out_channels=4, kernel_size=3, stride=1, padding=1),
                                nn.BatchNorm2d(4), 
                                nn.ReLU(),
                                nn.Conv2d(in_channels=4, out_channels=1, kernel_size=3, stride=1, padding=1)
                            )
        
    def forward(self, x): 
        # b x 4 x 64 x 96 -> b x 1 x 64 x 96
        return self.conv_module(x) 


class NeuralNetsRoutePlanner(nn.Module):
    def __init__(self, num_classes=1):
        super(NeuralNetsRoutePlanner, self).__init__()
        self.conv_module = ConvNet(num_classes)
        
        # for fine-tuning 
        # self.conv_module.load_state_dict(torch.load('/workspace/neural-nets-route-planner/checkpoints/U-resnet34/last_model.ckpt')) 
        # for param in self.conv_module.parameters(): 
        #     param.requires_grad = False 

        self.rnn_module = RNNModule(64, 96, num_layers=2) 
        self.conv_head = ConvModule(num_classes)

    def forward(self, x):
        x = self.conv_module(x).squeeze(1) 
        x = self.rnn_module(x) 
        x = self.conv_head(x)
        return x 


class NNRoutePlanner: 
    def __init__(self, parking_lot, arch, weight, device): 

        self.device = device 
        self.neural_nets = getattr(import_module('route_planner.neural_nets_route_planner'), arch)(num_classes=1) 
        self.neural_nets.load_state_dict(torch.load(weight))
        self.neural_nets.eval().to(device)

        self.parking_lot = parking_lot 
        self.coordinates = []

    def get_grid_map(self, start, goal): 
        w, h = self.parking_lot.lot_width+1, self.parking_lot.lot_height+1 
        zero_array = np.zeros((h, w)) 

        start_x, start_y = start 
        goal_x, goal_y = goal 

        start_channel = zero_array.copy() 
        start_channel[h - start_y - 1, start_x] = 1.0 
        goal_channel = zero_array.copy() 
        goal_channel[h - goal_y - 1, goal_x] = 1.0 
        obstacle_channel = zero_array.copy() 
        for obstacle in self.parking_lot.obstacles: 
            obstacle_channel[h - obstacle[1] - 1, obstacle[0]] = 1.0 

        grid_map = np.stack([start_channel, goal_channel, obstacle_channel], axis=0) 

        return grid_map 
    
    def get_data(self, start, goal): 

        grid_map = self.get_grid_map(start, goal) 

        data_transform = transforms.Compose([
            transforms.ToTensor(),  
            ChannelwisePad(target_height=64, target_width=96, padding_values=[0.0, 0.0, 1.0]),
        ])

        data = data_transform(np.transpose(grid_map.astype(np.float32), (1, 2, 0))) 

        return data 

    def find_min_distance_node(self, curr_node): 
        min_distance = 15
        min_distance_node = None 
        idx = None  

        for i, node in enumerate(self.coordinates): 
            if curr_node == node: 
                del self.coordinates[i] 
                continue

            distance = math.sqrt((curr_node[0] - node[0])**2 + (curr_node[1] - node[1])**2) 
            if distance < min_distance: 
                min_distance = distance 
                min_distance_node = node 
                idx = i 
        
        if idx is not None:
            del self.coordinates[idx] 

        return min_distance_node

    def search_route(self, start, goal): 

        w, h = self.parking_lot.lot_width+1, self.parking_lot.lot_height+1 
        rx, ry = [], [] 
        with torch.no_grad(): 
            data = self.get_data(start, goal) 
            output_logit = self.neural_nets(data.unsqueeze(0).to(self.device)) 
            # if self.neural_nets.__class__.__name__ == "ConvNet": 
            output_logit.squeeze_(1) 
            output_logit.squeeze_(0) 

            output_pred = torch.sigmoid(output_logit) 
            output = (output_pred > 0.5).float() 

            rx.append(goal[0]) 
            ry.append(goal[1])

            coordinates = np.argwhere(output.cpu().numpy() == 1) 
            self.coordinates = [[x, h-y-1] for y, x in coordinates] 

            next_node = goal 
            while len(self.coordinates) and next_node is not None: 
                next_node = self.find_min_distance_node(next_node) 
                if next_node is not None: 
                    rx.append(next_node[0]) 
                    ry.append(next_node[1]) 

        return rx, ry 


if __name__ == '__main__': 
    import torchsummary 
    import torchinfo 

    model = NeuralNetsRoutePlanner() 
    model = model.to('cuda') 
    # torchsummary.summary(model, (3, 64, 96)) 
    torchinfo.summary(model, (1, 3, 64, 96))   

