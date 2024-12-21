import torch 
import numpy as np 
import torchvision.transforms as transforms

from dataset import TestDataset, CustomDataset 
from utils import vis_numpy_array 
from augmentations import ChannelwisePad 
from route_planner.neural_nets_route_planner import ConvNet, NeuralNetsRoutePlanner 
from route_planner.nn_a_star_route_planner import NNAStarRoutePlanner  


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') 


def inference_test(model, start, goal): 
    data_transform = transforms.Compose([
        transforms.ToTensor(),  
        ChannelwisePad(target_height=64, target_width=96, padding_values=[0.0, 0.0, 1.0]), 
    ])
    print(start,'->', goal)
    test_dataset = TestDataset(root='./dataset', 
                               start=start, 
                               goal=goal, 
                               transform=data_transform) 
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=1, shuffle=False) 

    with torch.no_grad():
        model.eval() 
        data, grid_map = next(iter(test_loader)) 
        data = data.to(device)

        outputs_logit = model(data)
        outputs_pred = torch.sigmoid(outputs_logit)
        outputs = (outputs_pred > 0.5).float() 

        # if model.__class__.__name__ == "NeuralNetsRoutePlanner": 
        #     outputs.unsqueeze_(1) 
        #     outputs_logit.unsqueeze_(1)

        print(grid_map[0, 0:2, :, :].cpu().numpy().shape, outputs[0, :1, :, :].cpu().numpy().shape)
        result = np.concatenate([grid_map[0, 0:2, :, :].cpu().numpy(), outputs[0, :1, :, :].cpu().numpy()], axis=0) 
        vis_numpy_array(result)
        vis_numpy_array(outputs[0, 0, :, :].cpu().numpy())
        vis_numpy_array(outputs_logit[0, 0, :, :].cpu().numpy())


if __name__ == '__main__':

    start = [14, 4] # [14, 4],  # [10, 20], 
    # start = [40, 50]
    goal = [69, 59]

    # [29] Dice: 0.78 
    # start = [4, 22] 
    # goal = [81, 22] 

    start = [10, 50] 
    goal = [72, 4] 

    # start = [75, 54] 
    # goal = [5, 30] 

    start = [28, 11] 
    goal = [57, 43]

    # model = ConvNet() 
    # model.load_state_dict(torch.load('/workspace/neural-nets-route-planner/checkpoints/U-resnet34/last_model.ckpt')) 

    model = NeuralNetsRoutePlanner(1)
    model.load_state_dict(torch.load('/workspace/neural-nets-route-planner/checkpoints/U-resnet34-lstm-urdl-2-conv-000001/last_model.ckpt')) 
    # model.load_state_dict(torch.load('/workspace/neural-nets-route-planner/checkpoints/450_model.ckpt'))
    model = model.to(device) 

    inference_test(model, start, goal)

