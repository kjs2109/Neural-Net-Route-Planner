import torch 
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms

from dataset import CustomDataset 
from augmentations import ChannelwisePad 
from route_planner.neural_nets_route_planner import NeuralNetsRoutePlanner, ConvNet, RNNModule  
from utils import save_numpy_array, save_plot, calc_dice, check_and_mkdir    

    
def train(model, train_loader, test_loader, criterion, optimizer, num_epochs, device):

    # Train the model
    total_step = len(train_loader)
    dice_score_log = [] 

    
    for epoch in range(num_epochs):
        model.train() 
        for i, (images, labels) in enumerate(train_loader):
            images = images.to(device)
            labels = labels.to(device) # if model.__class__.__name__ == "ConvNet" else labels.to(device).squeeze(1)
            
            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            if (i+1) % 1 == 0:
                print ('Epoch [{}/{}], Step [{}/{}], Loss: {}' 
                    .format(epoch+1, num_epochs, i+1, total_step, loss.item()))
                
        # calculate dice score 
        with torch.no_grad(): 
            model.eval() 
            dice_score_list = []
            for i, (images, labels) in enumerate(test_loader): 
                images = images.to(device)
                labels = labels.to(device) # if model.__class__.__name__ == "ConvNet" else labels.to(device).squeeze(1)
                outputs = model(images)
                outputs_pred = torch.sigmoid(outputs)
                outputs = (outputs_pred > 0.5).float()

                dice_score = calc_dice(outputs.cpu().numpy(), labels.cpu().numpy())
                dice_score_list.append(dice_score)
                
            avg_dice_score = sum(dice_score_list) / len(dice_score_list)
            print(f"Epoch [{epoch+1}/{num_epochs}], Dice Score: {avg_dice_score}")
            dice_score_log.append(avg_dice_score)
        
        if epoch % 50 == 0: 
            check_and_mkdir(f"./checkpoints") 
            torch.save(model.state_dict(), f'./checkpoints/{epoch}_model.ckpt') 

            cnt = 1
            for output in outputs: 
                output_logit = output.squeeze()
                output_pred = (torch.sigmoid(output_logit) > 0.5)
                np_output = output_pred.detach().cpu().numpy()
                save_numpy_array(np_output, save_path=f"./dataset/inference/{epoch}_{cnt}_inference.png")
                cnt += 1 

                if cnt > 10: 
                    break 

        save_plot(dice_score_log, save_path="./dataset/inference/dice_score_log.png")
    
    # Save the model checkpoint
    torch.save(model.state_dict(), 'last_model.ckpt') 

    return dice_score_log 


def test(model, test_loader, device):

    model.eval() 
    with torch.no_grad():
        dice_score_list = [] 
        for (idx, (images, labels))in enumerate(test_loader):

            images = images.to(device)
            labels = labels.to(device) # if model.__class__.__name__ == "ConvNet" else labels.to(device).squeeze(1)
            output_logit = model(images) # .unsqueeze(1)  # batch channel  
            output_pred = torch.sigmoid(output_logit)
            output = (output_pred > 0.5).float()

            save_numpy_array(output_logit.cpu().numpy(), save_path=f"./dataset/inference/output_logit_figure_{idx}.png")
            save_numpy_array(output.cpu().numpy(), save_path=f"./dataset/inference/output_binary_figure_{idx}.png") 

            dice_score = calc_dice(output.cpu().numpy(), labels.cpu().numpy())
            dice_score_list.append(dice_score)

            print(f'[{idx}] Dice Score: {dice_score}') 

        avg_dice_score = sum(dice_score_list) / len(dice_score_list)
        print(f"Test Dice Score: {avg_dice_score}")


def main(hyperparameters, data_transform, device): 

    num_epochs = hyperparameters["num_epochs"]
    num_classes = hyperparameters["num_classes"]
    batch_size = hyperparameters["batch_size"]
    learning_rate = hyperparameters["learning_rate"]


    train_dataset = CustomDataset(root='./dataset', 
                                train='train', 
                                transform=data_transform) 
    test_dataset = CustomDataset(root='./dataset', 
                                train='val', 
                                transform=data_transform)

    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                            batch_size=batch_size, 
                                            shuffle=True)

    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                            batch_size=10, 
                                            shuffle=False)

    model = NeuralNetsRoutePlanner(num_classes).to(device)
    # model.load_state_dict(torch.load('/workspace/neural-nets-route-planner/checkpoints/350_model.ckpt'))
    # model = ConvNet(num_classes).to(device)
    # model.load_state_dict(torch.load('/workspace/neural-nets-route-planner/checkpoints/U-resnet34/last_model.ckpt'))
    # model = RNNModule(64, 96, 1) 

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)


    dice_score_log = train(model, train_loader, test_loader, criterion, optimizer, num_epochs, device) 
    test(model, test_loader, device)  

    save_plot(dice_score_log, save_path="./dataset/inference/dice_score_log.png")


if __name__ == '__main__': 

    # Device configuration
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # Hyper parameters
    hyperparameters = {
        "num_epochs": 1000,
        "num_classes": 1,
        "batch_size": 400,
        "learning_rate": 0.0001, 
    }

    # Data augmentation 
    data_transform = transforms.Compose([
        transforms.ToTensor(),  
        ChannelwisePad(target_height=64, target_width=96, padding_values=[0.0, 0.0, 1.0]), 
    ])

    main(hyperparameters, data_transform, device) 