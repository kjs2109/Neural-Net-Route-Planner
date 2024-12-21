import os 
import math 
import numpy as np 
import matplotlib.pyplot as plt 
import matplotlib 
matplotlib.use('TkAgg')


def check_and_mkdir(target_path):
    target_path = os.path.abspath(target_path)
    path_to_targets = os.path.split(target_path)

    if '.' in path_to_targets[-1]: 
        path_to_targets = path_to_targets[:-1] 
    
    path_history = '/'
    for path in path_to_targets:
        path_history = os.path.join(path_history, path)
        if not os.path.exists(path_history):
            os.mkdir(path_history)


def save_plot(dice_score_log, save_path): 

    check_and_mkdir(save_path)

    plt.figure(figsize=(10, 8))
    plt.plot(dice_score_log)
    plt.xlabel("Epoch")
    plt.ylabel("Dice Score")
    plt.title("Dice Score of ConvNet")
    plt.savefig(save_path)
    plt.close()


def vis_numpy_array(np_array, pause_time=None):

    if np_array.__class__ == str:
        np_array = np.load(np_array)

    if len(np_array.shape) == 4: 
        np_array = np_array.squeeze(0) 

    if (np_array.shape[-1] != 3) and (len(np_array.shape) == 3): 
        np_array = np_array.transpose((1, 2, 0)) 

    # 시각화
    plt.figure(figsize=(10, 8))
    plt.imshow(np_array, cmap='viridis', interpolation='nearest')
    plt.colorbar(label='Value')  

    # 그리드 추가
    plt.grid(color='white', linestyle='--', linewidth=0.5)

    # 그리드 맞추기
    plt.xticks(np.arange(-0.5, np_array.shape[1], 1), labels=[])
    plt.yticks(np.arange(-0.5, np_array.shape[0], 1), labels=[])
    plt.gca().set_xticks(np.arange(-0.5, np_array.shape[1], 1), minor=True)
    plt.gca().set_yticks(np.arange(-0.5, np_array.shape[0], 1), minor=True)
    plt.gca().grid(which='minor', color='white', linestyle='--', linewidth=0.5)

    plt.title("Grid Visualization of Output Array")
    plt.xlabel("X-axis")
    plt.ylabel("Y-axis")

    if pause_time is not None: 
        plt.pause(pause_time) 
        plt.close() 

    plt.show() 

def save_numpy_array(np_array, save_path):
    if len(np_array.shape) == 4: 
        np_array = np_array.squeeze(0) 

    if len(np_array.shape) == 3: 
        np_array = np_array.transpose((1, 2, 0)) 

    check_and_mkdir(save_path) 

    # 시각화
    plt.figure(figsize=(10, 8))
    plt.imshow(np_array, cmap='viridis', interpolation='nearest')
    plt.colorbar(label='Value')  

    # 그리드 추가
    plt.grid(color='white', linestyle='--', linewidth=0.5)

    # 그리드 맞추기
    plt.xticks(np.arange(-0.5, np_array.shape[1], 1), labels=[])
    plt.yticks(np.arange(-0.5, np_array.shape[0], 1), labels=[])
    plt.gca().set_xticks(np.arange(-0.5, np_array.shape[1], 1), minor=True)
    plt.gca().set_yticks(np.arange(-0.5, np_array.shape[0], 1), minor=True)
    plt.gca().grid(which='minor', color='white', linestyle='--', linewidth=0.5)

    plt.title("Grid Visualization of Output Array")
    plt.xlabel("X-axis")
    plt.ylabel("Y-axis")

    # 저장
    plt.savefig(save_path)
    plt.close() 


def calc_dice(im1, im2):
    im1 = np.asarray(im1).astype(np.bool_)
    im2 = np.asarray(im2).astype(np.bool_)

    if im1.shape != im2.shape:
        raise ValueError("Shape mismatch: im1 and im2 must have the same shape.")

    intersection = np.logical_and(im1, im2)
    dice_score = 2. * intersection.sum() / (im1.sum() + im2.sum())

    return dice_score if not math.isnan(dice_score) else 0.0 



if __name__ == '__main__':
    vis_numpy_array('./dataset/data/a_star_1-1_16-12_4-28.npy')