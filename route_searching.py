import sys 
sys.path.append('..') 
import time 
import matplotlib.pyplot as plt 
from route_planner.parking_lot import ParkingLot 
from route_planner.a_star_route_planner import AStarRoutePlanner  
from route_planner.neural_nets_route_planner import NNRoutePlanner 
from route_planner.nn_a_star_route_planner import NNAStarRoutePlanner 


def route_visualization(parking_lot, start, goal, rx, ry, title): 

    fig, ax = plt.subplots() 

    obstacle_x = [obstacle[0] for obstacle in parking_lot.obstacles] 
    obstacle_y = [obstacle[1] for obstacle in parking_lot.obstacles] 
    ax.plot(obstacle_x, obstacle_y, '.k') 

    print('start:', start, 'goal:', goal) 

    ax.plot(start[0], start[1], 'og') 
    ax.plot(goal[0], goal[1], 'ob') 
    ax.set_title(title) 
    ax.grid(True) 
    ax.set_xlabel('X [m]') 
    ax.set_ylabel('Y [m]') 
    ax.axis('equal') 

    ax.plot(rx, ry, '-r') 

    return fig


if __name__=='__main__': 

    parking_lot = ParkingLot() 
    arch = "NeuralNetsRoutePlanner" 
    weight = "/workspace/neural-nets-route-planner/checkpoints/U-resnet34-lstm-urdl-2-conv-00001/600_model.ckpt"
    # weight = "/workspace/neural-nets-route-planner/checkpoints/450_model.ckpt"


    nn_route_planner_cuda = NNRoutePlanner(parking_lot, arch, weight, 'cuda') 
    nn_a_star_route_planner = NNAStarRoutePlanner(parking_lot)
    
    a_star_route_planner = AStarRoutePlanner(parking_lot) 

    # start = [4, 22] 
    # goal = [81, 22] 

    # start = [14, 4] # [14, 4],  # , 
    # start = [10, 20] # [40, 50]
    # goal = [69, 59] 

    start = [10, 50] 
    goal = [80, 10]

    start = [10, 50] 
    goal = [72, 4]

    ### 
    start = [75, 54] 
    goal = [5, 54] 
    goal = [5, 49] 
    goal = [5, 30] 

    tick = time.time()
    rx, ry = nn_route_planner_cuda.search_route(start, goal) 
    print(f"NeuralNetsRoutePlanner: {time.time() - tick:.3f} sec") 
    figure1 = route_visualization(parking_lot, start, goal, rx, ry, 'NeuralNetsRoutePlanner') 

    tick = time.time() 
    rx, ry = nn_a_star_route_planner.search_route(start, goal) 
    print(f"NeuralNetsRoutePlanner: {time.time() - tick:.3f} sec") 
    figure2 = route_visualization(parking_lot, start, goal, rx, ry, 'NeuralNetsRoutePlanner')

    
    tick = time.time()
    rx, ry = a_star_route_planner.search_route(start, goal, False) 
    print(f"AStarRoutePlanner: {time.time() - tick:.3f} sec") 
    figure3 = route_visualization(parking_lot, start, goal, rx, ry, 'AStarRoutePlanner') 

    figure1.show() 
    figure2.show() 
    figure3.show() 
    plt.show()