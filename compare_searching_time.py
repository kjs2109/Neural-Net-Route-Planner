import time 
import matplotlib.pyplot as plt 
from route_planner.parking_lot import ParkingLot 
from route_planner.a_star_route_planner import AStarRoutePlanner 
from route_planner.nn_a_star_route_planner import NNAStarRoutePlanner 
from route_planner.neural_nets_route_planner import NNRoutePlanner 
import math 
import random 
import numpy as np 
import pandas as pd 


def sampling(sample_num=10000): 
    parking_lot = ParkingLot() 

    a_star_route_planner = AStarRoutePlanner(parking_lot) 
    chkpt_path = "/workspace/neural-nets-route-planner/checkpoints/U-resnet34/last_model.ckpt" 
    nn_a_star_route_planner = NNAStarRoutePlanner(parking_lot, 'ConvNet', chkpt_path, 50, 'cuda') 
    chkpt_path = '/workspace/neural-nets-route-planner/checkpoints/U-resnet34-lstm-urdl-2-conv-000001/last_model.ckpt'
    nn_route_planner_= NNRoutePlanner(parking_lot, 'NeuralNetsRoutePlanner', chkpt_path, 'cuda')  

    print("Start sampling...!")
    time.sleep(3) 

    cnt = 1
    sampling_data = [] 
    check_list = [] 
    while cnt <= sample_num: 

        start_x = random.randint(0, parking_lot.lot_width) 
        start_y = random.randint(0, parking_lot.lot_height) 
        goal_x = random.randint(0, parking_lot.lot_width) 
        goal_y = random.randint(0, parking_lot.lot_height) 

        if ((start_x, start_y) not in parking_lot.obstacles) and ((goal_x, goal_y) not in parking_lot.obstacles): 
                distance = math.sqrt((start_x - goal_x)**2 + (start_y - goal_y)**2)
                start = [start_x, start_y] 
                goal = [goal_x, goal_y]
                if (distance > 8): 

                    tick = time.time() 
                    rx, ry = a_star_route_planner.search_route(start, goal, False) 
                    a_star_time = time.time() - tick 

                    tick = time.time() 
                    rx, ry = nn_a_star_route_planner.search_route(start, goal, False) 
                    nn_cpu_time = time.time() - tick 

                    tick = time.time() 
                    rx, ry = nn_route_planner_.search_route(start, goal) 
                    nn_cuda_time = time.time() - tick 

                    a_star_searching_node_count = a_star_route_planner.searching_node_count 
                    nn_a_star_searching_node_count = nn_a_star_route_planner.searching_node_count 
                    if a_star_searching_node_count < nn_a_star_searching_node_count: 
                        check_list.append([a_star_searching_node_count, nn_a_star_searching_node_count, start[0], start[1], goal[0], goal[1]]) 

                    sampling_data.append([a_star_time, nn_cpu_time, nn_cuda_time, a_star_searching_node_count, nn_a_star_searching_node_count])

                    print(f"Sample[{cnt}]: AStar: {a_star_time:.5f} sec, NNAStar: {nn_cpu_time:.5f} sec, NN: {nn_cuda_time:.5f} sec | [{a_star_searching_node_count} | {nn_a_star_searching_node_count}]")
                    cnt += 1 
        
    df = pd.DataFrame(sampling_data, columns=['AStar', 'NNAStar', 'NN', 'AStarSearchingNodeCount', 'NNAStarSearchingNodeCount']) 
    df.to_csv('./sampling_time.csv', index=False) 

    df = pd.DataFrame(check_list, columns=['a_star', 'nn_a_star', 'start_x', 'start_y', 'goal_x', 'goal_y']) 
    df.to_csv('./check_list.csv', index=False) 


if __name__=='__main__': 
    sample_num = 20000
    sampling(sample_num)  
