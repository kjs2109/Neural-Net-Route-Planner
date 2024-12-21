import os 
import math 
import time 
import random 
import numpy as np 
import matplotlib.pyplot as plt 
from route_planner.parking_lot import ParkingLot 
from route_planner.a_star_route_planner import AStarRoutePlanner 
from route_planner.hybrid_a_star_route_planner import HybridAStarRoutePlanner, Pose 
from utils import vis_numpy_array, check_and_mkdir 


def generate_training_data(v, data_root, cnt, parking_lot, start_point, goal_point, rx, ry, show):  

    w, h = parking_lot.lot_width+1, parking_lot.lot_height+1
    # obstacle_x = [obstacle[0] for obstacle in parking_lot.obstacles]
    # obstacle_y = [obstacle[1] for obstacle in parking_lot.obstacles]
    zero_array = np.zeros((h, w))

    start_x, start_y = start_point 
    goal_x, goal_y = goal_point
    data_fname = f"a_star_{v}-{cnt}_{start_x}-{start_y}_{goal_x}-{goal_y}.npy" 
    gt_fname = f"a_star_gt_{v}-{cnt}_{start_x}-{start_y}_{goal_x}-{goal_y}.npy"

    start_channel = zero_array.copy() 
    start_channel[h - start_y - 1, start_x] = 1.0 
    goal_channel = zero_array.copy() 
    goal_channel[h - goal_y - 1, goal_x] = 1.0 
    obstacle_channel = zero_array.copy() 

    gt_channel = zero_array.copy() 

    for obstacle in parking_lot.obstacles: 
        obstacle_channel[h - obstacle[1] - 1, obstacle[0]] = 1.0 

    for gt_point in zip(rx, ry): 
        gt_channel[h - gt_point[1] - 1, gt_point[0]] = 1.0 

    data = np.stack([start_channel, goal_channel, obstacle_channel], axis=0) 
    gt = gt_channel 

    c1 = start_channel + goal_channel 
    c2 = obstacle_channel 

    if show: 
        vis_numpy_array(np.stack([c1, c2, gt_channel], axis=0), pause_time=1.0) 

    np.save(os.path.join(data_root, "data", data_fname), data) 
    np.save(os.path.join(data_root, "label", gt_fname), gt)


def main(v, number_of_data, parking_lot, searching, show, show_process):

    data_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "test_dataset") 
    check_and_mkdir(os.path.join(data_root, "data")) 
    check_and_mkdir(os.path.join(data_root, "label"))
    
    cnt = 1 
    while cnt <= number_of_data: 
        obstacle_x = [obstacle[0] for obstacle in parking_lot.obstacles] 
        obstacle_y = [obstacle[1] for obstacle in parking_lot.obstacles] 
        plt.plot(obstacle_x, obstacle_y, ".k")

        # start and goal point
        start_x = random.randint(0, parking_lot.lot_width) 
        start_y = random.randint(0, parking_lot.lot_height) 
        goal_x = random.randint(0, parking_lot.lot_width) 
        goal_y = random.randint(0, parking_lot.lot_height) 

        if ((start_x, start_y) not in parking_lot.obstacles) and ((goal_x, goal_y) not in parking_lot.obstacles): 
            distance = math.sqrt((start_x - goal_x)**2 + (start_y - goal_y)**2)
            start_point = [start_x, start_y] 
            goal_point = [goal_x, goal_y]
            if (distance > 7): 

                print(f"[{cnt}] Start A Star Route Planner (start {start_point}, end {goal_point})", end=" : ") 

                if searching.__class__.__name__ == "HybridAStarRoutePlanner": 
                    start_point = Pose(start_point[0], start_point[1], math.radians(0))
                    goal_point = Pose(goal_point[0], goal_point[1], math.radians(359))

                rx, ry = searching.search_route(start_point, goal_point, show_process) 
                if rx == []: 
                    continue 

                if show: 
                    plt.plot(start_point[0], start_point[1], "og")
                    plt.plot(goal_point[0], goal_point[1], "xb")
                    plt.title("A Star Route Planner")
                    plt.grid(True)
                    plt.xlabel("X [m]")
                    plt.ylabel("Y [m]")
                    plt.axis("equal")


                    plt.plot(rx, ry, "-r")
                    plt.pause(1.0)
                    plt.close() 

                generate_training_data(v, data_root, cnt, parking_lot, start_point, goal_point, rx, ry, show) 
                cnt += 1 


if __name__ == '__main__': 
    start = time.time() 

    v = 1  # version of dataset 
    number_of_data = 20000 
    parking_lot = ParkingLot()
    searching = AStarRoutePlanner(parking_lot)  # searching = HybridAStarRoutePlanner(parking_lot)
    show = False 
    show_process = False

    main(v, number_of_data, parking_lot, searching, show, show_process) 
    # vis_numpy_array(np.random.rand(63, 82))

    print(f"Elapsed Time: {time.time() - start:.2f} sec")