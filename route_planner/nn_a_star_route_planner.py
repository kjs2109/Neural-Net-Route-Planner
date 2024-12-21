import sys 
sys.path.append('..') 
import math
from importlib import import_module 

import matplotlib.pyplot as plt
from route_planner.parking_lot import ParkingLot 

import numpy as np 
import torch 
from torchvision import transforms

from route_planner.neural_nets_route_planner import ConvNet 
from augmentations import ChannelwisePad 


class Node:
    def __init__(self, x, y, cost, parent_node_index):
        self.x = x  # index of grid
        self.y = y  # index of grid
        self.cost = cost
        self.parent_node_index = parent_node_index


class NNAStarRoutePlanner:
    def __init__(self, parking_lot, arch, weight, alpha, device): 
        self.device = device 
        self.parking_lot: ParkingLot = parking_lot 
        self.neural_nets = getattr(import_module('route_planner.neural_nets_route_planner'), arch)(num_classes=1) 
        self.neural_nets.load_state_dict(torch.load(weight))
        self.neural_nets.eval().to(device)  
        self.cost_map = None 
        self.alpha = alpha 

        # Motion Model: dx, dy, cost
        self.motions = [
            [1, 0, 1],
            [0, 1, 1],
            [-1, 0, 1],
            [0, -1, 1],
            [-1, -1, math.sqrt(2)],
            [-1, 1, math.sqrt(2)],
            [1, -1, math.sqrt(2)],
            [1, 1, math.sqrt(2)],
        ]

        self.goal_node: Node = Node(0, 0, 0.0, -1) 
        self.searching_node_count = 0 

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

    def get_cost_map(self, start, goal):  
        data = self.get_data(start, goal) 
        data = data.unsqueeze(0).to(self.device)
        with torch.no_grad():
            outputs_logit = self.neural_nets(data)
            outputs_pred = torch.sigmoid(outputs_logit) 
            cost_map = 1 - outputs_pred[0, 0, :, :].cpu().numpy()

        return cost_map 

    def search_route(self, start_point, goal_point, show_process=True):
        start_node = Node(start_point[0], start_point[1], 0.0, -1)
        self.goal_node = Node(goal_point[0], goal_point[1], 0.0, -1) 

        self.cost_map = self.get_cost_map(start_point, goal_point)

        open_set = {self.parking_lot.get_grid_index(start_node.x, start_node.y): start_node}
        closed_set = {}

        while open_set:
            current_node_index = min(
                open_set,
                key=lambda o: open_set[o].cost + self.calculate_heuristic_cost(open_set[o]),
            )
            current_node = open_set[current_node_index]

            if show_process:
                self.plot_process(current_node, closed_set)

            if current_node.x == self.goal_node.x and current_node.y == self.goal_node.y:
                # print("Find goal")
                self.searching_node_count = len(closed_set.keys()) 
                self.goal_node = current_node
                return self.process_route(closed_set)

            # Remove the item from the open set
            del open_set[current_node_index]

            # Add it to the closed set
            closed_set[current_node_index] = current_node

            # expand_grid search grid based on motion model
            for motion in self.motions:
                next_node = Node(
                    current_node.x + motion[0],
                    current_node.y + motion[1],
                    current_node.cost + motion[2],
                    current_node_index,
                )
                next_node_index = self.parking_lot.get_grid_index(
                    next_node.x, next_node.y
                )

                if self.parking_lot.is_not_crossed_obstacle(
                        (current_node.x, current_node.y),
                        (next_node.x, next_node.y),
                ):
                    if next_node_index in closed_set:
                        continue

                    if next_node_index not in open_set:
                        open_set[next_node_index] = next_node  # discovered a new node
                    else:
                        if open_set[next_node_index].cost > next_node.cost:
                            # This path is the best until now. record it
                            open_set[next_node_index] = next_node

        print("Cannot find Route")
        return [], []

    def process_route(self, closed_set):
        rx = [round(self.goal_node.x)]
        ry = [round(self.goal_node.y)]
        parent_node = self.goal_node.parent_node_index
        while parent_node != -1:
            n = closed_set[parent_node]
            rx.append(n.x)
            ry.append(n.y)
            parent_node = n.parent_node_index
        return rx, ry

    def calculate_heuristic_cost(self, node):
        distance = math.sqrt(
            (node.x - self.goal_node.x) ** 2
            + (node.y - self.goal_node.y) ** 2
        )

        x_, y_ = node.x, self.parking_lot.lot_height - node.y

        cost = distance + self.cost_map[y_, x_] * self.alpha  
        return cost

    @staticmethod
    def plot_process(current_node, closed_set):
        # show graph
        plt.plot(current_node.x, current_node.y, "xc")
        # for stopping simulation with the esc key.
        plt.gcf().canvas.mpl_connect(
            "key_release_event",
            lambda event: [exit(0) if event.key == "escape" else None],
        )
        if len(closed_set.keys()) % 10 == 0:
            plt.pause(0.01)
            


def main(arch, weight, alpha, device, show_process, start_point, goal_point):
    parking_lot = ParkingLot()
    obstacle_x = [obstacle[0] for obstacle in parking_lot.obstacles]
    obstacle_y = [obstacle[1] for obstacle in parking_lot.obstacles]
    plt.plot(obstacle_x, obstacle_y, ".k")

    print(f"Start A Star Route Planner (start {start_point}, end {goal_point})")

    plt.plot(start_point[0], start_point[1], "og")
    plt.plot(goal_point[0], goal_point[1], "xb")
    plt.title("A Star Route Planner")
    plt.grid(True)
    plt.xlabel("X [m]")
    plt.ylabel("Y [m]")
    plt.axis("equal")

    a_star = NNAStarRoutePlanner(parking_lot, arch, weight, alpha, device)
    rx, ry = a_star.search_route(start_point, goal_point, show_process)
    print(f"Searching Node Count: {a_star.searching_node_count}") 

    plt.plot(rx, ry, "-r")
    plt.pause(0.01)
    plt.show()


if __name__ == "__main__":

    arch = "ConvNet" 
    weight = "/workspace/neural-nets-route-planner/checkpoints/U-resnet34/last_model.ckpt" 
    alpha = 50
    device = 'cuda' if torch.cuda.is_available() else 'cpu' 
    show_process = True  

    start_point = [14, 4] 
    goal_point = [69, 59]

    main(arch, weight, alpha, device, show_process, start_point, goal_point)
