#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2
from std_msgs.msg import String
import numpy as np
import torch
import torch.nn as nn
import struct

class SceneClassifier(nn.Module):
    def __init__(self, num_classes=4):
        super(SceneClassifier, self).__init__()
        self.block1 = nn.Sequential(nn.Conv2d(1, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(True), nn.MaxPool2d(2))
        self.block2 = nn.Sequential(nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(True), nn.MaxPool2d(2))
        self.block3 = nn.Sequential(nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(True), nn.MaxPool2d(2))
        self.block4 = nn.Sequential(nn.Conv2d(128, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(True), nn.MaxPool2d(2))
        self.classifier = nn.Sequential(nn.Linear(128 * 4 * 4, 256), nn.ReLU(True), nn.Dropout(0.5), nn.Linear(256, num_classes))

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)

class SceneClassifierNode(Node):
    def __init__(self):
        super().__init__('scene_classifier_node')
        self.declare_parameter('model_path', '')
        self.declare_parameter('range_m', 5.0)
        self.declare_parameter('grid_size', 64)

        self.model_path = self.get_parameter('model_path').get_parameter_value().string_value
        self.range_m = self.get_parameter('range_m').get_parameter_value().double_value
        self.grid_size = self.get_parameter('grid_size').get_parameter_value().integer_value
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.classes = ['open_space', 'caution_zone', 'narrow_corridor', 'dynamic_obstacle']
        
        self.model = SceneClassifier()
        if self.model_path:
            self.model.load_state_dict(torch.load(self.model_path, map_location=self.device))
        self.model.to(self.device).eval()

        self.subscription = self.create_subscription(PointCloud2, '/rslidar_points', self.pointcloud_callback, 10)
        self.publisher = self.create_publisher(String, '/scene_label', 10)
        self.get_logger().info(f'Scene Classifier Node started. Model: {self.model_path}')

    def pointcloud_callback(self, msg):
        # Extract X and Y coordinates
        points = np.frombuffer(msg.data, dtype=np.float32).reshape(-1, msg.point_step // 4)
        x = points[:, 0]
        y = points[:, 1]

        # Filter points within range
        mask = (x >= -self.range_m) & (x <= self.range_m) & (y >= -self.range_m) & (y <= self.range_m)
        x, y = x[mask], y[mask]

        # Map to grid
        grid = np.zeros((self.grid_size, self.grid_size), dtype=np.float32)
        if len(x) > 0:
            ix = ((x + self.range_m) / (2 * self.range_m) * (self.grid_size - 1)).astype(int)
            iy = ((y + self.range_m) / (2 * self.range_m) * (self.grid_size - 1)).astype(int)
            grid[iy, ix] = 1.0

        # Inference
        input_tensor = torch.from_numpy(grid).unsqueeze(0).unsqueeze(0).to(self.device)
        with torch.no_grad():
            output = self.model(input_tensor)
            _, predicted = output.max(1)
            label = self.classes[predicted.item()]

        self.publisher.publish(String(data=label))

def main(args=None):
    rclpy.init(args=args)
    node = SceneClassifierNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
