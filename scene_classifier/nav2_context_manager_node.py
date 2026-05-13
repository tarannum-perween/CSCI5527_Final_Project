#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from rcl_interfaces.msg import Parameter, ParameterValue, ParameterType

class Nav2ContextManager(Node):
    def __init__(self):
        super().__init__('nav2_context_manager_node')
        
        self.current_scene = None
        self.params_config = {
            'open_space':      {'max_vel_x': 0.5,  'inflation_radius': 0.4},
            'narrow_corridor': {'max_vel_x': 0.15, 'inflation_radius': 0.1},
            'caution_zone':    {'max_vel_x': 0.25, 'inflation_radius': 0.3},
            'dynamic_obstacle':{'max_vel_x': 0.0,  'inflation_radius': 0.5}
        }
        
        self.subscription = self.create_subscription(String, '/scene_label', self.scene_callback, 10)
        self.get_logger().info('Nav2 Context Manager Node started.')

    def set_nav2_params(self, scene):
        config = self.params_config[scene]
        
        # Update Controller Server
        self.set_remote_param('/controller_server', 'FollowPath.max_vel_x', config['max_vel_x'])
        
        # Update Local Costmap
        self.set_remote_param('/local_costmap/local_costmap', 'inflation_layer.inflation_radius', config['inflation_radius'])
        
        self.get_logger().info(f'Applied config for {scene}: max_vel_x={config["max_vel_x"]}, inflation={config["inflation_radius"]}')

    def set_remote_param(self, node_name, param_name, value):
        # Using a simplified approach for setting parameters via service client
        from rcl_interfaces.srv import SetParameters
        client = self.create_client(SetParameters, f'{node_name}/set_parameters')
        
        if not client.service_is_ready():
            self.get_logger().warn(f'Service {node_name}/set_parameters not ready')
            return

        req = SetParameters.Request()
        val = ParameterValue(type=ParameterType.PARAMETER_DOUBLE, double_value=float(value))
        req.parameters = [Parameter(name=param_name, value=val)]
        client.call_async(req)

    def scene_callback(self, msg):
        scene = msg.data
        if scene != self.current_scene:
            self.get_logger().info(f'Scene changed: {self.current_scene} -> {scene}')
            if scene in self.params_config:
                self.set_nav2_params(scene)
                self.current_scene = scene

def main(args=None):
    rclpy.init(args=args)
    node = Nav2ContextManager()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
