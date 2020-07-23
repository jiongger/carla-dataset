import glob
import os
import sys
import time

try:
    sys.path.append(glob.glob('carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass

import carla
# carla strandard

import argparse
import logging
import random

from math import sin, cos, tan, pi, sqrt

# define some carla color
red = carla.Color(255, 0, 0)
green = carla.Color(0, 255, 0)
blue = carla.Color(47, 210, 231)
cyan = carla.Color(0, 255, 255)
yellow = carla.Color(255, 255, 0)
orange = carla.Color(255, 162, 0)
white = carla.Color(255, 255, 255)


def main():
    argparser = argparse.ArgumentParser(
        description=__doc__)
    argparser.add_argument(
        '--host',
        metavar='H',
        default='127.0.0.1',
        help='IP of the host server (default: 127.0.0.1)')
    argparser.add_argument(
        '-p', '--port',
        metavar='P',
        default=2000,
        type=int,
        help='TCP port to listen to (default: 2000)')
    argparser.add_argument(
        '--safe',
        action='store_true',
        help='avoid spawning vehicles prone to accidents')
    argparser.add_argument(
        '--filterv',
        metavar='PATTERN',
        default='vehicle.*',
        help='vehicles filter (default: "vehicle.*")')
    argparser.add_argument(
        '--tm-port',
        metavar='P',
        default=8000,
        type=int,
        help='port to communicate with TM (default: 8000)')
    argparser.add_argument(
        '--sync',
        action='store_true',
        help='Synchronous mode execution')
    argparser.add_argument(
        '-n',
        '--number_of_vehicles',
        metavar='N',
        default=80,
        type=int,
        help='number of vehicles in map'
    )
    argparser.add_argument(
        '-s',
        '--spawn_point',
        metavar='S',
        default=-1,
        type=int,
        help='designate spawn point for your floating car'
    )
    argparser.add_argument(
        '--hybrid',
        action='store_true',
        help='Enanble')
    argparser.add_argument(
        '--path',
        metavar='P',
        default='../logs',
        type=str,
        help='path/to/save/sensor/data')
    argparser.add_argument(
        '--mode',
        metavar='M',
        default='common',
        type=str,
        choices=['debug-ego', 'debug-all', 'common'],
        help='debug mode: disable all file output')
    argparser.add_argument(
        '--save_as_kitti_format',
        action='store_true',
        help='save data as kitti format')
    argparser.add_argument(
        '--following',
        action='store_true'
    )
    argparser.add_argument(
        '--autopilot',
        action='store_true'
    )
    args = argparser.parse_args()

    logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.INFO)

    vehicles_list = []
    ego_vehicles_list = []
    sensors_list = []
    client = carla.Client(args.host, args.port)
    client.set_timeout(10.0)
    synchronous_master = False

    try:
        world = client.get_world()

        traffic_manager = client.get_trafficmanager(args.tm_port)
        traffic_manager.set_global_distance_to_leading_vehicle(2.0)
        if args.hybrid:
            traffic_manager.set_hybrid_physics_mode(True)

        if args.sync:
            settings = world.get_settings()
            traffic_manager.set_synchronous_mode(True)
            if not settings.synchronous_mode:
                synchronous_master = True
                settings.synchronous_mode = True
                settings.fixed_delta_seconds = 0.05
                world.apply_settings(settings)
            else:
                synchronous_master = False

        blueprints = world.get_blueprint_library().filter(args.filterv)

        if args.safe:
            blueprints = [x for x in blueprints if int(x.get_attribute('number_of_wheels')) == 4]
            blueprints = [x for x in blueprints if not x.id.endswith('isetta')]
            blueprints = [x for x in blueprints if not x.id.endswith('carlacola')]
            blueprints = [x for x in blueprints if not x.id.endswith('cybertruck')]
            blueprints = [x for x in blueprints if not x.id.endswith('t2')]
        
        if args.mode == 'common':
            if os.path.exists(args.path) == False:
                os.makedirs(args.path)

        import numpy as np
        spawn_points = world.get_map().get_spawn_points()
        if args.spawn_point >= len(spawn_points) or args.spawn_point < 0:
            if args.spawn_point >= len(spawn_points):
                print('\nWARNING: the designated spawn point does not exist\n')
            selected_spawn_points = np.random.choice(spawn_points, args.number_of_vehicles+1)
            ego_spawn_point = selected_spawn_points[0]
            selected_spawn_points = selected_spawn_points[1:]
        else:
            ego_spawn_point = spawn_points[args.spawn_point]
            if args.spawn_point == 0:
                spawn_points = spawn_points[1:]
            elif args.spawn_point == len(spawn_points)-1:
                spawn_points = spawn_points[:-1]
            else:
                spawn_points = spawn_points[0:args.spawn_point] + spawn_points[args.spawn_point+1:]
            selected_spawn_points = np.random.choice(spawn_points, args.number_of_vehicles)

        # @todo cannot import these directly.
        SpawnActor = carla.command.SpawnActor
        SetAutopilot = carla.command.SetAutopilot
        FutureActor = carla.command.FutureActor

        # spawn civil vehicles
        batch = []
        for spawn_point in spawn_points:
            blueprint = random.choice(blueprints)
            if blueprint.has_attribute('color'):
                color = random.choice(blueprint.get_attribute('color').recommended_values)
                blueprint.set_attribute('color', color)
            if blueprint.has_attribute('driver_id'):
                driver_id = random.choice(blueprint.get_attribute('driver_id').recommended_values)
                blueprint.set_attribute('driver_id', driver_id)
            blueprint.set_attribute('role_name', 'autopilot')
            batch.append(SpawnActor(blueprint, spawn_point).then(SetAutopilot(FutureActor, True, traffic_manager.get_port())))

        # find ego vehicle blueprint
        ego_bp = world.get_blueprint_library().find('vehicle.tesla.model3')
        ego_bp.set_attribute('role_name','ego')
        ego_color = random.choice(ego_bp.get_attribute('color').recommended_values)
        ego_bp.set_attribute('color',ego_color)

        # spawn ego
        ego_transform = ego_spawn_point
        if args.autopilot:
            batch.append(SpawnActor(ego_bp,ego_transform).then(SetAutopilot(FutureActor, True, traffic_manager.get_port())))
        else:
            batch.append(SpawnActor(ego_bp,ego_transform))
        
        # carla work (convert Carla.SpawnActor to Carla.Vehicle)        
        for response in client.apply_batch_sync(batch, synchronous_master):
            if response.error:
                logging.error(response.error)
            else:
                vehicles_list.append(response.actor_id)
                ego = response # ego should be the last vehicle
            
        # adjust spectator to proper position       
        spectator = world.get_spectator()
        # set spectator above the center 
        location = ego_transform.location + carla.Location(x=-30,z=30)
        rotation = ego_transform.rotation + carla.Rotation(pitch=-45)
        transform = carla.Transform(location, rotation)
        spectator.set_transform(transform)

        # calculate vehicle 3d parameters
        ego_bbox = ego.bounding_box
        ego_length = ego.bounding_box.extent.x*2
        ego_width = ego.bounding_box.extent.y*2
        ego_height = ego.bounding_box.extent.z*2
        ego_center = ego.bounding_box.location
        
        # find lidar blueprint
        lidar_bp = world.get_blueprint_library().find('sensor.lidar.ray_cast')
        lidar_bp.set_attribute('channels',str(64))
        lidar_bp.set_attribute('points_per_second',str(960000))
        lidar_bp.set_attribute('rotation_frequency',str(10))
        lidar_bp.set_attribute('sensor_tick', str(0.1))
        lidar_bp.set_attribute('upper_fov', str(0))
        lidar_bp.set_attribute('lower_fov', str(-24))
        lidar_bp.set_attribute('range',str(60))
        ego_lidar_location = carla.Location(0,0, tan(pi/180*25)*ego_length/2 + ego_height)
        lidar_rotation = carla.Rotation(0,0,0)
        ego_lidar_transform = carla.Transform(ego_lidar_location,lidar_rotation)

        # set lidar @ ego
        ego_lidar = world.spawn_actor(lidar_bp,ego_lidar_transform,attach_to=ego,attachment_type=carla.AttachmentType.Rigid)
        def ego_lidar_callback(LidarMeasurement):
            if args.mode == 'common':
                if args.save_as_kitti_format:
                    import struct
                    save = open(os.path.join(args.path, '%06d.bin' %(7500+LidarMeasurement.frame)), 'wb')
                else:
                    save = open(os.path.join(args.path, 'lidar_measurement_%d.txt' %LidarMeasurement.frame), 'w')
                    LidarMeasurement.save_to_disk(os.path.join(args.path, 'lidar_measurement_%d.ply' %LidarMeasurement.frame))
            else:
                save = None
            if not args.save_as_kitti_format:
                print(LidarMeasurement.frame, LidarMeasurement.timestamp,
                LidarMeasurement.transform.location.x, LidarMeasurement.transform.location.y, LidarMeasurement.transform.location.z,
                LidarMeasurement.transform.rotation.roll, LidarMeasurement.transform.rotation.pitch, LidarMeasurement.transform.rotation.yaw, 
                LidarMeasurement.horizontal_angle, LidarMeasurement.channels, file=save)
            if args.mode == 'common':
                for point in LidarMeasurement:
                    if args.save_as_kitti_format:
                        save.write(struct.pack('ffff', -point.y,point.x,-point.z,0.5))
                    else:
                        print(point.x, point.y, point.z, file=save)
        sensors_list.append(ego_lidar)
        # lidar: listen
        ego_lidar.listen(ego_lidar_callback)
        
        DEBUG_MODE = False
        # debug mode
        if args.mode.find('debug') >= 0:
            
            some_time = world.get_settings().fixed_delta_seconds
            debug = world.debug
            def draw_transform(debug, trans, col=carla.Color(255, 0, 0), lt=-1):
                debug.draw_arrow(
                    trans.location, trans.location + trans.get_forward_vector(),
                    thickness=0.05, arrow_size=0.1, color=col, life_time=lt)
            def draw_waypoint_union(debug, w0, w1, color=carla.Color(255, 0, 0), lt=5):
                debug.draw_line(
                    w0.transform.location + carla.Location(z=0.25),
                    w1.transform.location + carla.Location(z=0.25),
                    thickness=0.1, color=color, life_time=lt, persistent_lines=False)
                debug.draw_point(w1.transform.location + carla.Location(z=0.25), 0.1, color, lt, False)

            current_map = world.get_map()
            debug_list = []
            # find debug vehicle
            if args.mode == 'debug-all':
                debug_list = world.get_actors(vehicles_list)
            elif args.mode == 'debug-ego':
                debug_list = ego
            # preprocess debug vehicle waypoints
            current_w = []
            for vehicle in debug_list:
                current_w.append(current_map.get_waypoint(vehicle.get_location()))
            DEBUG_MODE = True
            
        while True:
            # debug loop
            if DEBUG_MODE == True:

                # find next debug vehicle waypoints
                next_w = []
                for vehicle in debug_list:
                    next_w.append(current_map.get_waypoint(vehicle.get_location(), lane_type=carla.LaneType.Driving | carla.LaneType.Shoulder | carla.LaneType.Sidewalk ))
                # draw vehicle trail
                for index, vehicle in enumerate(debug_list):
                    # Check if the vehicle is moving
                    if next_w[index] != current_w[index]:
                        vector = vehicle.get_velocity()
                        # Check if the vehicle is on a sidewalk
                        if current_w[index].lane_type == carla.LaneType.Sidewalk:
                            draw_waypoint_union(debug, current_w[index], next_w[index], cyan if current_w[index].is_junction else red, 10)
                        else:
                            draw_waypoint_union(debug, current_w[index], next_w[index], cyan if current_w[index].is_junction else green, 10)
                        debug.draw_string(current_w[index].transform.location, str('id %d\n%15.0f km/h' % (vehicle.id, 3.6 * sqrt(vector.x**2 + vector.y**2 + vector.z**2))), False, orange, some_time)
                        draw_transform(debug, current_w[index].transform, white, 10)

                # draw vehicle bounding box
                for vehicle in debug_list:
                    bbox = carla.BoundingBox(vehicle.get_transform().location, vehicle.bounding_box.extent) # generate bounding box
                    debug.draw_box(bbox, vehicle.get_transform().rotation, 0.1, red, some_time)

                # Update the current waypoint and sleep for some time
                current_w = next_w.copy()                 
                time.sleep(some_time)
            
            # following mode
            if args.following == True:
                ego_transform = ego.get_transform()
                location = ego_transform.location + carla.Location(x=-30,z=30)
                rotation = ego_transform.rotation + carla.Rotation(pitch=-45)
                transform = carla.Transform(location, rotation)
                spectator.set_transform(transform)
            
            if args.sync and synchronous_master:
                world.tick()
            else:
                world.wait_for_tick()
            
    
    finally:

        if args.sync and synchronous_master:
            settings = world.get_settings()
            settings.synchronous_mode = False
            settings.fixed_delta_seconds = None
            world.apply_settings(settings)

        print('\ndestroying %d vehicles' % len(vehicles_list))
        client.apply_batch([carla.command.DestroyActor(x) for x in vehicles_list])
        
        for sensor in sensors_list:
            sensor.stop()
        print('\ndestroying %d sensors' % len(sensors_list))
        client.apply_batch([carla.command.DestroyActor(x.id) for x in sensors_list])

        time.sleep(0.5)


if __name__ == '__main__':

    try:
        main()
    except KeyboardInterrupt:
        pass
    finally:
        print('\ndone.\n')

