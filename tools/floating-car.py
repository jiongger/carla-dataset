import glob
import os
import sys
import time

try:
    sys.path.append(glob.glob('carla/dist/carla-*%d.%d-%s.egg' % (
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

from math import sin, cos, tan, pi, sqrt, atan, asin, acos
import numpy as np

# define some carla color
red = carla.Color(255, 0, 0)
green = carla.Color(0, 255, 0)
blue = carla.Color(47, 210, 231)
cyan = carla.Color(0, 255, 255)
yellow = carla.Color(255, 255, 0)
orange = carla.Color(255, 162, 0)
white = carla.Color(255, 255, 255)

def get_matrix(transform):
    """
    Creates matrix from carla transform.
    """
    rotation = transform.rotation
    location = transform.location
    c_y = np.cos(np.radians(rotation.yaw))
    s_y = np.sin(np.radians(rotation.yaw))
    c_r = np.cos(np.radians(rotation.roll))
    s_r = np.sin(np.radians(rotation.roll))
    c_p = np.cos(np.radians(rotation.pitch))
    s_p = np.sin(np.radians(rotation.pitch))
    matrix = np.matrix(np.identity(4))
    matrix[0, 3] = location.x
    matrix[1, 3] = location.y
    matrix[2, 3] = location.z
    matrix[0, 0] = c_p * c_y
    matrix[0, 1] = -c_p * s_y
    matrix[0, 2] = s_p
    matrix[1, 0] = c_r * s_y + s_r * s_p * c_y
    matrix[1, 1] = c_r * c_y - s_r * s_p * s_y
    matrix[1, 2] = -s_r * c_p
    matrix[2, 0] = s_r * s_y - c_r * s_p * c_y
    matrix[2, 1] = s_r * c_y + c_r * s_p * s_y
    matrix[2, 2] = c_r * c_p
    return matrix

def _create_bb_points(vehicle):
    """
    Returns 3D bounding box for a vehicle.
    """
    cords = np.zeros((9, 4))
    extent = vehicle.bounding_box.extent
    cords[0, :] = np.array([extent.x, extent.y, -extent.z, 1])
    cords[1, :] = np.array([-extent.x, extent.y, -extent.z, 1])
    cords[2, :] = np.array([-extent.x, -extent.y, -extent.z, 1])
    cords[3, :] = np.array([extent.x, -extent.y, -extent.z, 1])
    cords[4, :] = np.array([extent.x, extent.y, extent.z, 1])
    cords[5, :] = np.array([-extent.x, extent.y, extent.z, 1])
    cords[6, :] = np.array([-extent.x, -extent.y, extent.z, 1])
    cords[7, :] = np.array([extent.x, -extent.y, extent.z, 1])
    cords[8, :] = np.array([0,0,0,1])
    return cords

def _vehicle_to_world(cords, vehicle):
    """
    Transforms coordinates of a vehicle bounding box to world.
    """
    bb_transform = carla.Transform(vehicle.bounding_box.location)
    bb_vehicle_matrix = get_matrix(bb_transform)
    vehicle_world_matrix = get_matrix(vehicle.get_transform())
    bb_world_matrix = np.dot(vehicle_world_matrix, bb_vehicle_matrix)
    world_cords = np.dot(bb_world_matrix, np.transpose(cords))
    return world_cords

def _world_to_sensor(cords, sensor):
    """
    Transforms world coordinates to sensor.
    """
    sensor_world_matrix = get_matrix(sensor.get_transform())
    world_sensor_matrix = np.linalg.inv(sensor_world_matrix)
    sensor_cords = np.dot(world_sensor_matrix, cords)
    return sensor_cords

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
        help='number of vehicles in map (default: 80)'
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
        '--autopilot',
        action='store_true'
    )
    argparser.add_argument(
        '--kitti_split',
        type=str,
        default='training',
        choices=['training', 'testing', 'val'],
        help="kitti split (functional when 'save_as_kitti_format' is true)"
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
            if args.save_as_kitti_format:
                VELO_DIR = os.path.join(args.path, 'object', args.kitti_split, 'velodyne')
                IMAGE_DIR = os.path.join(args.path, 'object', args.kitti_split, 'image_2')
                LABEL_DIR = os.path.join(args.path, 'object', args.kitti_split, 'label_2')
                CALIB_DIR = os.path.join(args.path, 'object', args.kitti_split, 'calib')        
                import shutil
                # create empty folders
                if os.path.exists(CALIB_DIR):
                    shutil.rmtree(CALIB_DIR)
                os.makedirs(CALIB_DIR)
                if os.path.exists(IMAGE_DIR):
                    shutil.rmtree(IMAGE_DIR)
                os.makedirs(IMAGE_DIR)
                if os.path.exists(VELO_DIR):
                    shutil.rmtree(VELO_DIR)
                os.makedirs(VELO_DIR)
                if os.path.exists(LABEL_DIR):
                    shutil.rmtree(LABEL_DIR)
                os.makedirs(LABEL_DIR)

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

        # find ego vehicle blueprint
        ego_bp = world.get_blueprint_library().find('vehicle.tesla.model3')
        ego_bp.set_attribute('role_name','ego')
        ego_color = random.choice(ego_bp.get_attribute('color').recommended_values)
        ego_bp.set_attribute('color',ego_color)

        # spawn ego
        batch = []
        ego_transform = ego_spawn_point
        if args.autopilot:
            batch.append(SpawnActor(ego_bp,ego_transform).then(SetAutopilot(FutureActor, True, traffic_manager.get_port())))
        else:
            batch.append(SpawnActor(ego_bp,ego_transform))

        # spawn civil vehicles
        for spawn_point in selected_spawn_points:
            blueprint = random.choice(blueprints)
            if blueprint.has_attribute('color'):
                color = random.choice(blueprint.get_attribute('color').recommended_values)
                blueprint.set_attribute('color', color)
            if blueprint.has_attribute('driver_id'):
                driver_id = random.choice(blueprint.get_attribute('driver_id').recommended_values)
                blueprint.set_attribute('driver_id', driver_id)
            blueprint.set_attribute('role_name', 'cvils')
            batch.append(SpawnActor(blueprint, spawn_point).then(SetAutopilot(FutureActor, True, traffic_manager.get_port())))
        
        # carla work (convert Carla.SpawnActor to Carla.Vehicle)        
        for response in client.apply_batch_sync(batch, synchronous_master):
            if response.error:
                logging.error(response.error)
            else:
                vehicles_list.append(response.actor_id)
                #print(world.get_actor(response.actor_id).attributes['role_name'])
        ego = world.get_actor(vehicles_list[0]) # ego should be the first vehicle
        assert ego.attributes['role_name'] == 'ego'

        # calculate vehicle 3d parameters
        ego_bbox = ego.bounding_box
        ego_length = ego.bounding_box.extent.x*2
        ego_width = ego.bounding_box.extent.y*2
        ego_height = ego.bounding_box.extent.z*2
        ego_center = ego.bounding_box.location

        IMAGEX = 1600
        IMAGEY = 900

        # Find the blueprint of the sensor
        camera_bp = world.get_blueprint_library().find('sensor.camera.rgb')
        camera_bp.set_attribute('image_size_x', str(IMAGEX))
        camera_bp.set_attribute('image_size_y', str(IMAGEY))
        camera_bp.set_attribute('fov', '120')
        camera_bp.set_attribute('sensor_tick', '1')
        ego_camera_location = carla.Location(0.8,0,1.7)
        ego_camera_rotation = carla.Rotation(0,0,0)
        ego_camera_transform = carla.Transform(ego_camera_location, ego_camera_rotation)

        # set camera @ ego
        ego_camera = world.spawn_actor(camera_bp, ego_camera_transform, attach_to=ego, attachment_type=carla.AttachmentType.Rigid)
        def ego_camera_callback(RGBMeasurement):
            if args.mode == 'common':
                if args.save_as_kitti_format:
                    RGBMeasurement.save_to_disk(os.path.join(IMAGE_DIR, '%06d.png' %RGBMeasurement.frame))
                    #calib_file = open(os.path.join(CALIB_DIR, '%06d.txt' %RGBMeasurement.frame), 'w')
                    #print('P0: %.12e %.12e %.12e %.12e %.12e %.12e %.12e %.12e %.12e %.12e %.12e %.12e' %())
                    #calib_file.close()
                else:
                    RGBMeasurement.save_to_disk(os.path.join(args.path, 'rgb_measurement_%d.png' %RGBMeasurement.frame))
        sensors_list.append(ego_camera)
        # camera : listen
        ego_camera.listen(ego_camera_callback)

        # find lidar blueprint
        lidar_bp = world.get_blueprint_library().find('sensor.lidar.ray_cast')
        lidar_bp.set_attribute('channels',str(64))
        lidar_bp.set_attribute('points_per_second',str(1280000))
        lidar_bp.set_attribute('rotation_frequency',str(10))
        lidar_bp.set_attribute('sensor_tick', str(0.1))
        lidar_bp.set_attribute('upper_fov', str(0))
        lidar_bp.set_attribute('lower_fov', str(-24))
        lidar_bp.set_attribute('range',str(120))
        lidar_height = tan(pi/180*25)*ego_length/2 + ego_height
        ego_lidar_location = carla.Location(0,0, lidar_height)
        ego_lidar_rotation = carla.Rotation(0,0,0)
        ego_lidar_transform = carla.Transform(ego_lidar_location,ego_lidar_rotation)

        # set lidar @ ego
        ego_lidar = world.spawn_actor(lidar_bp,ego_lidar_transform,attach_to=ego,attachment_type=carla.AttachmentType.Rigid)
        def ego_lidar_callback(LidarMeasurement):
            if args.mode == 'common':
                if args.save_as_kitti_format:
                    import struct
                    save = open(os.path.join(VELO_DIR, '%06d.bin' %(LidarMeasurement.frame)), 'wb')
                else:
                    save = open(os.path.join(args.path, 'ego_lidar_measurement_%d.txt' %LidarMeasurement.frame), 'w')
                    LidarMeasurement.save_to_disk(os.path.join(args.path, 'ego_lidar_measurement_%d.ply' %LidarMeasurement.frame))
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
                        save.write(struct.pack('ffff', -point.y,-point.x,-point.z,0.5))
                    else:
                        print(point.x, point.y, point.z, file=save)
                save.close()
            
            # generate kitti-style annotation
            if args.mode == 'common' and args.save_as_kitti_format == True:
                cvil_vehicles = world.get_actors(vehicles_list[1:])
                save = open(os.path.join(LABEL_DIR, '%06d.txt' %(LidarMeasurement.frame)), 'w')

                # ego pos := lidar pos
                ego_pos = ego_lidar.get_transform()
                ego_x, ego_y, ego_z = ego_pos.location.x, -ego_pos.location.y, ego_pos.location.z
                ego_pitch, ego_roll = -ego.get_transform().rotation.pitch, ego.get_transform().rotation.roll
                ego_yaw = -ego.get_transform().rotation.yaw

                for vehicle in cvil_vehicles:
                    veh_pos = vehicle.get_transform()
                    #veh_x, veh_y, veh_z = veh_pos.location.x, veh_pos.location.y, veh_pos.location.z
                    veh_pitch, veh_roll = -vehicle.get_transform().rotation.pitch, vehicle.get_transform().rotation.roll
                    veh_yaw = -veh_pos.rotation.yaw
                    #print(ego_lidar.get_transform())       >EQUAL
                    #print(ego.get_transform())             >EQUAL
                    #print(veh_x,veh_y,veh_z)

                    # transform vehicle space to lidar space
                    veh_bbox = _vehicle_to_world(_create_bb_points(vehicle), vehicle)
                    veh_box = _world_to_sensor(veh_bbox, ego_lidar).transpose()
                    
                    # sensor(>CAR?) to KITTI velodyne
                    # i.e. x:= forward y:=right z:=up ==> x:=forward y:=left z:up
                    for i in range(len(veh_box)):
                        veh_box[i,0], veh_box[i,1], veh_box[i,2] = veh_box[i,0],-veh_box[i,1],veh_box[i,2]
                    veh_x, veh_y, veh_z = veh_box[8,0], veh_box[8,1], veh_box[8,2]
                    #print('->>>',veh_x,veh_y,veh_z)

                    veh_length = vehicle.bounding_box.extent.x*2
                    veh_width = vehicle.bounding_box.extent.y*2
                    veh_height = vehicle.bounding_box.location.z*2

                    # refine bounding box extent
                    veh_w = veh_width*cos((veh_roll-ego_roll)/180*pi)+veh_height*sin(abs(veh_roll-ego_roll)/180*pi) + 0.2
                    veh_l = veh_length*cos((veh_pitch-ego_pitch)/180*pi)+veh_height*sin(abs(veh_pitch-ego_pitch)/180*pi) + 0.2
                    veh_h = veh_length*sin(abs(veh_pitch-ego_pitch)/180*pi)+veh_height*cos((veh_roll-ego_roll)/180*pi)+veh_width*sin(abs(veh_roll-ego_roll)/180*pi) + 0.8
                    #print('====',veh_h,veh_l,veh_w)

                    # calculate rotation-y(ry)
                    yaw = -pi/2 - (veh_yaw-ego_yaw)/180*pi
                    if yaw < -pi: yaw = yaw + 2*pi
                    if yaw > pi: yaw = yaw - 2*pi
                    
                    print('%s %.2f %d %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f'
                        %('Car', 0, 0, atan(-(veh_x-ego_x)/(veh_y-ego_y)), # alpha(rad)
                            -100, -100, -100, -100, # no corresponding images
                            veh_h, veh_w, veh_l, # size := h,w,l
                            -veh_y, -veh_z, veh_x, # x,y,z <- location:-y,-z,x
                            yaw, 1), # yaw(rad)
                            file=save)
                save.close()

        sensors_list.append(ego_lidar)
        # lidar : listen
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

        if args.mode == 'common' and args.save_as_kitti_format == True:
            import lib.carla_utils as carla_utils
            
            print('\nexecuting clean up\n')
            carla_utils.clean_up(args.path, args.kitti_split)
            print('\ndone clean up\n')
            
            print('\nexecuting refinement\n')
            carla_utils.refinement(args.path, args.kitti_split)
            print('\ndone refinement\n')

        time.sleep(0.5)


if __name__ == '__main__':

    try:
        main()
    except KeyboardInterrupt:
        pass
    finally:
        print('\ndone.\n')

