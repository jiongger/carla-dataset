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
import lib.carla_utils as carla_utils

try:
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + '/carla')
except IndexError:
    pass

import argparse
import logging
import random

from math import sin, cos, tan, pi, sqrt, atan
import numpy as np

# define some carla color
red = carla.Color(255, 0, 0)
green = carla.Color(0, 255, 0)
blue = carla.Color(47, 210, 231)
cyan = carla.Color(0, 255, 255)
yellow = carla.Color(255, 255, 0)
orange = carla.Color(255, 162, 0)
white = carla.Color(255, 255, 255)

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
    '--hybrid',
    action='store_true',
    help='Enanble')
argparser.add_argument(
    '--path',
    metavar='P',
    default='logs',
    type=str,
    help='path/to/save/sensor/data')
argparser.add_argument(
    '--split',
    type=str,
    default='training',
    help=''
)
argparser.add_argument(
    '--imu',
    action='store_true',
    default=False,
    help='set imu on ego vehicle(s)'
)
argparser.add_argument(
    '--gnss',
    action='store_true',
    default=False,
    help='set gnss on ego vehicle(s)'
)
argparser.add_argument(
    '--lidar',
    action='store_true',
    default=False,
    help='set lidar on ego vehicle(s)'
)
argparser.add_argument(
    '--camera', '--rgb_camera',
    action='store_true',
    default=False,
    help='set RGE camera on ego vehicle(s)'
)
argparser.add_argument(
    '--sence',
    type=str,
    default='opposite'
)
args = argparser.parse_args()
PATH = args.path
SPLIT = args.split


def main():

    logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.INFO)

    client = carla.Client(args.host, args.port)
    client.set_timeout(10.0)
    synchronous_master = False

    # @todo cannot import these directly.
    SpawnActor = carla.command.SpawnActor
    SetAutopilot = carla.command.SetAutopilot
    FutureActor = carla.command.FutureActor

    try:
        world = client.get_world()
    
        # static hyper parameters
        SPAWN_RANGE = 180
        SPAWN_RATIO = 0.6
        PATH_DISTANCE = 120

        # collect objects
        vehicles_list = [] # vehicle id >> list: int
        sensors_list = [] # sensor >> list: carla.sensor
        egos = [] # list: carla.vehicle
        cvil_vehicles = [] # list: carla.vehicle
        ego_names = []

        spawn_points = world.get_map().get_spawn_points() # list: carla.transform
        selected_spawn_points = [] # list: carla.transform
        if args.sence == 'opposite':
            selected_ego_spawn_points_index = [62,13] # list: int
            selected_ego_destination_points_index = [110,57] # list: int        
        elif args.sence == 'cross':
            selected_ego_spawn_points_index = [62,228] # list: int
            selected_ego_destination_points_index = [110,200] # list: int
        elif args.sence == 'parallel':
            selected_ego_spawn_points_index = [63,62] # list: int
            selected_ego_destination_points_index = [109,110] # list: int
        else:
            selected_ego_spawn_points_index = [62,228] # list: int
            selected_ego_destination_points_index = [110,200] # list: int
        # opposite: 62,13 -> 110,57
        # cross: 62,228 -> 110,200
        # parallel: 63,62 -> 109,110
        selected_ego_spawn_points = [] # list: carla.transform

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
        
        if os.path.exists(args.path) == False:
            os.makedirs(args.path)
        else:
            import shutil
            shutil.rmtree(args.path)

        # generate selected spawn points: list
        spawn_center = np.array([0,0,0])
        for point_index in selected_ego_spawn_points_index:
            spawn_point = spawn_points[point_index]
            spawn_location = spawn_point.location
            spawn_center = spawn_center + np.array([spawn_location.x, spawn_location.y, spawn_location.z])
            selected_ego_spawn_points.append(spawn_point)
        spawn_center = spawn_center/len(selected_ego_spawn_points_index)
        for index,spawn_point in enumerate(spawn_points):
            if index in selected_ego_spawn_points_index: continue
            spawn_location = spawn_point.location
            if sqrt((spawn_location.x - spawn_center[0])**2 + (spawn_location.y - spawn_center[1])**2 + (spawn_location.z - spawn_center[2])**2) <= SPAWN_RANGE:
                selected_spawn_points.append(spawn_point)
        selected_spawn_points = np.random.choice(selected_spawn_points, size=int(len(selected_spawn_points)*SPAWN_RATIO), replace=False) # choose some spawn points randomly

        batch = []
        STAT_ID = 0
        # spawn ego @ selected points
        for spawn_point in selected_ego_spawn_points:
            STAT_ID = STAT_ID + 1
            ego_bp = world.get_blueprint_library().find('vehicle.tesla.model3')
            ego_bp.set_attribute('role_name',str(STAT_ID))
            ego_color = random.choice(ego_bp.get_attribute('color').recommended_values)
            ego_bp.set_attribute('color',ego_color)
            ego_transform = spawn_point
            batch.append(SpawnActor(ego_bp,ego_transform))
        
        # carla work (convert Carla.SpawnActor to Carla.Vehicle)
        from lib.carla.agents.navigation.basic_agent import BasicAgent
        agents = []
        for response,destination_index in zip(client.apply_batch_sync(batch, synchronous_master),selected_ego_destination_points_index):
            if response.error:
                logging.error(response.error)
            else:
                vehicles_list.append(response.actor_id)
                ego = world.get_actor(response.actor_id)
                egos.append(ego)
                ego_names.append(ego.attributes['role_name'])
                agent = BasicAgent(ego)
                if destination_index is not None:
                    ego_destination = spawn_points[destination_index].location
                else:
                    ego_vector = ego.get_transform().rotation.get_forward_vector() * PATH_DISTANCE
                    ego_startup = ego.get_location()
                    ego_destination = ego_startup + ego_vector
                agent.set_destination((ego_destination.x,ego_destination.y,ego_destination.z))
                agents.append(agent)

        # spawn civil vehicles
        batch = []
        for spawn_point in selected_spawn_points:
            STAT_ID = STAT_ID + 1
            blueprint = random.choice(blueprints)
            if blueprint.has_attribute('color'):
                color = random.choice(blueprint.get_attribute('color').recommended_values)
                blueprint.set_attribute('color', color)
            if blueprint.has_attribute('driver_id'):
                driver_id = random.choice(blueprint.get_attribute('driver_id').recommended_values)
                blueprint.set_attribute('driver_id', driver_id)
            blueprint.set_attribute('role_name', str(STAT_ID))
            batch.append(SpawnActor(blueprint, spawn_point).then(SetAutopilot(FutureActor, True, traffic_manager.get_port())))

        for response in client.apply_batch_sync(batch, synchronous_master):
            if response.error:
                logging.error(response.error)
            else:
                vehicles_list.append(response.actor_id)
                cvil_vehicles.append(world.get_actor(response.actor_id))

        # adjust spectator to proper position       
        spectator = world.get_spectator()
        # set spectator above the center 
        location = carla.Location(spawn_center[0]-30,spawn_center[1],spawn_center[2]+50)
        rotation = carla.Rotation(-45,0,0)
        transform = carla.Transform(location, rotation)
        spectator.set_transform(transform)
        
        # find sensor blueprints
        if args.gnss:
            # find gnss blueprint
            gnss_bp = world.get_blueprint_library().find('sensor.other.gnss')
            gnss_bp.set_attribute("sensor_tick",str(0.1))
        if args.imu:
            # find imu blueprint
            imu_bp = world.get_blueprint_library().find('sensor.other.imu')
            imu_bp.set_attribute("sensor_tick",str(0.1))
        if args.lidar:
            # find lidar blueprint
            lidar_bp = world.get_blueprint_library().find('sensor.lidar.ray_cast')
            lidar_bp.set_attribute('channels',str(64))
            lidar_bp.set_attribute('points_per_second',str(960000))
            lidar_bp.set_attribute('rotation_frequency',str(10))
            lidar_bp.set_attribute('sensor_tick', str(0.1))
            lidar_bp.set_attribute('upper_fov', str(0))
            lidar_bp.set_attribute('lower_fov', str(-24))
            lidar_bp.set_attribute('range',str(60))
        if args.camera:
            IMAGEX = 1920
            IMAGEY = 1080
            # find rgb camera blueprint
            camera_bp = world.get_blueprint_library().find('sensor.camera.rgb')
            camera_bp.set_attribute('image_size_x', str(IMAGEX))
            camera_bp.set_attribute('image_size_y', str(IMAGEY))
            camera_bp.set_attribute('fov', '120')
            camera_bp.set_attribute('sensor_tick', str(0.1))


        if args.gnss:
            pass
            # set gnss @ ego
            gnss_location = carla.Location(0,0,0)
            gnss_rotation = carla.Rotation(0,0,0)
            gnss_transform = carla.Transform(gnss_location,gnss_rotation)
            from lib.carla_utils import ego_gnss_callback
            for ego in egos:
                ego_gnss = world.spawn_actor(gnss_bp,gnss_transform,attach_to=ego, attachment_type=carla.AttachmentType.Rigid)
                if os.path.exists(os.path.join(PATH, ego.attributes['role_name'])) == False:
                    os.makedirs(os.path.join(PATH, ego.attributes['role_name']))
                ego_gnss_log = open(os.path.join(PATH, ego.attributes['role_name'], 'gnss.log'), 'w')
                sensors_list.append(ego_gnss)
                # gnss : listen
                ego_gnss.listen(lambda gnss: ego_gnss_callback(gnss,ego_gnss_log))


        if args.imu:
            # set imu @ ego
            imu_location = carla.Location(0,0,0)
            imu_rotation = carla.Rotation(0,0,0)
            imu_transform = carla.Transform(imu_location,imu_rotation)
            from lib.carla_utils import ego_imu_callback
            ego = egos[0]
            ego_imu = world.spawn_actor(imu_bp,imu_transform,attach_to=ego, attachment_type=carla.AttachmentType.Rigid)
            if os.path.exists(os.path.join(PATH, ego.attributes['role_name'])) == False:
                os.makedirs(os.path.join(PATH, ego.attributes['role_name']))
            ego_imu_log1 = open(os.path.join(PATH, ego.attributes['role_name'], 'imu.log' ), 'w')
            sensors_list.append(ego_imu)
            # imu : listen
            ego_imu.listen(lambda imu: ego_imu_callback(imu,ego_imu_log1))
            ego = egos[1]
            ego_imu = world.spawn_actor(imu_bp,imu_transform,attach_to=ego, attachment_type=carla.AttachmentType.Rigid)
            if os.path.exists(os.path.join(PATH, ego.attributes['role_name'])) == False:
                os.makedirs(os.path.join(PATH, ego.attributes['role_name']))
            ego_imu_log2 = open(os.path.join(PATH, ego.attributes['role_name'], 'imu.log' ), 'w')
            sensors_list.append(ego_imu)
            # imu : listen
            ego_imu.listen(lambda imu: ego_imu_callback(imu,ego_imu_log2))


        if args.lidar:
            def ego_lidar_callback(LidarMeasurement,VELO_DIR,LABEL_DIR,ego):
                import struct
                save = open(os.path.join(VELO_DIR, '%06d.bin' %(LidarMeasurement.frame)), 'wb')
                for point in LidarMeasurement:
                    save.write(struct.pack('ffff', -point.y,-point.x,-point.z,0.5))
                save.close()
                # generate ground truth
                ego_x, ego_y, ego_z = ego.get_location().x, -ego.get_location().y, ego.get_location().z
                ego_yaw = -ego.get_transform().rotation.yaw
                vehicles = world.get_actors(vehicles_list)
                save = open(os.path.join(LABEL_DIR, '%06d.txt' %(LidarMeasurement.frame)), 'w')
                for vehicle in vehicles:
                    veh_bbox = vehicle.bounding_box
                    veh_x, veh_y, veh_z = vehicle.get_location().x, -vehicle.get_location().y, vehicle.get_location().z
                    #print(veh_bbox.location.z)
                    veh_yaw = -vehicle.get_transform().rotation.yaw
                    yaw = -pi/2 -(veh_yaw-ego_yaw)/180*pi
                    if yaw < -pi: yaw = yaw + 2*pi
                    if yaw > pi: yaw = yaw - 2*pi
                    print('%s %s %.2f %.2f %.2f %.2f %.2f %.2f %.2f'
                        %('Car', vehicle.attributes['role_name'], 
                          veh_bbox.extent.z*2,veh_bbox.extent.y*2,veh_bbox.extent.x*2, # size := h,w,l
                          -((veh_y-ego_y)*cos((ego_yaw)/180*pi)-(veh_x-ego_x)*sin((ego_yaw)/180*pi)), # x -> location:-y
                          -((veh_z+veh_bbox.location.z)-(ego_z+lidar_height)), # y -> location:-z
                          (veh_x-ego_x)*cos((ego_yaw)/180*pi)+(veh_y-ego_y)*sin((ego_yaw)/180*pi), # z -> location:x
                          yaw), # yaw(rad)
                        file=save)
                save.close()
            # set lidar @ ego
            ego1 = egos[0]
            VELO_DIR1 = os.path.join(PATH, ego1.attributes['role_name'], 'object', SPLIT, 'velodyne')
            LABEL_DIR1 = os.path.join(PATH, ego1.attributes['role_name'], 'object', SPLIT, 'label_2')
            if os.path.exists(VELO_DIR1) == False:
                os.makedirs(VELO_DIR1)
            if os.path.exists(LABEL_DIR1) == False:
                os.makedirs(LABEL_DIR1)
            lidar_height = tan(pi/180*25)*ego1.bounding_box.extent.x + ego1.bounding_box.extent.z*2
            ego1_lidar_location = carla.Location(0,0, lidar_height)
            lidar_rotation = carla.Rotation(0,0,0)
            ego1_lidar_transform = carla.Transform(ego1_lidar_location,lidar_rotation)
            ego1_lidar = world.spawn_actor(lidar_bp,ego1_lidar_transform,attach_to=ego1,attachment_type=carla.AttachmentType.Rigid)
            sensors_list.append(ego1_lidar)
            # lidar : listen
            ego1_lidar.listen(lambda lidar: ego_lidar_callback(lidar,VELO_DIR1,LABEL_DIR1,ego1))
            ego2 = egos[1]
            VELO_DIR2 = os.path.join(PATH, ego2.attributes['role_name'], 'object', SPLIT, 'velodyne')
            LABEL_DIR2 = os.path.join(PATH, ego2.attributes['role_name'], 'object', SPLIT, 'label_2')
            if os.path.exists(VELO_DIR2) == False:
                os.makedirs(VELO_DIR2)
            if os.path.exists(LABEL_DIR2) == False:
                os.makedirs(LABEL_DIR2)
            lidar_height = tan(pi/180*25)*ego2.bounding_box.extent.x + ego2.bounding_box.extent.z*2
            ego2_lidar_location = carla.Location(0,0, lidar_height)
            lidar_rotation = carla.Rotation(0,0,0)
            ego2_lidar_transform = carla.Transform(ego2_lidar_location,lidar_rotation)
            ego2_lidar = world.spawn_actor(lidar_bp,ego2_lidar_transform,attach_to=ego2,attachment_type=carla.AttachmentType.Rigid)
            sensors_list.append(ego2_lidar)
            # lidar : listen
            ego2_lidar.listen(lambda lidar: ego_lidar_callback(lidar,VELO_DIR2,LABEL_DIR2,ego2))
        

        if args.camera:
            ego_camera_location = carla.Location(0.8,0,1.7)
            ego_camera_rotation = carla.Rotation(0,0,0)
            ego_camera_transform = carla.Transform(ego_camera_location, ego_camera_rotation)
            def ego_camera_callback(RGBMeasurement,IMAGE_DIR,CALIB_DIR):
                RGBMeasurement.save_to_disk(os.path.join(IMAGE_DIR, '%06d.png' %RGBMeasurement.frame))
                #calib_file = open(os.path.join(CALIB_DIR, '%06d.txt' %RGBMeasurement.frame), 'w')
                #print('P0: %.12e %.12e %.12e %.12e %.12e %.12e %.12e %.12e %.12e %.12e %.12e %.12e' %())
                #calib_file.close()
            # set camera @ ego
            ego = egos[0]
            ego_camera = world.spawn_actor(camera_bp, ego_camera_transform, attach_to=ego, attachment_type=carla.AttachmentType.Rigid)
            IMAGE_DIR1 = os.path.join(PATH, ego.attributes['role_name'], 'object', SPLIT, 'image_2')
            CALIB_DIR1 = os.path.join(PATH, ego.attributes['role_name'], 'object', SPLIT, 'calib')
            if os.path.exists(IMAGE_DIR1) == False:
                os.makedirs(IMAGE_DIR1)
            if os.path.exists(CALIB_DIR1) == False:
                os.makedirs(CALIB_DIR1)
            sensors_list.append(ego_camera)
            # camera : listen
            ego_camera.listen(lambda RGBMeasurement: ego_camera_callback(RGBMeasurement,IMAGE_DIR1,CALIB_DIR1))
            ego = egos[1]
            ego_camera = world.spawn_actor(camera_bp, ego_camera_transform, attach_to=ego, attachment_type=carla.AttachmentType.Rigid)
            IMAGE_DIR2 = os.path.join(PATH, ego.attributes['role_name'], 'object', SPLIT, 'image_2')
            CALIB_DIR2 = os.path.join(PATH, ego.attributes['role_name'], 'object', SPLIT, 'calib')
            if os.path.exists(IMAGE_DIR2) == False:
                os.makedirs(IMAGE_DIR2)
            if os.path.exists(CALIB_DIR2) == False:
                os.makedirs(CALIB_DIR2)
            sensors_list.append(ego_camera)
            # camera : listen
            ego_camera.listen(lambda RGBMeasurement: ego_camera_callback(RGBMeasurement,IMAGE_DIR2,CALIB_DIR2))
    

        # debug mode
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
        debug_list = egos
        # preprocess debug vehicle waypoints
        current_w = []
        for vehicle in debug_list:
            current_w.append(current_map.get_waypoint(vehicle.get_location()))
        
        while True:

            for agent in agents:
                if agent.done():
                    raise EOFError

            if not args.camera:
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
                    debug.draw_box(bbox, vehicle.get_transform().rotation, 0.5, red, some_time*2)

                # Update the current waypoint and sleep for some time
                current_w = next_w.copy()
                time.sleep(some_time)

            # update egos control
            for agent,ego in zip(agents,egos):
                control = agent.run_step()
                control.manual_gear_shift = False
                ego.apply_control(control)

            # carla tick
            if args.sync and synchronous_master:
                world.tick()
            else:
                world.wait_for_tick()

    
    finally:

        print('\npost processing...')
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

        import lib.carla_utils as carla_utils
            
        print('\nexecuting clean up')
        for ego_name in ego_names:
            CPATH = os.path.join(PATH, ego_name)
            carla_utils.clean_up(CPATH, SPLIT)
        print('\ndone clean up')

        time.sleep(0.5)


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print('\naborted\n')
    finally:
        print('\ndone.\n')

