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
        '--hybrid',
        action='store_true',
        help='Enanble')
    argparser.add_argument(
        '-d', '--delay',
        metavar='D',
        default=2.0,
        type=float,
        help='delay in seconds between spawns (default: 2.0)')
    argparser.add_argument(
        '--mode',
        metavar='M',
        default='common',
        type=str,
        choices=['debug-ego', 'debug-all', 'common'],
        help='debug mode disable all file output')
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
            synchronous_master = True
            settings.synchronous_mode = True
            settings.fixed_delta_seconds = 0.05
            world.apply_settings(settings)

        blueprints = world.get_blueprint_library().filter(args.filterv)

        if args.safe:
            blueprints = [x for x in blueprints if int(x.get_attribute('number_of_wheels')) == 4]
            blueprints = [x for x in blueprints if not x.id.endswith('isetta')]
            blueprints = [x for x in blueprints if not x.id.endswith('carlacola')]
            blueprints = [x for x in blueprints if not x.id.endswith('cybertruck')]
            blueprints = [x for x in blueprints if not x.id.endswith('t2')]

        spawn_points = world.get_map().get_spawn_points()
        selected_spawn_points = [8,9,10,11,12,13,14,15,30,31,58,59,60,61,85,90,91,98,109,110,136,228,229,232,254,255,256,257,258,259,260,261]
        selected_ego_spawn_points = [31,85]
        for ego_spawn_point in selected_spawn_points:
            if ego_spawn_point not in selected_spawn_points:
                selected_spawn_points.append(ego_spawn_point)
        selected_spawn_points.sort()
        selected_ego_spawn_points.sort()
        # @todo cannot import these directly.
        SpawnActor = carla.command.SpawnActor
        SetAutopilot = carla.command.SetAutopilot
        FutureActor = carla.command.FutureActor

        # spawn civil vehicles
        batch = []
        for spawn_point_index in selected_spawn_points:
            if spawn_point_index in selected_ego_spawn_points:
                continue
            blueprint = random.choice(blueprints)
            if blueprint.has_attribute('color'):
                color = random.choice(blueprint.get_attribute('color').recommended_values)
                blueprint.set_attribute('color', color)
            if blueprint.has_attribute('driver_id'):
                driver_id = random.choice(blueprint.get_attribute('driver_id').recommended_values)
                blueprint.set_attribute('driver_id', driver_id)
            blueprint.set_attribute('role_name', 'autopilot')
            batch.append(SpawnActor(blueprint, spawn_points[spawn_point_index]).then(SetAutopilot(FutureActor, True, traffic_manager.get_port())))

        for response in client.apply_batch_sync(batch, synchronous_master):
            if response.error:
                logging.error(response.error)
            else:
                vehicles_list.append(response.actor_id)

        # adjust spectator to proper position       
        spectator = world.get_spectator()
        x = 0.0
        y = 0.0
        z = 0.0
        # calculate center of spawn points
        for spawn_point_index in selected_spawn_points:
            x = x + spawn_points[spawn_point_index].location.x/len(selected_spawn_points)
            y = y + spawn_points[spawn_point_index].location.y/len(selected_spawn_points)
            z = z + spawn_points[spawn_point_index].location.z/len(selected_spawn_points)
        # set spectator above the center 
        location = carla.Location(x-30,y,z+30)
        rotation = carla.Rotation(-45,0,0)
        transform = carla.Transform(location, rotation)
        spectator.set_transform(transform)

        # find ego vehicle blueprint
        ego_bp = world.get_blueprint_library().find('vehicle.tesla.model3')
        ego_bp.set_attribute('role_name','ego')
        ego_color = random.choice(ego_bp.get_attribute('color').recommended_values)
        ego_bp.set_attribute('color',ego_color)

        batch = []
        # spawn ego1 @ point #31
        ego1_transform = spawn_points[selected_ego_spawn_points[0]]
        batch.append(SpawnActor(ego_bp,ego1_transform).then(SetAutopilot(FutureActor, True, traffic_manager.get_port())))
        # spawn ego2 @ point #85
        ego2_transform = spawn_points[selected_ego_spawn_points[1]]
        batch.append(SpawnActor(ego_bp,ego2_transform).then(SetAutopilot(FutureActor, True, traffic_manager.get_port())))
        
        # carla work (convert Carla.SpawnActor to Carla.Vehicle)
        ego = []
        for response in client.apply_batch_sync(batch, synchronous_master):
            if response.error:
                logging.error(response.error)
            else:
                vehicles_list.append(response.actor_id)
                ego_vehicles_list.append(response.actor_id)
                ego.append(world.get_actor(response.actor_id))
        # calculate vehicle 3d parameters
        ego_bbox = []
        ego_length = []
        ego_width = []
        ego_height = []
        ego_center = []
        for v in ego:
            ego_bbox.append(v.bounding_box)
            ego_length.append(v.bounding_box.extent.x + v.bounding_box.location.x)
            ego_width.append(v.bounding_box.extent.y + v.bounding_box.location.y)
            ego_height.append(v.bounding_box.extent.z + v.bounding_box.location.z)
            ego_center.append(v.bounding_box.location)
        
        # find gnss blueprint
        gnss_bp = world.get_blueprint_library().find('sensor.other.gnss')
        gnss_location = carla.Location(0,0,0)
        gnss_rotation = carla.Rotation(0,0,0)
        gnss_transform = carla.Transform(gnss_location,gnss_rotation)
        gnss_bp.set_attribute("sensor_tick",str(1.0))
        # find imu blueprint
        imu_bp = world.get_blueprint_library().find('sensor.other.imu')
        imu_location = carla.Location(0,0,0)
        imu_rotation = carla.Rotation(0,0,0)
        imu_transform = carla.Transform(imu_location,imu_rotation)
        imu_bp.set_attribute("sensor_tick",str(1.0))
        # find lidar blueprint
        lidar_bp = world.get_blueprint_library().find('sensor.lidar.ray_cast')
        lidar_bp.set_attribute('channels',str(256))
        lidar_bp.set_attribute('points_per_second',str(180000))
        lidar_bp.set_attribute('rotation_frequency',str(10))
        lidar_bp.set_attribute('sensor_tick', str(1))
        lidar_bp.set_attribute('upper_fov', str(60))
        lidar_bp.set_attribute('lower_fov', str(-30))
        lidar_bp.set_attribute('range',str(50))
        ego1_lidar_location = carla.Location(0,0,tan(pi/6)*ego_length[0]/2 + ego_height[0] - ego_center[0].z + 0.1)
        ego2_lidar_location = carla.Location(0,0,tan(pi/6)*ego_length[1]/2 + ego_height[1] - ego_center[1].z + 0.1)
        lidar_rotation = carla.Rotation(90,0,0)
        ego1_lidar_transform = carla.Transform(ego1_lidar_location,lidar_rotation)
        ego2_lidar_transform = carla.Transform(ego2_lidar_location,lidar_rotation)

        # set gnss @ ego1
        ego1_gnss = world.spawn_actor(gnss_bp,gnss_transform,attach_to=ego[0], attachment_type=carla.AttachmentType.Rigid)
        if args.mode == 'common':
            ego1_gnss_log = open('ego1_gnss.log', 'w')
        else:
            ego1_gnss_log = None
        def ego1_gnss_callback(gnss):
            print(gnss.frame, gnss.timestamp, 
            gnss.transform.location.x, gnss.transform.location.y, gnss.transform.location.z,
            gnss.transform.rotation.roll, gnss.transform.rotation.pitch, gnss.transform.rotation.yaw,
            gnss.latitude, gnss.longitude, gnss.altitude, file=ego1_gnss_log)
        ego1_gnss.listen(lambda gnss: ego1_gnss_callback(gnss))
        sensors_list.append(ego1_gnss)
        # set gnss @ ego2
        ego2_gnss = world.spawn_actor(gnss_bp,gnss_transform,attach_to=ego[1], attachment_type=carla.AttachmentType.Rigid)
        if args.mode == 'common':
            ego2_gnss_log = open('ego2_gnss.log', 'w')
        else:
            ego2_gnss_log = None
        def ego2_gnss_callback(gnss):
            print(gnss.frame, gnss.timestamp, 
            gnss.transform.location.x, gnss.transform.location.y, gnss.transform.location.z,
            gnss.transform.rotation.roll, gnss.transform.rotation.pitch, gnss.transform.rotation.yaw,
            gnss.latitude, gnss.longitude, gnss.altitude, file=ego2_gnss_log)
        ego2_gnss.listen(lambda gnss: ego2_gnss_callback(gnss))
        sensors_list.append(ego2_gnss)

        # set imu @ ego1
        ego1_imu = world.spawn_actor(imu_bp,imu_transform,attach_to=ego[0], attachment_type=carla.AttachmentType.Rigid)
        if args.mode == 'common':
            ego1_imu_log = open('ego1_imu.log', 'w')
        else:
            ego1_imu_log = None
        def ego1_imu_callback(imu):
            print(imu.frame, imu.timestamp, 
            imu.transform.location.x, imu.transform.location.y, imu.transform.location.z, 
            imu.transform.rotation.roll, imu.transform.rotation.pitch, imu.transform.rotation.yaw, 
            imu.accelerometer.x, imu.accelerometer.y, imu.accelerometer.z, 
            imu.gyroscope.x, imu.gyroscope.y, imu.gyroscope.z, 
            imu.compass, file=ego1_imu_log)
        ego1_imu.listen(lambda imu: ego1_imu_callback(imu))
        sensors_list.append(ego1_imu)
        # set imu @ ego2
        ego2_imu = world.spawn_actor(imu_bp,imu_transform,attach_to=ego[1], attachment_type=carla.AttachmentType.Rigid)
        if args.mode == 'common':
            ego2_imu_log = open('ego2_imu.log', 'w')
        else:
            ego2_imu_log = None
        def ego2_imu_callback(imu):
            print(imu.frame, imu.timestamp, 
            imu.transform.location.x, imu.transform.location.y, imu.transform.location.z, 
            imu.transform.rotation.roll, imu.transform.rotation.pitch, imu.transform.rotation.yaw, 
            imu.accelerometer.x, imu.accelerometer.y, imu.accelerometer.z, 
            imu.gyroscope.x, imu.gyroscope.y, imu.gyroscope.z, 
            imu.compass, file=ego2_imu_log)
        ego2_imu.listen(lambda imu: ego2_imu_callback(imu))
        sensors_list.append(ego2_imu)

        # set lidar @ ego1
        ego1_lidar = world.spawn_actor(lidar_bp,ego1_lidar_transform,attach_to=ego[0],attachment_type=carla.AttachmentType.SpringArm)
        def ego1_lidar_callback(LidarMeasurement):
            if args.mode == 'common':
                save = open('ego1_lidar_measurement_%d.txt' %LidarMeasurement.frame, 'w')
                LidarMeasurement.save_to_disk('ego1_lidar_measurement_%d.ply' %LidarMeasurement.frame)
                print(LidarMeasurement.frame, LidarMeasurement.timestamp,
                LidarMeasurement.transform.location.x, LidarMeasurement.transform.location.y, LidarMeasurement.transform.location.z,
                LidarMeasurement.transform.rotation.roll, LidarMeasurement.transform.rotation.pitch, LidarMeasurement.transform.rotation.yaw, 
                LidarMeasurement.horizontal_angle, LidarMeasurement.channels, file=save)
                for point in LidarMeasurement:
                    print(point.x, point.y, point.z, file=save)
        ego1_lidar.listen(ego1_lidar_callback)
        sensors_list.append(ego1_lidar)
        # set lidar @ ego2
        ego2_lidar = world.spawn_actor(lidar_bp,ego2_lidar_transform,attach_to=ego[1],attachment_type=carla.AttachmentType.SpringArm)
        def ego2_lidar_callback(LidarMeasurement):
            if args.mode == 'common':
                save = open('ego2_lidar_measurement_%d.txt' %LidarMeasurement.frame, 'w')
                LidarMeasurement.save_to_disk('ego2_lidar_measurement_%d.ply' %LidarMeasurement.frame)
                print(LidarMeasurement.frame, LidarMeasurement.timestamp,
                LidarMeasurement.transform.location.x, LidarMeasurement.transform.location.y, LidarMeasurement.transform.location.z,
                LidarMeasurement.transform.rotation.roll, LidarMeasurement.transform.rotation.pitch, LidarMeasurement.transform.rotation.yaw, 
                LidarMeasurement.horizontal_angle, LidarMeasurement.channels, file=save)
                for point in LidarMeasurement:
                    print(point.x, point.y, point.z, file=save)
        ego2_lidar.listen(ego2_lidar_callback)
        sensors_list.append(ego2_lidar)
        
        # debug mode
        if args.mode.find('debug') >= 0:
            
            some_time = 0.1
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
            # debug loop
            while True:
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
                    
                # carla tick
                if args.sync and synchronous_master:
                    world.tick()
                else:
                    world.wait_for_tick()

        else:
            while True:
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
        print('\ndone.')

