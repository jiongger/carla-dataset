import os
import sys
import time

if os.path.exists('tools/lib/carla/dist/carla-0.9.9-py3.7-win-amd64.egg'):
    sys.path.append(os.path.abspath('tools/lib/carla/dist/carla-0.9.9-py3.7-win-amd64.egg'))
elif os.path.exists('lib/carla/dist/carla-0.9.9-py3.7-win-amd64.egg'):
    sys.path.append(os.path.abspath('lib/carla/dist/carla-0.9.9-py3.7-win-amd64.egg'))
if os.path.exists('tools'):
    sys.path.append(os.path.abspath('tools'))
    sys.path.append(os.path.abspath('tools/lib'))
    sys.path.append(os.path.abspath('tools/lib/carla'))
#print(sys.path)

from lib.carla.agents.navigation.basic_agent import BasicAgent
import carla

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
    '--debug',
    default=False,
    action='store_true')
args = argparser.parse_args()
PATH = args.path
print('exporting data to', os.path.abspath(PATH))


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

        # collect objects
        vehicles_list = [] # vehicle id >> list: int
        sensors_list = [] # sensor >> list: carla.sensor
        egos = [] # list: carla.vehicle
        cvil_vehicles = [] # list: carla.vehicle
        vehicles = [] # list: carla.vehicle
        ego_names = []

        spawn_points = world.get_map().get_spawn_points() # list: carla.transform
        selected_civil_routes = [
            [spawn_points[61],spawn_points[107]],
            [spawn_points[190],spawn_points[109]],
            [spawn_points[12],spawn_points[9]],
            [spawn_points[11],spawn_points[57]],
            [spawn_points[136],spawn_points[59]],
            [spawn_points[85],spawn_points[28]],
            [spawn_points[59],spawn_points[262]],
            [spawn_points[259],spawn_points[155]]
        ]
        selected_ego_routes = [
            [spawn_points[62],spawn_points[110]]
        ]
        roadside_camera_transforms = [
            carla.Transform(
                carla.Location(-90,20,8),
                carla.Rotation(-45,-45,0)
            )
        ]

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
        for route in selected_civil_routes:
            spawn_point = route[0]
            spawn_location = spawn_point.location
            spawn_center = spawn_center + np.array([spawn_location.x, spawn_location.y, spawn_location.z])
        spawn_center = spawn_center/len(selected_civil_routes)

        # adjust spectator to proper position       
        spectator = world.get_spectator()
        # set spectator above the center 
        location = carla.Location(spawn_center[0]-30,spawn_center[1],spawn_center[2]+50)
        rotation = carla.Rotation(-45,0,0)
        transform = carla.Transform(location, rotation)
        spectator.set_transform(transform)        

        batch = []
        STAT_ID = 0
        # spawn ego @ selected points
        for route in selected_ego_routes:
            STAT_ID = STAT_ID + 1
            ego_bp = world.get_blueprint_library().find('vehicle.tesla.model3')
            ego_bp.set_attribute('role_name',str(STAT_ID))
            ego_color = random.choice(ego_bp.get_attribute('color').recommended_values)
            ego_bp.set_attribute('color',ego_color)
            batch.append(SpawnActor(ego_bp, route[0]))
        # spawn civil vehicles
        for route in selected_civil_routes:
            STAT_ID = STAT_ID + 1
            blueprint = random.choice(blueprints)
            if blueprint.has_attribute('color'):
                color = random.choice(blueprint.get_attribute('color').recommended_values)
                blueprint.set_attribute('color', color)
            blueprint.set_attribute('role_name', str(STAT_ID))
            batch.append(SpawnActor(blueprint, route[0]))
        #print(len(selected_ego_routes), len(selected_civil_routes), len(batch))

        routes = selected_ego_routes + selected_civil_routes
        # carla work (convert Carla.SpawnActor to Carla.Vehicle)
        agents = []
        for i, (response, route) in enumerate(zip(client.apply_batch_sync(batch, synchronous_master), routes)):
            if response.error:
                logging.error(response.error)
            else:
                vehicles_list.append(response.actor_id)
                if i < len(selected_ego_routes):
                    ego = world.get_actor(response.actor_id)
                    egos.append(ego)
                    vehicles.append(ego)
                    ego_names.append(ego.attributes['role_name'])
                    agent = BasicAgent(ego)
                else:
                    civil = world.get_actor(response.actor_id)
                    cvil_vehicles.append(civil)
                    vehicles.append(civil)
                    agent = BasicAgent(civil)
                agent.set_destination((route[1].location.x,route[1].location.y,route[1].location.z))
                agents.append(agent)

        # find sensor blueprints
        IMAGEX = 1920
        IMAGEY = 1080
        # find rgb camera blueprint
        camera_bp = world.get_blueprint_library().find('sensor.camera.rgb')
        camera_bp.set_attribute('image_size_x', str(IMAGEX))
        camera_bp.set_attribute('image_size_y', str(IMAGEY))
        camera_bp.set_attribute('fov', '120')
        camera_bp.set_attribute('sensor_tick', str(0.1))

        ego_camera_location = carla.Location(0.8,0,1.7)
        ego_camera_rotation = carla.Rotation(0,0,0)
        ego_camera_transform = carla.Transform(ego_camera_location, ego_camera_rotation)
        def camera_callback(RGBMeasurement,IMAGE_DIR):
            RGBMeasurement.save_to_disk(os.path.join(IMAGE_DIR, '%06d.png' %RGBMeasurement.frame))
        import shutil
        # set camera @ ego
        ego = egos[0]
        ego_camera = world.spawn_actor(camera_bp, ego_camera_transform, attach_to=ego, attachment_type=carla.AttachmentType.Rigid)
        IMAGE_DIR1 = os.path.join(PATH, 'image', 'ego_'+ego.attributes['role_name'])
        if os.path.exists(IMAGE_DIR1):
            shutil.rmtree(IMAGE_DIR1)
        os.makedirs(IMAGE_DIR1)
        sensors_list.append(ego_camera)
        # camera : listen
        ego_camera.listen(lambda RGBMeasurement: camera_callback(RGBMeasurement,IMAGE_DIR1))
        # set camera @ roadside
        roadside_camera = world.spawn_actor(camera_bp, roadside_camera_transforms[0])
        IMAGE_DIR2 = os.path.join(PATH, 'image', 'roadside_'+ego.attributes['role_name'])
        if os.path.exists(IMAGE_DIR2):
            shutil.rmtree(IMAGE_DIR2)
        os.makedirs(IMAGE_DIR2)
        sensors_list.append(roadside_camera)
        # camera : listen
        roadside_camera.listen(lambda RGBMeasurement: camera_callback(RGBMeasurement,IMAGE_DIR2))
    
        if args.debug:

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
            
            if args.debug:
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

            # update vehicles control
            for agent,vehicle in zip(agents,vehicles):
                control = agent.run_step()
                control.manual_gear_shift = False
                vehicle.apply_control(control)

            # carla tick
            if args.sync and synchronous_master:
                world.tick()
            else:
                world.wait_for_tick()

    except KeyboardInterrupt:
        print('\naborted\n')
    
    finally:

        print('\npost processing...')

        print('\ndestroying %d vehicles' % len(vehicles_list))
        client.apply_batch([carla.command.DestroyActor(x) for x in vehicles_list])
        time.sleep(0.5)
        
        print('\ndestroying %d sensors' % len(sensors_list))
        for sensor in sensors_list:
            sensor.stop()
        client.apply_batch([carla.command.DestroyActor(x.id) for x in sensors_list])
        time.sleep(0.5)

        if args.sync and synchronous_master:
            settings = world.get_settings()
            settings.synchronous_mode = False
            settings.fixed_delta_seconds = None
            world.apply_settings(settings)
        
        print('\ndone\n')


if __name__ == '__main__':
    main()
