import carla
import random

class Create_Actors:
    def __init__(self,world, ego_vehicle, number=100):
        self.world = world
        self.map = world.get_map()
        self.waypoints = self.map.generate_waypoints(distance=1.0)
        self.ego_vehicle = ego_vehicle

        spawn_points = world.get_map().get_spawn_points()

        blueprint_library = world.get_blueprint_library()
        vehicle_bp = blueprint_library.find('vehicle.tesla.model3')
        walker_list = []
        for bl in blueprint_library:
            if 'walker' in bl.id.split('.'):
                walker_list.append(bl)

        firetruck_blueprint = blueprint_library.find('vehicle.carlamotors.firetruck')
        charger_police_blueprint = blueprint_library.find('vehicle.dodge.charger_police')
        charger_police_2020_blueprint = blueprint_library.find('vehicle.dodge.charger_police_2020')
        ambulance_blueprint = blueprint_library.find('vehicle.ford.ambulance')
        crossbike_blueprint = blueprint_library.find('vehicle.bh.crossbike')
        constructioncone = blueprint_library.find('static.prop.constructioncone')
        trafficwarning = blueprint_library.find('static.prop.trafficwarning')
        warningconstruction = blueprint_library.find('static.prop.warningconstruction')
        dirtdebris02 = blueprint_library.find('static.prop.dirtdebris02')

        bp_list = [firetruck_blueprint, charger_police_blueprint, charger_police_2020_blueprint, ambulance_blueprint,
                   crossbike_blueprint]

        static_list = [constructioncone, trafficwarning, warningconstruction, dirtdebris02]


        spawn_walker_list = []
        for _ in range(500):
            spawn_walker_list.append(random.choice(walker_list))

        close_waypoint_list = []
        close_waypoint_list_2 = []

        for wp in self.waypoints:
            if wp.transform.location.distance(ego_vehicle.get_transform().location) < 300:
                close_waypoint_list.append(wp)

            if wp.transform.location.distance(ego_vehicle.get_transform().location) < 100:
                close_waypoint_list_2.append(wp)

        new_spawn_points = []
        for sp in spawn_points:
            if sp.location.distance(ego_vehicle.get_transform().location) < 500:
                new_spawn_points.append(sp)

        world.get_map().get_spawn_points()
        self.vehicle_index = 0

        self.create_vehicle(firetruck_blueprint, new_spawn_points, spawn_point_label=True, amount=50)
        self.create_vehicle(ambulance_blueprint, new_spawn_points, spawn_point_label=True, amount=50)
        self.create_vehicle(charger_police_2020_blueprint, close_waypoint_list)
        self.create_vehicle(crossbike_blueprint, close_waypoint_list)
        #self.create_vehicle(constructioncone, close_waypoint_list, driving_area=True)
        #self.create_vehicle(trafficwarning, close_waypoint_list,driving_area=True)
        #self.create_vehicle(warningconstruction, close_waypoint_list,driving_area=True)
        #self.create_vehicle(dirtdebris02, close_waypoint_list,driving_area=True)

        blueprints = world.get_blueprint_library().filter("vehicle.*")
        for index in range(100):
            bp = random.choice(blueprints)
            random_waypoint = random.choice(close_waypoint_list)
            # Extract the location from the waypoint
            spawn_point = random_waypoint.transform#.location
            #try:
            if (str(random_waypoint.lane_type) == 'Driving') and not self.is_occupied_with_dimensions(random_waypoint):
                self.create_actor(bp, spawn_point)
                self.vehicle_index += 1
                print("vehicle_index created: ",self.vehicle_index, bp)






        self.walker_index = 0
        for bp in spawn_walker_list:
            random_waypoint = random.choice(close_waypoint_list)
            # Extract the location from the waypoint
            spawn_point = random_waypoint.transform#.location

            if not self.is_occupied_with_dimensions(random_waypoint):
                self.create_actor(bp, spawn_point)
                self.walker_index += 1
                print("walker created: ",self.walker_index)

        self.set_settings = True

    def create_vehicle(self, bp, close_waypoint_list, driving_area=True, spawn_point_label=False, check_driving_area=True, amount=50):
        for index in range(amount):
            random_waypoint = random.choice(close_waypoint_list)
            # Extract the location from the waypoint
            if spawn_point_label:
                spawn_point = random_waypoint#.transform#.location
                occupied = False
            else:
                spawn_point = random_waypoint.transform#.location
                occupied = self.is_occupied_with_dimensions(random_waypoint)


            if not occupied:
                self.create_actor(bp, spawn_point)
                self.vehicle_index += 1
                print("vehicle_index created: ",self.vehicle_index, bp)


        asd = 0
    def get_vehicle_number(self):
        return self.vehicle_index

    def get_walker_number(self):
        return self.walker_index

    def set_actor_settings(self):
        if self.set_settings:
            self.set_settings = False
            actor_list = self.world.get_actors()
            vehicle_actors = [actor for actor in actor_list if actor.type_id.startswith('vehicle.')]

            for vec in vehicle_actors:
                if vec != self.ego_vehicle:
                    vec.set_simulate_physics(True)
                    vec.set_autopilot(True)

            self.move_walker(actor_list)


    def move_walker(self, actor_list=None):
        #if actor_list is None:
        actor_list = self.world.get_actors()

        walker_actors = [actor for actor in actor_list if actor.type_id.startswith('walker.')]
        for wl in walker_actors:
            try:
                x_list = list(range(-500, 500))
                x = random.choice(x_list)
                y_list = list(range(-500, 500))
                y = random.choice(y_list)
                z_list = list(range(-500, 500))
                z = random.choice(z_list)
                destination = carla.Location(x=x, y=y, z=z)  # You can specify your own destination
            except:
                destination = carla.Location(x=10, y=0, z=0)  # You can specify your own destination

            # Function to move the walker
            def move_to_destination(walker, destination):
                speed_list = list(range(1, 10))
                speed = random.choice(speed_list)
                walker_control = carla.WalkerControl()
                walker_control.speed = float(speed)
                direction = destination - walker.get_location()
                direction = direction / direction.length()  # Normalize the direction vector
                walker_control.direction = direction
                walker.apply_control(walker_control)

            move_to_destination(wl, destination)

    def create_actor(self, bp, spawn_point):
        actor = self.world.try_spawn_actor(bp, spawn_point)#npc_vehicle = world.spawn_actor(npc_vehicle_bp, spawn_point)


    def destroy_actors(self, actors_to_destroy):

        for ac in actors_to_destroy:
            if ac is not None:
                ac.destroy()

    def is_occupied_with_dimensions(self, waypoint, threshold=8.0):
        waypoint_loc = waypoint.transform.location
        actors = self.world.get_actors()
        for actor in actors:
            if actor.is_alive:
                bbox = actor.bounding_box
                actor_transform = actor.get_transform()
                actor_location = actor_transform.transform(bbox.location)  # Get the global bbox center
                distance = waypoint_loc.distance(actor_location)

                # Consider the size of the bounding box
                size = max(bbox.extent.x,
                           bbox.extent.y)  # Using the larger dimension for a rough circular approximation
                if distance - size <= threshold:
                    return True
        return False