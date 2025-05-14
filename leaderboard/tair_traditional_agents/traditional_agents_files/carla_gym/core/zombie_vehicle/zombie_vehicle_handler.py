import numpy as np
import carla
import logging
from .zombie_vehicle import ZombieVehicle


class ZombieVehicleHandler(object):

    def __init__(self, client, tm_port=8000, spawn_distance_to_ev=10.0):
        self._logger = logging.getLogger(__name__)
        self.zombie_vehicles = {}
        self._client = client
        self._world = client.get_world()
        self._spawn_distance_to_ev = spawn_distance_to_ev
        self._tm_port = tm_port

    def reset(self, num_zombie_vehicles, ev_spawn_locations):
        if type(num_zombie_vehicles) is list:
            n_spawn = np.random.randint(num_zombie_vehicles[0], num_zombie_vehicles[1])
        else:
            n_spawn = num_zombie_vehicles
        filtered_spawn_points = self._filter_spawn_points(ev_spawn_locations)
        np.random.shuffle(filtered_spawn_points)

        self._spawn_vehicles(filtered_spawn_points[0:n_spawn])

    def _filter_spawn_points(self, ev_spawn_locations):
        all_spawn_points = self._world.get_map().get_spawn_points()

        def proximity_to_ev(transform): return any([ev_loc.distance(transform.location) < self._spawn_distance_to_ev
                                                    for ev_loc in ev_spawn_locations])

        filtered_spawn_points = [transform for transform in all_spawn_points if not proximity_to_ev(transform)]

        return filtered_spawn_points

    def create_random_vehicle(self, lane_info_dic, autopilot=False):
        spawn_points = lane_info_dic['random_vec_transform']
        zombie_vehicle_ids = self._spawn_vehicles(spawn_points, autopilot=autopilot)
        return zombie_vehicle_ids

    def _spawn_vehicles(self, spawn_transforms,autopilot=True):
        zombie_vehicle_ids = []
        create_new_zombie_vehicle_ids = []
        blueprints = self._world.get_blueprint_library().filter("vehicle.*")
        SpawnActor = carla.command.SpawnActor
        SetAutopilot = carla.command.SetAutopilot
        FutureActor = carla.command.FutureActor

        batch = []
        for transform in spawn_transforms:
            blueprint = np.random.choice(blueprints)
            if blueprint.has_attribute('color'):
                color = np.random.choice(blueprint.get_attribute('color').recommended_values)
                blueprint.set_attribute('color', color)
            if blueprint.has_attribute('driver_id'):
                driver_id = np.random.choice(blueprint.get_attribute('driver_id').recommended_values)
                blueprint.set_attribute('driver_id', driver_id)
            blueprint.set_attribute('role_name', 'zombie_vehicle')
            if autopilot:
                batch.append(SpawnActor(blueprint, transform).then(SetAutopilot(FutureActor, True, self._tm_port)))
            else:
                batch.append(SpawnActor(blueprint, transform))

        for response in self._client.apply_batch_sync(batch, do_tick=True):
            if not response.error:
                zombie_vehicle_ids.append(response.actor_id)
                actor = self._world.get_actor(response.actor_id)
                create_new_zombie_vehicle_ids.append(actor)

        for zv_id in zombie_vehicle_ids:
            self.zombie_vehicles[zv_id] = ZombieVehicle(zv_id, self._world)

        self._logger.debug(f'Spawned {len(zombie_vehicle_ids)} zombie vehicles. '
                           f'Should spawn {len(spawn_transforms)}')


        return create_new_zombie_vehicle_ids

    def _spawn_vehicles_until_created(self, spawn_transforms,autopilot=True):
        zombie_vehicle_ids = []
        create_new_zombie_vehicle_ids = []
        blueprints = self._world.get_blueprint_library().filter("vehicle.*")
        SpawnActor = carla.command.SpawnActor
        SetAutopilot = carla.command.SetAutopilot
        FutureActor = carla.command.FutureActor

        batch = []
        for transform in spawn_transforms:
            blueprint = np.random.choice(blueprints)
            if blueprint.has_attribute('color'):
                color = np.random.choice(blueprint.get_attribute('color').recommended_values)
                blueprint.set_attribute('color', color)
            if blueprint.has_attribute('driver_id'):
                driver_id = np.random.choice(blueprint.get_attribute('driver_id').recommended_values)
                blueprint.set_attribute('driver_id', driver_id)
            blueprint.set_attribute('role_name', 'zombie_vehicle')
            if autopilot:
                batch.append(SpawnActor(blueprint, transform).then(SetAutopilot(FutureActor, True, self._tm_port)))
            else:
                batch.append(SpawnActor(blueprint, transform))

        while len(create_new_zombie_vehicle_ids) == 0:
            response = self._client.apply_batch_sync(batch, do_tick=True)
            if not response.error:
                zombie_vehicle_ids.append(response.actor_id)
                create_new_zombie_vehicle_ids.append(response.actor_id)

        for zv_id in zombie_vehicle_ids:
            self.zombie_vehicles[zv_id] = ZombieVehicle(zv_id, self._world)

        self._logger.debug(f'Spawned {len(zombie_vehicle_ids)} zombie vehicles. '
                           f'Should spawn {len(spawn_transforms)}')

        return create_new_zombie_vehicle_ids

    def tick(self):
        pass

    def clean(self):
        live_vehicle_list = [vehicle.id for vehicle in self._world.get_actors().filter("*vehicle*")]
        # batch1 = []
        # batch2 = []
        # SetAutopilot = carla.command.SetAutopilot
        # DestroyActor = carla.command.DestroyActor
        # batch1.append(SetAutopilot(zv_id, False))
        # batch1.append(DestroyActor(zv_id))
        # self._client.apply_batch_sync(batch1, do_tick=True)
        # self._client.apply_batch_sync(batch2, do_tick=True)
        for zv_id, zv in self.zombie_vehicles.items():
            if zv_id in live_vehicle_list:
                zv.clean()
        self.zombie_vehicles = {}
