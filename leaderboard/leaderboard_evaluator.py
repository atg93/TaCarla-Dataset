#!/usr/bin/env python
# Copyright (c) 2018-2019 Intel Corporation.
# authors: German Ros (german.ros@intel.com), Felipe Codevilla (felipe.alcm@gmail.com)
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

"""
CARLA Challenge Evaluator Routes

Provisional code to evaluate Autonomous Agents for the CARLA Autonomous Driving challenge
"""
from __future__ import print_function

import traceback
import argparse
from argparse import RawTextHelpFormatter
from distutils.version import LooseVersion
import importlib
import os
import pkg_resources
import sys
import carla
import signal
#sys.path.append('/home/tg22/remote-pycharm/leaderboard2.0/')

from srunner.scenariomanager.carla_data_provider import *
from srunner.scenariomanager.timer import GameTime
from srunner.scenariomanager.watchdog import Watchdog


from leaderboard.scenarios.scenario_manager import ScenarioManager
from leaderboard.scenarios.route_scenario import RouteScenario
from leaderboard.envs.sensor_interface import SensorConfigurationInvalid
from leaderboard.autoagents.agent_wrapper import AgentError, validate_sensor_configuration
from leaderboard.utils.statistics_manager import StatisticsManager, FAILURE_MESSAGES
from leaderboard.utils.route_indexer import RouteIndexer
from leaderboard.utils.check_existing_data import Check_Existing_Data

#from leaderboard.autoagents.traditional_agents_files.utils.llm_model import Llm_model
from leaderboard.autoagents.traditional_agents_files.utils.save_il_data import Save_Il_Data
from leaderboard.autoagents.traditional_agents_files.utils.run_route_log import Run_Route_Log
from leaderboard.utils.log_traffic_manager import Log_Traffic_Manager

sensors_to_icons = {
    'sensor.camera.rgb':        'carla_camera',
    'sensor.camera.semantic_segmentation':  'carla_camera',
    'sensor.camera.instance_segmentation':  'carla_camera',
    'sensor.camera.depth':  'carla_camera',
    'sensor.lidar.ray_cast':    'carla_lidar',
    'sensor.other.radar':       'carla_radar',
    'sensor.other.gnss':        'carla_gnss',
    'sensor.other.imu':         'carla_imu',
    'sensor.opendrive_map':     'carla_opendrive_map',
    'sensor.speedometer':       'carla_speedometer'
}



class LeaderboardEvaluator(object):
    """
    Main class of the Leaderboard. Everything is handled from here,
    from parsing the given files, to preparing the simulation, to running the route.
    """

    # Tunable parameters
    client_timeout = 10.0  # in seconds
    frame_rate = 20.0      # in Hz

    def __init__(self, args, statistics_manager, current_path):
        """
        Setup CARLA client and world
        Setup ScenarioManager
        """
        os.environ['TOWN_NAME'] = "Town12" #os.environ.get('DEACTIVATE_TRAFFIC', 0)
        os.environ['TEAMCODE_PATH'] = '/home/tg22/remote-pycharm/TaCarla/leaderboard/autoagents/traditional_agents_files/pdm_lite/team_code/'
        os.environ['DATASAVEPATH'] = '/workspace/tg22/'
        os.environ['code_path'] = "/home/tg22/remote-pycharm/TaCarla"
        #os.environ['code_path'] = "/home/tg22/remote-pycharm/data_submit_code_leaderboard2.0"
        #os.environ["CHECK_PASS"] = True
        self.world = None
        self.manager = None
        self.sensors = None
        self.sensors_initialized = False
        self.sensor_icons = []
        self.agent_instance = None
        self.route_scenario = None

        self.statistics_manager = statistics_manager

        # This is the ROS1 bridge server instance. This is not encapsulated inside the ROS1 agent because the same
        # instance is used on all the routes (i.e., the server is not restarted between routes). This is done
        # to avoid reconnection issues between the server and the roslibpy client.
        self._ros1_server = None

        # Setup the simulation
        self.client, self.client_timeout, self.traffic_manager = self._setup_simulation(args)

        dist = pkg_resources.get_distribution("carla")
        if dist.version != 'leaderboard':
            if LooseVersion(dist.version) < LooseVersion('0.9.10'):
                raise ImportError("CARLA version 0.9.10.1 or newer required. CARLA version found: {}".format(dist))

        # Load agent
        module_name = os.path.basename(args.agent).split('.')[0]
        #sys.path.insert(0, os.path.dirname(args.agent))
        #self.module_agent = importlib.import_module(module_name)
        self.module_agent = importlib.import_module('.'.join(args.agent.split('.')[0].split('/')[1:]))#tugrul
        print("self.module_agent is created:",)

        # Create the ScenarioManager
        self.manager = ScenarioManager(args.timeout, self.statistics_manager, current_path, args.debug)

        # Time control for summary purposes
        self._start_time = GameTime.get_time()
        self._end_time = None

        # Prepare the agent timer
        self._agent_watchdog = None
        signal.signal(signal.SIGINT, self._signal_handler)

        self._client_timed_out = False

        self.current_path = current_path
        self.llm_model = None #Llm_model()
        self.save_il_data = None #Save_Il_Data(current_path)
        self.run_route_log = None #Run_Route_Log() #self.run_route_log.current_scenario_name
        self.log_traffic_manager = None #Log_Traffic_Manager(self.run_route_log, self.world) #self.log_traffic_manager._log_traffic_manager_tick

        self.check_existing_data = Check_Existing_Data()

        self.counter = 16770
        self.episode_number = 0
        self.create_new_file()

        self.new_run = True

        self.set_register = True

        asd = 0

    def create_new_file(self):
        """path = "/workspace/tg22/leaderboard_data/detection/"
        self.file_name_pickle = path + 'episode_' + str(self.episode_number) + '_pickle_file'
        self.file_name_image = path + 'episode_' + str(self.episode_number) + '_image_file'
        os.makedirs(self.file_name_pickle, exist_ok=True)
        os.makedirs(self.file_name_image, exist_ok=True)

        self.file_name_image_front = self.file_name_image + '/' + 'front'
        os.makedirs(self.file_name_image_front, exist_ok=True)
        self.file_name_image_back = self.file_name_image + '/' + 'back'
        os.makedirs(self.file_name_image_back, exist_ok=True)
        self.file_name_image_front_right = self.file_name_image + '/' + 'front_right'
        os.makedirs(self.file_name_image_front_right, exist_ok=True)
        self.file_name_image_front_left = self.file_name_image + '/' + 'front_left'
        os.makedirs(self.file_name_image_front_left, exist_ok=True)"""

        self.episode_number += 1

    def _signal_handler(self, signum, frame):
        """
        Terminate scenario ticking when receiving a signal interrupt.
        Either the agent initialization watchdog is triggered, or the runtime one at scenario manager
        """
        if self._agent_watchdog and not self._agent_watchdog.get_status():#tugrul
            raise RuntimeError("Timeout: Agent took longer than {}s to setup".format(self.client_timeout))
        elif self.manager:
            self.manager.signal_handler(signum, frame)

    def __del__(self):
        """
        Cleanup and delete actors, ScenarioManager and CARLA world
        """
        if hasattr(self, 'manager') and self.manager:
            del self.manager
        if hasattr(self, 'world') and self.world:
            del self.world

    def _get_running_status(self):
        """
        returns:
           bool: False if watchdog exception occured, True otherwise
        """
        if self._agent_watchdog:
            return self._agent_watchdog.get_status()
        return False

    def _cleanup(self):
        """
        Remove and destroy all actors
        """
        CarlaDataProvider.cleanup()

        if self._agent_watchdog:
            self._agent_watchdog.stop()

        try:
            if self.agent_instance:
                #self.counter, self.new_run = self.agent_instance.get_counter_value()
                self.create_new_file()
                self.agent_instance.destroy()
                self.agent_instance = None
        except Exception as e:
            print("\n\033[91mFailed to stop the agent:")
            print(f"\n{traceback.format_exc()}\033[0m")

        if self.route_scenario:
            self.route_scenario.remove_all_actors()
            self.route_scenario = None
            if self.statistics_manager:
                self.statistics_manager.remove_scenario()

        if self.manager:
            self._client_timed_out = not self.manager.get_running_status()
            self.manager.cleanup()

        # Make sure no sensors are left streaming
        alive_sensors = self.world.get_actors().filter('*sensor*')
        for sensor in alive_sensors:
            sensor.stop()
            sensor.destroy()

    def _setup_simulation(self, args):
        """
        Prepares the simulation by getting the client, and setting up the world and traffic manager settings
        """
        client = carla.Client(args.host, args.port)
        if args.timeout:
            client_timeout = args.timeout
        client.set_timeout(client_timeout)

        settings = carla.WorldSettings(
            synchronous_mode = True,
            fixed_delta_seconds = 1.0 / self.frame_rate,
            deterministic_ragdolls = True,#spectator_as_ego = False
        )
        client.get_world().apply_settings(settings)#tugrul

        traffic_manager = client.get_trafficmanager(args.traffic_manager_port)
        traffic_manager.set_synchronous_mode(True)
        traffic_manager.set_hybrid_physics_mode(True)

        self.client = client

        return client, client_timeout, traffic_manager

    def _reset_world_settings(self):
        """
        Changes the modified world settings back to asynchronous
        """
        # Has simulation failed?
        if self.world and self.manager and not self._client_timed_out:
            # Reset to asynchronous mode
            self.world.tick()  # TODO: Make sure all scenario actors have been destroyed
            settings = self.world.get_settings()
            settings.synchronous_mode = False
            settings.fixed_delta_seconds = None
            settings.deterministic_ragdolls = False
            settings.spectator_as_ego = True
            self.world.apply_settings(settings)

            # Make the TM back to async
            self.traffic_manager.set_synchronous_mode(False)
            self.traffic_manager.set_hybrid_physics_mode(False)

    def _load_and_wait_for_world(self, args, town):
        """
        Load a new CARLA world without changing the settings and provide data to CarlaDataProvider
        """
        self.world = self.client.load_world(town, reset_settings=False)

        # Large Map settings are always reset, for some reason
        settings = self.world.get_settings()
        settings.tile_stream_distance = 650
        settings.actor_active_distance = 650
        self.world.apply_settings(settings)

        self.world.reset_all_traffic_lights()
        CarlaDataProvider.set_client(self.client)
        CarlaDataProvider.set_traffic_manager_port(args.traffic_manager_port)
        CarlaDataProvider.set_world(self.world)

        # This must be here so that all route repetitions use the same 'unmodified' seed
        self.traffic_manager.set_random_device_seed(args.traffic_manager_seed)

        # Wait for the world to be ready
        self.world.tick()


        map_name = CarlaDataProvider.get_map().name.split("/")[-1]
        if map_name != town:
            raise Exception("The CARLA server uses the wrong map!"
                            " This scenario requires the use of map {}".format(town))

        if self.save_il_data != None:
            self.save_il_data.save_world(self.world)

        if self.log_traffic_manager != None:
            self.log_traffic_manager.save_world(self.world)

    def _register_statistics(self, route_index, entry_status, crash_message="", current_statistics=False):
        """
        Computes and saves the route statistics
        """
        print("\033[1m> Registering the route statistics\033[0m")
        self.statistics_manager.save_entry_status(entry_status)
        if current_statistics:
            score_composed, score_route, score_penalty = self.statistics_manager.current_compute_route_statistics(
                route_index, self.manager.scenario_duration_system, self.manager.scenario_duration_game, crash_message
            )

            return score_composed, score_route, score_penalty

        else:
            self.score_composed, self.score_route, self.score_penalty = self.statistics_manager.compute_route_statistics(
                route_index, self.manager.scenario_duration_system, self.manager.scenario_duration_game, crash_message
            )


    def _load_and_run_scenario(self, args, config):
        """
        Load and run the scenario given by config.

        Depending on what code fails, the simulation will either stop the route and
        continue from the next one, or report a crash and stop.
        """
        crash_message = ""
        entry_status = "Started"

        print("\n\033[1m========= Preparing {} (repetition {}) =========\033[0m".format(config.name, config.repetition_index))

        # Prepare the statistics of the route
        route_name = f"{config.name}_rep{config.repetition_index}"
        self.statistics_manager.create_route_data(route_name, config.index)

        print("\033[1m> Loading the world\033[0m")

        # Load the world and the scenario
        try:#args.routes
            self._load_and_wait_for_world(args, config.town)
            self.route_scenario = RouteScenario(world=self.world, config=config, debug_mode=args.debug, log_traffic_manager=None)
            if self.save_il_data != None:
                self.save_il_data.update_current_scenario_info(config.name, config.town, expert='rule',file_name=self.file_name_without_extension)
            self.statistics_manager.set_scenario(self.route_scenario) ##self.route_scenario.actorblockedtest.test_status

        except Exception:
            # The scenario is wrong -> set the ejecution to crashed and stop
            print("\n\033[91mThe scenario could not be loaded:")
            print(f"\n{traceback.format_exc()}\033[0m")

            entry_status, crash_message = FAILURE_MESSAGES["Simulation"]
            self._register_statistics(config.index, entry_status, crash_message)
            self._cleanup()
            return True

        print("\033[1m> Setting up the agent\033[0m")

        # Set up the user's agent, and the timer to avoid freezing the simulation
        try:
            self._agent_watchdog = Watchdog(args.timeout)
            self._agent_watchdog.start()
            agent_class_name = getattr(self.module_agent, 'get_entry_point')()
            agent_class_obj = getattr(self.module_agent, agent_class_name)

            # Start the ROS1 bridge server only for ROS1 based agents.
            if getattr(agent_class_obj, 'get_ros_version')() == 1 and self._ros1_server is None:
                from leaderboard.autoagents.ros1_agent import ROS1Server
                self._ros1_server = ROS1Server()
                self._ros1_server.start()


            self.agent_instance = agent_class_obj(args.host, args.port, args.debug)
            self.agent_instance.set_client(self.client)#

            self.agent_instance.set_path(self.current_path)

            if self.save_il_data != None:
                self.agent_instance.set_save_func(self.save_il_data.save_control,file_name_without_extension=self.file_name_without_extension)

            if self.run_route_log != None:
                try:
                    self.agent_instance.save_control_func(self.run_route_log.get_control, self.run_route_log.loc_array,self.run_route_log.rot_array,self.run_route_log.speed_array)
                except:
                    self.agent_instance.save_control_func(self.run_route_log.get_control, None, None, None)


            if self.log_traffic_manager != None:
                self.agent_instance.set_log_traffic_tick(self.log_traffic_manager._log_traffic_manager_tick)
            self.agent_instance.set_global_plan(self.route_scenario.gps_route, self.route_scenario.route)#self.world
            self.agent_instance.set_global_plan_wp_list(self.route_scenario.wp_list, self.route_scenario.gps_route, self.route_scenario.route_loc)
            self.agent_instance.setup(args.agent_config,  self.traffic_manager) #tugrul
            #self.agent_instance.set_path(self.file_name_pickle, self.file_name_image) #tugrul self.counter
            is_scenario_change = self.agent_instance.set_scenario_gt(self.route_scenario.check_current_scenario_name)
            if is_scenario_change:
                dummy_output = self._register_statistics(config.index, entry_status, crash_message)
                self.statistics_manager.clear_records()

            # Check and store the sensors
            if not self.sensors:
                self.sensors = self.agent_instance.sensors()
                track = self.agent_instance.track

                validate_sensor_configuration(self.sensors, track, args.track)

                self.sensor_icons = [sensors_to_icons[sensor['type']] for sensor in self.sensors]
                self.statistics_manager.save_sensors(self.sensor_icons)
                self.statistics_manager.write_statistics()

                self.sensors_initialized = True

            self._agent_watchdog.stop()
            self._agent_watchdog = None

        except SensorConfigurationInvalid as e:
            # The sensors are invalid -> set the ejecution to rejected and stop
            print("\n\033[91mThe sensor's configuration used is invalid:")
            print(f"{e}\033[0m\n")

            entry_status, crash_message = FAILURE_MESSAGES["Sensors"]
            self._register_statistics(config.index, entry_status, crash_message)
            self._cleanup()
            return True

        except Exception:
            # The agent setup has failed -> start the next route
            print("\n\033[91mCould not set up the required agent:")
            print(f"\n{traceback.format_exc()}\033[0m")

            entry_status, crash_message = FAILURE_MESSAGES["Agent_init"]
            self._register_statistics(config.index, entry_status, crash_message)
            self._cleanup()
            return False

        print("\033[1m> Running the route\033[0m")
        #print("self._register_statistics:",self._register_statistics)
        #self.manager.set_register_statistics(self._register_statistics, config.index, entry_status, crash_message)
        # Run the scenario
        try:
            # Load scenario and run it
            try:
                if self.set_register:
                    self.manager.set_register_statistics(self._register_statistics, config.index, entry_status,
                                                         crash_message)
                    self.set_register = False
                asd = 0
            except:
                asd = 0

            if args.record:
                self.client.start_recorder("{}/{}_rep{}.log".format(args.record, config.name, config.repetition_index))
            self.manager.load_scenario(self.route_scenario, self.agent_instance, config.index, config.repetition_index, self.file_name_without_extension)
            #self.agent_instance.planning.plant_agent.cfg_routes
            self.manager.run_scenario(self.file_name_without_extension)#_agent_wrapper._sensors_list

        except AgentError:
            # The agent has failed -> stop the route
            print("\n\033[91mStopping the route, the agent has crashed:")
            print(f"\n{traceback.format_exc()}\033[0m")

            entry_status, crash_message = FAILURE_MESSAGES["Agent_runtime"]

        except Exception:
            print("\n\033[91mError during the simulation:")
            print(f"\n{traceback.format_exc()}\033[0m")

            entry_status, crash_message = FAILURE_MESSAGES["Simulation"]

        # Stop the scenario
        try:
            print("\033[1m> Stopping the route\033[0m")
            self.manager.stop_scenario()
            self._register_statistics(config.index, entry_status, crash_message)

            if args.record:
                self.client.stop_recorder()
            print("self.route_scenario.actorblockedtest.test_status:", self.route_scenario.actorblockedtest.save_test_status)
            if self.save_il_data != None:
                self.save_il_data.save_agentblocked(self.route_scenario.actorblockedtest.save_test_status)
            if self.agent_instance.plant_dataset_collection:
                self.agent_instance.plant_data_save_score(config.name, self.file_name_without_extension, self.agent_instance.planning.plant_agent.cfg_routes, self.score_composed, self.score_route, self.score_penalty, self.statistics_manager)
            #save_file_name = '/workspace/tg22/route_dataset_log/'+'lead_12' +'/final_data'
            #self.agent_instance.plant_data_save_score(config.name, self.file_name_without_extension, save_file_name, self.score_composed, self.score_route, self.score_penalty)

            self._cleanup()

        except Exception:
            print("\n\033[91mFailed to stop the scenario, the statistics might be empty:")
            print(f"\n{traceback.format_exc()}\033[0m")

            _, crash_message = FAILURE_MESSAGES["Simulation"]

        # If the simulation crashed, stop the leaderboard, for the rest, move to the next route
        if self.save_il_data != None:
            print("self.score_composed:",self.score_composed)
            self.save_il_data.save_score(self.score_composed, self.score_route, self.score_penalty)
            self.save_il_data.close_h5_file()

        if self.run_route_log != None:
            self.run_route_log.reset()
            #self.score_composed

        return crash_message == "Simulation crashed"

    def run(self, args):
        """
        Run the challenge mode
        """
        route_indexer = RouteIndexer(args.routes, args.repetitions, args.routes_subset)

        if self.run_route_log != None:
            assert self.run_route_log.current_task == args.routes[14:-4]
        #self.check_existing_data
        self.file_name_without_extension, self.route_id_list = route_indexer.get_filename() #'RouteScenario_010'
        if args.resume:
            resume = route_indexer.validate_and_resume(args.checkpoint)
        else:
            resume = False

        if resume:
            self.statistics_manager.add_file_records(args.checkpoint)
        else:
            self.statistics_manager.clear_records()
        self.statistics_manager.save_progress(route_indexer.index, route_indexer.total)
        self.statistics_manager.write_statistics()

        crashed = False
        current_route_index = 0
        while route_indexer.peek() and not crashed:

            # Run the scenario
            config = route_indexer.get_next_config()
            if self.run_route_log != None and config.name != self.run_route_log.current_scenario_name:
                continue

            print("self.route_id_list:",self.route_id_list,"current_route_index:",current_route_index)
            if self.check_existing_data(self.file_name_without_extension+'_'+self.route_id_list[current_route_index]):
                current_route_index += 1
                print("pass ", self.file_name_without_extension+'_'+self.route_id_list[current_route_index])
                continue

            crashed = self._load_and_run_scenario(args, config)
            # Save the progress and write the route statistics
            self.statistics_manager.save_progress(route_indexer.index, route_indexer.total)
            self.statistics_manager.write_statistics()

            current_route_index += 1

        # Shutdown ROS1 bridge server if necessary
        if self._ros1_server is not None:
            self._ros1_server.shutdown()

        # Go back to asynchronous mode
        self._reset_world_settings()

        if not crashed:
            # Save global statistics
            print("\033[1m> Registering the global statistics\033[0m")
            self.statistics_manager.compute_global_statistics()
            self.statistics_manager.validate_and_write_statistics(self.sensors_initialized, crashed)

        return crashed

import datetime
import os

def get_output_file_name():
    # Get the current date and time
    now = datetime.datetime.now()

    # Format the date and time into a string suitable for a filename
    formatted_date_time = now.strftime("%Y-%m-%d_%H-%M-%S")

    # Create a filename
    filename = formatted_date_time

    # Define the path for the file
    file_path = os.path.join("outputs", filename)

    os.mkdir(file_path)

    return file_path

def main():


    description = "CARLA AD Leaderboard Evaluation: evaluate your Agent in CARLA scenarios\n"

    # general parameters
    parser = argparse.ArgumentParser(description=description, formatter_class=RawTextHelpFormatter)
    parser.add_argument('--host', default='localhost',
                        help='IP of the host server (default: localhost)')
    parser.add_argument('--port', default=2000, type=int,
                        help='TCP port to listen to (default: 2000)')
    parser.add_argument('--traffic-manager-port', default=8000, type=int,
                        help='Port to use for the TrafficManager (default: 8000)')
    parser.add_argument('--traffic-manager-seed', default=12, type=int,
                        help='Seed used by the TrafficManager (default: 0)')
    parser.add_argument('--debug', type=int,
                        help='Run with debug output', default=0)
    parser.add_argument('--record', type=str, default='',
                        help='Use CARLA recording feature to create a recording of the scenario')
    parser.add_argument('--timeout', default=300.0, type=float,
                        help='Set the CARLA client timeout value in seconds')

    # simulation setup
    parser.add_argument('--routes', required=True,
                        help='Name of the routes file to be executed.')
    parser.add_argument('--routes-subset', default='', type=str,
                        help='Execute a specific set of routes')
    parser.add_argument('--repetitions', type=int, default=1,
                        help='Number of repetitions per route.')

    # agent-related options
    parser.add_argument("-a", "--agent", type=str,
                        help="Path to Agent's py file to evaluate", required=True)
    parser.add_argument("--agent-config", type=str,
                        help="Path to Agent's configuration file", default="")

    parser.add_argument("--track", type=str, default='SENSORS',
                        help="Participation track: SENSORS, MAP")
    parser.add_argument('--resume', type=bool, default=False,
                        help='Resume execution from last checkpoint?')
    parser.add_argument("--checkpoint", type=str, default='./simulation_results.json',
                        help="Path to checkpoint used for saving statistics and resuming")
    parser.add_argument("--debug-checkpoint", type=str, default='./live_results.txt',
                        help="Path to checkpoint used for saving live results")

    arguments = parser.parse_args()
    current_path = get_output_file_name()
    arguments.checkpoint = './' + current_path + '/simulation_results.json'
    arguments.debug_checkpoint = './' + current_path + '/live_results.txt'
    statistics_manager = StatisticsManager(arguments.checkpoint, arguments.debug_checkpoint)
    leaderboard_evaluator = LeaderboardEvaluator(arguments, statistics_manager, current_path)
    crashed = leaderboard_evaluator.run(arguments)

    del leaderboard_evaluator

    if crashed:
        sys.exit(-1)
    else:
        sys.exit(0)

if __name__ == '__main__':
    main()
