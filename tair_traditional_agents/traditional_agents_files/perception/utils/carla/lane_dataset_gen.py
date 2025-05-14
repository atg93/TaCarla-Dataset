import custom_client.simulation_module
from custom_client.simulation_module import ReadSimulationParameters

from lane_data_iterative import LaneDataIterative
import argparse
import yaml


class DatasetGenerator:
    def __init__(self, arg):
        # self.sim_parameters = ReadSimulationParameters(arg.sim_config, 'lane_simulation')
        with open('small_train_cfg.yaml') as fp:
            self.config = yaml.load(fp)

        self.towns = ['Town06',
                    'Town10HD',
                     'Town07',
                     'Town01',
                     'Town02',
                     'Town03',
                     'Town04',
                     'Town05']


    def collect(self):
        for i, town in enumerate(self.towns):
            self.config['sim_parameters']['town'] = town
            with LaneDataIterative(**self.config) as ld:
                ld.game_loop()


if __name__ == '__main__':
    argument_parser = argparse.ArgumentParser(
        description=__doc__)

    argument_parser.add_argument(
        '--sim-config',
        default="simulation_config.json",
        help='config file')

    argument_parser.add_argument(
        '--config',
        default="small_train_cfg.yaml",
        help='config file')

    argument_parser.add_argument(
        '--sim',
        default="lane_simulation",
        help='simulation selection')

    args = argument_parser.parse_args()
    dg = DatasetGenerator(args)
    dg.collect()