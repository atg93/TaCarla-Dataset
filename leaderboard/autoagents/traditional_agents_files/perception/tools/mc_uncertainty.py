from tairvision.models.segmentation.mc import mc_predictions, mc_dropout
import argparse
from Initialize import load_trainer_model

parser = argparse.ArgumentParser(description='Train Code for Lane Detection')
parser.add_argument('yaml_name', metavar='FILE', help='path to the yaml')
parser.add_argument('--forward-passes', default=10, type=int, dest='forward_passes',
                    help='Number of forward passes with mc dropout')

args = parser.parse_args()

trainer = load_trainer_model(args.yaml_name)

mc_predictions(trainer, forward_passes=args.forward_passes, model=trainer.model,
                                n_classes=trainer.number_of_classes)