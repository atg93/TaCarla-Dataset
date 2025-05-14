import argparse
from Initialize import load_trainer_model
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')

parser = argparse.ArgumentParser(description='Train Code for Lane Detection')
parser.add_argument('yaml_name', metavar='FILE', help='path to the yaml')
parser.add_argument('-e', '--eval', dest='eval', help='In order to implement evaluation',
                    action='store_true', default=False)
args = parser.parse_args()

trainer = load_trainer_model(args.yaml_name)

if args.eval:
    trainer.eval()
else:
    trainer.train()


