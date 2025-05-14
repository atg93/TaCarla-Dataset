from collections import defaultdict, deque
import datetime
import errno
import os
import time

import torch
import torch.distributed as dist

from tairvision.references.detection.coco_utils import get_coco, get_coco_kp
from tairvision.references.detection.openimages_utils import get_openimages
from tairvision.references.detection.widerface_utils import get_widerface
from tairvision.references.detection.nuscenes_mono_utils import get_nuscenes_mono
from tairvision.references.detection.eda_utils import get_eda
from tairvision.references.detection.voc_utils import get_voc
from tairvision.references.detection.bdd_utils import get_bdd, get_bdd_10k
from tairvision.references.detection.shift_utils import get_shift
from tairvision.references.detection.nuimages_utils import get_nuimages

import threading
import random
import queue

try:
    import carla
except:
    print('carla python package is not available')

from multiprocessing import Process


class CARLA(threading.Thread):
    def __init__(self, q, transform=None, device=None):
        super(CARLA, self).__init__()
        self.q = q
        self.transform = transform
        self.device = device
        # random.seed(0)
        client = carla.Client("127.0.0.1", 2000)
        client.set_timeout(10)
        # client = carla.Client("10.29.40.161", 2000)
        world = client.get_world()
        bp_lib = world.get_blueprint_library()

        # Get the map spawn points
        spawn_points = world.get_map().get_spawn_points()

        # spawn vehicle
        vehicle_bp = bp_lib.find('vehicle.lincoln.mkz_2020')
        vehicle = world.try_spawn_actor(vehicle_bp, random.choice(spawn_points))

        # spawn camera
        camera_bp = bp_lib.find('sensor.camera.rgb')
        camera_bp.set_attribute('image_size_x', str(1333))
        camera_bp.set_attribute('image_size_y', str(800))
        camera_bp.set_attribute('fov', str(90))

        camera_init_trans = carla.Transform(carla.Location(z=3))
        camera = world.spawn_actor(camera_bp, camera_init_trans, attach_to=vehicle)
        vehicle.set_autopilot(True)

        # create npc cars and set autopilot
        for i in range(50):
            vehicle_bp = random.choice(bp_lib.filter('vehicle'))
            npc = world.try_spawn_actor(vehicle_bp, random.choice(spawn_points))
            if npc:
                npc.set_autopilot(True)

        # # Set up the simulator in synchronous mode
        settings = world.get_settings()
        settings.synchronous_mode = True  # Enables synchronous mode
        settings.fixed_delta_seconds = 0.025

        # Set up the TM in synchronous mode
        traffic_manager = client.get_trafficmanager()
        traffic_manager.set_synchronous_mode(True)

        # Set a seed so behaviour can be repeated if necessary
        # traffic_manager.set_random_device_seed(123)

        # weather = carla.WeatherParameters(
        #     cloudiness=80.0,
        #     precipitation=70.0,
        #     sun_altitude_angle=70.0)

        # world.set_weather(carla.WeatherParameters.HardRainSunset)
        world.set_weather(carla.WeatherParameters.ClearSunset)
        # world.set_weather(carla.WeatherParameters.CloudySunset)
        print(world.get_weather())
        world.apply_settings(settings)

        # Create a queue to store and retrieve the sensor data
        self.image_queue = queue.Queue()
        self.camera = camera
        camera.listen(self.image_queue.put)
        time.sleep(1)
        self.world = world


class SmoothedValue(object):
    """Track a series of values and provide access to smoothed values over a
    window or the global series average.
    """

    def __init__(self, window_size=20, fmt=None):
        if fmt is None:
            fmt = "{median:.4f} ({global_avg:.4f})"
        self.deque = deque(maxlen=window_size)
        self.total = 0.0
        self.count = 0
        self.fmt = fmt

    def update(self, value, n=1):
        self.deque.append(value)
        self.count += n
        self.total += value * n

    def synchronize_between_processes(self):
        """
        Warning: does not synchronize the deque!
        """
        if not is_dist_avail_and_initialized():
            return
        t = torch.tensor([self.count, self.total], dtype=torch.float64, device='cuda')
        dist.barrier()
        dist.all_reduce(t)
        t = t.tolist()
        self.count = int(t[0])
        self.total = t[1]

    @property
    def median(self):
        d = torch.tensor(list(self.deque))
        return d.median().item()

    @property
    def avg(self):
        d = torch.tensor(list(self.deque), dtype=torch.float32)
        return d.mean().item()

    @property
    def global_avg(self):
        return self.total / self.count

    @property
    def max(self):
        return max(self.deque)

    @property
    def value(self):
        return self.deque[-1]

    def __str__(self):
        return self.fmt.format(
            median=self.median,
            avg=self.avg,
            global_avg=self.global_avg,
            max=self.max,
            value=self.value)


def all_gather(data):
    """
    Run all_gather on arbitrary picklable data (not necessarily tensors)
    Args:
        data: any picklable object
    Returns:
        list[data]: list of data gathered from each rank
    """
    world_size = get_world_size()
    if world_size == 1:
        return [data]
    data_list = [None] * world_size
    dist.all_gather_object(data_list, data)
    return data_list


def reduce_dict(input_dict, average=True):
    """
    Args:
        input_dict (dict): all the values will be reduced
        average (bool): whether to do average or sum
    Reduce the values in the dictionary from all processes so that all processes
    have the averaged results. Returns a dict with the same fields as
    input_dict, after reduction.
    """
    world_size = get_world_size()
    if world_size < 2:
        return input_dict
    with torch.no_grad():
        names = []
        values = []
        # sort the keys so that they are consistent across processes
        for k in sorted(input_dict.keys()):
            names.append(k)
            values.append(input_dict[k])
        values = torch.stack(values, dim=0)
        dist.all_reduce(values)
        if average:
            values /= world_size
        reduced_dict = {k: v for k, v in zip(names, values)}
    return reduced_dict


class MetricLogger(object):
    def __init__(self, delimiter="\t"):
        self.meters = defaultdict(SmoothedValue)
        self.delimiter = delimiter

    def update(self, **kwargs):
        for k, v in kwargs.items():
            if isinstance(v, torch.Tensor):
                v = v.item()
            assert isinstance(v, (float, int))
            self.meters[k].update(v)

    def __getattr__(self, attr):
        if attr in self.meters:
            return self.meters[attr]
        if attr in self.__dict__:
            return self.__dict__[attr]
        raise AttributeError("'{}' object has no attribute '{}'".format(
            type(self).__name__, attr))

    def __str__(self):
        loss_str = []
        for name, meter in self.meters.items():
            loss_str.append(
                "{}: {}".format(name, str(meter))
            )
        return self.delimiter.join(loss_str)

    def synchronize_between_processes(self):
        for meter in self.meters.values():
            meter.synchronize_between_processes()

    def add_meter(self, name, meter):
        self.meters[name] = meter

    def log_every(self, iterable, print_freq, header=None):
        i = 0
        if not header:
            header = ''
        start_time = time.time()
        end = time.time()
        iter_time = SmoothedValue(fmt='{avg:.4f}')
        data_time = SmoothedValue(fmt='{avg:.4f}')
        space_fmt = ':' + str(len(str(len(iterable)))) + 'd'
        if torch.cuda.is_available():
            log_msg = self.delimiter.join([
                header,
                '[{0' + space_fmt + '}/{1}]',
                'eta: {eta}',
                '{meters}',
                'time: {time}',
                'data: {data}',
                'max mem: {memory:.0f}'
            ])
        else:
            log_msg = self.delimiter.join([
                header,
                '[{0' + space_fmt + '}/{1}]',
                'eta: {eta}',
                '{meters}',
                'time: {time}',
                'data: {data}'
            ])
        MB = 1024.0 * 1024.0
        for obj in iterable:
            data_time.update(time.time() - end)
            yield obj
            iter_time.update(time.time() - end)
            if i % print_freq == 0 or i == len(iterable) - 1:
                eta_seconds = iter_time.global_avg * (len(iterable) - i)
                eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
                if torch.cuda.is_available():
                    print(log_msg.format(
                        i, len(iterable), eta=eta_string,
                        meters=str(self),
                        time=str(iter_time), data=str(data_time),
                        memory=torch.cuda.max_memory_allocated() / MB))
                else:
                    print(log_msg.format(
                        i, len(iterable), eta=eta_string,
                        meters=str(self),
                        time=str(iter_time), data=str(data_time)))
            i += 1
            end = time.time()
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print('{} Total time: {} ({:.4f} s / it)'.format(
            header, total_time_str, total_time / len(iterable)))


def collate_fn(batch):
    return tuple(zip(*batch))


def warmup_lr_scheduler(optimizer, warmup_iters, warmup_factor):

    def f(x):
        if x >= warmup_iters:
            return 1
        alpha = float(x) / warmup_iters
        return warmup_factor * (1 - alpha) + alpha

    return torch.optim.lr_scheduler.LambdaLR(optimizer, f)


def mkdir(path):
    try:
        os.makedirs(path)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise


def setup_for_distributed(is_master):
    """
    This function disables printing when not in master process
    """
    import builtins as __builtin__
    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print


def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


def is_main_process():
    return get_rank() == 0


def save_on_master(*args, **kwargs):
    if is_main_process():
        torch.save(*args, **kwargs)


def init_distributed_mode(args):
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ['WORLD_SIZE'])
        args.gpu = int(os.environ['LOCAL_RANK'])
    elif 'SLURM_PROCID' in os.environ:
        args.rank = int(os.environ['SLURM_PROCID'])
        args.gpu = args.rank % torch.cuda.device_count()
    else:
        print('Not using distributed mode')
        args.distributed = False
        return

    args.distributed = True

    torch.cuda.set_device(args.gpu)
    args.dist_backend = 'nccl'
    print('| distributed init (rank {}): {}'.format(
        args.rank, args.dist_url), flush=True)
    torch.distributed.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                         world_size=args.world_size, rank=args.rank)
    torch.distributed.barrier()
    setup_for_distributed(args.rank == 0)


def initialize_wandb(args):

    import wandb

    yaml_location: str = args.cfg
    yaml_name = yaml_location.split('/')[-2]

    config_dict = vars(args)

    if args.wandb_id:
        experiment_id = args.wandb_id
    else:
        import datetime

        current_time = datetime.datetime.now()
        experiment_id = "{:4d}{:02d}{:02d}{:02d}{:02d}".format(current_time.year, current_time.month,
                                                               current_time.day, current_time.hour,
                                                               current_time.minute)

    wandb_settings_dict = {'project': "tairvision_detection",
                           'entity': 'tair',
                           'resume': 'allow',
                           'name': yaml_name,
                           'id': experiment_id,
                           'config': config_dict}

    if args.distributed:
        if args.rank == 0:
            wandb_object = wandb.init(**wandb_settings_dict)
    else:
        wandb_object = wandb.init(**wandb_settings_dict)

    return wandb_object


def get_dataset(dir_path, name, image_set, transform):
    paths = {
        "coco": (dir_path, get_coco, collate_fn, None),
        "coco_kp": (dir_path, get_coco_kp, collate_fn, None),
        "widerface": (dir_path, get_widerface, collate_fn, None),
        "bdd": (dir_path, get_bdd, collate_fn, None),
        "bdd10k": (dir_path, get_bdd_10k, collate_fn, None),
        "shift": (dir_path, get_shift, collate_fn, None),
        "bdd_with_others": (dir_path, get_bdd, collate_fn, False),
        "openimages": (dir_path, get_openimages, collate_fn, None),
        "nuscenes_mono": (dir_path, get_nuscenes_mono, collate_fn, None),
        "nuimages": (dir_path, get_nuimages, collate_fn, None),
        "eda": (dir_path, get_eda, collate_fn, None),
        "voc": (dir_path, get_voc, collate_fn, None)
    }
    p, ds_fn, collate, eliminate_others = paths[name]

    if name == "nuscenes_mono":
        ds, num_classes, num_keypoints = ds_fn(root=p, image_set=image_set, version='v1.0-trainval', transforms=transform)
        return ds, num_classes, collate, num_keypoints
    if eliminate_others is None:
        ds, num_classes, num_keypoints = ds_fn(p, image_set=image_set, transforms=transform)
    else:
        ds, num_classes, num_keypoints = ds_fn(p, image_set=image_set, transforms=transform,
                                               eliminate_others=eliminate_others)
    return ds, num_classes, collate, num_keypoints


def get_backbone_weights(checkpoint):
    keys_to_pop = [key for key in checkpoint['model'].keys() if key.startswith('head')]
    for key in keys_to_pop:
        checkpoint['model'].pop(key)
    keys_to_rename = list(checkpoint['model'].keys())
    for key in keys_to_rename:
        new_key = key.lstrip('backbone').lstrip('.')
        checkpoint['model'][new_key] = checkpoint['model'].pop(key)
    return checkpoint
