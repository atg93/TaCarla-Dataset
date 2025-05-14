from collections import defaultdict, deque
import datetime
import time
import torch
import torch.distributed as dist
import numpy as np
from .deployment_utils import to_numpy
from math import floor
from .panoptic_evaluation import pq_compute_single_image


# TODO, Delete these functions after moving them to tairvision utils.
# If we want to design loss functions and logger inside model, we need move them into the tairvision
def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


class SmoothedValue(object):
    """Track a series of values and provide access to smoothed values over a
    window or the global series average.
    """

    def __init__(self, window_size=20, fmt=None):
        if fmt is None:
            fmt = "{global_avg:.3f}"
        self.deque = deque(maxlen=window_size)
        self.total = 0.0
        self.count = 0
        self.print_count = None
        self.print_total = None
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
        self.print_count = int(t[0])
        self.print_total = t[1]
        # self.count = int(t[0])
        # self.total = t[1]

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
        if self.print_total is None:
            return self.total / self.count
        else:
            return self.print_total / self.print_count

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
            yield i, obj
            iter_time.update(time.time() - end)
            if i % print_freq == 0:
                self.synchronize_between_processes()
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
        print('{} Total time: {}'.format(header, total_time_str))


class ConfusionMatrix(object):
    def __init__(self, num_classes, class_names):
        # Initialize number of classes and class names
        self.num_classes = num_classes
        self.class_names = class_names
        # Initialize matrix for storing IoU values
        self.mat = None
    def update(self, gt, pred):
        # Get number of classes
        n = self.num_classes
        # Initialize matrix for storing IoU values
        if self.mat is None:
            self.mat = torch.zeros((n, n), dtype=torch.int64, device=gt.device)
        # Calculate IoU values
        with torch.no_grad():
            # Mask out values that are not in range of the number of classes.
            # Ignore class is also handled if ignore class > n
            mask = (gt >= 0) & (gt < n)
            # Extract predicted and ground truth bounding box values from mask
            gt_masked = gt[mask]
            pred_masked = pred[mask]
            # Calculate IoU using masked values
            inds = n * gt_masked.to(torch.int64) + pred_masked
            self.mat += torch.bincount(inds, minlength=n ** 2).reshape(n, n)
    def reset(self):
        # Reset confusion matrix to all zeros
        self.mat.zero_()
    def compute(self):
        # Convert matrix to float
        h = self.mat.float()
        # Calculate global accuracy and class-wise accuracy and IoU
        acc_global = torch.diag(h).sum() / h.sum()
        acc = torch.diag(h) / h.sum(1)
        iu = torch.diag(h) / (h.sum(1) + h.sum(0) - torch.diag(h))
        return acc_global, acc, iu

    def mean_iou(self):
        # Convert matrix to float
        h = self.mat.float()
        # Calculate mean IoU
        iu = torch.diag(h) / (h.sum(1) + h.sum(0) - torch.diag(h))
        return iu.mean().item() * 100

    def reduce_from_all_processes(self):
        if not is_dist_avail_and_initialized():
            return
        dist.barrier()
        dist.all_reduce(self.mat)

    def __str__(self):
        acc_global, acc, iu = self.compute()
        return (
            'global correct: {:.1f}\n'
            'average row correct: {}\n'
            'IoU: {}\n'
            'mean IoU: {:.1f}').format(
                acc_global.item() * 100,
                [f'{cls_name}: {acc:.2f}' for cls_name, acc in zip(self.class_names, (acc * 100).tolist())],
                [f'{cls_name}: {iu:.2f}' for cls_name, iu in zip(self.class_names, (iu * 100).tolist())],
                iu.mean().item() * 100)

class PrecisionCalculation(object):
    def __init__(self, class_indices, class_names, iou_match_threshold):
        self.class_indices = class_indices
        class_name_list = []
        for idx, name in enumerate(class_names):
            if idx in self.class_indices:
                class_name_list.append(name)
        self.class_names = class_name_list
        self.iou_match_threshold = iou_match_threshold
        self.tp = None
        self.fp = None
        self.fn = None
        self.f1 = None
        self.mean_f1 = None

    def update(self, prediction_mask, target_mask):
        if self.tp is None:
            self.tp = torch.zeros((len(self.class_indices)), dtype=torch.int64, device=prediction_mask.device)
            self.fp = torch.zeros((len(self.class_indices)), dtype=torch.int64, device=prediction_mask.device)
            self.fn = torch.zeros((len(self.class_indices)), dtype=torch.int64, device=prediction_mask.device)

        for i, class_label in enumerate(self.class_indices):
            batch_size = prediction_mask.shape[0]

            prediction_mask_filtered = prediction_mask.clone()
            target_mask_filtered = target_mask.clone()

            prediction_mask_filtered[prediction_mask != class_label] = 0
            prediction_mask_filtered[prediction_mask == class_label] = 1

            target_mask_filtered[target_mask != class_label] = 0
            target_mask_filtered[target_mask == class_label] = 1

            prediction_mask_filtered = prediction_mask_filtered.view(batch_size, -1)
            target_mask_filtered = target_mask_filtered.view(batch_size, -1)

            numerator = 2 * (prediction_mask_filtered * target_mask_filtered).sum(1)

            predictions_sums = prediction_mask_filtered.sum(-1)
            targets_sums = target_mask_filtered.sum(-1)

            denominator = predictions_sums + targets_sums
            iou = (numerator + 1) / (denominator + 1)

            proposals = predictions_sums > 0
            gt_existance = targets_sums > 0

            tp_boolean = torch.logical_and(iou > self.iou_match_threshold, proposals)
            fp_boolean = torch.logical_and(iou <= self.iou_match_threshold, proposals)

            fn_boolean = torch.logical_and(iou <= self.iou_match_threshold, gt_existance)

            self.tp[i] += torch.sum(tp_boolean).item()
            self.fp[i] += torch.sum(fp_boolean).item()
            self.fn[i] += torch.sum(fn_boolean).item()

    def reset(self):
        self.tp = None
        self.fp = None
        self.fn = None

    def compute_f1_metric(self):
        self.reduce_from_all_processes()
        self.f1 = self.tp / (self.tp + 0.5 * (self.fp + self.fn)) * 100
        self.mean_f1 = self.f1.mean().item()
        return self.mean_f1

    def reduce_from_all_processes(self):
        if not is_dist_avail_and_initialized():
            return
        dist.barrier()
        dist.all_reduce(self.tp)
        dist.all_reduce(self.fp)
        dist.all_reduce(self.fn)

    def __str__(self):
        return_string = "True Positive: {} \n" \
                        "False Positives: {} \n" \
                        "False Negatives: {} \n" \
                        "F1 Metrics: {} \n".format(
            [f'{cls_name}: {tp}' for cls_name, tp in zip(self.class_names, (self.tp).tolist())],
            [f'{cls_name}: {fp}' for cls_name, fp in zip(self.class_names, (self.fp).tolist())],
            [f'{cls_name}: {fp}' for cls_name, fp in zip(self.class_names, (self.fn).tolist())],
            [f'{cls_name}: {fp:.2f}' for cls_name, fp in zip(self.class_names, (self.f1).tolist())],
        )
        mean_string = f"mean f1 metric: {self.mean_f1: 0.2f}"
        return_string = return_string + mean_string
        return return_string


def cat_list(images, fill_value=0):
    max_size = tuple(max(s) for s in zip(*[img.shape for img in images]))
    batch_shape = (len(images),) + max_size
    batched_imgs = images[0].new(*batch_shape).fill_(fill_value)
    for img, pad_img in zip(images, batched_imgs):
        pad_img[..., :img.shape[-2], :img.shape[-1]].copy_(img)
    return batched_imgs


class PanopticEvaluationInTrain:
    def __init__(self, ignore_label, categories):
        self.ignore_label = ignore_label
        self.categories = {el['id']: el for el in categories}
        self.pq_stat = PQStat(categories=self.categories)
        self.metrics = [("All", None), ("Things", True), ("Stuff", False)]
        self.results = None

    def create_segment_info(self, outputs):
        panoptic = to_numpy(outputs['panoptic'][0])
        segment_info_list = outputs["segments_info"]
        for segment in segment_info_list:
            segment["area"] = np.sum(panoptic == segment['id'])
            if segment["area"] == 0:
                # TODO, this error should not be handled like this. What is the reason
                segment_info_list.remove(segment)

        segment_info = {
            'segments_info': segment_info_list,
        }

        return panoptic, segment_info

    def update(self, ground_truths, outputs):
        pan_pred, pred_ann = self.create_segment_info(outputs)
        pan_gt, gt_ann = self.create_segment_info(ground_truths)

        pq_stat_single = pq_compute_single_image(pan_gt, pan_pred, gt_ann, pred_ann, self.categories, self.ignore_label)
        self.pq_stat += pq_stat_single

    def compute(self):
        results = {}
        for name, isthing in self.metrics:
            results[name], per_class_results = self.pq_stat.pq_average(self.categories, isthing=isthing)
            if name == 'All':
                results['per_class'] = per_class_results
        self.results = results

    def reduce_from_all_processes(self):
        self.pq_stat.reduce_from_all_processes()

    def __str__(self):
        return_string = ""
        return_string += "{:10s}| {:>5s}  {:>5s}  {:>5s} {:>5s} \n".format("", "PQ", "SQ", "RQ", "N")
        return_string += "-" * (10 + 7 * 4)
        return_string += '\n'

        for name, _isthing in self.metrics:
            return_string += "{:10s}| {:5.1f}  {:5.1f}  {:5.1f} {:5d} \n".format(
                name,
                100 * self.results[name]['pq'],
                100 * self.results[name]['sq'],
                100 * self.results[name]['rq'],
                self.results[name]['n'])

        return_string += "{:20}| {:>5s}  {:>5s}  {:>5s} \n".format("", "PQ", "SQ", "RQ")
        print("-" * (20 + 7 * 3))
        return_string += '\n'
        for key in self.results['per_class']:
            name = self.categories[key]["name"]
            pq = self.results['per_class'][key]["pq"]
            sq = self.results['per_class'][key]["sq"]
            rq = self.results['per_class'][key]["rq"]
            return_string += "{:20s}| {:5.1f}  {:5.1f}  {:5.1f} \n".format(
                name, 100 * pq, 100 * sq, 100 * rq)

        return return_string


class PQStatCat():
    def __init__(self):
        self.iou = torch.zeros(1, dtype=torch.float64, device="cuda")
        self.tp = torch.zeros(1, dtype=torch.int64, device="cuda")
        self.fp = torch.zeros(1, dtype=torch.int64, device="cuda")
        self.fn = torch.zeros(1, dtype=torch.int64, device="cuda")

    def __iadd__(self, pq_stat_cat):
        self.iou += pq_stat_cat.iou
        self.tp += pq_stat_cat.tp
        self.fp += pq_stat_cat.fp
        self.fn += pq_stat_cat.fn
        return self

    def reduce_from_all_processes(self):
        if not is_dist_avail_and_initialized():
            return
        dist.barrier()
        dist.all_reduce(self.iou)
        dist.all_reduce(self.tp)
        dist.all_reduce(self.fp)
        dist.all_reduce(self.fn)


class PQStat():
    def __init__(self, categories):
        self.categories = categories
        self.pq_per_cat = {}
        for key in self.categories:
            self.pq_per_cat.update({key: PQStatCat()})

    def __getitem__(self, i):
        return self.pq_per_cat[i]

    def __iadd__(self, pq_stat):
        for label, pq_stat_cat in pq_stat.pq_per_cat.items():
            self.pq_per_cat[label] += pq_stat_cat
        return self

    def reduce_from_all_processes(self):
        for label, pq_stat_cat in self.pq_per_cat.items():
            self.pq_per_cat[label].reduce_from_all_processes()

    def pq_average(self, categories, isthing):
        pq, sq, rq, n = 0, 0, 0, 0
        per_class_results = {}
        for label, label_info in categories.items():
            if isthing is not None:
                cat_isthing = label_info['isthing'] == 1
                if isthing != cat_isthing:
                    continue
            iou = self.pq_per_cat[label].iou
            tp = self.pq_per_cat[label].tp
            fp = self.pq_per_cat[label].fp
            fn = self.pq_per_cat[label].fn
            if tp + fp + fn == 0:
                per_class_results[label] = {'pq': 0.0, 'sq': 0.0, 'rq': 0.0}
                continue
            n += 1
            pq_class = iou / (tp + 0.5 * fp + 0.5 * fn)
            sq_class = iou / tp if tp != 0 else 0
            rq_class = tp / (tp + 0.5 * fp + 0.5 * fn)

            if isinstance(pq_class, torch.Tensor):
                pq_class = pq_class.item()
            if isinstance(sq_class, torch.Tensor):
                sq_class = sq_class.item()
            if isinstance(rq_class, torch.Tensor):
                rq_class = rq_class.item()

            per_class_results[label] = {'pq': pq_class, 'sq': sq_class, 'rq': rq_class}
            pq += pq_class
            sq += sq_class
            rq += rq_class

        return {'pq': pq / n, 'sq': sq / n, 'rq': rq / n, 'n': n}, per_class_results
