import glob
import motmetrics as mm
from os import path as osp
from collections import OrderedDict
from pathlib import Path

class Evaluator:

    def __init__(self, data_root, out_root, tracker):
        #self.gt_type = '_val_half'
        self.gt_type = ''
        self.gtfiles = glob.glob(osp.join(data_root, '*/gt/gt{}.txt'.format(self.gt_type)))
        self.tsfiles = [f for f in glob.glob(osp.join(osp.join(out_root, tracker, 'track_results'), '*.txt'))
                        if not osp.basename(f).startswith('eval')]
        self.gt = OrderedDict([(Path(f).parts[-3], mm.io.loadtxt(f, fmt='mot15-2D', min_confidence=1))
                          for f in self.gtfiles])
        self.ts = OrderedDict([(osp.splitext(Path(f).parts[-1])[0], mm.io.loadtxt(f, fmt='mot15-2D', min_confidence=-1))
                          for f in self.tsfiles])


    def evaluate(self):

        mh = mm.metrics.create()
        accs, names = self._compare_dataframes(self.gt, self.ts)

        metrics = ['recall', 'precision', 'num_unique_objects', 'mostly_tracked',
                   'partially_tracked', 'mostly_lost', 'num_false_positives', 'num_misses',
                   'num_switches', 'num_fragmentations', 'mota', 'motp', 'num_objects']
        summary = mh.compute_many(accs, names=names, metrics=metrics, generate_overall=True)

        div_dict = {'num_objects': ['num_false_positives', 'num_misses', 'num_switches', 'num_fragmentations'],
                    'num_unique_objects': ['mostly_tracked', 'partially_tracked', 'mostly_lost']}
        for divisor in div_dict:
            for divided in div_dict[divisor]:
                summary[divided] = (summary[divided] / summary[divisor])

        fmt = mh.formatters
        change_fmt_list = ['num_false_positives', 'num_misses', 'num_switches', 'num_fragmentations', 'mostly_tracked',
                           'partially_tracked', 'mostly_lost']
        for k in change_fmt_list:
            fmt[k] = fmt['mota']
        print(mm.io.render_summary(summary, formatters=fmt, namemap=mm.io.motchallenge_metric_names))

        metrics = mm.metrics.motchallenge_metrics + ['num_objects']
        summary = mh.compute_many(accs, names=names, metrics=metrics, generate_overall=True)
        print(mm.io.render_summary(summary, formatters=mh.formatters, namemap=mm.io.motchallenge_metric_names))

    def _compare_dataframes(self, gts, ts):
        accs = []
        names = []
        for k, tsacc in ts.items():
            if k in gts:
                print('Comparing {}...'.format(k))
                accs.append(mm.utils.compare_to_groundtruth(gts[k], tsacc, 'iou', distth=0.5))
                names.append(k)
            else:
                print('No ground truth for {}, skipping.'.format(k))

        return accs, names