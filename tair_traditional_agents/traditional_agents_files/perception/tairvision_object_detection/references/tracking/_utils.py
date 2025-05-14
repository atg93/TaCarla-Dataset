import torch
import time
import os
import os.path as osp


RANK = int(os.getenv('RANK', -1))
FONT = 'Arial.ttf'

def prepare_directories(demo_path, output_format, tracker, vid_name):
    result_root = osp.join(demo_path, tracker)
    frame_dir = None if output_format == 'text' else osp.join(result_root, 'frame')
    frame_dir = osp.join(frame_dir, vid_name.split('/')[-1].split('.')[0])
    seq = vid_name.split('/')[-1].split('.')[0] + '.txt'
    result_filename = osp.join(result_root, 'track_results', seq)
    mkdir_if_missing(result_root)
    mkdir_if_missing(frame_dir)
    return result_root, frame_dir, result_filename

def time_sync():
    # PyTorch-accurate time
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return time.time()


def mkdir_if_missing(d):
    if not osp.exists(d):
        os.makedirs(d)


##### TODO: Taken from mm, will be deleted
import pandas as pd
def load_motchallenge(fname, gt_flag, **kwargs):
    r"""Load MOT challenge data.

    Params
    ------
    fname : str
        Filename to load data from

    Kwargs
    ------
    sep : str
        Allowed field separators, defaults to '\s+|\t+|,'
    min_confidence : float
        Rows with confidence less than this threshold are removed.
        Defaults to -1. You should set this to 1 when loading
        ground truth MOTChallenge data, so that invalid rectangles in
        the ground truth are not considered during matching.

    Returns
    ------
    df : pandas.DataFrame
        The returned dataframe has the following columns
            'X', 'Y', 'Width', 'Height', 'Confidence', 'ClassId', 'Visibility'
        The dataframe is indexed by ('FrameId', 'Id')
    """

    sep = kwargs.pop('sep', r'\s+|\t+|,')
    min_confidence = kwargs.pop('min_confidence', -1)
    df = pd.read_csv(
        fname,
        sep=',',
        index_col=[0, 1],
        skipinitialspace=True,
        header=None,
        names=['FrameId', 'Id', 'X', 'Y', 'Width', 'Height', 'Confidence', 'ClassId', 'Visibility', 'unused'],
        engine='python'
    )

    # Account for matlab convention.
    df[['X', 'Y']] -= (1, 1)

    if gt_flag == True:
        df = df[df['ClassId'] == 1]

    # Removed trailing column
    del df['unused']

    # Remove all rows without sufficient confidence
    return df[df['Confidence'] >= min_confidence]