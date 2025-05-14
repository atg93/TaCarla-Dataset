
def get_targets_line(batch, receptive_field=1):

    t = receptive_field - 1
    line_instances = [l[t:t+1] for l in batch['line_instances']]
    line_classes = [l[t:t+1] for l in batch['line_classes']]


    targets_static = {'line_instances': line_instances,
                      'line_classes': line_classes
                      }

    return targets_static