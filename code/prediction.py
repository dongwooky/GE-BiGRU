import numpy as np
import dateutil.parser

def put_labels(distance, threshold):
    return (distance > threshold).astype(int)
    
def fill_blank(check_ts, labels, total_ts):
    parsed_check_ts = [dateutil.parser.parse(t) for t in check_ts]
    parsed_total_ts = [dateutil.parser.parse(t) for t in total_ts]
    
    label_dict = dict(zip(parsed_check_ts, labels))
    final_labels = [label_dict.get(t, 0) for t in parsed_total_ts]

    return np.array(final_labels, dtype=np.int8)