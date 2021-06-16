import numpy as np
import sys
import json
import pandas as pd
import os.path as op
from utilities import files
from mne.io import read_raw_brainvision
from mne.channels import make_standard_montage
from mne import events_from_annotations, write_events

try:
    index = int(sys.argv[1])
except:
    print("incorrect arguments")
    sys.exit()

try:
    ses = int(sys.argv[2])
except:
    print("incorrect arguments")
    sys.exit()

try:
    json_file = sys.argv[3]
    print("USING:", json_file)
except:
    json_file = "settings.json"
    print("USING:", json_file)

# opening a json file
with open(json_file) as pipeline_file:
    parameters = json.load(pipeline_file)


durations = {
    "rest": 4,
    "cue": 3,
    "movement": 4
}

multigrasp_triggers = {
    13: 1,
    14: 255,
    8: 10,
    1: 15,
    2: 15,
    6: 15,
    10: 15,
    11: 21,
    21: 22,
    61: 23
}

reaching_triggers = {
    13: 1,
    14: 255,
    8: 10,
    9: 15,
    1: 15,
    2: 15,
    3: 15,
    4: 15,
    5: 15,
    6: 15,
    11: 31,
    21: 32,
    31: 33,
    41: 34,
    51: 35,
    61: 36
}

twist_triggers = {
    13: 1,
    14: 255,
    8: 10,
    9: 15,
    10: 15,
    91: 41,
    101: 42
}

event_labels = {
    1: "start",
    255: "end",
    10: "rest",
    15: "cue",
    21: "grasp_cup",
    22: "grasp_ball",
    23: "grasp_card",
    31: "reach_forward",
    32: "reach_backward",
    33: "reach_left",
    34: "reach_right",
    35: "reach_up",
    36: "reach_down",
    41: "pronation",
    42: "supination"
}

ds_path = parameters["dataset_path"]
ds_raw = op.join(ds_path, "raw")
ds_analysis = op.join(ds_path, "analysis")
files.make_folder(ds_analysis)

all_hdrs = files.get_files(ds_raw, "", ".vhdr")[2]
all_hdrs.sort()

split_ = [i.split("/")[-1].split("_") for i in all_hdrs]
session_ = [int(i[0][-1]) for i in split_]
subject_ = [int(i[1][-1]) for i in split_]
task_ = [i[2] for i in split_]
mode_ = [i[3].split(".")[0] for i in split_]

mode_ = ["imagery" if i=="MI" else i for i in mode_]
mode_ = ["movement" if i=="realMove" else i for i in mode_]

files_dict = {
    "subject": subject_,
    "session": session_,
    "task": task_,
    "type": mode_,
    "filename": all_hdrs
}

files_df = pd.DataFrame.from_dict(files_dict)

file_ = files_df.iloc[1]

raw = read_raw_brainvision(
    file_.filename,
    eog=["hEOG_L", "hEOG_R", "vEOG_U", "vEOG_D"],
    misc=["EMG_1", "EMG_2", "EMG_3", "EMG_4", "EMG_5", "EMG_6", "EMG_ref"]
)

raw = raw.resample(250)
montage = make_standard_montage("biosemi64")
raw = raw.set_montage(montage)

trig_dict = {
    "multigrasp": multigrasp_triggers,
    "reaching": reaching_triggers,
    "twist": twist_triggers,
}

triggers = trig_dict[file_.task]

events, *_ = events_from_annotations(raw)
to_remove = np.hstack([np.where(events == i)[0] for i in [99999, 10001]])
events = np.delete(events, to_remove, axis=0)
events[:,2] = [triggers[i] for i in events[:,2]]


# to do:
# - write_events
# - save raw
# - run ICA
# - epoch (-3,3)