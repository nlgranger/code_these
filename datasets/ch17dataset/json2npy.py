#!/bin/env python

import os
import argparse
import json
import numpy as np

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('dir', metavar='DIR', type=str)
parser.add_argument('output', metavar='OUTPUT', type=str)
args = parser.parse_args()

poses = []
fnames = sorted(os.listdir(args.dir))
for fname in fnames:
    t = int(fname[:-len("_pose.json")])
    with open(os.path.join(args.dir, fname)) as f:
        j = json.load(f)
        for p in j['people']:
            poses.append([float(t)] + p['body_parts'])

np.save(args.output, np.array(poses))
