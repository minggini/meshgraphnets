# Lint as: python3
# pylint: disable=g-bad-file-header
# Copyright 2020 DeepMind Technologies Limited. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or  implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""Plots a cloth trajectory rollout."""
import os

import pickle

import pathlib

from absl import app
from absl import flags

from matplotlib import animation
import matplotlib.pyplot as plt

import torch

root_dir = pathlib.Path(__file__).parent.resolve()
output_dir = os.path.join(root_dir, 'output', 'flag_simple')
all_subdirs = [os.path.join(output_dir, d) for d in os.listdir(output_dir) if
               os.path.isdir(os.path.join(output_dir, d))]
latest_subdir = max(all_subdirs, key=os.path.getmtime)
rollout_path = os.path.join(latest_subdir, 'rollout', 'rollout.pkl')

FLAGS = flags.FLAGS
# flags.DEFINE_string('rollout_path', '/home/i53/student/ruoheng_ma/meshgraphnets/output/flag_simple/Thu-Nov-11-10-25-26-2021/rollout/rollout.pkl', 'Path to rollout pickle file')
flags.DEFINE_string('rollout_path', rollout_path, 'Path to rollout pickle file')


def main(unused_argv):
    print("Ploting run", FLAGS.rollout_path)
    with open(FLAGS.rollout_path, 'rb') as fp:
        rollout_data = pickle.load(fp)
    fig = plt.figure(figsize=(19.2, 10.8))
    ax = fig.add_subplot(111, projection='3d')
    skip = 10
    num_steps = rollout_data[0]['gt_pos'].shape[0]
    # print(num_steps)
    num_frames = num_steps

    # compute bounds
    bounds = []
    index_temp = 0
    for trajectory in rollout_data:
        index_temp += 1
        # print("bb_min shape", trajectory['gt_pos'].shape)
        bb_min = torch.squeeze(trajectory['gt_pos'], dim=0).cpu().numpy().min(axis=(0, 1))
        bb_max = torch.squeeze(trajectory['gt_pos'], dim=0).cpu().numpy().max(axis=(0, 1))
        bounds.append((bb_min, bb_max))

    def animate(num):
        # step = (num * skip) % num_steps
        traj = 7
        # traj = (num * 3) // num_steps
        step = (num * 1) % num_steps
        ax.cla()
        bound = bounds[traj]

        ax.set_xlim([bound[0][0], bound[1][0]])
        ax.set_ylim([bound[0][1], bound[1][1]])
        ax.set_zlim([bound[0][2], bound[1][2]])

        pos = torch.squeeze(rollout_data[traj]['pred_pos'], dim=0)[step].to('cpu')
        original_pos = torch.squeeze(rollout_data[traj]['gt_pos'], dim=0)[step].to('cpu')
        # print(pos[10])
        faces = torch.squeeze(rollout_data[traj]['faces'], dim=0)[step].to('cpu')
        ax.plot_trisurf(pos[:, 0], pos[:, 1], faces, pos[:, 2], shade=True)
        ax.plot_trisurf(original_pos[:, 0], original_pos[:, 1], faces, original_pos[:, 2], shade=True, alpha=0.3)
        ax.set_title('Trajectory %d Step %d' % (traj, step))
        return fig,

    _ = animation.FuncAnimation(fig, animate, frames=num_frames * 100, interval=100)
    plt.show(block=True)


if __name__ == '__main__':
    app.run(main)
