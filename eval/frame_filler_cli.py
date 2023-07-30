# Copyright 2022 Google LLC

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     https://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
r"""Runs the FILM frame interpolator on missing frames from a directory.
"""

import functools
import math
import os
import re
from typing import List, Sequence

from . import interpolator as interpolator_lib
from . import util
from absl import app
from absl import flags
from absl import logging
import apache_beam as beam
import mediapy as media
import natsort
import numpy as np
import tensorflow as tf
from tqdm.auto import tqdm

# import libc
from ctypes import cdll, c_int, c_float, create_string_buffer, byref
if os.name == "nt":
    libc = cdll.msvcrt
else:
    # assuming Unix-like environment
    libc = cdll.LoadLibrary("libc.so.6")

# Controls TF_CCP log level.
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'


_INPUT_DIR = flags.DEFINE_string(
    name='input_dir',
    default=None,
    help='Input Directory containing missing frames.',
    required=True)
_FORMAT_STR = flags.DEFINE_string(
    name='format_str',
    default=None,
    help='printf style format for the frame file names. Example: frame_%05d.png',
    required=True)
_MODEL_PATH = flags.DEFINE_string(
    name='model_path',
    default=None,
    help='The path of the TF2 saved model to use.',
    required=True)
_ALIGN = flags.DEFINE_integer(
    name='align',
    default=64,
    help='If >1, pad the input size so it is evenly divisible by this value.')
_BLOCK_HEIGHT = flags.DEFINE_integer(
    name='block_height',
    default=1,
    help='An int >= 1, number of patches along height, '
    'patch_height = height//block_height, should be evenly divisible.')
_BLOCK_WIDTH = flags.DEFINE_integer(
    name='block_width',
    default=1,
    help='An int >= 1, number of patches along width, '
    'patch_width = width//block_width, should be evenly divisible.')

# Add other extensions, if not either.
_INPUT_EXT = ['png', 'jpg', 'jpeg']


def _output_frames(frames: List[np.ndarray], frames_dir: str):
  """Writes PNG-images to a directory.

  If frames_dir doesn't exist, it is created. If frames_dir contains existing
  PNG-files, they are removed before saving the new ones.

  Args:
    frames: List of images to save.
    frames_dir: The output directory to save the images.

  """
  if tf.io.gfile.isdir(frames_dir):
    old_frames = tf.io.gfile.glob(f'{frames_dir}/frame_*.png')
    if old_frames:
      logging.info('Removing existing frames from %s.', frames_dir)
      for old_frame in old_frames:
        tf.io.gfile.remove(old_frame)
  else:
    tf.io.gfile.makedirs(frames_dir)
  for idx, frame in tqdm(
      enumerate(frames), total=len(frames), ncols=100, colour='green'):
    util.write_image(f'{frames_dir}/frame_{idx:03d}.png', frame)
  logging.info('Output frames saved in %s.', frames_dir)


def find_missing_ranges():
    """Returns a list of first and last _existing_ frame numbers of missing frame ranges.
    For example, if a dir has frame numbers: [1, 2, 4, 5], then this will return [[2,4]]
    """
    existing_frames = []
    for f in os.listdir(_INPUT_DIR.value):
      n = c_int()
      print('f:', f)
      if libc.sscanf(bytes(f, 'ascii'), bytes(_FORMAT_STR.value, 'ascii'), byref(n)):
        existing_frames.append(n.value)
    existing_frames.sort()

    missing_ranges = []
    for i in range(0, len(existing_frames)-1):
      a = existing_frames[i]
      b = existing_frames[i+1]
      if a + 1 != b:
        missing_ranges.append([a, b])

    return missing_ranges


class ProcessMissing(beam.DoFn):
  """DoFn for running the interpolator on a single directory at the time."""


  def setup(self):
    self.interpolator = interpolator_lib.Interpolator(
        _MODEL_PATH.value, _ALIGN.value,
        [_BLOCK_HEIGHT.value, _BLOCK_WIDTH.value])

  def proc_existing(self, a, b, t):
    print('TREV proc_existing:', a, b, t)
    # return
    """Open frame a and b, and return an image buffer t between"""
    image_a = util.read_image(_INPUT_DIR.value + '/' + (_FORMAT_STR.value % a))
    image_b = util.read_image(_INPUT_DIR.value + '/' + (_FORMAT_STR.value % b))
    time = np.full(shape=(1,), fill_value=t, dtype=np.float32)
    return self.interpolator.interpolate(image_a[np.newaxis, ...], image_b[np.newaxis, ...], time)

  def fast(self, missing_range: list[int]):
    image_a = util.read_image(_INPUT_DIR.value + '/' + (_FORMAT_STR.value % missing_range[0]))
    image_b = util.read_image(_INPUT_DIR.value + '/' + (_FORMAT_STR.value % missing_range[1]))
    diff = missing_range[1] - missing_range[0]
    for t in range(1, diff):
      time = np.full(shape=(1,), fill_value=(t / diff), dtype=np.float32)
      new_image = self.interpolator.interpolate(image_a[np.newaxis, ...], image_b[np.newaxis, ...], time)[0]
      util.write_image(_INPUT_DIR.value + '/' + (_FORMAT_STR.value % (missing_range[0] + t)), new_image)

  def process(self, missing_range: list[int]):
    print('TREV_TEST: ', missing_range)
    # self.fast(missing_range)

    a = missing_range[0]
    b = missing_range[1]
    if a + 1 == b:
        return
    midp = a + ((b - a) / 2.0)
    if midp == math.floor(midp):
        new_image = self.proc_existing(a, b, 0.5)[0]
        util.write_image(_INPUT_DIR.value + '/' + (_FORMAT_STR.value % midp), new_image)
        self.process([a, midp])
        self.process([midp, b])
    else:
        new_image = self.proc_existing(a, b, (math.floor(midp) - a) / (b - a))[0]
        util.write_image(_INPUT_DIR.value + '/' + (_FORMAT_STR.value % math.floor(midp)), new_image)
        self.process([a, math.floor(midp)])
        self.process([math.floor(midp), b])
        new_image = self.proc_existing(a, b, (math.ceil(midp) - a) / (b - a))[0]
        util.write_image(_INPUT_DIR.value + '/' + (_FORMAT_STR.value % math.ceil(midp)), new_image)
        self.process([a, math.ceil(midp)])
        self.process([math.ceil(midp), b])


def _run_pipeline() -> None:
  pipeline = beam.Pipeline('DirectRunner')
  (pipeline | beam.Create(find_missing_ranges()) | 'Process directories' >> beam.ParDo(ProcessMissing()))

  result = pipeline.run()
  result.wait_until_finish()


def main(argv: Sequence[str]) -> None:
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')
  _run_pipeline()


if __name__ == '__main__':
  app.run(main)
