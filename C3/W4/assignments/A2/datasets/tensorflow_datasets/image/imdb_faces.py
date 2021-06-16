# coding=utf-8
# Copyright 2019 The TensorFlow Datasets Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""IMDB Faces dataset."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import collections
import os
import re
import tensorflow as tf
import tensorflow_datasets.public_api as tfds
_DESCRIPTION = """Since the publicly available face image datasets are often of small to medium size, rarely exceeding tens of thousands of images, this is an attempt to put together a diverse dataset in that domain."""
_URL = ("https://data.vision.ee.ethz.ch/cvl/rrothe/imdb-wiki/")
_DATASET_ROOT_DIR = 'imdb_crop'
_ANNOTATION_FILE = 'imdb.mat'
_CITATION = """@article{Rothe-IJCV-2016,
  author = {Rasmus Rothe and Radu Timofte and Luc Van Gool},
  title = {Deep expectation of real and apparent age from a single image without facial landmarks},
  journal = {International Journal of Computer Vision (IJCV)},
  year = {2016},
  month = {July},
}
@InProceedings{Rothe-ICCVW-2015,
  author = {Rasmus Rothe and Radu Timofte and Luc Van Gool},
  title = {DEX: Deep EXpectation of apparent age from a single image},
  booktitle = {IEEE International Conference on Computer Vision Workshops (ICCVW)},
  year = {2015},
  month = {December},
}
"""
# Source URL of the IMDB faces dataset
_TARBALL_URL = "https://data.vision.ee.ethz.ch/cvl/rrothe/imdb-wiki/static/imdb_crop.tar"
class ImdbFaces(tfds.core.GeneratorBasedBuilder):
  """IMDB Faces dataset."""
  VERSION = tfds.core.Version("0.1.0")
  def _info(self):
    return tfds.core.DatasetInfo(
        builder=self,
        description=_DESCRIPTION,
        # Describe the features of the dataset by following this url
        # https://www.tensorflow.org/datasets/api_docs/python/tfds/features
        features=tfds.features.FeaturesDict({
            "image": tfds.features.Image(),
            "gender":  tfds.features.ClassLabel(num_classes=2),
            "dob": tf.int32,
            "photo_taken": tf.int32,
            "face_location": tfds.features.BBoxFeature(),
            "face_score": tf.float32,
            "second_face_score": tf.float32,
            "celeb_id": tf.int32
        }),
        supervised_keys=("image", "gender"),
        urls=[_URL],
        citation=_CITATION)
  def _split_generators(self, dl_manager):
    # Download the dataset and then extract it.
    download_path = dl_manager.download([_TARBALL_URL])
    extracted_path = dl_manager.download_and_extract([_TARBALL_URL])
    # Parsing the mat file which contains the list of train images
    def parse_mat_file(file_name):
      with tf.io.gfile.GFile(file_name, "rb") as f:
        # Add a lazy import for scipy.io and import the loadmat method to 
        # load the annotation file
        dataset = tfds.core.lazy_imports.scipy.io.loadmat(file_name)['imdb']
      return dataset
    # Parsing the mat file by using scipy's loadmat method
    # Pass the path to the annotation file using the downloaded/extracted paths above
    meta = parse_mat_file(os.path.join(extracted_path[0], _DATASET_ROOT_DIR, _ANNOTATION_FILE))
    # Get the names of celebrities from the metadata
    celeb_names = meta[0, 0]['celeb_names'][0]
    # Create tuples out of the distinct set of genders and celeb names
    self.info.features['gender'].names = ('Female', 'Male')
    self.info.features['celeb_id'].names = tuple([x[0] for x in celeb_names])
    return [
        tfds.core.SplitGenerator(
            name=tfds.Split.TRAIN,
            gen_kwargs={
                "image_dir": extracted_path[0],
                "metadata": meta,
            })
    ]
  def _get_bounding_box_values(self, bbox_annotations, img_width, img_height):
    """Function to get normalized bounding box values.
    Args:
      bbox_annotations: list of bbox values in kitti format
      img_width: image width
      img_height: image height
    Returns:
      Normalized bounding box xmin, ymin, xmax, ymax values
    """
    ymin = bbox_annotations[0] / img_height
    xmin = bbox_annotations[1] / img_width
    ymax = bbox_annotations[2] / img_height
    xmax = bbox_annotations[3] / img_width
    return ymin, xmin, ymax, xmax
  def _get_image_shape(self, image_path):
    image = tf.io.read_file(image_path)
    image = tf.image.decode_image(image, channels=3)
    shape = image.shape[:2]
    return shape
  def _generate_examples(self, image_dir, metadata):
    # Add a lazy import for pandas here (pd)
    pd = tfds.core.lazy_imports.pandas
    # Extract the root dictionary from the metadata so that you can query all the keys inside it
    root = metadata[0, 0]
    """Extract image names, dobs, genders,  
               face locations, 
               year when the photos were taken,
               face scores (second face score too),
               celeb ids
    """
    image_names = root["full_path"][0]
    # Do the same for other attributes (dob, genders etc)
    dobs = root['dob'][0]
    genders = root['gender'][0]
    face_locations = root['face_location'][0]
    photo_taken_years = root['photo_taken'][0]
    face_scores = root['face_score'][0]
    second_face_scores = root['second_face_score'][0]
    celeb_id = root['celeb_id'][0]
    # Now create a dataframe out of all the features like you've seen before
    df = pd.DataFrame(
        list(zip(image_names, 
                dobs,
                genders,
                face_locations,
                photo_taken_years,
                face_scores,
                second_face_scores,
                celeb_id)),
        columns=['image_names', 'dobs', 'genders', 'face_locations', 'photo_taken_years',
                'face_scores', 'second_face_scores', 'celeb_id'])
    # Filter dataframe by only having the rows with face_scores > 1.0
    df = df[df['face_scores'] > 1.0]
    # Remove any records that contain Nulls/NaNs by checking for NaN with .isna()
    df = df[~df['genders'].isna()]
    df = df[~df['second_face_scores'].isna()]
    # Cast genders to integers so that mapping can take place
    df.genders = df.genders.astype(int)
    # Iterate over all the rows in the dataframe and map each feature
    for _, row in df.iterrows():
      # Extract filename, gender, dob, photo_taken, 
      # face_score, second_face_score and celeb_id
      filename = os.path.join(image_dir, _DATASET_ROOT_DIR, row['image_names'][0])
      gender = row['genders']
      dob = row['dobs']
      photo_taken = row['photo_taken_years']
      face_score = row['face_scores']
      second_face_score = row['second_face_scores']
      celeb_id = root['celeb_id']
      # Get the image shape
      image_width, image_height = self._get_image_shape(filename)
      # Normalize the bounding boxes by using the face coordinates and the image shape
      bbox = self._get_bounding_box_values(row['face_locations'][0], 
                                           image_width, image_height)
      # Yield a feature dictionary 
      yield filename, {
          "image": filename,
          "gender": gender,
          "dob": dob,
          "photo_taken": photo_taken,
          "face_location": tfds.features.BBox(
                          ymin=min(bbox[0], 1),
                          xmin=min(bbox[0], 1),
                          ymax=min(bbox[0], 1),
                          xmax=min(bbox[0], 1)
          ),
          "face_score": face_score,
          "second_face_score": second_face_score,
          "celeb_id": celeb_id
      }
