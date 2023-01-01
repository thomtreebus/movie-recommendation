import os
import pprint
import tempfile

from typing import Dict, Text

import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
import tensorflow_recommenders as tfrs

ratings = tfds.load("movielens/100k-ratings", split="train")
movies = tfds.load("movielens/100k-movies", split="train")

for rating in ratings.take(1).as_numpy_iterator():
	pprint.pprint(rating)

for movie in movies.take(1).as_numpy_iterator():
	pprint.pprint(movie)

	