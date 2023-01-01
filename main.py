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

ratings = ratings.map(lambda x: {
	"movie_title": x["movie_title"],
	"user_id": x["user_id"],
})

movies = movies.map(lambda x: x["movie_title"])


tf.random.set_seed(42)
shuffled = ratings.shuffle(100_000, seed=42, reshuffle_each_iteration=False)

train = shuffled.take(80_000)
test = shuffled.skip(80_000).take(20_000)

movie_titles = movies.batch(1_000)
user_ids = ratings.batch(1_000_000).map(lambda x: x["user_id"])

unique_movie_titles = np.unique(np.concatenate(list(movie_titles)))
unique_user_ids = np.unique(np.concatenate(list(user_ids)))


