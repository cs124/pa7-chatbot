"""Utility methods to load movie data from data files.

Ported to Python 3 by Matt Mistele (@mmistele) and Sam Redmond (@sredmond).

Intended for PA7 in Stanford's CS124.
"""
import csv
from typing import Tuple, List, Dict

import numpy as np


def load_ratings(src_filename, delimiter: str = '%',
                 header: bool = False) -> Tuple[List, np.ndarray]:
    title_list = load_titles('data/movies.txt')
    user_id_set = set()
    with open(src_filename, 'r') as f:
        content = f.readlines()
        for line in content:
            user_id = int(line.split(delimiter)[0])
            if user_id not in user_id_set:
                user_id_set.add(user_id)
    num_users = len(user_id_set)
    num_movies = len(title_list)
    mat = np.zeros((num_movies, num_users))

    with open(src_filename) as f:
        reader = csv.reader(f, delimiter=delimiter, quoting=csv.QUOTE_MINIMAL)
        if header:
            next(reader)
        for line in reader:
            mat[int(line[1])][int(line[0])] = float(line[2])
    return title_list, mat


def load_titles(src_filename: str, delimiter: str = '%',
                header: bool = False) -> List:
    title_list = []
    with open(src_filename, 'r', encoding='utf-8') as f:
        reader = csv.reader(f, delimiter=delimiter, quoting=csv.QUOTE_MINIMAL)
        if header:
            next(reader)

        for line in reader:
            movieID, title, genres = int(line[0]), line[1], line[2]
            if title[0] == '"' and title[-1] == '"':
                title = title[1:-1]
            title_list.append([title, genres])
    return title_list


def load_sentiment_dictionary(src_filename: str, delimiter: str = ',',
                              header: bool = False) -> Dict:
    with open(src_filename, 'r') as f:
        reader = csv.reader(f, delimiter=delimiter, quoting=csv.QUOTE_MINIMAL)
        if header:
            next(reader)
        return dict(reader)
