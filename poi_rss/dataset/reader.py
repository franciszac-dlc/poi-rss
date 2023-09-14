"""
Submodule of poi_rss.dataset for reading the raw dataset files and processing
them into MongoDB.

Ideally this should replace `datasetgen.py`.
"""

import collections
import time
import pickle
import math
from typing import List, Mapping, Any, Tuple

import orjson  # Faster json package
import pandas as pd
import numpy as np
from tqdm import tqdm
import sklearn
import sklearn.neighbors
import pymongo as mg

from poi_rss.dataset.connections import MongoDBConnection
import poi_rss.algorithms.library.areamanager as areamanager
import poi_rss.algorithms.library.cat_utils as cat_utils
import poi_rss.algorithms.library.geo_utils as geo_utils
from poi_rss.algorithms.library.parallel_util import run_parallel
from poi_rss.algorithms.library.constants import geocat_constants,experiment_constants

DB_WRITE_BATCH_SIZE = 1024

## Categories

def category_filter(categories):
    """
    Filters out a level of categories. I've no idea which though.
    """
    tmp_cat_list=list()
    if categories != None:
        for category in categories:
            try:
                if dict_alias_depth[dict_alias_title[category]] <= 2:
                    tmp_cat_list.append(dict_alias_title[category])
            except:
                pass
        tmp_cat_list=cat_utils.get_most_detailed_categories(tmp_cat_list,dict_alias_title,dict_alias_depth)
    return tmp_cat_list


def category_normalization(categories):
    if categories != None:
        return categories
    else:
        return []


## Businesses

def read_yelp_business_data(fname: str) -> Tuple[List[str], Mapping[str, str]]:
    """
    Given a path to a business.json file, read the file into MongoDB with
    appropriate preprocessing.
    """

    start_time = time.time()

    with MongoDBConnection('poi_rss_yelp') as db:
        poi_data = db['poi_data']
        poi_ids = [
            x['business_id']
            for x in poi_data.find(projection=['business_id'])
        ]
        poi_city = {}

        if len(poi_ids) > 0:
            print(f"Collection exists with {len(poi_ids)} records, returning...")
            return poi_ids

        poi_data.create_index(
            [("business_id", 1)],
            unique=True
        )

        _batch = []

        with open(fname, 'r') as fbusiness:
            for i, line in enumerate(tqdm(fbusiness)):  
                obj_json = orjson.loads(line)
                poi_ids.append(obj_json['business_id'])
                poi_city[obj_json['business_id']] = obj_json['city']
                _batch.append({
                    'business_id': obj_json['business_id'],
                    'latitude': obj_json['latitude'],
                    'longitude': obj_json['longitude'],
                    'categories': (None
                                if obj_json['categories'] is None
                                else obj_json['categories'].split(', ')),
                    'city': obj_json['city']
                })

                if len(_batch) >= DB_WRITE_BATCH_SIZE:
                    poi_data.insert_many(_batch)
                    _batch = []

        if len(_batch) > 0:
            poi_data.insert_many(_batch)
            _batch = []

    print(f"Reading business.json took {time.time()-start_time:.3f}s")
    return poi_ids, poi_city


## Cities

def read_yelp_city_data(cities, poi_ids):
    """
    Given a list of POIs and cities, do something.
    """

    areas=dict()
    cities_pid_in_area=dict()

    with MongoDBConnection('poi_rss_yelp') as db:
        poi_data_cursor = db.poi_data.find({'business_id': {'$in': poi_ids}})
        poi_data = {
            x['business_id']: x
            for x in poi_data_cursor
        }

    for city in cities:
        areas[city]=areamanager.delimiter_area(city)

    start_time=time.time()

    for city in cities:
        area=areas[city]
        pid_in_area=collections.defaultdict(bool)

        for poi_id in poi_ids:
            if areamanager.poi_in_area(area, poi_data[poi_id]):

                pid_in_area[poi_id]=True

        cities_pid_in_area[city]=pid_in_area
        print(f"{city} has {len(pid_in_area)} POIs")

    print(f"Generating the PID->area mapping took {time.time()-start_time:.3f}s")
    return areas, cities_pid_in_area


## Users

def read_yelp_user_data(fname: str) -> List[str]:
    """
    Given a path to a user.json file, read the file into MongoDB with
    appropriate preprocessing.
    """

    start_time = time.time()
    exclude_from_user_data = {'friends','elite','name'}

    with MongoDBConnection('poi_rss_yelp') as db:
        user_data = db['user_data']
        user_friend = db['user_friend']

        user_ids = [
            x['user_id']
            for x in user_data.find(projection=['user_id'])
        ]

        if len(user_ids) > 0:
            print(f"Collection exists with {len(user_ids)} records, returning...")
            return user_ids

        user_data.create_index(
            [("user_id", 1)],
            unique=True
        )
        user_friend.create_index(
            [("user_id", 1)],
            unique=True
        )

        _uf_batch = []
        _ud_batch = []

        with open(fname, 'r') as fuser:
            for i, line in enumerate(fuser):  
                obj_json = orjson.loads(line)
                user_ids.append(obj_json['user_id'])
                _uf_batch.append({
                    'user_id': obj_json['user_id'],
                    'friends': obj_json['friends'].split(', ')
                })
                _ud_batch.append({
                    k: v
                    for k, v in obj_json.items()
                    if k not in exclude_from_user_data
                })

                if len(_ud_batch) >= DB_WRITE_BATCH_SIZE:
                    user_friend.insert_many(_uf_batch)
                    user_data.insert_many(_ud_batch)
                    _uf_batch = []
                    _ud_batch = []

        if len(_ud_batch) > 0:
            user_friend.insert_many(_uf_batch)
            user_data.insert_many(_ud_batch)
            _uf_batch = []
            _ud_batch = []

    print(f"Reading user.json took {time.time()-start_time:.3f}s")
    return user_ids


## Reviews

def read_yelp_checkin_data(fname: str,
                           poi_city: Mapping[str, str]) -> None:
    """
    Given a path to a review.json or tip.json file, read the file into MongoDB
    with appropriate preprocessing.
    """

    start_time = time.time()

    with MongoDBConnection('poi_rss_yelp') as db:
        checkin_data = db['checkin_data']

        # if (checkin_count := db.checkin_data.count_documents({})) > 0:
        #     print(f"Collection exists with {checkin_count} records, returning...")
        #     break

        checkin_data.create_index([("city", 1)])

        _batch = []

        with open(fname, 'r') as freview:
            for i, line in enumerate(tqdm(freview)):  
                obj_json = orjson.loads(line)
                _batch.append({
                    'city': poi_city[obj_json['business_id']],
                    'user_id': obj_json['user_id'],
                    'poi_id': obj_json['business_id'],
                    'date': obj_json['date'],
                })

                if len(_batch) >= DB_WRITE_BATCH_SIZE:
                    checkin_data.insert_many(_batch)
                    _batch = []

        if len(_batch) > 0:
            checkin_data.insert_many(_batch)
            _batch = []

    print(f"Reading json file took {time.time()-start_time:.3f}s")


if __name__ == "__main__":
    # We're still testing.
    # with mg.MongoClient() as client:
    #     client.drop_database('poi_rss_yelp')

    DATA_DIR = "/mnt/d/School/Thesis/Yelp_data/"
    SPLIT_YEAR = 2017
    earth_radius = 6371000/1000 # km in earth
    # cities=['lasvegas', 'phoenix', 'charlotte', 'madison']
    cities = ['montreal', 'pittsburgh']

    print("Reading Yelp dataset into MongoDB...")

    print("1. Setting up categories...")
    dict_alias_title,category_tree,dict_alias_depth = \
        cat_utils.cat_structs(DATA_DIR+"categories.json")
    undirected_category_tree = category_tree.to_undirected()

    TRAIN_SIZE = experiment_constants.TRAIN_SIZE
    TEST_SIZE = 1-TRAIN_SIZE

    print("2. Setting up business data...")
    poi_ids, poi_city = read_yelp_business_data(DATA_DIR+"business.json")

    # print("3. Setting up cities data...")
    # areas, cities_pid_in_area = read_yelp_city_data(cities, poi_ids)

    # The process gets killed here because the user dictionaries end up being too
    # large. For reference, user.json is 3GB.
    print("4. Setting up users data...")
    user_ids = read_yelp_user_data(DATA_DIR+"user.json")

    # This doesn't have uniqueness checks as this is only indexed. Comment out to make sure there's no duplication
    print("5. Setting up checkin data...")
    read_yelp_checkin_data(DATA_DIR+"review.json", poi_city)
    read_yelp_checkin_data(DATA_DIR+"tip.json", poi_city)

    print("6. Iterating over cities...")
    '''
    Just so you know, here are the top 10 cities in the dataset by number of POIs
        [{'_id': 'Philadelphia', 'count': 14569},
        {'_id': 'Tucson', 'count': 9250},
        {'_id': 'Tampa', 'count': 9050},
        {'_id': 'Indianapolis', 'count': 7540},
        {'_id': 'Nashville', 'count': 6971},
        {'_id': 'New Orleans', 'count': 6209},
        {'_id': 'Reno', 'count': 5935},
        {'_id': 'Edmonton', 'count': 5054},
        {'_id': 'Saint Louis', 'count': 4827},
        {'_id': 'Santa Barbara', 'count': 3829}]

    Here are the top 10 cities by number of check-ins
        [{'_id': 'Philadelphia', 'count': 1086098},
        {'_id': 'New Orleans', 'count': 720227},
        {'_id': 'Tampa', 'count': 518286},
        {'_id': 'Nashville', 'count': 504074},
        {'_id': 'Tucson', 'count': 462347},
        {'_id': 'Indianapolis', 'count': 416814},
        {'_id': 'Reno', 'count': 394280},
        {'_id': 'Santa Barbara', 'count': 298065},
        {'_id': 'Saint Louis', 'count': 296214},
        {'_id': 'Boise', 'count': 116798}]
    '''