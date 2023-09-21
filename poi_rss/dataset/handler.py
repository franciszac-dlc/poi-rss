"""
Submodule of poi_rss.dataset for generating the city dataset files, including
ID mappings, train/test splits, and city-specific contextual data.

Ideally this should replace the second half of `datasetgen.py`.
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
from poi_rss.algorithms.library.constants import geocat_constants,experiment_constants,DATA


TRAIN_SIZE = experiment_constants.TRAIN_SIZE
TEST_SIZE = 1-TRAIN_SIZE

## Categories

dict_alias_title,category_tree,dict_alias_depth = cat_utils.cat_structs(DATA+"/categories.json")
undirected_category_tree = category_tree.to_undirected()

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


class DatasetHandler:
    DATA_DIR = "/mnt/d/School/Thesis/Yelp_data/"
    SPLIT_YEAR = 2017
    earth_radius = 6371000/1000 # km in earth
    city = 'Reno'
    dataset = 'yelp'

    def __init__(self, city: str = 'Reno', dataset: str = 'yelp'):
        self.city = city
        self.dataset = dataset
        self.database = f"poi_rss_{dataset}"

        self.user_id_mapping = None
        self.poi_id_mapping = None
        self.checkins = None

    def get_city_ids(self, recompute: bool = False) -> Tuple[dict, dict]:
        """
        Given the preparations done previously and stored in the MongoDB database,
        generate the city-specific IDs split. This is done for a specific city.

        Returns the User and POI mappings in the following format:
        { <raw ID>: <sequential int ID> }
        """

        if ((not recompute) and (self.user_id_mapping is not None)
                            and (self.poi_id_mapping is not None)):
            return self.user_id_mapping, self.poi_id_mapping

        with MongoDBConnection(self.database) as db:
            checkin_data = list(db.checkin_data.find({'city': self.city}))

        print("checkin_data size: %d"%(len(checkin_data)))
        # columns: _id, city, user_id, poi_id, date
        df_checkin = pd.DataFrame.from_dict(checkin_data)

        # Filter out POIs with < 5 visitors
        df_diff_users_visited = df_checkin[['user_id','poi_id']] \
                                    .drop_duplicates() \
                                    .reset_index(drop=True) \
                                    .groupby('poi_id') \
                                    .count() \
                                    .reset_index() \
                                    .rename(columns={"user_id":"diffusersvisited"})
        df_diff_users_visited = df_diff_users_visited[
            df_diff_users_visited['diffusersvisited'] >= 5
        ]
        del df_diff_users_visited['diffusersvisited']
        print("checkin_data size after POI filter: %d"%(len(df_diff_users_visited)))

        # Filter out users with < 20 check-ins
        df_checkin = pd.merge(df_checkin, df_diff_users_visited, on='poi_id', how='inner')
        df_checkin['usercount'] = df_checkin.groupby(['user_id'])['user_id'].transform('count')
        df_checkin = df_checkin[df_checkin['usercount']>=20]
        del df_checkin['usercount']

        checkin_data = list(df_checkin.to_dict('index').values())
        print("checkin_data size after user filter: %d"%(len(df_checkin)))

        # Generate a main set of user and POI ids for each city
        users_id = list(set(x['user_id'] for x in checkin_data))
        user_num=len(users_id)
        pois_id = list(set(x['poi_id'] for x in checkin_data))
        poi_num=len(pois_id)
        print("user_num:%d, poi_num:%d"%(user_num,poi_num))

        users_id_to_int = {
            user_id: i
            for i, user_id in enumerate(users_id)
        }

        pois_id_to_int = {
            poi_id: i
            for i, poi_id in enumerate(pois_id)
        }

        with MongoDBConnection('poi_rss_yelp') as db:
            user_id_dataset = db.user_id_dataset
            poi_id_dataset = db.poi_id_dataset
            user_id_dataset.delete_one({ 'city': self.city })
            user_id_dataset.insert_one({
                'city': self.city,
                'data': users_id_to_int
            })
            poi_id_dataset.delete_one({ 'city': self.city })
            poi_id_dataset.insert_one({
                'city': self.city,
                'data': pois_id_to_int
            })

        for checkin in checkin_data:
            checkin['user_id'] = users_id_to_int[checkin['user_id']]
            checkin['poi_id'] = pois_id_to_int[checkin['poi_id']]
            checkin['date'] = pd.to_datetime(checkin['date'])

        self.user_id_mapping = users_id_to_int
        self.poi_id_mapping = pois_id_to_int
        self.checkins = checkin_data

        return users_id_to_int, pois_id_to_int

    def get_category_data(self, full: bool = False):
        """
        Method used to replace the files `poi_full/<city>.pkl` and
        `poi/<city>.pkl`.
        """

        if self.poi_id_mapping is None:
            self.get_city_ids()

        with MongoDBConnection(self.database) as db:
            poi_data_cursor = db.poi_data.find(
                {'business_id': {'$in': list(self.poi_id_mapping.keys())}}
            )
            poi_data = {
                x['business_id']: x
                for x in poi_data_cursor
            }

        if not full:
            city_poi_data = {
                poi_ndx: {
                    **poi_data[poi_id],
                    'categories': category_filter(poi_data[poi_id]['categories'])
                }
                for poi_id, poi_ndx in self.poi_id_mapping.items()
            }
            return city_poi_data
        else:
            city_poi_data_full = {
                poi_ndx: {
                    'categories': category_normalization(
                        category_filter(poi_data[poi_id]['categories'])
                    )
                }
                for poi_id, poi_ndx in self.poi_id_mapping.items()
            }
            return city_poi_data_full

    def get_user_data(self):
        """
        Method used to replace the file `user/<city>.pkl`
        """

        if self.user_id_mapping is None:
            self.get_city_ids()

        city_user_data = dict()
        with MongoDBConnection(self.database) as db:
            user_data_cursor = db.user_data.find(
                {'user_id': {'$in': list(self.user_id_mapping.keys())}}
            )
            for user_data_item in tqdm(iter(user_data_cursor)):
                _user_id = user_data_item['user_id']
                city_user_data[self.user_id_mapping[_user_id]] = user_data_item
        
        return city_user_data

    def get_checkin_data(self):
        """
        Method used to replace the file `checkin/<city>.pkl`
        """

        if self.checkins is None:
            self.get_city_ids()

        return self.checkins

    def get_neighbor_data(self, city_poi_data: dict | None = None):
        """
        Method used to replace the file `neighbor/<city>.pkl`
        """

        if city_poi_data is None:
            city_poi_data = self.get_category_data()

        poi_neighbors={}

        # Remember that these IDs are sequential
        pois_id = range(len(city_poi_data.keys()))
        pois_coos = np.array([(city_poi_data[pid]['latitude'],city_poi_data[pid]['longitude']) for pid in pois_id])*np.pi/180
        poi_coos_balltree = sklearn.neighbors.BallTree(pois_coos,metric="haversine")
        poi_neighbors = {
            lid: [
                int(x) for x in
                poi_coos_balltree.query_radius(
                    [pois_coos[lid]],geocat_constants.NEIGHBOR_DISTANCE/self.earth_radius
                )[0]
            ]
            for lid in pois_id
        }

        return poi_neighbors

    def get_friend_data(self):
        """
        Method used to replace the file `user/friend/<city>.pkl`
        """

        if self.user_id_mapping is None:
            self.get_city_ids()

        city_user_friend = dict()
        countusf = 0
        with MongoDBConnection(self.database) as db:
            user_friend_cursor = db.user_friend.find({'user_id': {'$in': list(self.user_id_mapping.keys())}})
            for user_friend_item in tqdm(iter(user_friend_cursor)):
                _user_id = user_friend_item['user_id']
                _friends = user_friend_item['friends']
                city_user_friend[self.user_id_mapping[_user_id]] = [
                    self.user_id_mapping[u] for u in _friends
                    if u in self.user_id_mapping
                ]
                countusf += len(city_user_friend[self.user_id_mapping[_user_id]])

        return city_user_friend

    def create_train_test_split(self):
        """
        Given a dataset of users, POIs, and checkins, create a train-test split
        in the format expected.

        1. For each user, gather the visited POIs by date of visit
        2. Split them by date into training and test sets for each user
        3. Remove POIs in the test set if they already appear in the training set
        4. Combine them into training and test sets
        """

        if ((self.user_id_mapping is None) or (self.poi_id_mapping is None)
                                           or (self.checkins is None)):
            raise ValueError("Mappings and checkins are null")

        tr_checkin_data=[]
        te_checkin_data=[]

        user_checkin_data = {
            v: []
            for v in self.user_id_mapping.values()
        }
        
        for checkin in self.checkins:
            user_checkin_data[checkin['user_id']].append({'poi_id':checkin['poi_id'],'date':checkin['date']})
        
        for i, user_id in enumerate(tqdm(self.user_id_mapping.values())):
            # user_id = users_id_to_int[users_id[i]]
            checkin_list=user_checkin_data[user_id]
            checkin_list=sorted(checkin_list, key = lambda x: x['date']) 
            train_size=math.ceil(len(checkin_list)*TRAIN_SIZE)
            #test_size=math.floor(len(checkin_list)*TEST_SIZE)
            count=1
            te_pois=set()
            tr_pois=set()
            initial_te_size=len(te_checkin_data)
            final_te_size=len(te_checkin_data)
            for checkin in checkin_list:
                if count<=train_size:
                    tr_pois.add(checkin['poi_id'])
                    tr_checkin_data.append({'user_id':user_id,'poi_id':checkin['poi_id'],'date':checkin['date']})
                else:
                    te_pois.add(checkin['poi_id'])
                    te_checkin_data.append({'user_id':user_id,'poi_id':checkin['poi_id'],'date':checkin['date']})
                    final_te_size+=1
                count+=1
            int_pois=te_pois&tr_pois
            rel_index=0
            for j in range(initial_te_size,final_te_size):
                j+=rel_index
                if te_checkin_data[j]['poi_id'] in int_pois:
                    te_checkin_data.pop(j)
                    rel_index-=1
        
        return tr_checkin_data, te_checkin_data