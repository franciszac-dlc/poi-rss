"""
Submodule of poi_rss.dataset for managing connections to the dataset store
(e.g. MongoDB)
"""

import pymongo as mg

class MongoDBConnection:
    def __init__(self, dbname="poi_rss"):
        self._dbname = dbname

    def __enter__(self):
        self._client = mg.MongoClient()
        return self._client[self._dbname]

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._client.close()