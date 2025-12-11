import numpy as np
import cv2
import gc
import sys

class X40Image(object):
    
    def __init__(self, data):
        self._data_source = data
        self._data = None 
        self._cache = dict()

    def load(self):
        if self._data is None:
            if isinstance(self._data_source, str):
                self._data = cv2.imread(self._data_source, cv2.IMREAD_UNCHANGED)
            elif isinstance(self._data_source, np.ndarray):
                self._data = self._data_source.copy()
                self._data_source = None  # Clear the source to free memory
            else:
                raise ValueError("Unsupported data source type. Must be a file path or numpy array.")
        return self
    
    def release(self):
        self._data = None
        self._data_source = None
        self._cache.clear()
        return self

    @property
    def data(self):
        if self._data is None:
            raise ValueError("Image data not loaded. Call load() before accessing data.")
        return self._data

    @property
    def is_loaded(self):
        return self._data is not None
    
    def get_cache(self, key):
        return self._cache.get(key)
    
    def set_cache(self, key, value):
        self._cache[key] = value
        return self
    
    def match_done(self):
        if 'match' in self._cache:
            del self._cache['match']
        self.release()