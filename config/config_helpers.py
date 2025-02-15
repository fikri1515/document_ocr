import json
import os
# from . import find_file_path

# config helpers class
class Config:
    def __init__(self, file_path):
        self._config = None

        self._file_path = file_path

        self.load_config()

    # @staticmethod
    # def file_path(file_path):
    #     file_dir = os.path.abspath(__file__)
    #     dir = os.path.dirname(file_dir)
    #     return f"{dir}\\{file_path}"

    def load_config(self):
        # load config.json file
        if not os.path.isfile(self._file_path):
            raise FileNotFoundError(f"config file not found: {self._file_path}")
        
        # error handling
        try:
            with open(self._file_path, 'r') as file:
                self._config = json.load(file)
        except json.JSONDecodeError as e:
            raise ValueError(f"error decoding JSON from the config file: {e}")
        
    # json value getter function
    def get(self, key, default=None):
        if self._config is None:
            raise ValueError("config not loaded, ensure 'load_config' is called")
        
        keys = key.split('.')
        value = self._config

        for k in keys:
            value = value.get(k, default)

        return value
    
# config_json = Config('common\\config.json')