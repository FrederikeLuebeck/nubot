import os
import pandas as pd


class Logger:
    def __init__(self, path):
        self.path = path

    def write_to_file(self, key, value):
        pass

    def write_summary(self, dict):
        file = "summary"
        df = pd.DataFrame(dict)
        path = os.path.join(self.path, file)
        df.to_csv(path, sep=";")
