import pandas as pd
import sys
import numpy as np
import yaml
from sklearn.model_selection import StratifiedKFold

input_file = sys.argv[1]  # input_file is a yaml compliant file

with open( input_file, 'r' ) as input_stream:
    input_data = yaml.load(input_stream, Loader=yaml.Loader)

trainDir = input_data.get("trainDir", "./DataBase/train/")
valDir = input_data.get("valDir", "./DataBase/validate/")
testDir = input_data.get("testDir", "./DataBase/test/")
directories = [trainDir, valDir, testDir]

fileExtension = input_data.get("fileExtension", ".xyz")