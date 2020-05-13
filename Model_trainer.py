from __future__ import print_function, division
import time
from matplotlib import rcParams
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from six import iteritems

%matplotlib inline

rcParams['figure.figsize'] = (13, 6)

from nilmtk import DataSet, TimeFrame, MeterGroup, HDFDataStore
from nilmtk.disaggregate import CombinatorialOptimisation, FHMM
import nilmtk.utils

train = DataSet('C:\\Users\\dl50129\\Desktop\\nilmtk\\data\\redd.h5')  
test = DataSet('C:\\Users\\dl50129\\Desktop\\nilmtk\\data\\redd.h5')   

building = 1

train.set_window(end="2011-04-30")
test.set_window(start="2011-04-30")

train_elec = train.buildings[1].elec
test_elec = test.buildings[1].elec


top_5_train_elec = train_elec.submeters().select_top_k(k=5)

np.random.seed(42)

params = {}
#classifiers = {'CO':CombinatorialOptimisation(), 'FHMM':FHMM()}
predictions = {}
sample_period = 120

co = CombinatorialOptimisation()
fhmm = FHMM()

## Train models
co.train(top_5_train_elec, sample_period=sample_period)
fhmm.train(top_5_train_elec, sample_period=sample_period)

## Export models
co.export_model(filename='co.h5')
fhmm.export_model(filename='fhmm.h5')

co.import_model(filename='co.h5')
fhmm.import_model(filename='fhmm.h5')



### Predictions
gt, predictions[clf_name] = predict(clf, test_elec, 120, train.metadata['timezone'])

rmse = {}
for clf_name in classifiers.keys():
    rmse[clf_name] = nilmtk.utils.compute_rmse(gt, predictions[clf_name], pretty=True)

rmse = pd.DataFrame(rmse)

print(rmse)