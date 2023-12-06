# -*- coding: utf-8 -*-
"""
Created on Mon Oct 28 09:58:16 2019

@author: zhangh
"""

import pandas as pd
def load_data_from_csv(input_csv):
    df = pd.read_csv(input_csv, header=0)
    csv_headings = list(df.columns.values)
    feature_names = csv_headings[:len(csv_headings) - 1]
    label_name = csv_headings[len(csv_headings) - 1:len(csv_headings)][0]
    df = df._get_numeric_data()
#    numpy_array = df.as_matrix() #deprecated
#    numpy_array = df.values #deprecated
    numpy_array = df.to_numpy()
    number_of_rows, number_of_columns = numpy_array.shape
    instances = numpy_array[:, 0:number_of_columns - 1]
    labels = []
    for label in numpy_array[:, number_of_columns - 1:number_of_columns].tolist():
        labels.append(label[0])
    return feature_names, instances, labels
    