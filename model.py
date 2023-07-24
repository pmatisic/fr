import os
import scipy.io
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
import matplotlib.pyplot as plt
from PIL import Image
from pathlib import Path
from tensorflow.keras.preprocessing.image import load_img
from sklearn.model_selection import train_test_split
from tensorflow.keras.initializers import random_uniform, glorot_uniform, constant, identity
from tensorflow.keras.layers import Dropout, Input, Add, Dense, Activation, BatchNormalization, Flatten, Conv2D, MaxPooling2D, GlobalMaxPooling2D
from tensorflow.keras.models import Model, load_model

# Putanje do slika
wiki_path = Path("data/wiki_crop/")
imdb_path = Path("data/imdb_crop/")

# Prikupljanje svih slika iz wiki_crop mape
wiki_filenames = [f.name for f in wiki_path.rglob('*.jpg')]

# Prikupljanje svih slika iz imdb_crop mape
imdb_filenames = [f.name for f in imdb_path.rglob('*.jpg')]

# Spajanje dvije liste u jednu
all_filenames = wiki_filenames + imdb_filenames

# Miješanje slika
np.random.seed(10)
np.random.shuffle(all_filenames)

print(len(all_filenames))
print(all_filenames[:3])

def matlab_to_year(matlab_datenum):
    return 1970 + (matlab_datenum - 719529) / 365.25

def extract_data(mat_data, source="wiki"):
    # Izvlačenje metapodataka
    photo_taken = mat_data[0][0][1][0]
    dob = mat_data[0][0][0][0]
    full_path = mat_data[0][0][2][0]
    gender = mat_data[0][0][3][0]
    
    # Izračunavanje dobi
    birth_years = [matlab_to_year(date) for date in dob]
    age = [photo - year + 0.5 for photo, year in zip(photo_taken, birth_years)]
    
    # Spajanje podataka u jednostavnu listu za svaku sliku
    data = []

    for i in range(len(dob)):
        if not np.isnan(gender[i]):  # Ignoriramo slike gdje je spol nepoznat
            image_path = full_path[i][0]
            person_age = age[i]
            person_gender = int(gender[i])
            data.append((image_path, person_age, person_gender))
    
    return data

# Učitavanje .mat datoteka
wiki_mat_path = "./data/wiki_crop/wiki.mat"
imdb_mat_path = "./data/imdb_crop/imdb.mat"

wiki_mat_data = scipy.io.loadmat(wiki_mat_path)['wiki']
imdb_mat_data = scipy.io.loadmat(imdb_mat_path)['imdb']

wiki_data = extract_data(wiki_mat_data)
imdb_data = extract_data(imdb_mat_data)

# Spajanje podataka
all_data = wiki_data + imdb_data

print(len(all_data))
print(all_data[:3])