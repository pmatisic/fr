import os
import warnings
import scipy.io
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
import matplotlib.pyplot as plt
from PIL import Image
from pathlib import Path
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.initializers import random_uniform, glorot_uniform, constant, identity
from tensorflow.keras.layers import Dropout, Input, Add, Dense, Activation, BatchNormalization, Flatten, Conv2D, MaxPooling2D, GlobalMaxPooling2D

# Uklanjanje upozorenja
warnings.simplefilter(action='ignore', category=FutureWarning)

# Putanje do slika
wiki_path = Path("data/wiki_crop/")
imdb_path = Path("data/imdb_crop/")

# Prikupljanje svih slika iz wiki_crop i imdb_crop mape
wiki_filenames = [f.name for f in wiki_path.rglob('*.jpg')]
imdb_filenames = [f.name for f in imdb_path.rglob('*.jpg')]

# Spajanje dvije liste u jednu
all_filenames = wiki_filenames + imdb_filenames

# Miješanje slika
np.random.seed(10)
np.random.shuffle(all_filenames)

print(len(all_filenames))
print(all_filenames[:3])

def is_valid_image(image_path):
    # Provjerava je li slika valjana i može li se učitati
    if (wiki_path / image_path).exists():
        full_path = wiki_path / image_path
    elif (imdb_path / image_path).exists():
        full_path = imdb_path / image_path
    else:
        print(f"Image {image_path} doesn't exist in both directories!")
        return False

    try:
        with Image.open(full_path) as img:
            img.verify()
        return True
    except Exception as e:
        print(f"Error with image {full_path}: {e}")
        return False

def matlab_to_year(matlab_datenum):
    return 1970 + (matlab_datenum - 719529) / 365.25

def extract_data(mat_data):
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
    problematic_entries = []  # Lista za spremanje upisa s negativnom dobi
    high_age_entries = []  # Lista za spremanje upisa s neobično visokom dobi

    for i in range(len(dob)):
        if not np.isnan(gender[i]):  
            image_path = full_path[i][0]
            person_age = age[i]
            person_gender = int(gender[i])
            
        if person_age >= 0 and person_age < 120:  # Pretpostavljamo da osoba ne može biti starija od 120 godina
            data.append((image_path, person_age, person_gender))
        elif person_age >= 120:
            high_age_entries.append((image_path, person_age, matlab_to_year(dob[i]), photo_taken[i]))
        else:
            # Spremite problematične unose za analizu
            problematic_entries.append((image_path, person_age, matlab_to_year(dob[i]), photo_taken[i]))

    print(f"Found {len(problematic_entries)} problematic entries with negative age.")
    for entry in problematic_entries:
        print(f"Image path: {entry[0]}, Calculated Age: {entry[1]}, Year of Birth: {entry[2]}, Photo Taken: {entry[3]}")
        
    print(f"\nFound {len(high_age_entries)} entries with unusually high age.")
    for entry in high_age_entries:
        print(f"Image path: {entry[0]}, Calculated Age: {entry[1]}, Year of Birth: {entry[2]}, Photo Taken: {entry[3]}")
    
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

# Konverzija liste podataka u DataFrame
df = pd.DataFrame(all_data, columns=['Image_Path', 'Age', 'Gender'])

original_num_images = len(df)

# Filtriranje podataka
df = df[df['Image_Path'].apply(is_valid_image)]

removed_images = original_num_images - len(df)
print(f"Removed {removed_images} images due to invalidity issues.")

# Provjera postoji li još uvijek negativna dob
assert df['Age'].min() >= 0, "There are still negative age values!"

# Provjera postoji li još uvijek nekonačna ili nedostajuća dob
assert not df['Age'].isin([np.inf, -np.inf, np.nan]).any(), "There are still infinite or NaN age values!"

print(df.head())

# Mapiranje spolnih vrijednosti
gender_dict = {1: "Male", 0: "Female"}

# Zamjena vrijednosti spola u skladu s mapiranjem
df['Gender'] = df['Gender'].map(gender_dict)

# Pretvaranje tipova podataka u DataFrame-u
df = df.astype({'Age': 'float32', 'Gender': 'category'})
print(df.dtypes)

# Otvaranje i prikazivanje slike
image_relative_path = df['Image_Path'][1]

# Provjera postoji li slika u wiki_path
if (wiki_path / image_relative_path).exists():
    img_path = wiki_path / image_relative_path
# U suprotnom pretpostavimo da je u imdb_path
else:
    img_path = imdb_path / image_relative_path

img = Image.open(img_path)
plt.imshow(img)
plt.title(f"Age: {df['Age'][1]}, Gender: {df['Gender'][1]}")
plt.show()

# Vizualizacija distribucije dobi koristeći Seaborn
sns.histplot(df['Age'], kde=True)
plt.xlabel('Age')
plt.ylabel('Density')
plt.title('Distribution of Ages')
plt.xlim(0, 150)  # Ovdje postavljamo granice na osi x
plt.show()

# Provjera postoji li direktorij "results"
if not os.path.exists('results'):
    os.makedirs('results')

# Spremanje podataka o distribuciji dobi
df.to_csv('results/dob.csv', index=False)

# Prikaz prvih 20 slika iz skupa podataka
files = df.iloc[0:20]

plt.figure(figsize=(15, 15))

for index, (image_relative_path, age, gender) in enumerate(files.values):

    # Provjera postoji li slika u wiki_path
    if (wiki_path / image_relative_path).exists():
        img_path = wiki_path / image_relative_path
    # U suprotnom pretpostavimo da je u imdb_path
    else:
        img_path = imdb_path / image_relative_path

    img = Image.open(img_path)
    plt.subplot(5, 4, index+1)  # 5 redaka po 4 slike
    plt.imshow(img)
    plt.title(f"Age: {age:.2f} Gender: {gender}")
    plt.axis('off')

plt.tight_layout()
plt.show()
