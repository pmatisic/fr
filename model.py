import os
import cv2
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


def is_valid_image_util(image_path, face_detector, wiki_path, imdb_path):
    """
    Utility function to check if the image is valid and can be loaded.
    
    Args:
    - image_path (str): Relative path of the image.
    - face_detector (function): A function to detect faces in the image.
    - wiki_path (Path): Path object for the wiki directory.
    - imdb_path (Path): Path object for the imdb directory.

    Returns:
    bool: True if the image is valid, False otherwise.
    """
    # Provjera je li slika valjana i može li se učitati
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
        
        # Provjera je li na slici lice
        if not face_detector(full_path):
            print(f"No face detected in {full_path}")
            return False
        return True
    except Exception as e:
        print(f"Error with image {full_path}: {e}")
        return False


class ImageProcessor:
    def __init__(self, wiki_path: str, imdb_path: str):
        """
        ImageProcessor Constructor.
        
        Args:
        - wiki_path (str): Path to the wiki_crop images directory.
        - imdb_path (str): Path to the imdb_crop images directory.
        """
        self.wiki_path = Path(wiki_path)
        self.imdb_path = Path(imdb_path)
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    def detect_face(self, image_path):
        """
        Detect faces in an image using OpenCV's pre-trained cascade classifier.
        
        Args:
        - image_path (str): Path to the image.

        Returns:
        bool: True if a face is detected, False otherwise.
        """
        img = cv2.imread(str(image_path))
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.1, 4)
        return len(faces) > 0

    def gather_images(self):
        """
        Collect all image filenames from the wiki_crop and imdb_crop directories.
        
        Returns:
        list: List containing filenames of all images.
        """
        # Sve slike iz wiki_crop i imdb_crop mape
        wiki_filenames = [f.name for f in self.wiki_path.rglob('*.jpg')]
        imdb_filenames = [f.name for f in self.imdb_path.rglob('*.jpg')]

        # Spajanje dvije liste u jednu
        all_filenames = wiki_filenames + imdb_filenames

        # Miješanje slika
        np.random.seed(10)
        np.random.shuffle(all_filenames)

        # Ispis broja slika i strukture prve tri slike
        print(len(all_filenames))
        print(all_filenames[:3])

        return all_filenames

    def is_valid_image(self, image_path):
        """
        Check if an image is valid and can be loaded using a utility function.
        
        Args:
        - image_path (str): Path to the image.

        Returns:
        bool: True if the image is valid, False otherwise.
        """
        return is_valid_image_util(image_path, self.detect_face, self.wiki_path, self.imdb_path)


class MetadataProcessor:
    @staticmethod
    def matlab_to_year(matlab_datenum):
        """
        Convert a MATLAB datenum to the corresponding year.
        
        Args:
        - matlab_datenum (float): MATLAB datenum value.

        Returns:
        float: Corresponding year.
        """
        return 1970 + (matlab_datenum - 719529) / 365.25

    @staticmethod
    def extract_data(mat_data):
        """
        Extract and process the metadata from a given MATLAB .mat data structure.
        
        Args:
        - mat_data (dict): MATLAB .mat file data loaded into Python dictionary.

        Returns:
        tuple: Three lists containing valid metadata, problematic metadata with negative ages, 
               and entries with unusually high ages.
        """
        # Izvlačenje metapodataka
        photo_taken = mat_data[0][0][1][0]
        dob = mat_data[0][0][0][0]
        full_path = mat_data[0][0][2][0]
        gender = mat_data[0][0][3][0]
        
        # Izračunavanje dobi
        birth_years = [MetadataProcessor.matlab_to_year(date) for date in dob]
        age = [photo - year + 0.5 for photo, year in zip(photo_taken, birth_years)]
        
        data = [] # Spajanje podataka u jednostavnu listu za svaku sliku
        problematic_entries = []  # Lista za spremanje upisa s negativnom dobi
        high_age_entries = []  # Lista za spremanje upisa s neobično visokom dobi

        for i in range(len(dob)):
            if not np.isnan(gender[i]):
                image_path = full_path[i][0][0]
                person_age = age[i]
                person_gender = int(gender[i])
                
                if 0 <= person_age < 120:  # Pretpostavimo da osoba ne može biti starija od 120 godina
                    data.append({
                        'full_path': image_path,
                        'gender': person_gender,
                        'age': person_age
                    })
                elif person_age >= 120:
                    # Spremanje podataka za neobično velike godine
                    high_age_entries.append((image_path, person_age, MetadataProcessor.matlab_to_year(dob[i]), photo_taken[i]))
                else:
                    # Spremanje problematičnih unosa za analizu
                    problematic_entries.append((image_path, person_age, MetadataProcessor.matlab_to_year(dob[i]), photo_taken[i]))

        print(f"Found {len(problematic_entries)} problematic entries with negative age.")
        for entry in problematic_entries:
            print(f"Image path: {entry[0]}, Calculated Age: {entry[1]}, Year of Birth: {entry[2]}, Photo Taken: {entry[3]}")
            
        print(f"\nFound {len(high_age_entries)} entries with unusually high age.")
        for entry in high_age_entries:
            print(f"Image path: {entry[0]}, Calculated Age: {entry[1]}, Year of Birth: {entry[2]}, Photo Taken: {entry[3]}")
        
        return data, problematic_entries, high_age_entries

    @staticmethod
    def load_metadata(file_path):
        """
        Load and extract metadata from a .mat file.
        
        Args:
        - file_path (str): Path to the .mat file.

        Returns:
        tuple: Extracted metadata, problematic entries, and high age entries.
        """
        mat_data = scipy.io.loadmat(file_path)
        data, problematic, high_age = MetadataProcessor.extract_data(mat_data)
        return data, problematic, high_age


class ImageAnalyzer:
    def __init__(self, data, wiki_path, imdb_path):
        """
        ImageAnalyzer Constructor.
        
        Args:
        - data (list): List of image metadata entries.
        - wiki_path (str): Path to the wiki_crop images directory.
        - imdb_path (str): Path to the imdb_crop images directory.
        """
        self.df = pd.DataFrame(data, columns=['full_path', 'age', 'gender'])
        self.wiki_path = wiki_path
        self.imdb_path = imdb_path
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml') # Koristimo Haar kaskade iz OpenCV
        self.filter_data()

    def filter_data(self):
        """
        Filter out invalid images and problematic metadata.
        Also, convert age to float and gender to category data type.
        """
        original_num_images = len(self.df)
        valid_images = self.df['full_path'].apply(self.is_valid_image)
        self.df = self.df[valid_images]
        removed_images = original_num_images - len(self.df)
        print(f"Removed {removed_images} images due to invalidity issues.")

        # Dodatne provjere
        assert self.df['age'].min() >= 0, "There are still negative age values!"
        assert not self.df['age'].isin([np.inf, -np.inf, np.nan]).any(), "There are still infinite or NaN age values!"

        # Mapiranje i zamjena vrijednosti spola u skladu s mapiranjem
        gender_dict = {1: "Male", 0: "Female"}
        self.df['gender'] = self.df['gender'].map(gender_dict)
        self.df = self.df.astype({'age': 'float32', 'gender': 'category'}) # Pretvorba tipova podataka

        print(self.df.dtypes)
        print(self.df.head())

    def detect_face(self, image_path):
        img = cv2.imread(str(image_path))
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.1, 4)
        return len(faces) > 0

    def is_valid_image(self, image_path):
        return is_valid_image_util(image_path, self.detect_face, self.wiki_path, self.imdb_path)
    
    def show_image(self, index):
        """
        Display an image with age and gender information.
        
        Args:
        - index (int): Index of the image in the dataframe.
        """
        image_relative_path = self.df['full_path'].iloc[index]
        img_path = self.get_image_path(image_relative_path)
        img = Image.open(img_path)
        plt.imshow(img)
        plt.title(f"Age: {self.df['age'].iloc[index]}, Gender: {self.df['gender'].iloc[index]}")
        plt.show()

    def plot_age_distribution(self):
        """Plot a histogram of age distribution in the dataset."""
        sns.histplot(self.df['age'], kde=True)
        plt.xlabel('Age')
        plt.ylabel('Density')
        plt.title('Distribution of Ages')
        plt.xlim(0, 150)
        plt.show()

    def save_age_data(self, filename="results/dob.csv"):
        """
        Save the age metadata to a CSV file.
        
        Args:
        - filename (str): Path and name of the CSV file to save data to.
        """
        if not os.path.exists('results'):
            os.makedirs('results')
        self.df.to_csv(filename, index=False)

    def show_sample_images(self, num_samples=20):
        """
        Display a sample of images from the dataset.
        
        Args:
        - num_samples (int): Number of images to display.
        """
        samples = self.df.iloc[:num_samples]
        plt.figure(figsize=(15, 15))
        for index, (image_relative_path, age, gender) in enumerate(samples.values):
            img_path = self.get_image_path(image_relative_path)
            img = Image.open(img_path)
            plt.subplot(5, 4, index + 1) # 5 redaka po 4 slike
            plt.imshow(img)
            plt.title(f"Age: {age:.2f} Gender: {gender}")
            plt.axis('off')
        plt.tight_layout()
        plt.show()

    def get_image_path(self, image_relative_path):
        """
        Get the absolute path of an image.
        
        Args:
        - image_relative_path (str): Relative path of the image.

        Returns:
        str: Absolute path of the image.
        """
        if (self.wiki_path / image_relative_path).exists():
            return self.wiki_path / image_relative_path
        return self.imdb_path / image_relative_path


if __name__ == '__main__':
    wiki_path = "data/wiki_crop/"
    imdb_path = "data/imdb_crop/"
    wiki_mat_path = "data/wiki_crop/wiki.mat"
    imdb_mat_path = "data/imdb_crop/imdb.mat"
    
    # Inicijalizacija ImageProcessor, zatim dohvaćanje svih slika te filtriranje
    image_processor = ImageProcessor(wiki_path, imdb_path)
    all_images = image_processor.gather_images()
    valid_images = [img for img in all_images if image_processor.is_valid_image(img)]
    
    # Učitavanje metapodataka za wiki
    wiki_metadata, wiki_problematic_metadata, wiki_high_age_metadata = MetadataProcessor.load_metadata(wiki_mat_path)
    
    # Učitavanje metapodataka za imdb
    imdb_metadata, imdb_problematic_metadata, imdb_high_age_metadata = MetadataProcessor.load_metadata(imdb_mat_path)

    # Spajanje podataka
    all_data = wiki_metadata + imdb_metadata
    print(all_data[:3])

    # Problematični podatci
    total_problematic = len(wiki_problematic_metadata) + len(imdb_problematic_metadata)
    total_high_age = len(wiki_high_age_metadata) + len(imdb_high_age_metadata)

    # Prikaz nekih osnovnih informacija
    print(f"Total valid images: {len(valid_images)}")
    print(f"Total data: {len(all_data)}")
    print(f"Total problematic metadata entries: {total_problematic}")
    print(f"Total high age metadata entries: {total_high_age}")

    # Vizualizacija
    analyzer = ImageAnalyzer(all_data, wiki_path, imdb_path)
    analyzer.show_image(1)
    analyzer.plot_age_distribution()
    analyzer.save_age_data()
    analyzer.show_sample_images()