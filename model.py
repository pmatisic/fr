import os
import cv2
import math
import json
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

warnings.simplefilter(action='ignore', category=FutureWarning)

class ImageProcessor:
    def __init__(self, wiki_path: str, imdb_path: str, seed: int = 10):
        self.wiki_path = Path(wiki_path)
        self.imdb_path = Path(imdb_path)
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.all_filenames = self._collect_all_filenames()
        self.invalid_images = self.load_invalid_images()
        np.random.seed(seed)
        np.random.shuffle(self.all_filenames)

    def _collect_all_filenames(self) -> list:
        wiki_filenames = [str(f.relative_to(self.wiki_path)) for f in self.wiki_path.rglob('*.jpg')]
        imdb_filenames = [str(f.relative_to(self.imdb_path)) for f in self.imdb_path.rglob('*.jpg')]
        return wiki_filenames + imdb_filenames

    def is_valid_image(self, image_relative_path: str) -> bool:
        if image_relative_path in self.invalid_images:
            return False

        if (self.wiki_path / image_relative_path).exists():
            full_path = self.wiki_path / image_relative_path
        elif (self.imdb_path / image_relative_path).exists():
            full_path = self.imdb_path / image_relative_path
        else:
            print(f"Image {image_relative_path} doesn't exist in both directories!")
            self.invalid_images.append(image_relative_path)
            return False

        try:
            with Image.open(full_path) as img:
                img.verify()
            if not self.detect_face(full_path):
                print(f"No face detected in {full_path}")
                self.invalid_images.append(image_relative_path)
                return False
            return True
        except Exception as e:
            print(f"Error with image {full_path}: {e}")
            self.invalid_images.append(image_relative_path)
            return False

    def detect_face(self, image_path: Path) -> bool:
        img = cv2.imread(str(image_path))
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.01, 4)
        return len(faces) > 0

    def save_invalid_images(self, invalid_images, json_path="temp/invalid_images.json"):
        with open(json_path, 'w') as f:
            json.dump(invalid_images, f)

    def load_invalid_images(self, json_path="temp/invalid_images.json"):
        if os.path.exists(json_path):
            with open(json_path, 'r') as f:
                return json.load(f)
        return []

    def get_valid_images(self):
        valid_images = [img for img in self.all_filenames if self.is_valid_image(img)]
        invalid_images_count = len(self.all_filenames) - len(valid_images)
        self.invalid_images.extend([img for img in self.all_filenames if img not in valid_images])
        self.save_invalid_images(self.invalid_images)
        return valid_images, invalid_images_count

    def display_info(self):
        print(len(self.all_filenames))
        print(self.all_filenames[:3])

class DataExtractor:
    def __init__(self, wiki_mat_path: str, imdb_mat_path: str):
        self.wiki_mat_data = scipy.io.loadmat(wiki_mat_path)['wiki']
        self.imdb_mat_data = scipy.io.loadmat(imdb_mat_path)['imdb']
        self.all_data = self._combine_data()

    @staticmethod
    def _matlab_to_year(matlab_datenum: float) -> float:
        return 1970 + (matlab_datenum - 719529) / 365.25

    def _extract_data(self, mat_data: np.ndarray) -> list:
        photo_taken = mat_data[0][0][1][0]
        dob = mat_data[0][0][0][0]
        full_path = mat_data[0][0][2][0]
        gender = mat_data[0][0][3][0]
        
        birth_years = [self._matlab_to_year(date) for date in dob]
        age = [photo - year + 0.5 for photo, year in zip(photo_taken, birth_years)]
        
        data = []
        problematic_entries = []
        high_age_entries = []

        for i in range(len(dob)):
            if not np.isnan(gender[i]):  
                image_path = full_path[i][0]
                person_age = age[i]
                person_gender = int(gender[i])

            if 0 <= person_age < 120:
                data.append((image_path, person_age, person_gender))
            elif person_age >= 120:
                high_age_entries.append((image_path, person_age, self._matlab_to_year(dob[i]), photo_taken[i]))
            else:
                problematic_entries.append((image_path, person_age, self._matlab_to_year(dob[i]), photo_taken[i]))

        self._display_problematic_entries(problematic_entries, "negative age")
        self._display_problematic_entries(high_age_entries, "unusually high age")

        return data

    @staticmethod
    def _display_problematic_entries(entries: list, problem_type: str):
        print(f"\nFound {len(entries)} entries with {problem_type}.")
        for entry in entries:
            print(f"Image path: {entry[0]}, Calculated Age: {entry[1]}, Year of Birth: {entry[2]}, Photo Taken: {entry[3]}")

    def _combine_data(self) -> list:
        wiki_data = self._extract_data(self.wiki_mat_data)
        imdb_data = self._extract_data(self.imdb_mat_data)
        return wiki_data + imdb_data

    def display_info(self):
        print(len(self.all_data))
        print(self.all_data[:3])

class DatasetAnalyzer:
    def __init__(self, data: list, invalid_images_count: int, wiki_path: str, imdb_path: str):
        self.data = data
        self.invalid_images_count = invalid_images_count
        self.wiki_path = Path(wiki_path)
        self.imdb_path = Path(imdb_path)
        self.df = self._prepare_dataframe()
        
    def _prepare_dataframe(self):
        print(f"Removed {self.invalid_images_count} images due to invalidity issues.")
        df = pd.DataFrame(self.data, columns=['Image_Path', 'Age', 'Gender'])
        assert df['Age'].min() >= 0, "There are still negative age values!"
        assert not df['Age'].isin([np.inf, -np.inf, np.nan]).any(), "There are still infinite or NaN age values!"
        gender_dict = {1: "Male", 0: "Female"}
        df['Gender'] = df['Gender'].map(gender_dict)
        df = df.astype({'Age': 'float32', 'Gender': 'category'})
        print(df.dtypes)
        return df

    def show_image(self, index: int):
        image_relative_path = self.df['Image_Path'][index]
        img_path = self._get_image_path(image_relative_path)
        img = Image.open(img_path)
        plt.imshow(img)
        plt.title(f"Age: {self.df['Age'][index]}, Gender: {self.df['Gender'][index]}")
        plt.show()

    def visualize_age_distribution(self):
        sns.histplot(self.df['Age'], kde=True)
        plt.xlabel('Age')
        plt.ylabel('Density')
        plt.title('Distribution of Ages')
        plt.xlim(0, 150)
        plt.show()

    def save_age_data(self, results_path='results/dob.csv'):
        if not os.path.exists('results'):
            os.makedirs('results')
        self.df.to_csv(results_path, index=False)

    def show_first_n_images(self, n=20):
        rows = math.ceil(math.sqrt(n))
        cols = math.ceil(n / rows)
        files = self.df.iloc[0:n]
        plt.figure(figsize=(15, 15))
        for index, (image_relative_path, age, gender) in enumerate(files.values):
            img_path = self._get_image_path(image_relative_path)
            img = Image.open(img_path)
            plt.subplot(rows, cols, index+1)
            plt.imshow(img)
            plt.title(f"Age: {age:.2f} Gender: {gender}")
            plt.axis('off')
        plt.tight_layout()
        plt.show()

    def _get_image_path(self, image_relative_path: str) -> Path:
        if (self.wiki_path / image_relative_path).exists():
            return self.wiki_path / image_relative_path
        else:
            return self.imdb_path / image_relative_path

if __name__ == '__main__':

    processor = ImageProcessor("data/wiki_crop/", "data/imdb_crop/")
    valid_images, invalid_images_count = processor.get_valid_images()
    processor.display_info()
    extractor = DataExtractor("data/wiki_crop/wiki.mat", "data/imdb_crop/imdb.mat")
    extractor.display_info()
    analyzer = DatasetAnalyzer(extractor.all_data, invalid_images_count, "data/wiki_crop/", "data/imdb_crop/")
    analyzer.show_image(1)
    analyzer.visualize_age_distribution()
    analyzer.save_age_data()
    analyzer.show_first_n_images()
