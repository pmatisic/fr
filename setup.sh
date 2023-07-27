#!/bin/bash

# Provjera potrebnih privilegija
if [ "$EUID" -ne 0 ]
then
    read -p "Ova skripta može zahtijevati sudo privilegije. Imate li sudo privilegije? (y/n): " sudo_priv
    if [ "$sudo_priv" != "y" ]; then
        echo "Nemate potrebnih privilegija. Završetak skripte."
        exit 1
    fi
fi

echo "Prije nastavka s ovom skriptom, potrebno je preuzeti i instalirati sljedeće komponente (ukoliko vaše računalo ima NVIDIA grafičku karticu):"
echo "1. CUDA: https://developer.nvidia.com/cuda-downloads"
echo "2. cuDNN: https://developer.nvidia.com/rdp/cudnn-download"
echo "-----------------------------------"

read -p "Jeste li preuzeli i instalirali navedene komponente? (y/n) " choice

if [ "$choice" != "y" ]; then
    echo "Molimo preuzmite i instalirajte komponente s navedenih poveznica prije nastavka s ovom skriptom."
    exit 1
fi

echo "Odlično! Nastavljamo s postavljanjem projekta."

# 1. Ažuriraj putanje
echo "Ažuriranje putanja za CUDA i cuDNN..."
update_path() {
    echo "export PATH=/opt/cuda/bin:$PATH" >> $1
    echo "export LD_LIBRARY_PATH=/opt/cuda/lib64:$LD_LIBRARY_PATH" >> $1
    echo "Napomena: Za ažurirane putanje restartirajte terminal ili izvršite 'source $1'."
}

if [[ $SHELL == *bash* ]]; then
    update_path ~/.bashrc
elif [[ $SHELL == *zsh* ]]; then
    update_path ~/.zshrc
else
    echo "Nepoznata ljuska. Ručno dodajte putanje za CUDA i cuDNN."
fi

# 2. Kreiraj direktorij "data"
echo "Kreiranje direktorija 'data'..."
mkdir -p "data" && cd "data" || { echo "Greška prilikom kreiranja direktorija 'data'. Završetak skripte."; exit 1; }

# 3. Preuzimanje datoteka
download_file() {
    if [ ! -f "$1" ]; then
        wget "$2" || { echo "Greška prilikom preuzimanja $1. Završetak skripte."; exit 1; }
    fi
}

echo "Preuzimanje potrebnih datoteka..."
download_file "imdb_crop.tar" "https://data.vision.ee.ethz.ch/cvl/rrothe/imdb-wiki/static/imdb_crop.tar"
download_file "wiki_crop.tar" "https://data.vision.ee.ethz.ch/cvl/rrothe/imdb-wiki/static/wiki_crop.tar"

# 4. Raspakiravanje datoteka
unpack_file() {
    if [ -f "$1" ]; then
        tar -xvf "$1" || { echo "Greška prilikom raspakiravanja $1. Završetak skripte."; exit 1; }
    fi
}

echo "Raspakiravanje datoteka..."
unpack_file "imdb_crop.tar"
unpack_file "wiki_crop.tar"

# 5. Vrati se u korijenski direktorij
cd ..

# 6. Instalacija python3.8
echo "Instalacija Python 3.8..."
sudo apt-get update
sudo apt-get install -y python3.8 || { echo "Greška prilikom instalacije Python 3.8. Završetak skripte."; exit 1; }

# 7. Instalacija conda
echo "Provjerava se postojanje Conde..."
if ! command -v conda &> /dev/null; then
    echo "Instalacija Conda..."
    wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
    bash Miniconda3-latest-Linux-x86_64.sh || { echo "Greška prilikom instalacije Conde. Završetak skripte."; exit 1; }
else
    echo "Conda je već instalirana."
fi

# 8. Kreiranje conda okruženja
echo "Kreiranje Conda okruženja 'tfgpue' s Python 3.8..."
conda create -y --name tfgpue python=3.8 || { echo "Greška prilikom kreiranja Conda okruženja. Završetak skripte."; exit 1; }

# 9. Instalacija tensorflow-gpu unutar conda okruženja
echo "Instalacija TensorFlow-GPU unutar Conda okruženja 'tfgpue'..."
conda install -y -n tfgpue tensorflow-gpu || { echo "Greška prilikom instalacije TensorFlow-GPU. Završetak skripte."; exit 1; }

# 10. Instrukcije za aktivaciju conda okruženja
echo "Molimo vas da ručno aktivirate conda okruženje pomoću 'conda activate tfgpue' i zatim nastavite s instalacijom paketa iz 'requirements.txt' koristeći 'pip install -r requirements.txt'."

echo "Postavljanje projekta je završeno!"
