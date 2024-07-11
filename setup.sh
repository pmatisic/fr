#!/bin/bash

# Provjera potrebnih privilegija
if [ "$EUID" -ne 0 ]; then
    if ! sudo -v; then
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
    if ! grep -q "/opt/cuda/bin" "$1"; then
        echo "export PATH=/opt/cuda/bin:\$PATH" >> "$1"
    fi
    if ! grep -q "/opt/cuda/lib64" "$1"; then
        echo "export LD_LIBRARY_PATH=/opt/cuda/lib64:\$LD_LIBRARY_PATH" >> "$1"
    fi
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
        wget "$2" -O "$1" || { echo "Greška prilikom preuzimanja $1. Završetak skripte."; exit 1; }
    else
        echo "$1 već postoji. Preskakanje preuzimanja."
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

# 6. Instalacija conda
echo "Provjerava se postojanje Conde..."
if ! command -v conda &> /dev/null; then
    echo "Instalacija Conda..."
    wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
    bash Miniconda3-latest-Linux-x86_64.sh -b -p $HOME/miniconda || { echo "Greška prilikom instalacije Conde. Završetak skripte."; exit 1; }
    export PATH="$HOME/miniconda/bin:$PATH"
    source ~/.bashrc || source ~/.zshrc
else
    echo "Conda je već instalirana."
fi

# 7. Dodaj Miniconda u PATH
if ! grep -q "$HOME/miniconda/bin" ~/.bashrc; then
    echo "export PATH=\"$HOME/miniconda/bin:\$PATH\"" >> ~/.bashrc
    source ~/.bashrc
fi

# 8. Kreiranje conda okruženja
echo "Kreiranje Conda okruženja 'fr' s Python 3.8..."
conda create -y --name fr python=3.8 || { echo "Greška prilikom kreiranja Conda okruženja. Završetak skripte."; exit 1; }

# 9. Aktivacija conda okruženja i instalacija zahtjeva
echo "Aktivacija Conda okruženja 'fr' i instalacija zahtjeva..."
source $HOME/miniconda/bin/activate fr || { echo "Greška prilikom aktivacije Conda okruženja. Završetak skripte."; exit 1; }
pip install -r requirements.txt || { echo "Greška prilikom instalacije paketa. Završetak skripte."; exit 1; }

echo "Postavljanje projekta je završeno!"
