#!/bin/bash

# Provjera je li nvidia-smi dostupan
if ! command -v nvidia-smi &> /dev/null
then
    echo "nvidia-smi nije pronađen. Provjerite je li NVIDIA drajver instaliran."
    exit 1
fi

# Provjera je li prime-run dostupan
if ! command -v prime-run &> /dev/null
then
    echo "prime-run nije pronađen. Provjerite je li Prime render offload instaliran."
    exit 1
fi

# Pokretanje aplikacije koristeći NVIDIA GPU
echo "Pokrećem model.py koristeći NVIDIA GPU..."
prime-run python model.py &

# Dohvaćanje PID-a aplikacije
APP_PID=$!

# Čekanje nekoliko sekundi da se aplikacija pokrene
sleep 5

# Provjera korištenja GPU-a
echo "Provjeravam korištenje NVIDIA GPU-a..."
nvidia-smi

# Čekanje da aplikacija završi
wait $APP_PID

# Provjera je li aplikacija uspješno pokrenuta
if [ $? -eq 0 ]; then
    echo "Aplikacija je uspješno pokrenuta."
else
    echo "Došlo je do greške prilikom pokretanja aplikacije."
fi
