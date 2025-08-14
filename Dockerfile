# Base image con CUDA 12.1 e Python preinstallato (compatibile con torch)
#FROM pytorch/pytorch:2.7.0-cuda11.8-cudnn9-runtime
FROM python:3.12.2

# Imposta la directory di lavoro
WORKDIR /app

# Copia requirements e installa le dipendenze
COPY requirements.txt .


RUN pip install --no-cache-dir -r requirements.txt

# Copia tutto il codice del progetto
COPY . .

# Comando di avvio: lancia il tuo server UHP
CMD ["python3", "-u","src/engine.py"]


#per creare l'immagine: docker build -t nome .
#per avviare il container: docker run -t nome
#per la versione interattiva: docker run -it nome