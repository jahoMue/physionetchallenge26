FROM python:3.10.20-bookworm

## DO NOT EDIT these 3 lines.
RUN mkdir /challenge
COPY ./ /challenge
WORKDIR /challenge

## Install your dependencies here using apt install, etc.

## Include the following line if you have a requirements.txt file.
RUN pip install -r requirements.txt

## Download Google Drive zip and extract into /challenge/model
ARG GDRIVE_FILE_ID="1i9ekH3G_UCJr6bRykRhEq98fqbIVFyM_"

RUN mkdir -p /challenge/model \
    && python -m gdown "${GDRIVE_FILE_ID}" -O /tmp/model.zip \
    && python - <<'PY'
import zipfile

zip_path = "/tmp/model.zip"
extract_dir = "/challenge"

with zipfile.ZipFile(zip_path, "r") as z:
    z.extractall(extract_dir)
PY