FROM python:3.10.20-bookworm

## DO NOT EDIT these 3 lines.
RUN mkdir /challenge
COPY ./ /challenge
WORKDIR /challenge

## Install your dependencies here using apt install, etc.

## Include the following line if you have a requirements.txt file.
RUN pip install -r requirements.txt

#ARG MODEL_URL="https://datashare.tu-dresden.de/s/rFdMM7HkzpirjLW/download"
#RUN printf 'MODEL_URL=<%s>\n' "${MODEL_URL}"

#RUN mkdir -p /challenge \
#    && curl -fL --retry 5 --retry-delay 20 --retry-all-errors \
#       -o /tmp/model.zip "${MODEL_URL}" \
#    && python -c "import zipfile; z=zipfile.ZipFile('/tmp/model.zip', 'r'); bad=z.testzip(); assert bad is None, f'Bad file in zip: {bad}'; z.extractall('/challenge')" \
#    && rm -f /tmp/model.zip
