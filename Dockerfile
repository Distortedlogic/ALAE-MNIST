FROM jupyter/minimal-notebook
COPY requirements.txt requirements.txt
USER root
RUN pip install -r requirements.txt
RUN jupyter nbextension enable --py widgetsnbextension
WORKDIR /notebooks
COPY . .
# RUN chown 1000:100 .
# USER 1000