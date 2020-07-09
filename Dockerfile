FROM jupyter/minimal-notebook
COPY requirements.txt requirements.txt
USER root
RUN pip install -r requirements.txt
RUN jupyter labextension install @jupyter-widgets/jupyterlab-manager
RUN jupyter nbextension enable --py --sys-prefix widgetsnbextension
WORKDIR /notebooks
COPY . .
# RUN chown 1000:100 .
# USER 1000