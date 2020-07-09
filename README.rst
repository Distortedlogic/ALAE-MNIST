ALAE implementation in pytorch for MNIST datasets

Quick Start
===========

Clone the repo and cd into it.

Without docker-compose

    docker build -t alae .

    docker run -p 8888:8888 -t alae jupyter lab --allow-root

or

With docker-compose

    docker-compose up --build

Copy and Paste the url with token from terminal output into your browser.

example:

    http://127.0.0.1:8888/?token=422ac908c97aae9d990baa9cb2ed5afdaa8ca4357a4b2e29

cd into the torch folder and open pytorch.ipynb