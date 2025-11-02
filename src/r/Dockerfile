FROM rocker/r-base:4.4.1
WORKDIR /app
RUN apt-get update && apt-get install -y libxml2-dev libssl-dev libcurl4-openssl-dev
COPY requirements-bin.txt .
RUN cat requirements-bin.txt | xargs apt-get install -y -qq
COPY requirements-src.R .
RUN Rscript requirements-src.R
RUN rm -rf /var/lib/apt/lists/*
COPY . /app
CMD ["Rscript", "main.r"]
