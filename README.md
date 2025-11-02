# Titanic Disaster Prediction

Logistic Regression Models in Python and R

1. Visit https://www.kaggle.com/competitions/titanic/data
2. Download:
   - `train.csv`
   - `test.csv`
3. Create a folder `data/` in the project root and place both files inside:
   ```
   titanic_disaster_prediction/data/train.csv
   titanic_disaster_prediction/data/test.csv
   ```
   Create a folder `src/r/data/` in the project root and place both files inside:
   ```
   titanic_disaster_prediction/src/r/data/train.csv
   titanic_disaster_prediction/src/r/data/test.csv
   ```

---

Now to build and run docker, you only need **Docker Desktop** (no local R or Python setup).
Run the following commands:

For python implementation,

#### Build

```bash
docker build -f Dockerfile_Python -t titanic-py .
```

#### Run

```bash
docker run --rm -v $(pwd)/data:/app/data titanic-py
```

For R imeplemntation,

Switch to a different folder

```bash
cd src/r
```

#### Build

```bash
docker build -f Dockerfile -t titanic-r .
```

#### Run

```bash
docker run --rm -v $(pwd)/data:/app/data titanic-r
```

Alternatively, we can build docker using `install_packages.R`, but takes significantly longer time using the following commands.

```bash
docker build -f Dockerfile_R_Alternative -t titanic-r-alt .
docker run --rm -v $(pwd)/data:/app/data titanic-r-alt
```

---

Quick Start Summary

```bash
# Clone repo
git clone https://github.com/<your-username>/titanic_disaster_prediction.git
cd titanic_disaster_prediction

# Add Kaggle data
mkdir -p data
cp ~/Downloads/train.csv data/
cp ~/Downloads/test.csv data/

# Build & run Python version
docker build -f Dockerfile_Python -t titanic-py .
docker run --rm -v $(pwd)/data:/app/data titanic-py

# Build & run R version
cd src/r
docker build -f Dockerfile -t titanic-r .
docker run --rm -v $(pwd)/data:/app/data titanic-r


```

`data/submission.csv` will appear in your local data folder.
