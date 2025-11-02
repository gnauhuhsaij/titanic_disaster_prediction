FROM python:3.11
WORKDIR /app
COPY . /app
RUN python -m venv /app/venv
RUN /app/venv/bin/pip install --upgrade pip
COPY requirements.txt /app/requirements.txt
RUN /app/venv/bin/pip install --no-cache-dir -r /app/requirements.txt
ENV PATH="/app/venv/bin:$PATH"
CMD ["python", "src/application/app.py"]