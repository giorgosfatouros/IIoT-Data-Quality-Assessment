FROM python:3.8-slim-buster

WORKDIR /iot

# Copy requirements first to leverage Docker cache
COPY requirements.txt requirements.txt
COPY pyLeanxcale-1.9.13_latest-py3-none-any.whl pyLeanxcale-1.9.13_latest-py3-none-any.whl

# Install pyLeanxcale wheel first
RUN pip install --no-cache-dir /iot/pyLeanxcale-1.9.13_latest-py3-none-any.whl
# Upgrade pip and install dependencies
RUN pip install --no-cache-dir --upgrade pip
# Install other dependencies from requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of your app's code into the container
COPY . .

# Make port 8501 available to the world outside this container
EXPOSE 8501

# Run Streamlit directly
CMD ["streamlit", "run", "--server.port=8501", "--server.address=0.0.0.0", "app.py"]
