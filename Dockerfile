FROM python:3.8-slim
LABEL authors="George Fatouros"
# Set the working directory in the container
WORKDIR /iot

# Copy the requirements file into the container
COPY requirements.txt requirements.txt

# Copy the local wheel file into the container
COPY pyLeanxcale-1.9.13_latest-py3-none-any.whl pyLeanxcale-1.9.13_latest-py3-none-any.whl

# Install the dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Install the local wheel file
RUN pip install pyLeanxcale-1.9.13_latest-py3-none-any.whl

# Copy the rest of your app's code into the container
COPY . .

# Make port 8501 available to the world outside this container
EXPOSE 8501

# Run the application
CMD ["streamlit", "run", "app.py"]
