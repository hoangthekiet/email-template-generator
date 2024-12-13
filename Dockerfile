# Use Python 3.11 as the base image
FROM python:3.11-slim

# Set the working directory to /app
WORKDIR /app

# Copy the requirements file into the container
COPY requirements.txt .

# Install the dependencies
RUN pip install -r requirements.txt

# Copy the application code into the container
COPY . .

# Run the service.py script when the container starts
CMD ["python", "service.py"]