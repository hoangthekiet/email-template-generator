### About
This repository offers two AI-assisted functionalities:

1. Auto-detecting the color scheme from the user's website.

2. Generating email content by learning the user's writing style.

### How To Run
#### Building the Docker Image
```
docker build -t email-template-wizard:latest .
```
#### Running the Docker Container
```
docker run -e GROQ_API_KEY=<key> [-e VERBOSE=true] -it email-template-wizard:latest
```
#### Running Service without Docker
```
pip install -r requirements.txt
python service.py
```