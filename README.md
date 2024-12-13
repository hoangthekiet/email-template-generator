### Building the Docker Image
```
docker build -t email-template-generator:latest .
```
### Running the Docker Container
```
docker run -e GROQ_API_KEY=<key> [-e VERBOSE=true] -it email-template-generator:latest
```