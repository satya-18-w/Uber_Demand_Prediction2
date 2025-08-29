#!/bin/bash
# Login to AWS ECR
aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin 525102257214.dkr.ecr.us-east-1.amazonaws.com
# Pull the latest image
docker pull 525102257214.dkr.ecr.us-east-1.amazonaws.com/demand1:latest

# Check if the container 'campusx-app' is running
if [ "$(docker ps -q -f name=my-app)" ]; then
    # Stop the running container
    docker stop my-app
fi

# Check if the container 'campusx-app' exists (stopped or running)
if [ "$(docker ps -aq -f name=my-app)" ]; then
    # Remove the container if it exists
    docker rm my-app
fi

# Run a new container
docker run -d -p 8000:8000 -e DAGSHUB_TOKEN="25476b0cf1226d6672acb8c6d26ef034715e6dc9" -e  DAGSHUB_USER="satyajitsamal198076"  --name my-app 051826734860.dkr.ecr.ap-southeast-2.amazonaws.com/campusx_ecr:latest