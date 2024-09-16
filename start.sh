#!/bin/bash

# Prompt the user for the OpenAI API key
read -p "Please enter your OpenAI API key: " OPENAI_API_KEY

# Export the API key as an environment variable
export OPENAI_API_KEY

# Build the Docker image
echo "Building the Docker image..."
docker-compose build

# Start the Docker container with the environment variable
echo "Starting the Docker container..."
OPENAI_API_KEY=$OPENAI_API_KEY docker-compose up