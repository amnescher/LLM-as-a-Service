# EscherCloud LLM-as-a-Service

## Description

Welcome to our project repository!

This project aims to provide a comprehensive **LLM-as-a-Service** (Language Model as a Service) solution. The repository contains a set of demos showcasing the capabilities of our **LLM-as-a-Service**.

## Features

- **Document and Video Retrieval**: Our service excels in document and video retrieval question answering tasks.
- **Easy Integration**: We have designed the service to be easily integrated into existing applications.

## Directory Structure

- `Backend`: Contains files for LLM and serves as the core of the language model demos.
- `FrontEnd`: Contains files for a Streamlit engine to provide a minimal UI for the LLM service.

## Getting Started

1. Clone the repository to your local machine.

2. Create an `env.env` file and store your Huggingface access token as the `Hugging_ACCESS_TOKEN` variable in it.

3. Build the Docker image using the following command:

`sudo docker compose -f docker-compose.yml --env-file ./env.env build`

4. Run the LLM service using the following command:

`sudo docker compose -f docker-compose.yml --env-file ./env.env up`


### Hardware Requirements

To run LLM as an inference service, a minimum A100 GPU is required.

