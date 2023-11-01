# LLM-as-a-Service Setup Guide

This guide provides the steps to set up and run the LLM-as-a-Service on your machine.

## Prerequisites

### Hardware Requirements
- **GPU**: NVIDIA A100 or equivalent.
- **Software**: CUDA version 11.8 or above.

### System Packages

Before proceeding, ensure the following packages are installed:

```
sudo apt install git vim python3 python3-pip
```

## Python Environment Setup
Install virtualenv:
```
sudo pip3 install virtualenv
```
Create a virtual environment:
```
virtualenv dev
```
Activate the virtual environment
```
source dev/bin/activate

```
Install required Python packages
```
pip install -r requirements.txt
```
## Configuration
Set the configuration path
Export the path to your cluster_conf.yaml
```
export CONFIG_PATH=/path/to/your/cluster_conf.yaml
```
#### Important:
In the cluster_conf.yaml, you must specify access tokens for both your Huggingface and Weights & Biases accounts.

If you wish not to monitor deployment logging, set WANDB_ENABLE to False.
When WANDB_ENABLE is set to False, there's no need to provide a Weights & Biases token.

## Running the Service
Initialize a RAY cluster
```
ray start --head
```
Build the configuration for your backend service (only once).
```
serve build backend:app -o config.yaml
```
Deploy the application
```
serve deploy config.yaml
```

### Run the authentication microservice
```
uvicorn API/app:app --reload --port 8080
```