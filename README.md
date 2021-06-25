# Lipohypertrophy Model Deployment
To see the app in action, click
[here](https://share.streamlit.io/xudongyang2/lipo_deploy/deployment/lipo_app.py).

## About
This is the repo for lipohypertrophy model deployment as a web application on Streamlit share. In this repo:
- 'best.pt' is the YOLOv5m model file trained on our data for object detection
- 'densenet_final.pth' is the densenet model file trained on our data for lipohypertrophy classification
- 'lipo_app.py' is the source code to build the web application
- 'requirements.txt' is the environment file for deployment on the server

## Usage
To re-deploy the application with new models:

1. Ask for permission to add as collaborator to this repo
2. replace the 'best.pt' with newly trained YOLOv5 model or replace 'densenet_final.pth' with newly trained classification model by Pytorch. 
3. Make a PR to the 'deployment' branch
4. Once there are new commits on the 'deployment' branch, the app will be automatically refreshed

## Dependencies
- torch==1.8.1+cpu
- torchvision==0.9.1+cpu
- opencv-python-headless==4.5.2.54
- streamlit==0.82.0
- plotly==4.14.3
- PyYAML==5.4.1
- tqdm==4.61.0
- matplotlib==3.4.2
- seaborn==0.11.1
