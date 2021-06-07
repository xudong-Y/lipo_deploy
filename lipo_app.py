import streamlit as st
import torch
import torchvision
from torch import nn, optim
from torchvision import transforms, models, datasets, utils
from PIL import Image
from collections import OrderedDict
import plotly.graph_objects as go

st.set_page_config(layout="wide")
st.title('Lipohypertrophy Prediction')

#load the model
cnn_model = models.densenet121(pretrained=True)
new_layers = nn.Sequential(OrderedDict([
            ('new1', nn.Linear(1024, 500)),
            ('relu', nn.ReLU()),
            ('new2', nn.Linear(500, 1))
        ]))
cnn_model.classifier = new_layers

cnn_model.load_state_dict(torch.load('densenet_final.pth', map_location=torch.device('cpu'))) #put the directory here where cnn_model.pt is located
cnn_model.eval()

#make prediction
uploaded_file = st.sidebar.file_uploader('Upload ultrasound image here!', accept_multiple_files = True)
option = st.sidebar.selectbox('file_names', uploaded_file, format_func = lambda x: x.name)

def get_prediction(image):
    "display image and get prediction"
    img = Image.open(image).convert('RGB')
    #st.image(img.resize((350, 350)), caption='Uploaded Image', use_column_width=False)
    image_tensor = transforms.functional.to_tensor(img.resize((300, 300))).unsqueeze(0)

    pos = float(torch.sigmoid(cnn_model(image_tensor)).detach().numpy())
    neg = 1 - pos
    pred = 'Positive' if pos >0.5 else 'Negative' 
    return pos, neg, pred

def display_results(image, pos, neg):
    "display prediction label and confidence"

    labels = ['Positive','Negative']
    val = [pos, neg]
    img = Image.open(image).convert('RGB')
    
    fig = go.Figure(go.Bar(
                x=val,
                y=labels,
                orientation='h',
                marker_color=['rgb(245, 133, 24)','rgb(76, 120, 168)'],
                text = [str(round(x*100,1))+'%' for x in val],
        textposition='outside',
        textfont={'color': ['rgb(245, 133, 24)','rgb(76, 120, 168)'], 'size':15}
    ))

    fig.update_yaxes(autorange="reversed")
    fig.update_xaxes({'showgrid': False, 'visible': False, 'range': [0, 1.2]})
    fig.update_layout(paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        width=500,
        height = 110, 
        title={
            'text': "<b>Prediction Confidence<b>",
            'y':0.99,
            'x':0.28,
            'xanchor': 'center',
            'yanchor': 'top'},
            title_font = {'size': 17},
            margin=dict(l=20, r=0, t=25, b=20, pad = 2)
            )
    right_column.image(img.resize((350, 350)), caption='Uploaded Image', use_column_width=False)
    left_column.text(" \n")
    left_column.plotly_chart(fig, use_container_width = False)


    
if len(uploaded_file)>0:
    pos, neg, pred = get_prediction(option)
    if pred == 'Positive':
        original_title  = '<p style="font-family:Courier; color:rgb(245, 133, 24); fontWeight:bold; font-size: 27px;">Positive</p>'
    else:
        original_title  = '<p style="font-family:Courier; color:rgb(76, 120, 168); fontWeight:bold; font-size: 27px;">Negative</p>'
    st.markdown(original_title, unsafe_allow_html=True)    
    left_column, right_column = st.beta_columns([2,2.4])
    display_results(option, pos, neg)       
else:
    pass
        