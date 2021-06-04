import streamlit as st
import pandas as pd
import torch
from torch import nn
from torchvision import transforms, models
from PIL import Image
import altair as alt
from collections import OrderedDict
import gc

st.title('Lipohypertrophy Prediction')

#load the model
cnn_model = models.densenet121(pretrained=True)
new_layers = nn.Sequential(OrderedDict([
            ('new1', nn.Linear(1024, 500)),
            ('relu', nn.ReLU()),
            ('new2', nn.Linear(500, 1))
        ]))
cnn_model.classifier = new_layers

cnn_model.load_state_dict(torch.load('densemodels.pth', map_location=torch.device('cpu'))) #put the directory here where cnn_model.pt is located
torch.set_grad_enabled(False)
cnn_model.eval()
left_column, right_column = st.beta_columns(2)

#make prediction
uploaded_file = left_column.file_uploader('Upload ultrasound image here!', accept_multiple_files = True)
option = left_column.selectbox('file_names', uploaded_file, format_func = lambda x: x.name)

def get_prediction(image):
    "display image and get prediction"
    img = Image.open(image).convert('RGB')
    st.image(img.resize((350, 350)), caption='Uploaded Image', use_column_width=False)
    image_tensor = transforms.functional.to_tensor(img.resize((300, 300))).unsqueeze(0)

    pos = float(torch.sigmoid(cnn_model(image_tensor)).detach().numpy())
    neg = 1 - pos
    pred = 'Positive' if pos >0.5 else 'Negative' 
    return pos, neg, pred

def display_results(pos, neg, pred):
    "display prediction label and confidence"
    st.write(f"Lipohypertrophy prediction: {pred}")
    chart_data = pd.DataFrame(
        {'label': ['Positive','Negative'],
        'value': [pos,neg],
        'perc':[str(round(pos*100,1))+'%', str(round(neg*100,1))+'%']})
    chart = alt.Chart(chart_data,title="Prediction Confidence").mark_bar().encode(
    alt.X("value", title = "", axis = None),
    alt.Y("label", title = "", sort=['Positive','Negative']),
    alt.Text('value'),
    alt.Color('label', legend = None)
    )
    text = chart.mark_text(
    align='left',
    baseline='middle',
    dx=3 
    ).encode(
        text='perc'
    )
    st.write(alt.layer(chart, text).configure_axis(
    grid=False).configure_view(
    strokeWidth=0
    ))

    
if len(uploaded_file)>0:
    left_column = st.empty()
    with left_column:
        pos, neg, pred = get_prediction(option)
        gc.collect()
    with right_column:
        display_results(pos, neg, pred)       
else:
    pass
        