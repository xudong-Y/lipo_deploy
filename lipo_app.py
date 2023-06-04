import streamlit as st
import torch
from torch import nn
from torchvision import transforms, models
from PIL import Image
from collections import OrderedDict
import plotly.graph_objects as go
import plotly.express as px

torch.autograd.set_grad_enabled(False)

st.set_page_config(layout="wide")
st.title('Lipohypertrophy Prediction')

#load the model
@st.cache(suppress_st_warning=True)
def create_model():
    cnn_model = models.densenet121(pretrained=True)
    new_layers = nn.Sequential(OrderedDict([
                ('new1', nn.Linear(1024, 500)),
                ('relu', nn.ReLU()),
                ('new2', nn.Linear(500, 1))
            ]))
    cnn_model.classifier = new_layers

    cnn_model.load_state_dict(torch.load('densenet_final.pth', map_location=torch.device('cpu'))) #put the directory here where cnn_model.pt is located
    return cnn_model

# @st.cache(suppress_st_warning=True)
def create_yolo_model():
#     model = torch.hub.load('ultralytics/yolov5', 'custom', path='best.pt', source='local')
    model = torch.hub.load('/opt/ml/model/code/', 'custom', source ='local', path='best.pt',force_reload=True)
    return model
    
@st.cache(suppress_st_warning=True)
def read_image(image):
    img = Image.open(image).convert('RGB')
    return img

def get_prediction(img, cnn_model):
    "get classification prediction"

    image_tensor = transforms.functional.to_tensor(img.resize((300, 300))).unsqueeze(0)

    pos = float(torch.sigmoid(cnn_model(image_tensor)).detach().numpy())
    neg = 1 - pos
    pred = 'Positive' if pos >0.5 else 'Negative' 
    return pos, neg, pred

def get_lipo_prediction(img, yolo_model):
    output = yolo_model(img)
    if len(output.pred[0]) > 0:
        cord = output.pred[0][0].numpy()
        xmin = cord[0]
        ymin = cord[1]
        xmax = cord[2]
        ymax = cord[3]
    else:
        xmin = None
        ymin = None
        xmax = None
        ymax = None
    return xmin, ymin, xmax, ymax

def display_results(img, pos, neg, xmin = None, ymin = None, xmax = None, ymax = None):
    "display prediction label and confidence"

    labels = ['Positive','Negative']
    val = [pos, neg]
    
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
        width=470,
        height = 110, 
        title={
            'text': "<b>Prediction Confidence<b>",
            'y':0.99,
            'x':0.295,
            'xanchor': 'center',
            'yanchor': 'top'},
            title_font = {'size': 17},
            margin=dict(l=20, r=0, t=25, b=20, pad = 2)
            )
    if pred == 'Positive': #add confidence
        figure = px.imshow(img)
        figure.update_xaxes({'showgrid': False, 'visible': False})
        figure.update_yaxes({'showgrid': False, 'visible': False})
        figure.update_layout(width = 500, margin=dict(l=0, r=120, t=5, b=40, pad = 0))
        figure.add_shape(type="rect",
            x0=xmin, y0=ymin, x1=xmax, y1=ymax,
            line=dict(color="red"),
        )
        right_column.plotly_chart(figure, config={"displayModeBar": False})
        left_column.text(" \n")
        left_column.plotly_chart(fig, use_container_width = False, config={"displayModeBar": False})
    else:
        figure = px.imshow(img)
        figure.update_xaxes({'showgrid': False, 'visible': False})
        figure.update_yaxes({'showgrid': False, 'visible': False})
        figure.update_layout(width = 500, margin=dict(l=0, r=120, t=5, b=40, pad = 0))
        right_column.plotly_chart(figure, config={"displayModeBar": False})
        left_column.text(" \n")
        left_column.plotly_chart(fig, use_container_width = False, config={"displayModeBar": False})
    
cnn_model = create_model()
cnn_model.eval()

yolo_model = create_yolo_model()
yolo_model.eval()

#make prediction
uploaded_file = st.sidebar.file_uploader('Upload ultrasound image here!', accept_multiple_files = True)
option = st.sidebar.selectbox('file_names', uploaded_file, format_func = lambda x: x.name)
    
if len(uploaded_file)>0:
    img = read_image(option)
    pos, neg, pred = get_prediction(img, cnn_model)
    if pred == 'Positive':
        original_title  = '<p style="font-family:Courier; color:rgb(245, 133, 24); fontWeight:bold; font-size: 27px;">Positive</p>'
        st.markdown(original_title, unsafe_allow_html=True)
        left_column, right_column = st.beta_columns([2,2.4])
        xmin, ymin, xmax, ymax = get_lipo_prediction(img, yolo_model)
        if xmin != None:
            right_column.write("Object Detection Available")
        else:
            right_column.write("Object Detection Not Available")
        display_results(img, pos, neg, xmin ,ymin, xmax, ymax)
    else:
        original_title  = '<p style="font-family:Courier; color:rgb(76, 120, 168); fontWeight:bold; font-size: 27px;">Negative</p>'    
        st.markdown(original_title, unsafe_allow_html=True)
        left_column, right_column = st.beta_columns([2,2.4])
        display_results(img, pos, neg)       
else:
    pass
        
