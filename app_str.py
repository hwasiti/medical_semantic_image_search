
#### To run the app:
#### streamlit run app_str.py
#### To avoid webpage loading stuck at "Loading...":
#### ssh with port forwarding so that the app can be accessed from the local machine
#### ssh -L 8501:localhost:8501 username@server_ip


####### Install dependencies #######
# pip install spacy ftfy==4.4.3
# python -m spacy download en
# conda install -c huggingface transformers==4.14.1 tokenizers==0.10.3 see:
# https://discuss.huggingface.co/t/importing-tokenizers-version-0-10-3-fails-due-to-openssl/17820/3
# pip install streamlit

import os
import streamlit as st
from PIL import Image
from math import floor, ceil

import datetime
from PIL import Image
import requests
from transformers import CLIPProcessor, CLIPModel, CLIPVisionModel, CLIPTextModel
import torch
from torch import autocast
import glob
from functools import partial
from pathlib import Path
import pandas as pd
import numpy as np
from  tqdm import tqdm
tqdm.pandas()

path = Path('images')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
model.to(device)
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
df = pd.read_pickle(str(path).replace('/', '-')+'.pickle')
img_embs = torch.stack(df.features.values.tolist())[:, -1, :].t().to(device)
logit_scale = model.logit_scale.exp()

def compute_probs_and_sort(text_embeds, n):
    preds = torch.matmul(text_embeds.detach(), img_embs) * logit_scale.detach()  # compute cosine similarity * 100 (100 is perfectly text matched to image)
    print(f'max, min cosine similarity of all images with the text prompt: {preds.max()} , {preds.min()}')
    sorted, indices = torch.sort(preds, descending=True)
    if device == 'cpu':
        probs = sorted[:, :n].numpy()
        idxs = indices[:, :n].numpy()
    else:
        probs = sorted[:, :n].cpu().numpy()
        idxs = indices[:, :n].cpu().numpy()
    return probs, idxs


def infer(prompt, img_cnt):
    input_text = processor(text=[prompt], return_tensors="pt", padding=True).to(device)
    output_text_features = model.get_text_features(**input_text)
    text_embeds = output_text_features / output_text_features.norm(p=2, dim=-1, keepdim=True)  
    probs, idxs = compute_probs_and_sort(text_embeds, img_cnt)
    print('done compute probabilities to all images')

    
    images = []
    metadata = []
    for i, idx in enumerate(idxs[0]):
        fn = df.iloc[idx, df.columns.get_loc('path')]
        try:    
            image = Image.open(fn)
        except Exception as e:
            print(f'Could not open the image {fn}')
            print(f'The error was: {str(e)}')
        images.append(image)
        print(fn)
        metadata.append((str(fn),str(probs[0][i])))
    print('done reading the images')
    return images, metadata


def show_images(images, grid_size, page, metadata):
    n_grid_images = grid_size * grid_size  # number of images in a page
    if page * n_grid_images <= len(images):  # if this page will NOT exceed the number of possible images to be shown
        last_y = ceil(n_grid_images / grid_size)
        n_images_to_show = n_grid_images
    else:
        last_img_n = len(images)
        last_y = ceil((last_img_n - (page - 1) * n_grid_images) / grid_size)
        n_images_to_show = last_img_n - (page - 1) * n_grid_images

    for y in range(last_y):
        cols = st.columns(grid_size)
        last_x = n_images_to_show - (last_y - 1) * grid_size
        for x in range(last_x):
            image = images[(page - 1) * n_grid_images + y * grid_size + x]
            if image.mode in ("RGBA", "P"):
                image = image.convert('RGB')  # discard the alpha channel
            cols[x].image(image, use_column_width=True)
            cols[x].markdown(
                '**' + metadata[x][0].split('/')[-1] + '**'  + ' (' + metadata[x][1] + ')'
            )
            # cols[x].dataframe(df_filtered[(df_filtered.filename == img_name)])
                


def main():
    st.set_page_config(
        page_title="Medical Semantic Image Search",
        page_icon="🧊",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    st.title("Medical Semantic Image Search")
    st.write("")

    st.sidebar.header("Search")
    search_query = st.sidebar.text_input('Write search query and hit enter', value='') #, label_visibility='hidden')

    st.sidebar.header("Grid size")
    grid_size = st.sidebar.slider('', 1, 8, 3, label_visibility='hidden')
    
    # st.sidebar.title("Top Results count")
    st.sidebar.header("Top Results count")
    img_cnt = st.sidebar.slider('', 1, 100, 18, label_visibility='hidden')  # len(df) as a max is too much

    if search_query != "":
        images, metadata = infer(search_query, img_cnt)
    else:
        images = []
        metadata = []

    # st.markdown('**Total images =** ' + str(len(images)))

    col1, col2, col3, col4, col5, col6, col7, col8, col9, col10, col11, col12, col13 = st.columns(
        (1, 1, 1, 1, 1, 1, 3, 1, 1, 1, 1, 1, 1))
    total_pages = ceil(len(images) / (grid_size * grid_size))
    if total_pages == 0:
        total_pages = 1
    pages = ['Page ' + str(i + 1) for i in range(total_pages)]  # number of pages of the gallery

    page = col7.selectbox('', pages, label_visibility='hidden')
    current_page = int(page[5:])

    # Hide 'Made with streamlit' footer and menu
    hide_streamlit_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            </style>
            """
    st.markdown(hide_streamlit_style, unsafe_allow_html=True) 

    if images is None:
        st.write("No results")
    elif len(images) == 0:
        st.write("No results")
    else:
        show_images(images, grid_size, current_page, metadata)


if __name__ == "__main__":
    main()

