
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
    if page * n_grid_images <= len(
            images):  # if this page will NOT exceed the number of possible images to be shown
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
            image = images[x]
            if image.mode in ("RGBA", "P"):
                image = image.convert('RGB')  # discard the alpha channel
            cols[x].image(image, use_column_width=True)
            for item in metadata:
                cols[x].markdown(
                    '**' + item[0] + '**: ' + item[1] 
            # cols[x].dataframe(df_filtered[(df_filtered.filename == img_name)]
                )


def main():
    st.set_page_config(
        page_title="Medial Semantic Image Search",
        page_icon="🧊",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    st.title("Semantic Image Search Results")
    st.write("")

    st.sidebar.header("Search")
    search_query = st.sidebar.text_input('Search in images', value='')

    st.sidebar.header("Grid size")
    grid_size = st.sidebar.slider('', 1, 8, 2)
    
    # st.sidebar.title("Top Results count")
    st.sidebar.header("Top Results count")
    img_cnt = st.sidebar.slider('', 1, 100, 8)  # len(df) as a max is too much

    images, metadata = infer(search_query, img_cnt)



    # columns = ['search term', 'date of download', 'time of download', 'filename', 'link',
    #            'title', 'description', 'date taken', 'owner name', 'path alias', 'owner',
    #            'latitude', 'longitude', 'country', 'city', 'machine tags', 'views', 'tags',
    #            'exif data', 'species prediction score', 'cage prediction score']

    # st.sidebar.header("Search term")
    # terms_selected = st.sidebar.multiselect('', ['monkey wild', 'monkey cage'], ['monkey wild', 'monkey cage'])

    # st.sidebar.header("Metadata")

    # captions = st.sidebar.multiselect(
    #     'Select what to show below images. Other info will be shown as a table below. For long data, just hover the '
    #     'mouse over it, a tool-tip will show all the cell content.',
    #     columns)

    # st.sidebar.header("DL prediction of Monkey")
    # pred_thr_monkey_low, pred_thr_monkey_high = st.sidebar.slider("1.0 = Highly confident", 0.0, 1.0, (0.45, 1.0), 0.05,
    #                                                               key='11')  # any rand unique key

    # st.sidebar.header("DL prediction of Cage")
    # pred_thr_cage_low, pred_thr_cage_high = st.sidebar.slider("", 0.0, 1.0, (0.45, 1.0), 0.05, key='12')
    # try:
    #     fn = 'df_100_per_search_encrypted.json'
    #     with open(fn, 'r') as file:  # Check file available?
    #         pd_data = file.read()
    #     df_decrypted = load_db_from_pd(key)
    # except FileNotFoundError:
    #     df_decrypted = load_db(passw, key)

    # st.sidebar.header("Country")
    # countries = df_decrypted.country.unique().tolist()
    # countries_selected = st.sidebar.multiselect('', countries, countries)

    # date_max = pd.to_datetime(df_decrypted['date taken'], format='%Y-%m-%d %H:%M:%S').apply(lambda x: x.date()).max()
    # date_min = pd.to_datetime(df_decrypted['date taken'], format='%Y-%m-%d %H:%M:%S').apply(lambda x: x.date()).min()
    # st.sidebar.header("Date Taken")
    # date_low, date_high = st.sidebar.slider("", date_min, date_max, (date_min, date_max))

    # view_max = df_decrypted.views.max().astype(int).item()
    # view_min = df_decrypted.views.min().astype(int).item()
    # st.sidebar.header("View Count")
    # view_low, view_high = st.sidebar.slider("", view_min, view_max, (view_min, view_max))


    # # Show Map
    # st.sidebar.header("Show Map")
    # chk_show_map = st.sidebar.checkbox('')

    # Refresh online database load
    # st.sidebar.header("Reload Online DB")
    # if st.sidebar.button('Refresh DB'):
    #     os.remove("df_100_per_search_encrypted.json")  # delete the local DB file
    #     st.experimental_rerun()

    # filtering the DB
    # df_filtered = df_filtered.loc[(df_decrypted['species prediction score'] >= pred_thr_monkey_low)
    #                               & (df_decrypted['cage prediction score'] >= pred_thr_cage_low)
    #                               & (df_decrypted['species prediction score'] <= pred_thr_monkey_high)
    #                               & (df_decrypted['cage prediction score'] <= pred_thr_cage_high)
    #                               & (df_decrypted['views'].astype(int) <= view_high)
    #                               & (df_decrypted['views'].astype(int) >= view_low)
    #                               & (pd.to_datetime(df_decrypted['date taken'], format='%Y-%m-%d %H:%M:%S').apply(
    #     lambda x: x.date()) <= date_high)
    #                               & (pd.to_datetime(df_decrypted['date taken'], format='%Y-%m-%d %H:%M:%S').apply(
    #     lambda x: x.date()) >= date_low)
    #                               & (df_decrypted['search term'].isin(terms_selected))
    #                               & (df_decrypted['country'].isin(countries_selected))
    #                               ]

    # df_filtered.reset_index(drop=True, inplace=True)

    st.markdown('**Total images = ** ' + str(len(images)))

    col1, col2, col3, col4, col5, col6, col7, col8, col9, col10, col11, col12, col13 = st.columns(
        (1, 1, 1, 1, 1, 1, 3, 1, 1, 1, 1, 1, 1))
    total_pages = ceil(len(images) / (grid_size * grid_size))
    if total_pages == 0:
        total_pages = 1
    pages = ['Page ' + str(i + 1) for i in range(total_pages)]  # number of pages of the gallery

    page = col7.selectbox('', pages)
    current_page = int(page[5:])

    # Hide Made with streamlit footer
    hide_streamlit_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            </style>
            """
    st.markdown(hide_streamlit_style, unsafe_allow_html=True) 

    if images is None:
        st.write("No results to show")
    elif len(images) == 0:
        st.write("No results to show")
    else:
        show_images(images, grid_size, current_page, metadata)
        # Draw map
        # if chk_show_map:
        #     expand1 = st.beta_expander('Show Map', True)
        #     df_geo = df_filtered[['search term', 'latitude', 'longitude']][
        #         (df_filtered['latitude'] != 0) & (df_filtered['longitude'] != 0)]  # ignore those without gps info
        #     df_geo.reset_index(drop=True, inplace=True)

        #     expand1.pydeck_chart(pdk.Deck(
        #         map_style='mapbox://styles/mapbox/light-v9',
        #         initial_view_state=pdk.ViewState(
        #             latitude=0,
        #             longitude=0,
        #             zoom=1,
        #             pitch=50,
        #         ),

        #         layers=[
        #             pdk.Layer(
        #                 'HexagonLayer',
        #                 data=df_geo,
        #                 get_position='[longitude, latitude]',
        #                 radius=200,
        #                 elevation_scale=4,
        #                 elevation_range=[0, 1000],
        #                 pickable=True,
        #                 extruded=True,
        #             ),
        #             pdk.Layer(
        #                 'ScatterplotLayer',
        #                 data=df_geo.loc[(df_geo['search term'] == 'monkey cage')],
        #                 get_position='[longitude, latitude]',
        #                 pickable=False,
        #                 opacity=0.8,
        #                 stroked=True,
        #                 filled=True,
        #                 radius_scale=6,
        #                 radius_min_pixels=3,
        #                 radius_max_pixels=100,
        #                 line_width_min_pixels=1,
        #                 get_radius="exits_radius",
        #                 get_fill_color=[255, 0, 0],
        #                 get_line_color=[255, 0, 0],
        #             ),
        #             pdk.Layer(
        #                 'ScatterplotLayer',
        #                 data=df_geo.loc[(df_geo['search term'] == 'monkey wild')],
        #                 get_position='[longitude, latitude]',
        #                 pickable=False,
        #                 opacity=0.8,
        #                 stroked=True,
        #                 filled=True,
        #                 radius_scale=6,
        #                 radius_min_pixels=3,
        #                 radius_max_pixels=100,
        #                 line_width_min_pixels=1,
        #                 get_radius="exits_radius",
        #                 get_fill_color=[0, 255, 0],
        #                 get_line_color=[0, 255, 0],
        #             ),
        #         ],
        #     ))
        #     expand1.write('The map shows the scatter-plot of the datapoints')
        #     expand1.markdown('* **location** by latitude, longitude coordinates')
        #     expand1.markdown('* **search term**  by color')
        #     expand1.write({"monkey wild": "green", "monkey cage": "red"})


    

if __name__ == "__main__":
    main()

