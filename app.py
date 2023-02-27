
####### Install dependencies #######
# pip install spacy ftfy==4.4.3
# python -m spacy download en
# conda install -c huggingface transformers==4.14.1 tokenizers==0.10.3 see:
# https://discuss.huggingface.co/t/importing-tokenizers-version-0-10-3-fails-due-to-openssl/17820/3
# pip install gradio

import datetime
import gradio as gr

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
    for i, idx in enumerate(idxs[0]):
        fn = df.iloc[idx, df.columns.get_loc('path')]
        try:    
            image = Image.open(fn)
        except Exception as e:
            print(f'Could not open the image {fn}')
            print(f'The error was: {str(e)}')
        images.append(image)
        print(fn)
        # images.append(str(fn))
        # images.append(str(probs[0][i]))
    print('done reading the images')
    return images


    
    
css = """
        .gradio-container {
            font-family: 'IBM Plex Sans', sans-serif;
        }
        .gr-button {
            color: white;
            border-color: black;
            background: black;
        }
        input[type='range'] {
            accent-color: black;
        }
        .dark input[type='range'] {
            accent-color: #dfdfdf;
        }
        .container {
            max-width: 1070px;
            margin: auto;
            padding-top: 2rem;
        }
        #gallery {
            min-height: 22rem;
            margin-bottom: 15px;
            border-bottom-right-radius: .5rem !important;
            border-bottom-left-radius: .5rem !important;
        }
        #gallery>div>.h-full {
            min-height: 20rem;
        }
        .details:hover {
            text-decoration: underline;
        }
        .gr-button {
            white-space: nowrap;
        }
        .gr-button:focus {
            border-color: rgb(147 197 253 / var(--tw-border-opacity));
            outline: none;
            box-shadow: var(--tw-ring-offset-shadow), var(--tw-ring-shadow), var(--tw-shadow, 0 0 #0000);
            --tw-border-opacity: 1;
            --tw-ring-offset-shadow: var(--tw-ring-inset) 0 0 0 var(--tw-ring-offset-width) var(--tw-ring-offset-color);
            --tw-ring-shadow: var(--tw-ring-inset) 0 0 0 calc(3px var(--tw-ring-offset-width)) var(--tw-ring-color);
            --tw-ring-color: rgb(191 219 254 / var(--tw-ring-opacity));
            --tw-ring-opacity: .5;
        }
        #advanced-btn {
            font-size: .7rem !important;
            line-height: 19px;
            margin-top: 24px;
            margin-bottom: 12px;
            padding: 2px 8px;
            border-radius: 14px !important;
        }
        #advanced-options {
            display: none;
            margin-bottom: 20px;
        }
        .footer {
            margin-bottom: 45px;
            margin-top: 35px;
            text-align: center;
            border-bottom: 1px solid #e5e5e5;
        }
        .footer>p {
            font-size: .8rem;
            display: inline-block;
            padding: 0 10px;
            transform: translateY(10px);
            background: white;
        }
        .dark .footer {
            border-color: #303030;
        }
        .dark .footer>p {
            background: #0b0f19;
        }
        .acknowledgments h4{
            margin: 1.25em 0 .25em 0;
            font-weight: bold;
            font-size: 115%;
        }
"""

block = gr.Blocks(css=css)


with block:
    gr.HTML(
        """
            <div style="text-align: center; max-width: 736px; margin: 0 auto;">
              <div
                style="
                  display: inline-flex;
                  align-items: center;
                  gap: 0.8rem;
                  font-size: 1.75rem;
                "
              >
                <svg
                  width="0.65em"
                  height="0.65em"
                  viewBox="0 0 115 115"
                  fill="none"
                  xmlns="http://www.w3.org/2000/svg"
                >
                  <rect width="23" height="23" fill="white"></rect>
                  <rect y="69" width="23" height="23" fill="white"></rect>
                  <rect x="23" width="23" height="23" fill="#AEAEAE"></rect>
                  <rect x="23" y="69" width="23" height="23" fill="#AEAEAE"></rect>
                  <rect x="46" width="23" height="23" fill="white"></rect>
                  <rect x="46" y="69" width="23" height="23" fill="white"></rect>
                  <rect x="69" width="23" height="23" fill="black"></rect>
                  <rect x="69" y="69" width="23" height="23" fill="black"></rect>
                  <rect x="92" width="23" height="23" fill="#D9D9D9"></rect>
                  <rect x="92" y="69" width="23" height="23" fill="#AEAEAE"></rect>
                  <rect x="115" y="46" width="23" height="23" fill="white"></rect>
                  <rect x="115" y="115" width="23" height="23" fill="white"></rect>
                  <rect x="115" y="69" width="23" height="23" fill="#D9D9D9"></rect>
                  <rect x="92" y="46" width="23" height="23" fill="#AEAEAE"></rect>
                  <rect x="92" y="115" width="23" height="23" fill="#AEAEAE"></rect>
                  <rect x="92" y="69" width="23" height="23" fill="white"></rect>
                  <rect x="69" y="46" width="23" height="23" fill="white"></rect>
                  <rect x="69" y="115" width="23" height="23" fill="white"></rect>
                  <rect x="69" y="69" width="23" height="23" fill="#D9D9D9"></rect>
                  <rect x="46" y="46" width="23" height="23" fill="black"></rect>
                  <rect x="46" y="115" width="23" height="23" fill="black"></rect>
                  <rect x="46" y="69" width="23" height="23" fill="black"></rect>
                  <rect x="23" y="46" width="23" height="23" fill="#D9D9D9"></rect>
                  <rect x="23" y="115" width="23" height="23" fill="#AEAEAE"></rect>
                  <rect x="23" y="69" width="23" height="23" fill="black"></rect>
                </svg>
               <h1 style="font-weight: 900; margin-bottom: 7px;">
                  Semantic Image Search v.3 [with filenames]
                </h1>
              </div>
              <p style="margin-bottom: 20px;">
                HELICS 2022
                
              </p>
            </div>
        """
    )
    with gr.Group():
        with gr.Box():
            with gr.Row().style(mobile_collapse=False, equal_height=True):
                text = gr.Textbox(
                    label="Enter your prompt",
                    show_label=False,
                    max_lines=1,
                    placeholder="Enter your prompt",
                ).style(
                    border=(True, False, True, True),
                    rounded=(True, False, False, True),
                    container=False,
                )
                btn = gr.Button("Find image").style(
                    margin=False,
                    rounded=(False, True, True, False),
                )

        gallery = gr.Gallery(
            label="images", show_label=False, elem_id="gallery"
        ).style(grid=(2,6), height="auto")

        with gr.Row():
            img_cnt = gr.Slider(label="Images", minimum=1, maximum=100, value=4, step=1) 
            
       
        text.submit(infer, inputs=[text, img_cnt], outputs=gallery)
        btn.click(infer, inputs=[text, img_cnt], outputs=gallery)
        # advanced_button.click(
            # None,
            # [],
            # text,
            # _js="""
            # () => {
                # const options = document.querySelector("body > gradio-app").querySelector("#advanced-options");
                # options.style.display = ["none", ""].includes(options.style.display) ? "flex" : "none";
            # }""",
        # )


###### Did not work to avoid issues of port not closed in previous run. Gradio seems still buggy and leave the port open ######
# the solution is to :
# vscode is automatically port forwarding when i use it as remote dev., so I can leave block.launch without specifying the port
# but in production I should specify the port

# while True:  
#     cnt = 0
#     try:
#         block.launch(server_name='0.0.0.0', server_port = 7862, auth=("hwasiti", "helics"))  # cannot use .queue(max_size=40) with password
#     except KeyboardInterrupt:
#         print('Interrupted by keyboard')
#     except Exception as e:
#         print(f'An Exception occured during launching the gradio server. Which is: {str(e)}')
#         print()
#         print('Will close the port and try again')
#         block.close()
#         cnt +=1
#         print(f'Attempt {cnt} at {datetime.datetime.now().isoformat()}')



##### in production I should specify the port. Add the arg: server_port = 7862 #####
block.launch(server_name='0.0.0.0', auth=("hwasiti", "helics"))  # cannot use .queue(max_size=40) with password