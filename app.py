####### Install dependencies #######
# pip install spacy ftfy==4.4.3
# python -m spacy download en
# conda install -c huggingface transformers==4.14.1 tokenizers==0.10.3 see:
# https://discuss.huggingface.co/t/importing-tokenizers-version-0-10-3-fails-due-to-openssl/17820/3
# pip install streamlit
# pip install multilingual-clip

### Multilingial CLIP model:
# https://huggingface.co/M-CLIP/XLM-Roberta-Large-Vit-L-14
# pip install transformers==4.8  #### this version or older is important to avoid an error when loading the model
# pip install gradio

# import datetime
import gradio as gr

from PIL import Image

from transformers import CLIPProcessor, CLIPModel
import transformers
from multilingual_clip import pt_multilingual_clip
import torch
from pathlib import Path
import pandas as pd
from  tqdm import tqdm
from functools import partial
tqdm.pandas()

# global last_multiling_chkbx, multiling
# last_multiling_chkbx = None 
# multiling = True

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")   # There is a bug when using gpu: tensors should be on the same device not on cuda:0 and cpu
device = torch.device("cpu")

class ClpModel:
    def __init__(self, multiling = True):
        self.multiling = multiling
        self.model = None
        self.processor = None
        self.img_embs = None
        self.logit_scale = None
        self.last_multiling_chkbx = None 
        self.df = None
        
    def load_model_and_emb(self, multiling = True):
        self.last_multiling_chkbx = multiling
        path = Path('images')

        if multiling:
            model_name = 'M-CLIP/XLM-Roberta-Large-Vit-L-14'
            img_emb_from_model = 'openai/clip-vit-large-patch14'
        else:
            model_name = "openai/clip-vit-base-patch32"   # preconfigured with image size = 224: https://huggingface.co/openai/clip-vit-base-patch32/blob/main/preprocessor_config.json
            # model_name = "openai/clip-vit-large-patch14-336"  # preconfigured with image size = 336: https://huggingface.co/openai/clip-vit-large-patch14-336/blob/main/preprocessor_config.json
            img_emb_from_model = model_name

        # Load Model & Tokenizer
        if multiling:
            self.model = pt_multilingual_clip.MultilingualCLIP.from_pretrained(model_name)
            self.processor = transformers.AutoTokenizer.from_pretrained(model_name)
        else:
            self.model = CLIPModel.from_pretrained(model_name)
            self.processor = CLIPProcessor.from_pretrained(model_name)
        self.model.to(device)

        fname = str(path).replace('/', '-') + '_' + img_emb_from_model.split('/')[-1]
        self.df = pd.read_pickle(fname + '.pickle')
        self.img_embs = torch.stack(self.df.features.values.tolist())[:, -1, :].t().to(device)
        self.logit_scale = self.model.logit_scale.exp() if not multiling else torch.tensor(100., dtype=torch.float32).to(device)

MyClpModel = ClpModel()

def compute_probs_and_sort(text_embeds, n):
    preds = torch.matmul(text_embeds.detach(), MyClpModel.img_embs) * MyClpModel.logit_scale.detach()  # compute cosine similarity * 100 (100 is perfectly text matched to image)
    print(f'max, min cosine similarity of all images with the text prompt: {preds.max()} , {preds.min()}')
    sorted, indices = torch.sort(preds, descending=True)
    if device == 'cpu':
        probs = sorted[:, :n].numpy()
        idxs = indices[:, :n].numpy()
    else:
        probs = sorted[:, :n].cpu().numpy()
        idxs = indices[:, :n].cpu().numpy()
    return probs, idxs


def infer(prompt, img_cnt, chkbx):
    if type(chkbx) != bool:
        multiling = chkbx.value
    else:
        multiling = chkbx
    
    if type(img_cnt) != int:
        image_count = img_cnt.value
    else:
        image_count = img_cnt

    if multiling != MyClpModel.last_multiling_chkbx:
        MyClpModel.load_model_and_emb(multiling)
        # last_multiling_chkbx = multiling
    # input_text = processor(text=[prompt], return_tensors="pt", padding=True).to(device)

    if multiling:
        output_text_features = MyClpModel.model.forward([prompt], MyClpModel.processor)
    else:
        input_text = MyClpModel.processor(text=[prompt], return_tensors="pt", padding=True).to(device)
        output_text_features = MyClpModel.model.get_text_features(**input_text)
    text_embeds = output_text_features / output_text_features.norm(p=2, dim=-1, keepdim=True)  
    probs, idxs = compute_probs_and_sort(text_embeds, image_count)
    print('done compute probabilities to all images')

    
    images = []
    metadata = []
    images_metadata = []
    for i, idx in enumerate(idxs[0]):
        df = MyClpModel.df
        fn = df.iloc[idx, df.columns.get_loc('path')]
        try:    
            image = Image.open(fn)
        except Exception as e:
            print(f'Could not open the image {fn}')
            print(f'The error was: {str(e)}')
        images.append(image)
        metad = f'({probs[0][i]}) {fn}'
        print(metad)
        metadata.append(metad)
        images_metadata.append((image, metad))
    print('done reading the images')
    return  images  # images_metadata # https://github.com/gradio-app/gradio/pull/2284

    
    
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
MyClpModel = ClpModel()

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
                  Semantic Medical Image Search
                </h1>
              </div>
              <p style="margin-bottom: 20px;">
                Search images of skin diseases using natural language                
              </p>
            </div>
        """
    )
    with gr.Group():
        with gr.Box():
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
            multiling_checkbox = gr.Checkbox(label="Multilingual (68 languages)", value=True, interactive=True) 

        gallery = gr.Gallery(
            label="images", show_label=False, elem_id="gallery"
        ).style(grid=(2,6), height="auto")

        with gr.Row():
            img_cnt = gr.Slider(label="Images", minimum=1, maximum=100, value=8, step=1) 
       
        text.submit(infer, inputs=[text, img_cnt, multiling_checkbox], outputs=gallery)
        btn.click(infer, inputs=[text, img_cnt, multiling_checkbox], outputs=gallery)
        # multiling_checkbox.change(load_model_and_emb, inputs=[multiling_checkbox.value], outputs=[])  # checkbox change event listener seems has a bug
        
        # Example inputs
        # Nice code example: https://huggingface.co/spaces/hysts/sample-008
        examples = [
            ['English', 'hands with rash'], 
            ['Finnish (skin rash on legs)', 'ihottuma jaloissa'],
            ['Swedish (abdomen with urticaria)', 'buken med urtikaria'], 
            ['Swedish (swollen eyes)', 'svullna ögon'],
            ['English', 'nail fungal infection'], 
            ['Finnish (opened mouth)', 'suu auki'],
            ['Arabic (skin bacterial infection)', 'عدوى الجلد البكتيرية'], 
            ['Greek (tattoo or marks on the skin)', 'τατουάζ ή σημάδια στο δέρμα'],
            ['Japanese (circular patches)', '円形パッチ']
            ]
        
        # Just a way to receive the language cols from the examples
        text_dummy = gr.Textbox(
                        label="Language",
                        show_label=False,
                        max_lines=1,
                        placeholder="",
                        visible=False
                    )


        def partial_infer(text_dummy, text):
            return infer(text, img_cnt=img_cnt, chkbx=multiling_checkbox)

        with gr.Row():
            with gr.Box():
                gr.Examples(examples=examples, inputs=[text_dummy , text], examples_per_page=10,
                            fn = partial_infer, outputs=gallery, run_on_click=False, cache_examples=False)  # No cache so that the examples can be updated and the model can be rerun if the checkbox is changed

                    
        with gr.Row():
            with gr.Box():
                gr.Markdown(''' 
                - **TODO:**<br />
                    [   ] Add more images to the database.<br />
                    [   ] Add a threshold to filter out unrelated images.<br />
                - The model lists the nearest images to the given prompt starting from top left to bottom right row-wise. Even if the prompt is not in English, the multi-lingual model will be able to search for the nearest images when the multilingual option is enabled.  
                - The database has only ~5k images for test purposes without any text or keywords. If no images are found, the model will try to search for the nearest images in the database. Due to the small size of the database, the model may not find any related images but it will still return the nearest images in the database. 
                - **Multilingual DL model supports 68 languages:** Afrikaans, Albanian, Amharic, Arabic, Armenian, Azerbaijani, Bengali, Bosnian, Bulgarian, Catalan, Chinese Simplified, Chinese Traditional, Croatian, Czech, Danish, Dutch, English, Estonian, Finnish, French, Georgian, German, Greek, Gujarati, Haitian, Hausa, Hebrew, Hindi, Hungarian, Icelandic, Indonesian, Italian, Japanese, Kannada, Kazakh, Korean, Latvian, Lithuanian, Macedonian, Malay, Malayalam, Maltese, Mongolian, Norwegian, Persian, Polish, Pushto, Portuguese, Romanian, Russian, Serbian, Sinhala, Slovak, Slovenian, Somali, Spanish, Swahili, Swedish, Tagalog, Tamil, Telugu, Thai, Turkish, Ukrainian, Urdu, Uzbek, Vietnamese, Welsh
                - **Contact for more information:** Haider Alwasiti at: [haider.alwasiti@helsinki.fi](mailto:haider.alwasiti@helsinki.fi)
                            ''')
        

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
block.launch(server_name='0.0.0.0') #, auth=("hwasiti", "helics"))  # cannot use .queue(max_size=40) with password