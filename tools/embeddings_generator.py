
# https://huggingface.co/M-CLIP/XLM-Roberta-Large-Vit-L-14

from PIL import Image
import requests
from transformers import CLIPProcessor, CLIPModel, CLIPVisionModel, CLIPTextModel
import transformers
from multilingual_clip import pt_multilingual_clip
import torch
import glob
import pandas as pd
from  tqdm import tqdm
from pathlib import Path

MULTILING = True
BS = 128  # batch size although does not seem to differ too much

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
path = Path('images')

if MULTILING:
    model_name = 'openai/clip-vit-large-patch14'
else:
    model_name = "openai/clip-vit-base-patch32"   # preconfigured with image size = 224: https://huggingface.co/openai/clip-vit-base-patch32/blob/main/preprocessor_config.json
    # model_name = "openai/clip-vit-large-patch14-336"  # preconfigured with image size = 336: https://huggingface.co/openai/clip-vit-large-patch14-336/blob/main/preprocessor_config.json

# Load Model & Tokenizer
model = CLIPModel.from_pretrained(model_name)
processor = CLIPProcessor.from_pretrained(model_name)

model.to(device)


def get_image_feats(images):
    input_images = processor(text=None, images=images, return_tensors="pt", padding=True).to(device)
    output_images_features = model.get_image_features(**input_images).detach()  # don't keep grad data and avoid run out of memory
    images_embeds = output_images_features / output_images_features.norm(p=2, dim=-1, keepdim=True)  # normalized features
    return images_embeds.cpu()


feats=[]
paths = []
for fn in tqdm(path.rglob('*.*')):
    try:
        image = Image.open(fn)
    except:
        print(f'Failed to open {fn}')
        continue
    paths.append(fn)
    feats.append(get_image_feats(image))

df = pd.DataFrame(zip(paths, feats), columns=['path', 'features'])

df.to_pickle(str(path).replace('/', '-')+'.pickle')




