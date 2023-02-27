
from PIL import Image
import requests
from transformers import CLIPProcessor, CLIPModel, CLIPVisionModel, CLIPTextModel
import torch
import glob
import pandas as pd
from  tqdm import tqdm
from pathlib import Path

path = Path('images')

BS = 128  # batch size although does not seem to differ too much
model_id = "openai/clip-vit-base-patch32"   # preconfigured with image size = 224: https://huggingface.co/openai/clip-vit-base-patch32/blob/main/preprocessor_config.json
# model_id = "openai/clip-vit-large-patch14-336"  # preconfigured with image size = 336: https://huggingface.co/openai/clip-vit-large-patch14-336/blob/main/preprocessor_config.json

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CLIPModel.from_pretrained(model_id)
model.to(device)
processor = CLIPProcessor.from_pretrained(model_id)


def get_images_feats(images):
    input_images = processor(text=None, images=images, return_tensors="pt", padding=True).to(device)
    output_images_features = model.get_image_features(**input_images).detach()  # don't keep grad data and avoid run out of memory
    images_embeds = output_images_features / output_images_features.norm(p=2, dim=-1, keepdim=True)  # normalized features
    return images_embeds.cpu()


def get_image_feats(image):
    input_image = processor(text=None, images=image, return_tensors="pt", padding=True).to(device)
    output_image_features = model.get_image_features(**input_image).detach()
    image_embeds = output_image_features / output_image_features.norm(p=2, dim=-1, keepdim=True)  # normalized features
    return image_embeds.cpu()


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




