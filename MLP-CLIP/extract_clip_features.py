import torch
import text_normalizer
import clip
from PIL import Image
import pandas as pd
import numpy as np

def check_number_of_tokens(text: str):

    start = 0
    end = int(len(text) / 2)
    output_text = text[start:end]

    return output_text.strip()

def extract_features(df, base_dir, model, preprocess):

    ids = df["image"].tolist()
    image_text_docs = df["claim"].tolist()
    #static_text='This is a Sentence.'
    #static_image='./blank.png'

    print(len(ids))

    features = []

    for id in ids:
        if ids.index(id)%300==0:
            print(ids.index(id))
        #image_path = base_dir + 'images_test/' + str(id)+'.jpg'
        image_path = base_dir + 'images/'+ str(id)
        #image_path = static_image
        raw_image_text = image_text_docs[ids.index(id)]
        #raw_image_text=static_text
        normalized_image_text = text_normalizer.preprocess(raw_image_text)
        image = preprocess(Image.open(image_path)).unsqueeze(0).to(device)

        while True:
            try:
                image_text = clip.tokenize(normalized_image_text).to(device)
                break
            except RuntimeError:
                normalized_image_text = check_number_of_tokens(normalized_image_text)

        with torch.no_grad():
            image_features = model.encode_image(image)
            image_text_features = model.encode_text(image_text)

            image_features = image_features.data.cpu().numpy()[0]
            image_text_features = image_text_features.data.cpu().numpy()[0]

            feature = np.concatenate((image_text_features, image_features))
            features.append(feature)

    return features

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device, jit=False)

dataset_path = r"C:\Users\tahmasebis\PycharmProjects/data/"
train_df = pd.read_csv(dataset_path +'train.csv')

print('Loaded the splits')
print('Processing training data')

train_features = extract_features(train_df, dataset_path, model, preprocess)

print('Saving training data')
np.save(dataset_path+'test_clip_features.npy', train_features)

