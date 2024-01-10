import os
import json
from collections import Counter
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import Dataset
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from madgrad import MADGRAD
from sklearn.metrics import f1_score, accuracy_score,classification_report
import pandas as pd
import clip
from matplotlib import pyplot as plt
import file_utils

from datetime import datetime
#from tqdm.notebook import tqdm
from tqdm.autonotebook import tqdm

from transformers import (
    AutoConfig,
    AutoModel,
    AutoTokenizer,
    MMBTConfig,
    MMBTModel,
    MMBTForClassification,
    get_linear_schedule_with_warmup,
)
num_image_embeds = 4
num_labels = 1
gradient_accumulation_steps = 10
data_dir = './data/well_distributed_data'
#data_dir = '/nfs/data/cleopatra/misogyny_detection/MAMI_2021'
now = datetime.now()
model_dir_path = now.strftime("%d-%m-%Y %H;%M;%S").replace(" ", "_")

if not file_utils.path_exists(data_dir+ "/" + "mmbt_results/"):
    file_utils.create_folder(data_dir+ "/" + "mmbt_results/")

if not file_utils.path_exists(data_dir+ "/" + "mmbt_results"+model_dir_path):
    file_utils.create_folder(data_dir+ "/" + "mmbt_results"+model_dir_path)
model_dir_path = data_dir+ "/" + "mmbt_results"+model_dir_path

max_seq_length = 80
max_grad_norm = 0.5
train_batch_size = 32
eval_batch_size = 32
image_encoder_size = 288
image_features_size = 640
num_train_epochs = 2

def prepare_test_predictions(predictions):
    reports = classification_report(predictions['labels'], predictions['prediction'], target_names=['fake', 'real'], output_dict=True)
    reports['avg precision'] = reports['weighted avg']['precision']
    reports['avg f1'] = reports['weighted avg']['f1-score']
    reports['avg recall'] = reports['weighted avg']['recall']
    reports['macro-precision'] = reports['macro avg']['precision']
    reports['macro-f1'] = reports['macro avg']['f1-score']
    reports['macro-recall'] = reports['macro avg']['recall']
    reports['avg_loss'] = predictions['loss']
    del reports['macro avg']
    del reports['weighted avg']


    # Save output Labels
    preds=predictions['prediction'].squeeze().astype(int)
    id = list(range(len(preds)))
    out = {'id': id, 'targets': predictions['labels'].squeeze(), 'predictions': preds}
    outdf = pd.DataFrame(data=out)
#    outdf.to_csv(eval_output_dir + '/labels.csv', index=False)
    return outdf,reports

def slice_image(im, desired_size):
    '''
    Resize and slice image
    '''
    old_size = im.size

    ratio = float(desired_size) / min(old_size)
    new_size = tuple([int(x * ratio) for x in old_size])

    im = im.resize(new_size, Image.ANTIALIAS)

    ar = np.array(im)
    images = []
    if ar.shape[0] < ar.shape[1]:
        middle = ar.shape[1] // 2
        half = desired_size // 2

        images.append(Image.fromarray(ar[:, :desired_size]))
        images.append(Image.fromarray(ar[:, middle - half:middle + half]))
        images.append(Image.fromarray(ar[:, ar.shape[1] - desired_size:ar.shape[1]]))
    else:
        middle = ar.shape[0] // 2
        half = desired_size // 2

        images.append(Image.fromarray(ar[:desired_size, :]))
        images.append(Image.fromarray(ar[middle - half:middle + half, :]))
        images.append(Image.fromarray(ar[ar.shape[0] - desired_size:ar.shape[0], :]))

    return images

def resize_pad_image(im, desired_size):
    '''
    Resize and pad image to a desired size
    '''
    old_size = im.size

    ratio = float(desired_size)/max(old_size)
    new_size = tuple([int(x*ratio) for x in old_size])

    im = im.resize(new_size, Image.ANTIALIAS)

    # create a new image and paste the resized on it
    new_im = Image.new("RGB", (desired_size, desired_size))
    new_im.paste(im, ((desired_size-new_size[0])//2,
                        (desired_size-new_size[1])//2))

    return new_im

class ClipEncoderMulti(nn.Module):
    def __init__(self, num_embeds, num_features=image_features_size):
        super().__init__()
        self.model = clip_model
        self.num_embeds = num_embeds
        self.num_features = num_features

    def forward(self, x):
        # 4x3x288x288 -> 1x4x640
        out = self.model.encode_image(x.view(-1,3,288,288))
        out = out.view(-1, self.num_embeds, self.num_features).float()
        return out  # Bx4x640

class JsonlDataset(Dataset):
    def __init__(self, data_path, csv_file_path, images_dir, tokenizer, transforms, max_seq_length):
        # self.data = [json.loads(l) for l in open(data_path)]
        self.data = pd.read_csv(data_path + '/' + csv_file_path)
        # self.data = self.data.head(100)

        self.data_dir = data_path
        self.images_dir = images_dir
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length

        self.transforms = transforms

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        text = self.data['tweetText'][index]
        image_path = os.path.join(self.data_dir, self.images_dir, self.data["imageId(s)"][index])

        sentence = torch.LongTensor(self.tokenizer.encode(text, add_special_tokens=True))
        start_token, sentence, end_token = sentence[0], sentence[1:-1], sentence[-1]
        sentence = sentence[:self.max_seq_length]

        if 'label' in self.data:
            label = torch.FloatTensor([self.data['label'][index]])
        else:
            label = torch.FloatTensor([0])

        image = Image.open(image_path).convert("RGB")
        sliced_images = slice_image(image, 288)
        sliced_images = [np.array(self.transforms(im)) for im in sliced_images]
        image = resize_pad_image(image, image_encoder_size)
        image = np.array(self.transforms(image))

        sliced_images = [image] + sliced_images
        sliced_images = torch.from_numpy(np.array(sliced_images)).to(device)

        return {
            "image_start_token": start_token,
            "image_end_token": end_token,
            "sentence": sentence,
            "image": sliced_images,
            "label": label
        }

    def get_label_frequencies(self):
        label_freqs = Counter()
        for row in self.data:
            label_freqs.update([row["label"]])
        return label_freqs

    def get_labels(self):
        labels = []
        for row in self.data:
            labels.append(row["label"])
        return labels

def collate_fn(batch):
    lens = [len(row["sentence"]) for row in batch]
    bsz, max_seq_len = len(batch), max(lens)

    mask_tensor = torch.zeros(bsz, max_seq_len, dtype=torch.long)
    text_tensor = torch.zeros(bsz, max_seq_len, dtype=torch.long)

    for i_batch, (input_row, length) in enumerate(zip(batch, lens)):
        text_tensor[i_batch, :length] = input_row["sentence"]
        mask_tensor[i_batch, :length] = 1

    img_tensor = torch.stack([row["image"] for row in batch])
    tgt_tensor = torch.stack([row["label"] for row in batch])
    img_start_token = torch.stack([row["image_start_token"] for row in batch])
    img_end_token = torch.stack([row["image_end_token"] for row in batch])

    return text_tensor, mask_tensor, img_tensor, img_start_token, img_end_token, tgt_tensor

def load_examples(tokenizer, split_name, images_dir):
    transforms = preprocess
    dataset = JsonlDataset(data_dir, split_name, images_dir, tokenizer, transforms, max_seq_length - num_image_embeds - 2)
    return dataset

def save_checkpoint(save_path, model, valid_loss):
    if save_path == None:
        return

    state_dict = {'model_state_dict': model.state_dict(),
                  'valid_loss': valid_loss}

    torch.save(state_dict, save_path+"/best.pt")
    print(f'Model saved to ==> {save_path}')

def pltorAcc_loss(History,outdir):
    plt.style.use("ggplot")
    plt.figure()
    plt.plot(H["train_loss"], label="train_loss")
    plt.plot(H["val_loss"], label="val_loss")
    #plt.plot(H["train_acc"], label="train_accuracy")
    plt.plot(H["val_acc"], label="valid_accuracy")
    plt.title("Training Loss and Accuracy on Dataset")
    plt.xlabel("Epoch #")
    plt.ylabel("Loss/Accuracy")
    plt.legend(loc="lower left")
    plt.savefig(outdir + '/loss_accuracy.pdf')

def load_checkpoint(load_path, model):
    if load_path == None:
        return

    state_dict = torch.load(load_path, map_location=device)
    print(f'Model loaded from <== {load_path}')

    model.load_state_dict(state_dict['model_state_dict'])
    return state_dict['valid_loss']

def evaluate(model, tokenizer, criterion, dataloader, tres=0.5):
    # Eval!
    eval_loss = 0.0
    nb_eval_steps = 0
    preds = None
    proba = None
    out_label_ids = None
    for batch in dataloader:
        model.eval()
        batch = tuple(t.to(device) for t in batch)
        with torch.no_grad():
            labels = batch[5]
            inputs = {
                "input_ids": batch[0],
                "input_modal": batch[2],
                "attention_mask": batch[1],
                "modal_start_tokens": batch[3],
                "modal_end_tokens": batch[4],
                "return_dict": False
            }
            outputs = model(**inputs)
            logits = outputs[0]  # model outputs are always tuple in transformers (see doc)
            tmp_eval_loss = criterion(logits, labels)
            eval_loss += tmp_eval_loss.mean().item()
        nb_eval_steps += 1
        if preds is None:
            preds = torch.sigmoid(logits).detach().cpu().numpy() > tres
            proba = torch.sigmoid(logits).detach().cpu().numpy()
            out_label_ids = labels.detach().cpu().numpy()
        else:
            preds = np.append(preds, torch.sigmoid(logits).detach().cpu().numpy() > tres, axis=0)
            proba = np.append(proba, torch.sigmoid(logits).detach().cpu().numpy(), axis=0)
            out_label_ids = np.append(out_label_ids, labels.detach().cpu().numpy(), axis=0)

    eval_loss = eval_loss / nb_eval_steps

    result = {
        "loss": eval_loss,
        "accuracy": accuracy_score(out_label_ids, preds),
        "macro_f1": f1_score(out_label_ids, preds, average="macro"),
        "prediction": preds,
        "labels": out_label_ids,
        "proba": proba
    }

    return result

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

clip_model, preprocess = clip.load("RN50x4", device=device, jit=False)
for p in clip_model.parameters():
    p.requires_grad = False

model_name = 'bert-base-uncased'
transformer_config = AutoConfig.from_pretrained(model_name)
transformer = AutoModel.from_pretrained(model_name, config=transformer_config)
img_encoder = ClipEncoderMulti(num_image_embeds)

tokenizer = AutoTokenizer.from_pretrained(model_name, do_lower_case=True)

config = MMBTConfig(transformer_config, num_labels=num_labels, modal_hidden_size=image_features_size)
model = MMBTForClassification(config, transformer, img_encoder)

model.to(device)

train_dataset = load_examples(tokenizer, 'train.csv', 'train_images')
eval_dataset = load_examples(tokenizer, 'valid.csv', 'valid_images')
test_dataset = load_examples(tokenizer, 'test.csv', 'test_images')

train_sampler = RandomSampler(train_dataset)
eval_sampler = SequentialSampler(eval_dataset)
test_sampler = SequentialSampler(test_dataset)

train_dataloader = DataLoader(
        train_dataset,
        sampler=train_sampler,
        batch_size=train_batch_size,
        collate_fn=collate_fn
)


eval_dataloader = DataLoader(
        eval_dataset,
        sampler=eval_sampler,
        batch_size=eval_batch_size,
        collate_fn=collate_fn)

test_dataloader = DataLoader(
        test_dataset,
        sampler=test_sampler,
        batch_size=eval_batch_size,
        collate_fn=collate_fn)

no_decay = ["bias",
            "LayerNorm.weight"
           ]
weight_decay = 0.0005

optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": weight_decay,
        },
        {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
    ]

t_total = (len(train_dataloader) // gradient_accumulation_steps) * num_train_epochs
warmup_steps = t_total // 10
learning_rate=2e-4

optimizer = MADGRAD(optimizer_grouped_parameters, lr=learning_rate)

scheduler = get_linear_schedule_with_warmup(
        optimizer, warmup_steps, t_total
    )

criterion = nn.BCEWithLogitsLoss()

optimizer_step = 0
global_step = 0
train_step = 0
tr_loss, logging_loss = 0.0, 0.0
best_valid_f1 = 0.0
global_steps_list = []
train_loss_list = []
val_loss_list = []
val_acc_list = []
val_f1_list = []
eval_every = len(train_dataloader) // 2
running_loss = 0
file_path = "models/"

model.zero_grad()

best_model = model

for i in range(num_train_epochs):
    print("Epoch", i + 1, f"from {num_train_epochs}")
    whole_y_pred = np.array([])
    whole_y_t = np.array([])
    for step, batch in enumerate(tqdm(train_dataloader)):
        model.train()
        batch = tuple(t.to(device) for t in batch)
        labels = batch[5]
        inputs = {
            "input_ids": batch[0],
            "input_modal": batch[2],
            "attention_mask": batch[1],
            "modal_start_tokens": batch[3],
            "modal_end_tokens": batch[4],
            "return_dict": False
        }
        outputs = model(**inputs)
        logits = outputs[0]  # model outputs are always tuple in transformers (see doc)
        loss = criterion(logits, labels)

        if gradient_accumulation_steps > 1:
            loss = loss / gradient_accumulation_steps

        loss.backward()

        tr_loss += loss.item()
        running_loss += loss.item()
        global_step += 1

        if (step + 1) % gradient_accumulation_steps == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()
            scheduler.step()  # Update learning rate schedule

            optimizer_step += 1
            optimizer.zero_grad()

        if (step + 1) % eval_every == 0:

            average_train_loss = running_loss / eval_every
            train_loss_list.append(average_train_loss)
            global_steps_list.append(global_step)
            running_loss = 0.0

            val_result = evaluate(model, tokenizer, criterion, eval_dataloader)

            val_loss_list.append(val_result['loss'])
            val_acc_list.append(val_result['accuracy'])
            val_f1_list.append(val_result['macro_f1'])

            # checkpoint
            if val_result['macro_f1'] > best_valid_f1:
                best_valid_f1 = val_result['macro_f1']
                val_loss = val_result['loss']
                val_acc = val_result['accuracy']
                best_model = model
                # model_path = f'{file_path}/model-embs{num_image_embeds}-seq{max_seq_length}-auc{best_valid_f1:.3f}-loss{val_loss:.3f}-acc{val_acc:.3f}.pt'
                print(f"AUC improved, so saving this model")
                save_checkpoint(model_dir_path, best_model, val_result['loss'])

            print("Train loss:", f"{average_train_loss:.4f}",
                  "Val loss:", f"{val_result['loss']:.4f}",
                  "Val macro_f1:", f"{val_result['macro_f1']:.4f}")
    print('\n')

test_result = evaluate(best_model, tokenizer, criterion, test_dataloader)
labels_results,Creports = prepare_test_predictions(test_result)
labels_results.to_csv(model_dir_path + '/labels.csv',index=False)
output_test_file = os.path.join(model_dir_path, "Test_results.json")
with open(output_test_file, "w") as fp:
    json.dump(Creports, fp)

plt.plot(global_steps_list, val_f1_list)
plt.grid()
plt.xlabel('Global Steps')
plt.ylabel('AUC')
plt.title('MMBT Macro F1')
plt.savefig(model_dir_path + "/history.png")
# plt.show()

    # update our training history
H = {
    "train_loss": [],
    "train_acc": [],
    "val_loss": [],
    "val_acc": []
}

H["train_loss"]=train_loss_list
#H["train_acc"].append(train_Acc)
H["val_loss"]=val_loss_list
H["val_acc"]=val_acc_list

#plot ACC and Loss
pltorAcc_loss(H, model_dir_path)
CFG={
    'learn_rate' : learning_rate,
    'max_Seq':max_seq_length,
    'max_gradNorm':max_grad_norm,
    'batch_train':train_batch_size,
    'batch_eval':eval_batch_size,
    'size_image_encode':image_encoder_size,
    'size_image_feature':image_features_size,
    'train_epoch':num_train_epochs,
    'eval_every':eval_every,
    "weight_decay": weight_decay,
    "text_model_name" : model_name,
    "warmup_steps":warmup_steps
}
cfg_df=pd.DataFrame(CFG,index=[0])
cfg_path=os.path.join(model_dir_path,'CFG.csv')
cfg_df.to_csv(cfg_path,index=False)

print("program completed")