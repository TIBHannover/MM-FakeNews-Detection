import os
import cv2
import pandas as pd
import itertools
import sklearn.metrics
from tqdm.autonotebook import tqdm
import albumentations as A
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import torch
from torch import nn
import torch.nn.functional as F
import timm
from transformers import BertModel,BertConfig,BertTokenizer
import matplotlib.pyplot as plt
from datetime import datetime
import pickle


def get_top_data(df,top_n):
    top_data_df_positive = df[df['label'] == 'fake'].head(top_n)
    top_data_df_negative = df[df['label'] == 'real'].head(top_n)
    top_data_df_small = pd.concat([top_data_df_positive, top_data_df_negative])
    return top_data_df_small

class CFG:
    debug = False
    image_path = "./2015_final/devset_images"
    image_path_test = "./2015_final/testset_images"
    currentTime=datetime.now().strftime("%m-%d-%Y_%H;%M;%S")
    batch_size = 32
    head_lr = 1e-3
    image_encoder_lr = 1e-4
    text_encoder_lr = 1e-5
    weight_decay = 1e-3
    patience = 1
    factor = 0.8
    epochs = 1
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_name = 'resnet50'
    image_embedding = 2048
    text_encoder_model = "distilbert-base-uncased"
    text_embedding = 768
    text_tokenizer = "distilbert-base-uncased"
    max_length = 80

    pretrained = True # for both image encoder and text encoder
    trainable = True # for both image encoder and text encoder
    temperature = 1.0

    # image size
    size = 224

    # for projection head; used for both image and text encoders
    num_projection_layers = 1
    projection_dim = 256
    dropout = 0.1

class MYDataset(torch.utils.data.Dataset):
    def __init__(self, image_filenames, text,labels, tokenizer, transforms,datatype):

        self.image_filenames = image_filenames
        self.captions = list(text)
        self.encoded_captions = tokenizer(
            list(text), padding=True, truncation=True, max_length=CFG.max_length
        )
        self.transforms = transforms
        self.labels=labels
        self.datatype=datatype

    def __getitem__(self, idx):
        item = {
            key: torch.tensor(values[idx])
            for key, values in self.encoded_captions.items()
        }
        if self.datatype=='train'or self.datatype=='valid':

            image = cv2.imread(f"{CFG.image_path}/{self.image_filenames[idx]}.jpg")
        else:
            image= cv2.imread(f"{CFG.image_path_test}/{self.image_filenames[idx]}.jpg")
        if image is None:
            x=self.image_filenames[idx]
            print('This image doesnt exist')
            print(x)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        image = self.transforms(image=image)['image']
        item['image'] = torch.tensor(image).permute(2, 0, 1).float()
        item['tweetText'] = self.captions[idx]
        item['labels']=0 if self.labels[idx]=='fake' else 1

        return item


    def __len__(self):
        return len(self.captions)

def get_transforms(mode="train"):
    if mode == "train":
        return A.Compose(
            [
                A.Resize(CFG.size, CFG.size, always_apply=True),
                A.Normalize(max_pixel_value=255.0, always_apply=True),
            ]
        )
    else:
        return A.Compose(
            [
                A.Resize(CFG.size, CFG.size, always_apply=True),
                A.Normalize(max_pixel_value=255.0, always_apply=True),
            ]
        )

class ImageEncoder(nn.Module):
    """
    Encode images to a fixed size vector
    """

    def __init__(
        self, model_name=CFG.model_name, pretrained=CFG.pretrained, trainable=CFG.trainable
    ):
        super().__init__()
        self.model = timm.create_model(
            model_name, pretrained, num_classes=0, global_pool="avg"
        )
        for p in self.model.parameters():
            p.requires_grad = trainable

    def forward(self, x):
        return self.model(x)

class TextEncoder(nn.Module):
    def __init__(self, model_name=CFG.text_encoder_model, pretrained=CFG.pretrained, trainable=CFG.trainable):
        super().__init__()
        if pretrained:
            self.model = BertModel.from_pretrained(model_name)
        else:
            self.model = BertModel(config=BertConfig())

        for p in self.model.parameters():
            p.requires_grad = trainable

        # using the CLS token hidden representation as the sentence's embedding
        self.target_token_idx = 0

    def forward(self, input_ids, attention_mask):
        output = self.model(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden_state = output.last_hidden_state
        return last_hidden_state[:, self.target_token_idx, :]

class ProjectionHead(nn.Module):
    def __init__(
            self,
            embedding_dim,
            projection_dim=CFG.projection_dim,
            dropout=CFG.dropout
    ):
        super().__init__()
        self.projection = nn.Linear(embedding_dim, projection_dim)
        self.gelu = nn.GELU()
        self.fc = nn.Linear(projection_dim, projection_dim)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(projection_dim)

    def forward(self, x):
        projected = self.projection(x)
        x = self.gelu(projected)
        x = self.fc(x)
        x = self.dropout(x)
        x = x + projected
        x = self.layer_norm(x)
        return x

class AvgMeter:
    def __init__(self, name="Metric"):
        self.name = name
        self.reset()

    def reset(self):
        self.avg, self.sum, self.count = [0] * 3

    def update(self, val, count=1):
        self.count += count
        self.sum += val * count
        self.avg = self.sum / self.count

    def __repr__(self):
        text = f"{self.name}: {self.avg:.4f}"
        return text

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group["lr"]

def build_loaders(dataframe, tokenizer, mode):
    transforms = get_transforms(mode=mode)
    #dataframe['binary_labels'] = dataframe['label'].apply(lambda x : 1 if x == "fake" else  0)
    dataset = MYDataset(
        dataframe["image_id"].values,
        dataframe["post_text"].values,
        dataframe["label"].values,
        tokenizer=tokenizer,
        transforms=transforms,
        datatype=mode

    )
    x = dataset[1]

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=CFG.batch_size,
        #num_workers=CFG.num_workers,
        shuffle=True if mode == "train" else False,
    )


    return dataloader

def train_epoch(model, train_loader, optimizer, lr_scheduler, step):
    totalprediction = []
    totalTargets = []
    loss_meter = AvgMeter()
    tqdm_object = tqdm(train_loader, total=len(train_loader))
    for batch in tqdm_object:
        batch = {k: v.to(CFG.device) for k, v in batch.items() if k != "tweetText"}
        loss,prediction = model(batch)
        totalprediction.append(prediction)
        totalTargets.append(batch['labels'])

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if step == "batch":
            lr_scheduler.step()

        count = batch["image"].size(0)
        loss_meter.update(loss.item(), count)

        tqdm_object.set_postfix(train_loss=loss_meter.avg, lr=get_lr(optimizer))
    return loss_meter,totalprediction,totalTargets

def valid_epoch(model, valid_loader):
    loss_meter = AvgMeter()
    totalprediction=[]
    totalTargets=[]

    tqdm_object = tqdm(valid_loader, total=len(valid_loader))
    for batch in tqdm_object:
        batch = {k: v.to(CFG.device) for k, v in batch.items() if k != "tweetText"}
        loss,prediction = model(batch)
        totalprediction.append(prediction)
        totalTargets.append(batch['labels'])

        count = batch["image"].size(0)
        loss_meter.update(loss.item(), count)

        tqdm_object.set_postfix(valid_loss=loss_meter.avg)
    return loss_meter,totalprediction,totalTargets

class simpleModel(nn.Module):
    def __init__(
        self,
        temperature=CFG.temperature,
        image_embedding=CFG.image_embedding,
        text_embedding=CFG.text_embedding,
    ):
        super().__init__()
        self.image_encoder = ImageEncoder()
        self.text_encoder = TextEncoder()
        self.image_projection = ProjectionHead(embedding_dim=image_embedding)
        self.text_projection = ProjectionHead(embedding_dim=text_embedding)
        self.combined_fc = nn.Linear(512, 128)
        self.output_fc = nn.Linear(128,2)
        self.temperature = temperature

    def forward(self, batch):
        # Getting Image and Text Features
        image_features = self.image_encoder(batch["image"])
        text_features = self.text_encoder(
            input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]
        )
        # Getting Image and Text Embeddings (with same dimension)
        image_embeddings = self.image_projection(image_features)
        text_embeddings = self.text_projection(text_features)

        # concatane their embedding
        combined_inp = torch.cat((image_embeddings, text_embeddings), 1)
        x_comb = F.relu(self.combined_fc(combined_inp))
        out = torch.sigmoid(self.output_fc(x_comb))

        # Calculating the Loss

        lossfunction= nn.CrossEntropyLoss()
        loss=lossfunction(out,batch["labels"])

        return loss,out

def round_output(predictions,targets):
    totalprediction=[]
    totaltargets=[]
    predictions=torch.cat(predictions,dim=0)
    for i,j in predictions:
        temp=max(i,j)
        if i>j:
            temp=0
        else:
            temp=1
        totalprediction.append(temp)

    totaltargets=torch.cat(targets,dim=-1)
    return totalprediction,totaltargets

def pltorAcc_loss(H,outdir):
    plt.style.use("ggplot")
    plt.figure()
    plt.plot(H["train_loss"], label="train_loss")
    plt.plot(H["val_loss"], label="val_loss")
    plt.plot(H["train_acc"], label="train_accuracy")
    plt.plot(H["val_acc"], label="valid_accuracy")
    plt.title("Training Loss and Accuracy on Dataset")
    plt.xlabel("Epoch #")
    plt.ylabel("Loss/Accuracy")
    plt.legend(loc="lower left")
    plt.savefig(outdir + '/loss_accuracy.pdf')

def report_to_json(reports,outdir):
    import json
    reports['avg precision'] = reports['weighted avg']['precision']
    reports['avg f1'] = reports['weighted avg']['f1-score']
    reports['avg recall'] = reports['weighted avg']['recall']
    reports['macro-precision'] = reports['macro avg']['precision']
    reports['macro-f1'] = reports['macro avg']['f1-score']
    reports['macro-recall'] = reports['macro avg']['recall']
    del reports['macro avg']
    del reports['weighted avg']
    with open(outdir+'/Creports.json', 'w') as fp:
        json.dump(reports, fp)

def save_object(obj, filename):

    CFG = {
        'batch_size': obj.batch_size,
        'head_lr': obj.head_lr,
        'image_lr': obj.image_encoder_lr,
        'text_lr': obj.text_encoder_lr,
        'weight_decay': obj.weight_decay,
        'patience': obj.patience,
        'factor': obj.factor,
        'epoch': obj.epochs,
        'model_name': obj.text_encoder_model,
        "max_length": obj.max_length,
        "dropout": obj.dropout,
        "temperture": obj.temperature,
        "num_projection_layer":obj.num_projection_layers
    }
    cfg_df = pd.DataFrame(CFG,index=[0])
    cfg_df.to_csv(filename,index=False )
# Main Loop
def main():
    df_train = pd.read_csv('./2015_final/train_broken.txt',delimiter='\t')
    df_valid = pd.read_csv('./2015_final/valid_broken.txt',delimiter='\t')
    df_test = pd.read_csv('./2015_final/test.txt',delimiter='\t')
    #df['tweetText'] = df['tweetText'].str.lstrip()
    #df_test['tweetText'] = df_test['tweetText'].str.lstrip()
    #df_train.to_csv("./well_distributed_data/train.csv", index=False)
    #df_valid.to_csv("./well_distributed_data/valid.csv", index=False)
    #df_test.to_csv("./well_distributed_data/test.csv", index=False)
    #make output directory based on date-time
    #df_train_small = get_top_data(df_train, top_n=100)
    #df_valid_small = get_top_data(df_valid, top_n=100)
    #df_test_small = get_top_data(df_test, top_n=100)
    d=CFG.currentTime
    os.umask(0)

    outdir=os.path.join('./2015_final/results',d)
    #outdir=os.path.join('./output_wellD_data',d)
    os.mkdir(outdir,mode=0o777)

    tokenizer = BertTokenizer.from_pretrained(CFG.text_tokenizer)
    train_loader = build_loaders(df_train, tokenizer, mode="train")
    test_loader = build_loaders(df_test, tokenizer, mode="test")
    valid_loader = build_loaders(df_valid, tokenizer, mode="valid")

    model = simpleModel().to(CFG.device)
    params = [
        {"params": model.image_encoder.parameters(), "lr": CFG.image_encoder_lr},
        {"params": model.text_encoder.parameters(), "lr": CFG.text_encoder_lr},
        {"params": itertools.chain(
            model.image_projection.parameters(), model.text_projection.parameters()
        ), "lr": CFG.head_lr, "weight_decay": CFG.weight_decay}
        ]
    optimizer = torch.optim.AdamW(params, weight_decay=0.)
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=CFG.patience, factor=CFG.factor
    )
    step = "epoch"
    # initialize a dictionary to store training history
    H = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": []
    }

    best_loss = float('inf')
    for epoch in range(CFG.epochs):
        print(f"Epoch: {epoch + 1}")
        model.train()
        train_loss,trainlabels,traintargets = train_epoch(model, train_loader, optimizer, lr_scheduler, step)
        trainlabels, traintargets = round_output(trainlabels, traintargets)
        ctraintargets=torch.tensor(traintargets,device = 'cpu')
        ctrainlabels=torch.tensor(trainlabels,device = 'cpu')
        trainAcc=sklearn.metrics.accuracy_score(ctraintargets,ctrainlabels, normalize = True)
        model.eval()
        with torch.no_grad():
            valid_loss,validlabels,validtargets = valid_epoch(model, valid_loader)
            validlabels, validtargets = round_output(validlabels, validtargets)
            cvalidtargets = torch.tensor(validtargets, device='cpu')
            cvalidlabels = torch.tensor(validlabels, device='cpu')
            validAcc = sklearn.metrics.accuracy_score(cvalidtargets, cvalidlabels, normalize=True)

        if valid_loss.avg < best_loss:
            best_loss = valid_loss.avg
            torch.save(model.state_dict(), outdir+"/best.pt")
            print("Saved Best Model!")
            filename=os.path.join(outdir,'CFG.csv')
            save_object(CFG(), filename)
            print('Saved Configuration')

        lr_scheduler.step(valid_loss.avg)
        # update our training history
        H["train_loss"].append(train_loss.avg)
        H["train_acc"].append(trainAcc)
        H["val_loss"].append(valid_loss.avg)
        H["val_acc"].append(validAcc)

    pltorAcc_loss(H,outdir)
    # Test the model
    model.eval()
    with torch.no_grad():
        test_loss,prediction,targets = valid_epoch(model, test_loader)

    # generate classification reports
    rounded_prediction,totaltargets=round_output(prediction,targets)
    crounded_prediction = torch.tensor(rounded_prediction, device='cpu')
    ctotaltargets = torch.tensor(totaltargets, device='cpu')
    reports=classification_report(ctotaltargets,crounded_prediction, target_names=['fake','real'], output_dict=True)
    report_to_json(reports,outdir)


    # save output Labels
    prob=torch.cat(prediction,dim=0)
    prob=prob.cpu().detach().numpy()
    id=list(range(len(totaltargets)))
    out={'id':id,'targets': ctotaltargets, 'predictions': crounded_prediction,'prob class fake':prob[:,0],'prob class real':prob[:,1]}
    outdf= pd.DataFrame(data=out)
    outdf.to_csv(outdir+'/labels.csv',index=False )

if __name__=="__main__":
    main()
