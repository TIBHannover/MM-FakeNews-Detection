import numpy as np
import pandas as pd
import torch
from torch import nn
import torch.nn.functional as F
from datetime import datetime
import os
import torch.utils as utils
from tqdm.autonotebook import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
import pickle
import argparse

class early_fusion_model(nn.Module):
    def __init__(self,dropout):
        super().__init__()
        self.projection1 = nn.Linear(1024, 512)
        self.projection2 = nn.Linear(512, 256)
        self.projection3 = nn.Linear(256, 128)
        self.dropout=nn.Dropout(dropout)
        self.out_fc = nn.Linear(128, 2)

    def forward(self, batch):
        combined_features=batch
        hidden1=F.relu(self.projection1(combined_features))
        hidden1=self.dropout(hidden1)
        hidden2=F.relu(self.projection2(hidden1))
        hidden3 = F.relu(self.projection3(hidden2))
        output=torch.sigmoid(self.out_fc(hidden3))

        return output

class late_fusion_model(nn.Module):
    def __init__(self,dropout):
        super().__init__()
        self.text_prejection1 = nn.Linear(512, 256)
        self.image_prejection1 = nn.Linear(512, 256)
        self.text_projection2 = nn.Linear(256, 128)
        self.image_projection2 = nn.Linear(256, 128)
        self.dropout=nn.Dropout(dropout)
        self.out_fc = nn.Linear(256, 2)

    def forward(self, batch):
        x=torch.chunk(batch,2,dim=1)
        text_features=x[0]
        image_features=x[1]
        text_hidden1=F.relu(self.text_prejection1(text_features))
        image_hidden1 = F.relu(self.image_prejection1(image_features))
        text_hidden1=self.dropout(text_hidden1)
        image_hidden1 = self.dropout(image_hidden1)
        text_hidden2=F.relu(self.text_projection2(text_hidden1))
        image_hidden2 = F.relu(self.image_projection2(image_hidden1))
        #text_hidden2=self.dropout(text_hidden2)
        #image_hidden2 = self.dropout(image_hidden2)
        combined_features=torch.cat((text_hidden2,image_hidden2),dim=1)
        output=torch.sigmoid(self.out_fc(combined_features))

        return output

def train_epoch(model, train_loader, optimizer,lossfunction,args):
    totalprediction = []
    totallabels=[]
    totalTargets = []
    traincorrect=0
    loss_meter = AvgMeter()
    tqdm_object = tqdm(train_loader, total=len(train_loader))
    for (x,y) in tqdm_object:
        (x, y) = (x.to(args.device), y.to(args.device,dtype=torch.int64))
        prediction = model(x)
        loss = lossfunction(prediction,y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        traincorrect += (prediction.argmax(1) == y).type(torch.float).sum().item()
        totalprediction.append(prediction)
        totalTargets.extend(y)
        totallabels.extend(prediction.argmax(1))


        count = x.size(0)
        loss_meter.update(loss.item(), count)
        tqdm_object.set_postfix(train_loss=loss_meter.avg)

    acc=traincorrect/loss_meter.count
    totalprediction = torch.cat(totalprediction, dim=0)
    return loss_meter,acc,totalprediction,totallabels,totalTargets

def valid_epoch(model, valid_loader,lossfunction,args):
    totalprediction = []
    totallabels=[]
    totalTargets = []
    traincorrect=0
    loss_meter = AvgMeter()
    tqdm_object = tqdm(valid_loader, total=len(valid_loader))
    for (x,y) in tqdm_object:
        (x, y) = (x.to(args.device), y.to(args.device,dtype=torch.int64))
        prediction = model(x)
        loss = lossfunction(prediction,y)


        traincorrect += (prediction.argmax(1) == y).type(torch.float).sum().item()
        totalprediction.append(prediction)
        totalTargets.extend(y)
        totallabels.extend(prediction.argmax(1))


        count = x.size(0)
        loss_meter.update(loss.item(), count)
        tqdm_object.set_postfix(valid_loss=loss_meter.avg)

    acc=traincorrect/len(valid_loader.dataset)
    totalprediction = torch.cat(totalprediction, dim=0)
    return loss_meter,acc,totalprediction,totallabels,totalTargets

def build_loaders(clip_features,labels,args,mode):
    data_x = torch.Tensor(clip_features)
    data_y = torch.Tensor(labels)
    train_dataset = utils.data.TensorDataset(data_x, data_y)
    dataloader = utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True if mode == "train" else False)
    return dataloader

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

def pltorAcc_loss(History,outdir):
    plt.style.use("ggplot")
    plt.figure()
    plt.xticks(np.arange(0, len(History['train_loss']), 1),fontsize=7)
    #plt.xlim([1,len(History['train_loss'])])
    plt.plot(History["train_loss"], label="train_loss")
    plt.plot(History["val_loss"], label="val_loss")
    plt.plot(History["train_acc"], label="train_accuracy")
    plt.plot(History["val_acc"], label="valid_accuracy")
    plt.title("Training Loss and Accuracy on Dataset")
    plt.xlabel("Epoch #")
    plt.ylabel("Loss/Accuracy")
    plt.legend(loc="best")
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
    with open(outdir+'/Creport.json', 'w') as fp:
        json.dump(reports, fp)

def save_object(obj, filename):
    with open(filename, 'wb') as outp:  # Overwrites any existing file.
        pickle.dump(obj, outp, pickle.HIGHEST_PROTOCOL)


def main():
    args = get_args()
    ############### Load train combined clip features##################################
    train_VN_FakeIm = np.load('./data/VisualNews/train_FakeIm_clip_features.npy')
    train_VN_with_EvRep = np.load('./data/VisualNews/train_plus_EvRep_clip_features.npy')
    train_ME_with_EvRep = np.load('./data/2015_data/train_plus_EvRep_clip_features.npy')
    train_ME_FakeIm = np.load('./data/2015_data/train_FakeIm_clip_features.npy')
    train_ME = np.load('./data/2015_data/train_clip_features.npy')

    ## find extra fake samples for ev_rep train VN:
    extra = pd.read_csv('./MediaEval_VN_data/VN_half_data/train.csv')
    extra_f = np.load('./MediaEval_VN_data/VN_half_data/VN_train_clip_features.npy')
    extra = extra[extra['label_name'] == 'fake']
    extra = extra[0:2000]
    idx = extra.index.to_list()
    extra_f = extra_f[idx]
#################### Load valid combined features ######################################
    valid_VN_FakeIm = np.load('./data/VisualNews/valid_FakeIm_clip_features.npy')
    valid_VN_with_EvRep = np.load('./data/VisualNews/valid_plus_EvRep_clip_features.npy')
    valid_ME_with_EvRep = np.load('./data/2015_data/valid_plus_EvRep_clip_features.npy')
    valid_ME_FakeIm = np.load('./data/2015_data/valid_FakeIm_clip_features.npy')
    valid_ME = np.load('./data/2015_data/valid_clip_features.npy')
################### Load test data features ###########################################
    test_clip_features = np.load('./data/2015_data/test_clip_features.npy')
###################################### Load labels ############################################
    df_train_VN_FakeIm = pd.read_csv('./data/VisualNews/train_FakeIm.csv')
    df_train_VN_with_EvRep = pd.read_csv('./data/VisualNews/train_plus_EvRep.csv')
    df_train_ME_with_EvRep = pd.read_csv('./data/2015_data/train_plus_EvRep.csv')
    df_train_ME_FakeIm = pd.read_csv('./data/2015_data/train_FakeIm.csv')
    df_train_ME = pd.read_csv('./data/2015_data/train.csv')

    df_valid_VN_FakeIm = pd.read_csv('./data/VisualNews/valid_FakeIm.csv')
    df_valid_VN_with_EvRep= pd.read_csv('./data/VisualNews/valid_plus_EvRep.csv')
    df_valid_ME_with_EvRep = pd.read_csv('./data/2015_data/valid_plus_EvRep.csv')
    df_valid_ME_FakeIm = pd.read_csv('./data/2015_data/valid_FakeIm.csv')
    df_valid_ME = pd.read_csv('./data/2015_data/valid.csv')

    df_test = pd.read_csv('./data/2015_data/test.csv')
################################## choose model and associated data ####################
    if args.model=='clip-mlp':
        train_clip_features=train_ME
        valid_clip_features=valid_ME
        df_train=df_train_ME
        df_valid=df_valid_ME

    elif args.model=='VNME-Img':
        train_clip_features = np.concatenate((train_VN_FakeIm,train_ME,train_ME_FakeIm), axis=0)
        valid_clip_features=np.concatenate((valid_VN_FakeIm,valid_ME,valid_ME_FakeIm), axis=0)
        df_train = pd.concat([df_train_VN_FakeIm, df_train_ME, df_train_ME_FakeIm], axis=0, ignore_index=True)
        df_valid= pd.concat([df_valid_VN_FakeIm,df_valid_ME, df_valid_ME_FakeIm], axis=0,
            ignore_index=True)

    elif args.model=='VNME-Evt':
        train_clip_features = np.concatenate((train_VN_with_EvRep,train_ME_with_EvRep), axis=0)
        valid_clip_features = np.concatenate((valid_VN_with_EvRep,valid_ME_with_EvRep), axis=0)
        df_train= pd.concat([df_train_VN_with_EvRep,df_train_ME_with_EvRep], axis=0,
            ignore_index=True)
        df_valid= pd.concat([df_valid_VN_with_EvRep,df_valid_ME_with_EvRep], axis=0,
            ignore_index=True)

    elif args.model=='VNME-All':
        #train_clip_features = np.concatenate((train_VN_with_FakeIm, train_VN_with_EvRep,
        #                                      train_ME_with_EvRep, train_ME_FakeIm,
        #                                      train_ME_with_EvRep[8671:], extra_f), axis=0)
        train_clip_features = np.concatenate((train_VN_FakeIm,train_VN_with_EvRep,
                                              train_ME_with_EvRep,train_ME_FakeIm), axis=0)
        valid_clip_features = np.concatenate((valid_VN_FakeIm,valid_VN_with_EvRep,
                                              valid_ME_with_EvRep, valid_ME_FakeIm), axis=0)
        df_train = pd.concat([df_train_VN_FakeIm, df_train_VN_with_EvRep, df_train_ME_with_EvRep, df_train_ME_FakeIm], axis=0, ignore_index=True)
        df_valid = pd.concat([df_valid_VN_FakeIm, df_valid_VN_with_EvRep, df_valid_ME_with_EvRep, df_valid_ME_FakeIm], axis=0,ignore_index=True)

    train_labels = df_train['label_name'].apply(lambda x: 1 if x == "real" else 0)
    valid_labels = df_valid['label_name'].apply(lambda x: 1 if x == "real" else 0)
    test_labels=df_test['label_name'].apply(lambda x : 1 if x == "real" else  0)
####################################################################################
    currentTime=datetime.now().strftime("%m-%d-%Y_%H;%M;%S")
    lossfunction=nn.CrossEntropyLoss()
    model = early_fusion_model(args.dropout).to(args.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    #make output directory based on date-time
    d=currentTime
    os.umask(0)
    if model._get_name()=='early_fusion_model':
        d=os.path.join('./MediaEval_VN_data/results\Im+Ev/early_fusion',d)
    elif model._get_name()=='late_fusion_model':
        d=os.path.join('./MediaEval_VN_data/results\Im+Ev/late_fusion',d)

    #outdir=os.path.join('./2015_final/results',d)
    outdir=d
    os.mkdir(outdir, mode=0o777)
    # build data Loaders
    train_loader=build_loaders(train_clip_features,train_labels,args,mode="train")
    valid_loader=build_loaders(valid_clip_features,valid_labels,args,mode="valid")
    test_loader=build_loaders(test_clip_features,test_labels,args,mode="test")

############################# MAIN LOOP############################################
    # initialize a dictionary to store training history
    H = {
         "train_loss": [],
         "train_acc": [],
        "val_loss": [],
        "val_acc": []
    }

    best_loss = float('inf')
    best_Acc=0
    for epoch in range(args.epochs):
        print(f"Epoch: {epoch + 1}")
        model.train()
        train_loss,train_Acc,_,trainlabels,traintargets = train_epoch(model, train_loader, optimizer,lossfunction,args)

        model.eval()
        with torch.no_grad():
            valid_loss,valid_Acc,_, validlabels, validtargets = valid_epoch(model, valid_loader,lossfunction,args)


        if valid_loss.avg < best_loss:
        #if valid_Acc> best_Acc:
            best_loss = valid_loss.avg
            best_Acc=valid_Acc
            bestmodel=model
            best_loss_acc={
                'best_loss':[best_loss],
                'best_Acc':[best_Acc]
            }
            best= pd.DataFrame(data=best_loss_acc)
            best.to_json(outdir+'/best_loss_acc.json')
            torch.save(bestmodel.state_dict(), outdir + "/best.pt")
            print("Saved Best Model!")
            CFG = {
                "batch_size": [args.batch_size],
                "epochs": [args.epochs],
                "lr": [args.lr],
                "dropout": [args.dropout]
            }
            cfg= pd.DataFrame(data=CFG)
            cfg.to_json(outdir+'/CFG.json')
            print('Saved Config')

        # update our training history
        H["train_loss"].append(train_loss.avg)
        H["train_acc"].append(train_Acc)
        H["val_loss"].append(valid_loss.avg)
        H["val_acc"].append(valid_Acc)

    #plot ACC and Loss
    pltorAcc_loss(H, outdir)
    # Test the model
    model=bestmodel
    model.eval()
    with torch.no_grad():
        test_loss,test_Acc,test_predictions,testlabels,testtargets = valid_epoch(model, test_loader,lossfunction)

    # generate classification reports
    testlabels = torch.tensor(testlabels, device='cpu')
    testtargets = torch.tensor(testtargets, device='cpu')
    reports=classification_report(testtargets,testlabels, target_names=['fake','real'], output_dict=True)
    report_to_json(reports,outdir)
    torch.save(model.state_dict(), outdir + "/best_last.pt")

    # save output Labels
    test_predictions=test_predictions.cpu().detach().numpy()
    id=list(range(len(testtargets)))
    out={'id':id,'targets': testtargets, 'predictions': testlabels,'prob class real':test_predictions[:,0],'prob class fake':test_predictions[:,1]}
    outdf= pd.DataFrame(data=out)
    outdf.to_csv(outdir+'/labels.csv',index=False )
    print('completed')
def get_args():

    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", default=64, type=int)
    parser.add_argument("--lr", default=1e-5)
    parser.add_argument("--epochs",default=30, type=int)
    parser.add_argument("--device", default=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    parser.add_argument("--dropout", default=0.4)
    parser.add_argument("--model", default='clip-mlp',help='choose between [clip-mlp,VNME-Img,VNME-Evt,VNME-Evt]')
    args = parser.parse_args()

    print(args)
    return args

if __name__=="__main__":
    main()





