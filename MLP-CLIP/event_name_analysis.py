import pandas as pd
import torch
from torch import nn
from main import early_fusion_model,build_loaders,valid_epoch
import re
import numpy as np
from sklearn.metrics import classification_report


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
    #with open(outdir + '/Creports.json', 'w') as fp:
    with open(outdir+'changed_image/Creports_100samples_test.json', 'w') as fp:
        json.dump(reports, fp)

rootdir_results='./MediaEval_VN_data/results\Im+Ev\early_fusion/10-04-2022_19;35;56/'
CFG=pd.read_json(rootdir_results+'CFG.json')
rootdir_data='./well_distributed_data/changed_image/'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
lossfunction=nn.CrossEntropyLoss()

# Load the model
model=early_fusion_model(CFG.dropout[0])
model.load_state_dict(torch.load(rootdir_results+'best.pt'))
model.to(device)
model.eval()
# Load the test Data features and labels
test_clip_features= np.load(rootdir_data+'100samples_test_features2.npy')
test_df=pd.read_csv(rootdir_data+'100samples_test2.csv')
#test_clip_features= np.load(rootdir_data+'test_clip_vitL_14_features.npy')
#test_df=pd.read_csv(rootdir_data+'test.csv')
test_labels=test_df['label']
test_labels = test_labels.apply(lambda x: 1 if x == "real" else 0)

# Build data_Loader
test_loader=build_loaders(test_clip_features,test_labels,mode="test")

# Test the model
with torch.no_grad():
    test_loss,test_Acc,test_predictions,testlabels,testtargets = valid_epoch(model, test_loader,lossfunction)

# generate classification reports
testlabels = torch.tensor(testlabels, device='cpu')
testtargets = torch.tensor(testtargets, device='cpu')
#reports=classification_report(testtargets,testlabels, target_names=['fake','real'], output_dict=True)
#report_to_json(reports,rootdir_results)

# save output Labels
test_predictions=test_predictions.cpu().detach().numpy()
id=list(range(len(testtargets)))
out={'id':id,'targets': testtargets, 'predictions': testlabels,'prob class real':test_predictions[:,0],'prob class fake':test_predictions[:,1]}
outdf= pd.DataFrame(data=out)
outdf.to_csv(rootdir_results+'/changed_image/labels_100samples_test2.csv',index=False )
#outdf.to_csv(rootdir_results+'/2/labels2.csv',index=False )
print('complete')

