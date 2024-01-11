import pandas as pd
import torch
from torch import nn
from BERTResNet import simpleModel,build_loaders,CFG,valid_epoch,round_output
from transformers import BertTokenizer
from sklearn.metrics import classification_report

rootdir_results='./output_wellD_data/03-29-2022_14;07;37/'
rootdir_data='./well_distributed_data/'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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
    with open(outdir+'/Creports_100samples_test_similar_image.json', 'w') as fp:
        json.dump(reports, fp)

# Load the model
model = simpleModel()
model.load_state_dict(torch.load(rootdir_results+'best.pt'))
model.to(device)
model.eval()
# Load the test Data and labels
test_df=pd.read_csv(rootdir_data+'100samples_test_similar_image.csv')
test_labels=test_df['label']

# Set static image and text for static experiment
#test_df['imageId(s)']='blank.png'
#test_df['tweetText']='This is a sentence.'
#test_df.to_csv(rootdir_data+'/test_only_text.csv',index=False )

tokenizer = BertTokenizer.from_pretrained(CFG.text_tokenizer)
# Build data_Loader
test_loader = build_loaders(test_df, tokenizer, mode="test")
# Test the model
with torch.no_grad():
    test_loss,prediction,targets = valid_epoch(model, test_loader)

# generate classification reports
rounded_prediction,totaltargets=round_output(prediction,targets)
crounded_prediction = torch.tensor(rounded_prediction, device='cpu')
ctotaltargets = torch.tensor(totaltargets, device='cpu')
reports=classification_report(ctotaltargets,crounded_prediction, target_names=['fake','real'], output_dict=True)
report_to_json(reports,rootdir_results)
# save output Labels
prob=torch.cat(prediction,dim=0)
prob=prob.cpu().detach().numpy()
id=list(range(len(totaltargets)))
out={'id':id,'targets': ctotaltargets, 'predictions': crounded_prediction,'prob class fake':prob[:,0],'prob class real':prob[:,1]}
outdf= pd.DataFrame(data=out)
outdf.to_csv(rootdir_results+'/labels_100samples_test_similar_image.csv',index=False )
print('program completed')
