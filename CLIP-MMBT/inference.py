import torch
import torch.nn as nn
from transformers import AutoConfig,AutoModel,MMBTConfig,MMBTForClassification,AutoTokenizer
from train_mmbt_clip_model import ClipEncoderMulti,prepare_test_predictions,evaluate,JsonlDataset,collate_fn, preprocess
from train_mmbt_clip_model import constants
from torch.utils.data import DataLoader,SequentialSampler
import os
import json

rootdir_results='./data/well_distributed_data/mmbt_results/15-02-2022_22;39;29/'
rootdir_data='./data/well_distributed_data/'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the model
model_name = 'bert-base-uncased'
transformer_config = AutoConfig.from_pretrained(model_name)
transformer = AutoModel.from_pretrained(model_name, config=transformer_config)
img_encoder = ClipEncoderMulti(constants['num_image_embeds'])
config = MMBTConfig(transformer_config, num_labels=constants['num_labels'], modal_hidden_size=constants['image_features_size'])
model = MMBTForClassification(config, transformer, img_encoder)

checkpoint = torch.load(rootdir_results+'best.pt')
model.load_state_dict(checkpoint['model_state_dict'])

model.to(device)
model.eval()
criterion = nn.BCEWithLogitsLoss()
tokenizer = AutoTokenizer.from_pretrained(model_name, do_lower_case=True)

# Load the test Data
transforms = preprocess
dataset = JsonlDataset(rootdir_data, '100samples_test_similar_image.csv', 'test_images',
                       tokenizer, transforms, constants['max_seq_length'] - constants['num_image_embeds'] - 2)
test_sampler = SequentialSampler(dataset)
# Build data_Loader
test_dataloader = DataLoader(dataset,sampler=test_sampler,batch_size=constants['eval_batch_size'],collate_fn=collate_fn)

# Test the model
with torch.no_grad():
    test_result = evaluate(model, tokenizer, criterion, test_dataloader)

# save output Labels
labels_results,Creports = prepare_test_predictions(test_result)
labels_results.to_csv(rootdir_results + '/labels_100samples_test_similar_image.csv',index=False)
output_test_file = os.path.join(rootdir_results, "100samples_Test_similar_image.json")
with open(output_test_file, "w") as fp:
    json.dump(Creports, fp)
print('complete')
