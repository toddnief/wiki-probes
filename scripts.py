from tqdm import tqdm
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer, LabelEncoder
from ast import literal_eval

import torch
from torch import Tensor
import torch.nn as nn
from torch.nn import BCEWithLogitsLoss
import torch.nn.functional as F
import torch.nn.init as init
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader, Dataset

import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer
from baukit import TraceDict

class ActivationsDataset(Dataset):
    def __init__(self, Xs, titles, categories, text):
        self._Xs = Xs
        self._titles = titles
        self._categories = categories
        self.text = text

    @property
    def Xs(self):
        return self._Xs

    @property
    def titles(self):
        return self._titles
    
    @property
    def categories(self):
        return self._categories

    def __len__(self):
        return len(self._Xs)

    def __getitem__(self, idx):
        return self._Xs[idx], self._titles[idx], self._categories[idx], self.text[idx]

class Linear(nn.Module):
    # TODO: figure out better way to handle the number of classes for default
    def __init__(self, hidden_size, n_classes):
        super().__init__()
        self.fc1 = nn.Linear(hidden_size, n_classes)

    def forward(self, x):
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = self.fc1(x)
        return x

class MLP(nn.Module):
    def __init__(self, hidden_size, n_classes):
        super().__init__()
        self.fc1 = nn.Linear(hidden_size, 120)
        self.fc2 = nn.Linear(120, n_classes)

    def forward(self, x):
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

def get_activations(prompts,tokenizer,model,device,layer="all"):
    """Returns a Numpy array of residual stream activations. 
    Based on https://github.com/likenneth/honest_llama
    
    David's uncertainties: I think these are the activations before the MLP sublayer?
    """
    tokenized = tokenizer(prompts, padding=True, truncation=True, return_tensors="pt")
    input_ids = tokenized.input_ids.to(device)
    attention_mask = tokenized.attention_mask.to(device)

    model.eval()
    outputs = model(
        input_ids,
        attention_mask=attention_mask, output_hidden_states = True
    )
    hidden_states = outputs.hidden_states
    if layer == "all":
         # (num_layers, batch_size, seq_length, hidden_dim)
        hidden_states = torch.stack(hidden_states, dim = 0).squeeze()
        hidden_states = hidden_states.detach().cpu().numpy()
    else:
         # (batch_size, seq_length, hidden_dim)
        hidden_states = hidden_states[layer].detach().cpu().numpy()
    return hidden_states

def get_fitted_label_encoder(df, labels):
    if labels == "categories":
        from ast import literal_eval
        unique_labels = set()
        for item in df['categories'].tolist():
            categories = literal_eval(item)
            for cat in categories:
                unique_labels.update([cat])
        unique_labels = list(unique_labels)
    elif labels == "title":
        unique_labels = list(df['title'].drop_duplicates())
    
    label_encoder = LabelEncoder()
    label_encoder.fit(unique_labels)

    return label_encoder

def parse_categories(cat_list, label_encoder):
    encoded_categories = []
    for cat in cat_list:
        encoded_cat = label_encoder.transform(literal_eval(cat)).tolist()
        encoded_categories.append(encoded_cat)
    return encoded_categories

def init_weights(m):
    if type(m) == nn.Linear:
        init.xavier_normal_(m.weight)
        init.constant_(m.bias, 0)

# TODO: call this something different
def return_activations(df, model, tokenizer, device, label_encoder_title, label_encoder_cats, save=False):
    # TODO: this is kinda ugly
    # title
    df['title_encoded'] = label_encoder_title.transform(df['title'])

    # categories
    df['label_encoded'] = parse_categories(df['categories'].tolist(), label_encoder_cats)
    def list_to_binary_vector(lst, dim=len(label_encoder_cats.classes_)):
        return [1 if i in lst else 0 for i in range(dim)]
    df['binary_labels'] = df['label_encoded'].apply(list_to_binary_vector)

    hidden_states = []
    titles = []
    categories = []
    text = []
    for i, row in df.iterrows():
        hidden_states.append(get_activations(row.text,tokenizer,model,device))
        titles.append(row.title_encoded)
        categories.append(row.binary_labels)
        text.append(row.text)
    
    # TODO: maybe set this up to save the activations
    if save:
        pass

    # TODO: should prob make this a class of some sort that stores all of this stuff

    return hidden_states, titles, categories, text

# TODO: call this something different
def get_hidden_states(filepath, model, tokenizer, device):
    df = pd.read_csv(filepath)

    label_encoder_title = get_fitted_label_encoder(df, labels="title")
    label_encoder_cats = get_fitted_label_encoder(df, labels="categories")

    data = return_activations(df, model, tokenizer, device, label_encoder_title, label_encoder_cats)

    return data, label_encoder_title, label_encoder_cats

# TODO: Set this up to handle layer = None
def create_dataset(hidden_states, titles, categories, text, layer=-1, aggregation="max"):
    Xs = []
    for hs in hidden_states:
        if len(hs.shape) == 2: # GPT-2 will lose a dimension if there's a single token
            hs = hs[:, np.newaxis, :]

        if layer is None:
            hs = hs.reshape(1,hs.shape[1],-1).squeeze(0)
        else:
            hs = hs[layer,:,:]

        if aggregation == "max":
            x = np.max(hs, axis=0)
        elif aggregation == "mean":
            x = np.mean(hs, axis=0)

        # TODO: add some PCA or dimensionality reduction or feature selection options here

        Xs.append(x)

    Xs_t = Tensor(np.asarray(Xs)).float()
    titles_t = Tensor(np.asarray(titles)).long() # cross entropy loss wants a long dtype
    categories_t = Tensor(np.asarray(categories)).float() # binary cross entropy loss wants a float dtype

    return ActivationsDataset(Xs_t, titles_t, categories_t, text)

def train_handler(model, train_dataset, val_dataset, label_encoder, probe_type="linear", labels="title", batch_size=4, epochs=200, print_progress=True):
    if labels == "title":
        n_classes = len(train_dataset.titles.unique())
    elif labels == "categories":
        n_classes = len(train_dataset.categores.unique())

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)

    hidden_size = next(iter(val_loader))[0].shape[1]

    if probe_type == "linear":
        probe = Linear(hidden_size=hidden_size, n_classes=n_classes)
    elif probe_type == "mlp":
        probe = MLP(hidden_size=hidden_size, n_classes=n_classes)
    probe.apply(init_weights)
    # TODO: add a LASSO option
    # Or does AdamW kind of do this already?

    if labels == "categories":
        criterion = BCEWithLogitsLoss(pos_weight=Tensor(torch.ones(n_classes * 20)))
    elif labels == "title":
        criterion = nn.CrossEntropyLoss()

    # TODO: add options for other optimizers
    optimizer = optim.AdamW(probe.parameters(), lr=0.001)

    # TODO: should prob have a train function that is called here
    for epoch in range(epochs):  
        probe.train()
        train_loss = 0.0
        for i, data in enumerate(train_loader):
            inputs, titles, categories, text = data
            # TODO: this is sloppy - handle the alternative dataloader better
            lbls = titles if labels == "title" else categories
            optimizer.zero_grad()
            outputs = probe(inputs)
            loss = criterion(outputs, lbls)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        if (epoch + 1) % 5 == 0:
            val_loss = 0.0
            val_total = 0
            correct = 0
            incorrect_examples = []
            probe.eval()
            with torch.no_grad():
                for i, data in enumerate(val_loader):  
                    inputs, titles, categories, text = data
                    lbls = titles if labels == "title" else categories
                    outputs = probe(inputs)
                    loss = criterion(outputs, lbls)
                    val_loss += loss.item()
                    if labels == "title":
                        _, predicted_labels = torch.max(outputs, 1)
                    elif labels == "categories":
                        predicted_labels = (torch.sigmoid(outputs) > .5).int()
                    graded_preds = predicted_labels == lbls
                    correct += (graded_preds).sum().item()
                    # TODO: set this up to be the correct encoder depending on label
                    if epoch + 1 == epochs and print_progress:
                        for txt, lbl, pred in zip(text, label_encoder.inverse_transform(lbls), label_encoder.inverse_transform(predicted_labels)):
                            print("text: ", txt)
                            print("labels: ", lbl)
                            print("predicted labels: ", pred)
                    val_total += lbls.size()[0]
                    val_accuracy = correct / val_total
            if print_progress:
                print(f'[Training][{epoch + 1}] loss: {train_loss / len(train_loader):.3f}')
                print(f'[Validation][{epoch + 1}] loss: {val_loss / val_total:.3f}')
                print(f'[Validation]{epoch + 1} accuracy: {val_accuracy:.3f}')
    return probe, val_accuracy

