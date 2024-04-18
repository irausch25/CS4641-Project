"""
Spring 2024 CS 4641 BERT Emotion Classifier Model
packages:
- torch
- transformers
- scikit-learn
- pandas
- numpy
- datetime

"""


import torch
from transformers import BertTokenizer, BertForSequenceClassification
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score
import numpy as np
import pandas as pd
from datetime import datetime


#setting up data
datasetPath = '../text.csv' #path to data
df = pd.read_csv(datasetPath) #convert from csv to pandas dataframe
#split data into texts and their corresponding labels
texts = df['text'].tolist()
labels = df['label'].tolist() #Labels=> 0:anger, 1:fear, 2:joy, 3:love, 4:sadness, 5:surprise
# Split data into training and validation sets
train_texts, val_texts, train_labels, val_labels = train_test_split(texts, labels, test_size=0.2, random_state=42)

# Define a custom dataset class
class CustomDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]

        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )

        return {
            'text': text,
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'label': torch.tensor(label, dtype=torch.long)
        }

# Initialize BERT tokenizer and model
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=6) 

# training parameters---- adjust these to fine tune model
batch_size = 64
max_len = 256
epochs = 6

# Create custom datasets and data loaders
train_dataset = CustomDataset(train_texts, train_labels, tokenizer, max_len)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# Define optimizer and loss function
optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
criterion = torch.nn.CrossEntropyLoss()


#Table header for result data
print(f'\n\n\nEpoch||Train Loss||Val Accuracy||Precision||Recall||F1 Score|')
print('-----||----------||------------||---------||------||--------|')

#keep track of results
#
epoch_metrics = []
# Training loop
model.train()
for epoch in range(epochs):
    total_loss = 0
    for batch in train_loader:
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        labels = batch['label']

        optimizer.zero_grad()
        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        total_loss += loss.item()
        loss.backward()
        optimizer.step()

    avg_train_loss = total_loss / len(train_loader)

    # Validation
    val_dataset = CustomDataset(val_texts, val_labels, tokenizer, max_len)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)


    val_predictions = []
    val_true_labels = []  
    val_accuracy = 0
    total_val_samples = 0  
    model.eval()
    
    for batch in val_loader:
        with torch.no_grad():
            input_ids = batch['input_ids']
            attention_mask = batch['attention_mask']
            labels = batch['label']

            outputs = model(input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            _, predicted = torch.max(logits, dim=1)
            
            val_accuracy += (predicted == labels).sum().item()
            total_val_samples += labels.size(0)
            val_predictions.extend(predicted.cpu().numpy())
            val_true_labels.extend(labels.cpu().numpy())

    # Convert lists to numpy arrays for sklearn metrics
    val_predictions = np.array(val_predictions)
    val_true_labels = np.array(val_true_labels)

    # Calculate precision, recall, and F1 scores
    precision = precision_score(val_true_labels, val_predictions, average='weighted', zero_division=1)
    recall = recall_score(val_true_labels, val_predictions, average='weighted')
    f1 = f1_score(val_true_labels, val_predictions, average='weighted')
    val_accuracy /= total_val_samples
   
    #print eval metrics
    print(f'{epoch+1}    ||  {avg_train_loss:.4f}  ||   {val_accuracy * 100:.2f}%   ||  {precision:.4f}  || {recall:.4f} || {f1:.4f} |')

    #save metrics
    epoch_metrics.append({'epoch': epoch, 'train_loss': avg_train_loss, 'val_accuracy': val_accuracy, 'precision': precision, 'recall': recall, 'f1':f1})


####################### BETA ANALYTICS ###############################################
    """ use metrics to stop model to prevent overfitting
    - moving average
    - moving std deviation
    - window size

    """
    #parameters
    window_size = 3
    moving_avg = {'epochs': (epoch - window_size, epoch), 'train_loss': avg_train_loss, 'val_accuracy': val_accuracy, 'precision': precision, 'recall': recall, 'f1':f1}
    moving_std_dev = 0

    if epoch >= window_size:
        vals = epoch_metrics[epoch - window_size]
        for key, item in moving_avg:
            

            moving_avg[key]= np.average(np.array(epoch_metrics[epoch - window_size:][key]))
####################### END BETA ANALYTICS ###########################################


#save model we just trained. 
records = open('bertModelsTracker.txt', 'w')
version = records.readline()[18:]

if version.isnumeric():
    versionNum = float(version)
    versionNum += 0.1
    records.write(f'Version Tracker: {versionNum:.1f}\nDate: {datetime.now()}')

else:
    versionInt = -1
    print("BERTMODELSTRAINING.TXT FILE CORRUPTED")

torch.save(model.state_dict(), f'bert_emotion_classifier_vT{int(versionNum)}.pth')

records.close()