"""
Spring 2024 CS 4641 BERT Emotion Classifier Model
packages:
- torch (torch torchvision torchaudio)
- transformers
- scikit-learn
- pandas
- numpy
"""


import torch
from transformers import BertTokenizer, BertForSequenceClassification
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score
import numpy as np
import pandas as pd
from datetime import datetime

class bertModel():
    def __init__(self, dataPath= '../sample.csv', testSize=0.2, randomState=42):
        self.batch_size = batchSize
        self.max_len = maxLen
        self.epochs = epochCount
        self.datasetPath = dataPath
        self.dataFrame = pd.read_csv(dataPath)
        self.texts = self.dataFrame['text'].tolist()
        self.labels = self.dataFrame['label'].tolist()
        self.train_texts, self.val_texts, self.train_labels, self.val_labels = train_test_split(self.texts, self.labels, test_size=testSize, random_state=randomState)
        
   
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=6) 

    # Split data into training and validation sets
    

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

    
    """       run Model method
    #---------------------------------- training parameters ------------------------------<<<
    #---------------------------- adjust these to fine tune model ------------------------<<<
    #----------- see comments at end of file for explanations on these parameters --------<<<
    """
    def runModel(self, batchSize, maxLen, epochCount):


   
        batch_size = batchSize
        max_len = maxLen
        epochs = epochCount
    
    #---------------------------------vvvvvvvvvvvvvvvv------------------------------------<<<


    # Create custom datasets and data loaders
        train_dataset = self.CustomDataset(self.train_texts, self.train_labels, self.tokenizer, max_len)
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
        print(f'{epoch+1}    ||  {avg_train_loss:.4f}  ||   {val_accuracy * 100:.2f}%   || {precision:.4f} || {recall:.4f}|| {f1:.4f} |')

        #save metrics
        epoch_metrics.append({'epoch': epoch, 'train_loss': avg_train_loss, 'val_accuracy': val_accuracy, 'precision': precision, 'recall': recall, 'f1':f1})

        epoch_metrics
    ####################### BETA ANALYTICS ###############################################
        """ use metrics to stop model to prevent overfitting
        - moving average
        - moving std deviation
        - window size

        """
        """
        #parameters
        window_size = 3
        moving_avg = {'epochs': (epoch - window_size, epoch), 'train_loss': avg_train_loss, 'val_accuracy': val_accuracy, 'precision': precision, 'recall': recall, 'f1':f1}
        moving_std_dev = 0

        if epoch >= window_size:
            vals = epoch_metrics[epoch - window_size]
            for key, item in moving_avg:
                

                moving_avg[key]= np.average(np.array(epoch_metrics[epoch - window_size:][key]))
    `   """
    ####################### END BETA ANALYTICS ###########################################

    """
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

    """
    ######################################
    """ Parameter Instructions 

    Determining the ideal training parameters such as batch size, max sequence length (max_len), and number of epochs involves a combination of empirical testing, understanding your dataset characteristics, and considering computational resources. Here are steps and considerations to help you determine these parameters effectively:

    1. Batch Size:
    Rule of Thumb: Start with a moderate batch size (e.g., 32, 64) as it balances between computational efficiency and model convergence. Very small batches may slow down training, while very large batches may lead to poor generalization.
    Resource Constraints: Consider your GPU memory capacity. Larger batch sizes require more memory. If memory is limited, reduce the batch size accordingly.
    Dataset Size: Larger datasets can generally benefit from larger batch sizes, but avoid exceeding GPU memory limits.
    2. Max Sequence Length (max_len):
    Tokenization Requirements: Set max_len to the maximum token length your model architecture supports. For BERT-like models, this is often around 512 tokens.
    Data Analysis: Analyze your dataset to determine the typical sequence lengths. Set max_len slightly higher than the maximum observed sequence length to avoid truncating important information.
    Resource Constraints: Longer sequences require more memory and computation. Balance between capturing context and computational efficiency.
    3. Number of Epochs:
    Learning Curve: Observe the training and validation loss/accuracy curves as training progresses. Stop training when validation performance starts to degrade (indicating overfitting).
    Early Stopping: Implement early stopping based on validation performance. Stop training when validation metrics plateau or start to decrease for a certain number of epochs.
    Training Time: Consider computational resources and training time constraints. More epochs generally lead to better convergence, but there are diminishing returns.
    """

bertModel()