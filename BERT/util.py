import argparse
import random
import numpy as np
import torch
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, f1_score
import pandas as pd

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--random_state', type=int, default=0)
    parser.add_argument('--data_file', type=str, default='user_features_label.csv')
    parser.add_argument('--label_type', type=int, default=0)
    parser.add_argument('--valid_size', type=float, default=0.1)
    parser.add_argument('--model_name', type=str, default='nghuyong/ernie-3.0-base-zh')
    parser.add_argument('--output_dir', type=str, default='checkpoints')
    parser.add_argument('--save_name', type=str, default='model.pt')
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--bert_lr', type=float, default=1e-5)
    parser.add_argument('--weight_decay', type=float, default=1e-2)
    parser.add_argument('--num_epochs', type=int, default=5)
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1)
    parser.add_argument('--warmup_ratio', type=float, default=0.06)
    parser.add_argument('--max_norm', type=float, default=1.0)
    parser.add_argument('--logging_dir', type=str, default='logs')

    args = parser.parse_args()

    return args
    
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # torch.backends.cudnn.enabled = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    
def load_data(data_file, label_type):
    df = pd.read_csv(data_file)
    texts = list(df['description'].apply(lambda x: str(x).replace(' ', '') + ' ') + df['all_weibo'].apply(str).values)
    # texts = list((df['description'].apply(lambda x: str(x) + ' ') + df['all_weibo'].apply(str)).values)
    features = df.drop(['uid', 'description', 'all_weibo', 'C3DV1_社交排斥', 'C3DV2_社交排斥', 'C3DV_恶意幽默', 'C3DV_内疚诱导'], axis=1).values
    labels = (
        list(df['C3DV1_社交排斥'].values),
        list(df['C3DV2_社交排斥'].values),
        list(df['C3DV_恶意幽默'].values),
        list(df['C3DV_内疚诱导'].values)
    )

    return texts, features, labels[label_type]

def evaluation(model, test_dataloader, criterion):
    epoch_loss = 0.
    all_preds, all_labels = [], []
    device = torch.device('cuda')
    for batch in test_dataloader:
        input_ids = batch['input_ids'].to(device)
        token_type_ids = batch['token_type_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        text_features = {
            'input_ids': input_ids,
            'token_type_ids': token_type_ids,
            'attention_mask': attention_mask
        }
        numerical_features = batch['features'].to(device)
        labels = batch['labels'].to(device)
        with torch.no_grad():
            logits = model(text_features, numerical_features)
        loss = criterion(logits, labels)
        epoch_loss += loss.item()
        all_preds += logits.argmax(1).tolist()
        all_labels += labels.tolist()

    return accuracy_score(all_labels, all_preds), \
           f1_score(all_labels, all_preds, average='macro'), \
           epoch_loss / len(test_dataloader)
