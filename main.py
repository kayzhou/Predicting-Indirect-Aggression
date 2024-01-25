from util import load_data, parse_args, set_seed, evaluation
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer, AutoModel, AdamW, get_linear_schedule_with_warmup
from dataset import Dataset
import os
import logging
import torch
from model import Model
from torch.utils.data import DataLoader
import torch.nn as nn
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
import torch.nn.functional as F

def main():
    args = parse_args()

    logging.basicConfig(
        level=logging.INFO,
        filename=f'./logs/bert_{args.label_type}.log',
        filemode='w'
    )
    logger = logging.getLogger(__name__)

    set_seed(args.seed)
    device = torch.device('cuda')

    texts, features, labels = load_data(args.data_file, args.label_type)

    output_dir = args.output_dir + f'/bert_{args.label_type}/'

    train_texts, valid_texts, train_features, valid_features, train_labels, valid_labels = train_test_split(texts, features, labels, test_size=args.valid_size, random_state=args.random_state, stratify=labels)
    
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    bert_model = AutoModel.from_pretrained(args.model_name, output_attentions=True)
    model = Model(bert_model)

    train_encodings = tokenizer(list(train_texts), padding='max_length', truncation=True, max_length=512)
    valid_encodings = tokenizer(list(valid_texts), padding='max_length', truncation=True, max_length=512)

    train_dataset = Dataset(train_encodings, train_features, train_labels)
    valid_dataset = Dataset(valid_encodings, valid_features, valid_labels)

    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    valid_dataloader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False)

    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    bert_parameters = ['bert']
    optimizer_grouped_parameters = [
        {
            'params': [p for n, p in model.named_parameters() if any(nd in n for nd in bert_parameters) and not any(nd in n for nd in no_decay)], 
            'weight_decay': args.weight_decay,
            'lr': args.bert_lr
        },
        {
            'params': [p for n, p in model.named_parameters() if any(nd in n for nd in bert_parameters) and any(nd in n for nd in no_decay)], 
            'weight_decay': 0.,
            'lr': args.bert_lr
        },
        {
            'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in bert_parameters) and not any(nd in n for nd in no_decay)], 
            'weight_decay': args.weight_decay,
            'lr': args.lr
        },
        {
            'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in bert_parameters) and any(nd in n for nd in no_decay)], 
            'weight_decay': 0.,
            'lr': args.lr
        },
    ]
    optimizer = AdamW(optimizer_grouped_parameters)
    num_training_steps = args.num_epochs * len(train_dataloader) // args.gradient_accumulation_steps
    num_warmup_steps = num_training_steps * args.warmup_ratio
    scheduler = get_linear_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps
    )

    model.to(device)
    criterion = nn.CrossEntropyLoss().to(device)

    best_acc = best_f1 = best_auc = 0.
    with tqdm(total=num_training_steps) as pbar:
        for epoch in range(args.num_epochs):
            epoch_loss = 0.
            all_preds, all_labels, all_logits = [], [], []
            for i, batch in enumerate(train_dataloader):
                model.train()
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
                
                logits = model(text_features, numerical_features)
                loss = criterion(logits, labels)
                loss = loss / args.gradient_accumulation_steps
                loss.backward()
                if (i + 1) % args.gradient_accumulation_steps == 0:
                    nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.max_norm)
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()

                    pbar.set_postfix_str(f'{loss.item():.4f}')
                    pbar.update(1)
                epoch_loss += loss.item()
                all_preds += logits.argmax(1).tolist()
                all_labels += labels.tolist()
                all_logits += F.softmax(logits).tolist()
            if epoch < 999:
                model.eval()
                test_acc, test_f1, test_auc, test_loss = evaluation(model, valid_dataloader, criterion)  
                if test_acc > best_acc:
                    best_acc = test_acc
                    best_f1 = test_f1
                    best_auc = test_auc
                    torch.save(model.state_dict(), output_dir + args.save_name)
                    model.bert_model.save_pretrained(os.path.join(output_dir, 'bert'))
                    tokenizer.save_pretrained(os.path.join(output_dir, 'bert'))
                train_acc, train_f1, train_auc, train_loss = accuracy_score(all_labels, all_preds), \
                                                  f1_score(all_labels, all_preds, average='macro'), \
                                                  roc_auc_score(all_labels, all_logits, average='macro', multi_class='ovr'), \
                                                  epoch_loss / len(train_dataloader)
                print(f'Epoch: {epoch+1:02}')
                print(f'\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc*100:.2f}% | Train F1: {train_f1*100:.2f}% | Train Auc: {train_auc*100:.2f}%')
                print(f'\tTest  Loss: {test_loss:.3f} | Test  Acc: {test_acc*100:.2f}% | Test  F1: {test_f1*100:.2f}% | Test Auc: {test_auc*100:.2f}%')
                logger.info(f'Epoch: {epoch+1:02}')
                logger.info(f'\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc*100:.2f}% | Train F1: {train_f1*100:.2f}% | Train Auc: {train_auc*100:.2f}%')
                logger.info(f'\tTest  Loss: {test_loss:.3f} | Test  Acc: {test_acc*100:.2f}% | Test  F1: {test_f1*100:.2f}% | Test Auc: {test_auc*100:.2f}%')
                
    print(f'Best Test Acc: {best_acc*100:.2f}% | Best Test F1: {best_f1*100:.2f}% | Best Test Auc: {best_auc*100:.2f}%')
    logger.info(f'Best Test Acc: {best_acc*100:.2f}% | Best Test F1: {best_f1*100:.2f}% | Best Test Auc: {best_auc*100:.2f}%')


if __name__ == '__main__':
    main()