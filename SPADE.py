import sys, os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import BertForMaskedLM, BertTokenizer, AdamW, get_linear_schedule_with_warmup
import argparse
import numpy as np
import warnings
from sklearn.metrics import precision_recall_fscore_support as score
from sklearn.metrics import accuracy_score
from tqdm import tqdm
import pickle

sys.path.append(os.getcwd())
from Process.lm_loadsplits import *

warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser()
parser.add_argument('--dataset_name', default='politifact', type=str)
parser.add_argument('--model_name', default='SPADE', type=str)
parser.add_argument('--iters', default=20, type=int)
parser.add_argument('--batch_size', default=16, type=int)
parser.add_argument('--n_epochs', default=3, type=int)
parser.add_argument('--n_samples', default=16, type=int)
parser.add_argument('--u_thres', default=5, type=int)
args = parser.parse_args()

device = torch.device("cuda")
torch.manual_seed(0)
np.random.seed(0)
torch.backends.cudnn.deterministic = True
torch.cuda.manual_seed_all(0)

datasetname = args.dataset_name
batch_size = args.batch_size
user_threshold = args.u_thres
n_samples = args.n_samples
max_len = 512
tokenizer = BertTokenizer.from_pretrained("./bert-base-uncased")

train_conf, adj = pickle.load(
    open('data/adjs/user_t' + str(user_threshold) + '/' + datasetname + '_nn_relations_' + str(n_samples) + '.pkl',
         'rb'))
A_nn = adj.todense()
A_nn = torch.FloatTensor(A_nn).to(device)
train_conf = torch.Tensor(train_conf).to(device)


def compute_js_divergence(p, q):
    m = 0.5 * (p + q)
    kl_p = F.kl_div(torch.log(m + 1e-9), p, reduction='none').sum(dim=1)
    kl_q = F.kl_div(torch.log(m + 1e-9), q, reduction='none').sum(dim=1)
    return 0.5 * (kl_p + kl_q)


class NewsDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len, template_type="p1"):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.template_type = template_type

    def __getitem__(self, item):
        text = self.texts[item]
        text = f"{self.tokenizer.mask_token}: " + text
        label_logit = self.labels[item]
        if label_logit == 0:
            label = np.array([1, 0])
        elif label_logit == 1:
            label = np.array([0, 1])

        encoding = self.tokenizer.encode_plus(text, add_special_tokens=True, max_length=self.max_len,
                                              pad_to_max_length=True, truncation=True, return_token_type_ids=False,
                                              return_attention_mask=True, return_tensors='pt')
        token_ids = encoding['input_ids']
        masked_position = (token_ids.squeeze() == tokenizer.mask_token_id).nonzero().item()

        return {
            'news_text': text,
            'input_ids': token_ids.flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.FloatTensor(label),
            'masked_pos': torch.tensor(masked_position, dtype=torch.long),
        }

    def __len__(self):
        return len(self.texts)


class BERTPrompt(nn.Module):
    def __init__(self):
        super(BERTPrompt, self).__init__()
        self.bert = BertForMaskedLM.from_pretrained('./bert-base-uncased')
        self.softmax = torch.nn.Softmax(dim=-1)

    def forward(self, input_ids, masked_position):
        input_ids = input_ids.squeeze(1)
        output = self.bert(input_ids=input_ids)
        last_hidden_state = output[0]
        mask_hidden_state = last_hidden_state.select(1, masked_position)
        mask_hidden_state = self.softmax(mask_hidden_state)
        real_id = tokenizer.convert_tokens_to_ids('news')
        fake_id = tokenizer.convert_tokens_to_ids('rumor')
        indices = torch.tensor([real_id, fake_id]).to(device)
        probs = mask_hidden_state.index_select(1, indices)
        return probs


def create_train_loader(contents, labels, tokenizer, max_len, batch_size, template_type="p1"):
    ds = NewsDataset(texts=contents, labels=np.array(labels), tokenizer=tokenizer,
                     max_len=max_len, template_type=template_type)
    return DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=0)


def create_eval_loader(contents, labels, tokenizer, max_len, batch_size, template_type="p1"):
    ds = NewsDataset(texts=contents, labels=np.array(labels), tokenizer=tokenizer,
                     max_len=max_len, template_type=template_type)
    return DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)


def train_single_model(template_type, x_train, x_test, y_train, y_test, tokenizer, max_len, n_epochs, batch_size,
                       seed_offset):
    torch.manual_seed(seed_offset)
    torch.cuda.manual_seed_all(seed_offset)

    model = BERTPrompt().to(device)
    optimizer = AdamW(model.parameters(), lr=5e-5)
    train_loader = create_train_loader(x_train, y_train, tokenizer, max_len, batch_size, template_type)
    test_loader = create_eval_loader(x_test, y_test, tokenizer, max_len, batch_size, template_type)

    total_steps = len(train_loader) * n_epochs
    if len(x_train) == 16:
        total_steps = 5 * n_epochs

    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)

    for epoch in range(n_epochs):
        model.train()
        for Batch_data in tqdm(train_loader, desc=f"Training {template_type} Ep {epoch}", leave=False):
            input_ids = Batch_data["input_ids"].to(device)
            targets = Batch_data["labels"].to(device)
            out_labels = model(input_ids=input_ids, masked_position=Batch_data["masked_pos"][0].item())
            loss_func = nn.BCELoss()
            loss = loss_func(out_labels, targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

    model.eval()
    y_probs = []
    y_targets = []
    with torch.no_grad():
        for Batch_data in test_loader:
            input_ids = Batch_data["input_ids"].to(device)
            targets = Batch_data["labels"].to(device)
            target_logits = torch.nonzero(targets)[:, -1]
            test_out = model(input_ids=input_ids, masked_position=Batch_data["masked_pos"][0].item())
            y_probs.append(test_out)
            y_targets.append(target_logits)

    return torch.cat(y_probs, dim=0), torch.cat(y_targets, dim=0)


def robust_refined_voting(probs_a, probs_b, temperature=2.0):
    entropy_a = -torch.sum(probs_a * torch.log(probs_a + 1e-9), dim=1)
    entropy_b = -torch.sum(probs_b * torch.log(probs_b + 1e-9), dim=1)

    weights = torch.stack([torch.exp(-entropy_a), torch.exp(-entropy_b)], dim=1)
    weights = F.softmax(weights, dim=1)

    w_a = weights[:, 0].unsqueeze(1)
    w_b = weights[:, 1].unsqueeze(1)

    probs_a_smooth = F.softmax(torch.log(probs_a + 1e-9) / temperature, dim=-1)
    probs_b_smooth = F.softmax(torch.log(probs_b + 1e-9) / temperature, dim=-1)

    final_probs = w_a * probs_a_smooth + w_b * probs_b_smooth
    return final_probs


def train_model(args, x_train, x_test, y_train, y_test, tokenizer, max_len, n_epochs, batch_size, datasetname, iter):
    probs_a, y_test_labels = train_single_model("Model A", x_train, x_test, y_train, y_test,
                                                tokenizer, max_len, n_epochs, batch_size, seed_offset=iter)

    probs_b, _ = train_single_model("Model B", x_train, x_test, y_train, y_test,
                                    tokenizer, max_len, n_epochs, batch_size, seed_offset=iter + 100)

    print(f"\n[Iteration {iter:02d}] INFO: Performing Robust Refined Voting (T=2)...")
    y_probs_voted = robust_refined_voting(probs_a, probs_b, temperature=2)

    confidence_scores, temp_preds = y_probs_voted.max(dim=1)
    disagreement = compute_js_divergence(probs_a, probs_b)

    penalty_factor = 0.5
    ranking_scores = confidence_scores - (penalty_factor * disagreement)

    num_test = y_probs_voted.shape[0]
    top_k_per_class = int(num_test * 0.30)
    if top_k_per_class < 5: top_k_per_class = 5

    print(f"[Iteration {iter:02d}] INFO: Anchor Selection (Top-K={top_k_per_class} per class)...")
    y_probs_pseudo = torch.zeros_like(y_probs_voted).to(device)

    class0_indices = (temp_preds == 0).nonzero(as_tuple=True)[0]
    if len(class0_indices) > 0:
        c0_scores = ranking_scores[class0_indices]
        k_0 = min(len(class0_indices), top_k_per_class)
        _, topk_local_idx = torch.topk(c0_scores, k_0)
        global_indices_0 = class0_indices[topk_local_idx]
        y_probs_pseudo[global_indices_0] = torch.tensor([1.0, 0.0]).to(device)

    class1_indices = (temp_preds == 1).nonzero(as_tuple=True)[0]
    if len(class1_indices) > 0:
        c1_scores = ranking_scores[class1_indices]
        k_1 = min(len(class1_indices), top_k_per_class)
        _, topk_local_idx = torch.topk(c1_scores, k_1)
        global_indices_1 = class1_indices[topk_local_idx]
        y_probs_pseudo[global_indices_1] = torch.tensor([0.0, 1.0]).to(device)

    pseudo_cnt = (y_probs_pseudo.sum(dim=1) > 0).sum().item()
    print(f"[Iteration {iter:02d}] SUCCESS: Generated {pseudo_cnt} reliable anchors via Silent Mechanism.")

    y_probs_final = torch.cat([train_conf, y_probs_pseudo], dim=0)
    print(f"[Iteration {iter:02d}] INFO: Executing Selective Graph Propagation...")

    y_probs_upd = torch.matmul(A_nn, y_probs_final)
    y_probs_upd = torch.matmul(A_nn, y_probs_upd)

    _, y_pred_upd = y_probs_upd.max(dim=1)

    prob_sums = y_probs_upd.sum(dim=1)

    is_silent_node = (prob_sums < 1e-6)

    _, y_pred_upd = y_probs_upd.max(dim=1)
    y_pred_upd[is_silent_node] = 0

    y_pred_upd = y_pred_upd[args.n_samples:]
    y_test_labels_sliced = y_test_labels

    acc = accuracy_score(y_test_labels_sliced.detach().cpu().numpy(), y_pred_upd.detach().cpu().numpy())
    precision, recall, fscore, _ = score(y_test_labels_sliced.detach().cpu().numpy(), y_pred_upd.detach().cpu().numpy(),
                                         average='macro')

    print(f"{'=' * 20} Iteration {iter:02d} Summary {'=' * 20}")
    print(f"Metrics: Acc: {acc:.4f} | Prec: {precision:.4f} | Rec: {recall:.4f} | F1: {fscore:.4f}")
    print(f"{'=' * 50}\n")

    return acc, precision, recall, fscore


n_epochs = args.n_epochs
batchsize = args.batch_size
iterations = args.iters
test_accs = []
prec_all, rec_all, f1_all = [], [], []

x_train, x_test, y_train, y_test = get_splits_fewshot(datasetname, args.n_samples)

for iter in range(iterations):
    acc, prec, recall, f1 = train_model(args,
                                        x_train, x_test, y_train, y_test,
                                        tokenizer,
                                        max_len,
                                        n_epochs,
                                        batch_size,
                                        datasetname,
                                        iter)
    test_accs.append(acc)
    prec_all.append(prec)
    rec_all.append(recall)
    f1_all.append(f1)

print("Total_Test_Accuracy: {:.4f}|Prec_Macro: {:.4f}|Rec_Macro: {:.4f}|F1_Macro: {:.4f}".format(
    sum(test_accs) / iterations, sum(prec_all) / iterations, sum(rec_all) / iterations, sum(f1_all) / iterations))

log_path = f'logs/log_{datasetname}_{args.n_samples}shot_SPADE_t{user_threshold}.iter{iterations}'

with open(log_path, 'a+') as f:
    f.write("-" * 30 + " Experiment Configuration " + "-" * 30 + "\n")
    f.write(f"Method: SPADE (Selective Propagation via Anchor-based Dual-seed Ensemble)\n")
    f.write(f"Dataset: {datasetname} | Shot: {args.n_samples} | User_Thres: {user_threshold}\n")
    f.write(f"Hyperparams: Epochs: {n_epochs} | Batch: {batchsize} | Total_Iters: {iterations}\n")

    f.write("-" * 30 + " Results per Iteration " + "-" * 30 + "\n")
    f.write(f"All Accuracies: {test_accs}\n")

    f.write("-" * 30 + " Overall Performance " + "-" * 30 + "\n")
    f.write(f"Average Accuracy:  {sum(test_accs) / iterations:.4f}\n")
    f.write(f"Average Precision: {sum(prec_all) / iterations:.4f}\n")
    f.write(f"Average Recall:    {sum(rec_all) / iterations:.4f}\n")
    f.write(f"Average F1-Macro:  {sum(f1_all) / iterations:.4f}\n")
    f.write("-" * 85 + "\n\n")