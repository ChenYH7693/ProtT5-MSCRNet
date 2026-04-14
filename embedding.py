# =====================================
# ProtT5 + CNN + Transformer for ACP
# Single Script Version
# =====================================
import os
from pathlib import Path

# =========================================================
# Frozen ProtT5 + CNN + Transformer for ACP Classification
# =========================================================

# =========================================================
# Frozen ProtT5 + CNN + Transformer for ACP
# Data from ACP-data/*.csv
# =========================================================

import torch
import pandas as pd
from tqdm import tqdm
from transformers import T5Tokenizer, T5EncoderModel

DEVICE = "cuda:0"  
MODEL_NAME = "Rostlab/prot_t5_xl_uniref50"
MAX_LEN = 512


def extract_and_save(csv_path, save_path):
    tokenizer = T5Tokenizer.from_pretrained(MODEL_NAME, do_lower_case=False)
    model = T5EncoderModel.from_pretrained(MODEL_NAME).to(DEVICE).eval()

    df = pd.read_csv(csv_path, header=None)
    labels = df.iloc[:, 0].tolist()
    seqs = df.iloc[:, 1].tolist()

    embeddings = []

    with torch.no_grad():
        for seq in tqdm(seqs, desc=f"Extracting {csv_path}"):
            
            seq_spaced = " ".join(list(seq))
            inputs = tokenizer(seq_spaced, return_tensors="pt", padding=True,
                               truncation=True, max_length=MAX_LEN).to(DEVICE)

            outputs = model(**inputs)
         
            emb = outputs.last_hidden_state.cpu().squeeze(0)  # (L, 1024)
            embeddings.append(emb)

  
    torch.save({"embeddings": embeddings, "labels": labels}, save_path)
    print(f"Saved to {save_path}")


extract_and_save("ACP-data/Train.fasta", "train_emb.pt")
extract_and_save("ACP-data/Test1.fasta", "test_emb1.pt")
extract_and_save("ACP-data/Test2.fasta", "test_emb2.pt")
