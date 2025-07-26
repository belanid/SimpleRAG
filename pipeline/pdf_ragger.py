import os
import time
import requests
import fitz  # PyMuPDF
from tqdm import tqdm
import pandas as pd
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from sentence_transformers import util, SentenceTransformer
import textwrap
from spacy.lang.en import English
from typing import List, Dict, Tuple, Optional

# =========================
# Utility Functions
# =========================
def text_formatter(text: str) -> str:
    """
    Replace newlines with spaces and trim whitespace at the end of the text.
    """
    return text.replace('\n', ' ').rstrip()

# =========================
# PDF Processing Functions
# =========================
def open_and_read_pdf(pdf_path: str) -> List[Dict]:
    """
    Open a PDF file and return a list of dicts with page metadata and text.
    """
    doc = fitz.open(pdf_path)
    pages = []
    for page_num, page in enumerate(tqdm(doc, desc="Reading PDF pages")):
        page_text = text_formatter(page.get_text())
        pages.append({
            "page_num": page_num - 41,  # TODO: Remove magic number or explain
            "page_char_count": len(page_text),
            "page_word_count": len(page_text.split()),
            "page_sent_count": len(page_text.split('. ')),
            "page_token_count": len(page_text) / 4,
            "text": page_text
        })
    doc.close()
    return pages

def split_pages_per_sentence(pdf_pages: List[Dict]) -> List[Dict]:
    """
    Split each page's text into sentences using spaCy's sentencizer.
    """
    nlp = English()
    nlp.add_pipe("sentencizer")
    for item in tqdm(pdf_pages, desc="Processing pages"):
        item["sentences"] = [s.text for s in nlp(item["text"]).sents]
        item["sent_count"] = len(item["sentences"])
    return pdf_pages

def chunk_sentences(pdf_pages: List[Dict], chunk_size: int) -> List[Dict]:
    """
    Chunk sentences into groups of a specified size.
    """
    for item in tqdm(pdf_pages, desc="Chunking sentences"):
        item["chunks"] = [item["sentences"][i:i + chunk_size] for i in range(0, len(item["sentences"]), chunk_size)]
        item["page_chunk_count"] = len(item["chunks"])
    return pdf_pages

def split_chunks_into_items(pages: List[Dict], token_count_threshold: int = 30) -> List[Dict]:
    """
    Split each chunk in the pages into separate items with metadata.
    Only keep chunks above the token_count_threshold.
    """
    chunk_items = []
    for page in tqdm(pages, desc="Splitting chunks into items"):
        page_num = page["page_num"]
        for chunk_idx, chunk in enumerate(page["chunks"]):
            chunk_text = " ".join(chunk).strip()
            chunk_metadata = {
                "page_num": page_num,
                "chunk_idx": chunk_idx,
                "chunk_size": len(chunk),
                "chunk_char_count": len(chunk_text),
                "chunk_word_count": len(chunk_text.split()),
                "chunk_sent_count": len(chunk),
                "chunk_token_count": len(chunk_text) / 4,
                "text": chunk_text,
                "sentences": chunk
            }
            chunk_items.append(chunk_metadata)
    filtered_chunk_items = [item for item in chunk_items if item['chunk_token_count'] >= token_count_threshold]
    return filtered_chunk_items

# =========================
# Embedding Functions
# =========================
def create_embedding_from_pdf(
    pdf_path: str,
    chunk_size: int = 10,
    token_count_threshold: int = 30,
    embedding_model_name: str = "all-mpnet-base-v2"
) -> Tuple[List[Dict], torch.Tensor, SentenceTransformer]:
    """
    Process a PDF, chunk its text, and create embeddings for each chunk.
    Returns the chunk metadata, embeddings tensor, and the embedder.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading embedder from {embedding_model_name} ...")
    embedder = SentenceTransformer(model_name_or_path=embedding_model_name, device=device)
    base, ext = os.path.splitext(pdf_path)
    fout_path = f"{base}_chunksize{chunk_size}_tokenthresh{token_count_threshold}_embedder_{embedding_model_name}.csv"
    if os.path.exists(fout_path):
        print(f"Embeddings file {fout_path} exists! Loading ...")
        pdf_chunks_df = pd.read_csv(fout_path)
        pdf_chunks_df["embedding"] = pdf_chunks_df["embedding"].apply(lambda x: np.fromstring(x.strip("[]"), sep=" "))
        pdf_chunks = pdf_chunks_df.to_dict(orient="records")
        embeddings = torch.tensor(np.stack(pdf_chunks_df["embedding"].values, axis=0), device=device, dtype=torch.float32)
        print("Loading complete!")
    else:
        pdf_pages = open_and_read_pdf(pdf_path)
        pdf_pages = split_pages_per_sentence(pdf_pages)
        pdf_pages = chunk_sentences(pdf_pages, chunk_size=chunk_size)
        pdf_chunks = split_chunks_into_items(pdf_pages, token_count_threshold=token_count_threshold)
        print("Creating embedding arrays ...")
        chunk_texts = [item['text'] for item in pdf_chunks]
        embeddings = embedder.encode(chunk_texts, show_progress_bar=True, convert_to_tensor=True)
        for chunk_idx, item in tqdm(list(enumerate(pdf_chunks)), desc="Adding embeddings to dictionary"):
            item["embedding"] = np.array(embeddings[chunk_idx].cpu())
        pdf_chunks_df = pd.DataFrame(pdf_chunks)
        pdf_chunks_df.to_csv(fout_path, index=False)
    return pdf_chunks, embeddings, embedder

# =========================
# Retrieval and LLM Functions
# =========================
def print_wrapped_text(text: str, width: int = 100):
    """
    Print the given text wrapped to the specified width for better readability.
    Existing newlines are preserved.
    """
    for line in text.splitlines():
        print(textwrap.fill(line, width=width, replace_whitespace=False))

def get_top_k_results(
    query: str,
    embeddings: torch.Tensor,
    embedder: SentenceTransformer,
    dict_chunks: Optional[List[Dict]] = None,
    top_k: int = 5,
    verbose: bool = True
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Retrieve the top_k most relevant chunks for a query using dot product similarity.
    """
    query_embedding = embedder.encode(query, convert_to_tensor=True)
    start_time = time.time()
    dot_product = util.dot_score(a=query_embedding, b=embeddings)[0]
    scores, indices = torch.topk(dot_product, k=top_k)
    end_time = time.time()
    if verbose and dict_chunks is not None:
        print(f"Top {top_k} results: ", indices.cpu().numpy())
        print(f"Top {top_k} scores: ", scores.cpu().numpy())
        print(f"Time taken with torch.topk: {(end_time - start_time) * 1000:.2f} ms")
        print("\n" + "_"*100 + "\n")
        for rank, idx in enumerate(indices.cpu().numpy().tolist()):
            print(f"Chunk {idx}:")
            print(f"Score: {scores[rank]:.4f}")
            print(f"page: {dict_chunks[idx]['page_num']}")
            print_wrapped_text(dict_chunks[idx]["text"])
            print("\n" + "-"*100 + "\n")
    return scores, indices

def ask_pdf(
    query: str,
    pdf_path: str,
    chunk_size: int = 10,
    token_count_threshold: int = 30,
    max_new_tokens: int = 50,
    top_k: int = 5,
    device: str = "auto",
    llm_model_name: str = "google/gemma-2-2b-it",
    embedding_model_name: str = "all-mpnet-base-v2"
):
    """
    Main entry point: Given a query and a PDF, retrieve relevant context and answer using an LLM.
    """
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    # 1. Load PDF and create embeddings
    pdf_chunks, embeddings, embedder = create_embedding_from_pdf(
        pdf_path,
        chunk_size=chunk_size,
        token_count_threshold=token_count_threshold,
        embedding_model_name=embedding_model_name
    )
    # 2. Get the top k closest embedding chunks to the query
    scores, indices = get_top_k_results(query, embeddings, embedder, pdf_chunks, top_k=top_k, verbose=False)
    # 3. Setup a quantization config
    quantization_config = BitsAndBytesConfig(load_in_4bit=True)
    # 4. Load the LLM model
    model = AutoModelForCausalLM.from_pretrained(
        llm_model_name,
        quantization_config=quantization_config,
        device_map=device,
        trust_remote_code=True
    )
    # 5. Load the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(llm_model_name, device_map=device)
    # 6. Format the query
    user_prompt = query
    chat = [
        {"role": "user", "content": user_prompt},
    ]
    prompt = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
    # 7. Tokenize the prompt
    inputs = tokenizer(prompt, add_special_tokens=False, return_tensors="pt").to(device)
    # 8. Generate the response
    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens
    )
    # 9. Decode (de-tokenize) the response and print
    print_wrapped_text(tokenizer.decode(outputs[0], skip_special_tokens=True))