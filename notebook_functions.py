import os
import time
from pydoc import describe
import requests
import fitz  # PyMuPDF
from tqdm import tqdm
import random
import pandas as pd
import numpy as np
from spacy.lang.en import English

import textwrap
import torch
from sentence_transformers import util, SentenceTransformer


def text_formatter(text: str) -> str:
    """
    Replace newlines with spaces and trim whitespace at the end of the text.
    """
    return text.replace('\n', ' ').rstrip()

def open_and_read_pdf(pdf_path: str) -> list:
    """
    Open a PDF file and return a list of dicts with page number as key and page text as value.
    """
    doc = fitz.open(pdf_path)
    pages = []
    for page_num, page in enumerate(tqdm(doc, desc="Reading PDF pages")):
        page_text = text_formatter(text=page.get_text())
        pages.append({
            "page_num": page_num - 41,
            "page_char_count": len(page_text),
            "page_word_count": len(page_text.split()),
            "page_sent_count": len(page_text.split('. ')),
            "page_token_count": len(page_text) / 4,
            "text": page_text
        })
    doc.close()
    return pages

def split_pages_per_sentence(pdf_pages: list[dict]) -> list[dict]:
    from spacy.lang.en import English

    nlp = English()
    nlp.add_pipe("sentencizer")

    for item in tqdm(pdf_pages, desc="Processing pages"):
        item["sentences"] = list(nlp(item["text"]).sents)
        item["sentences"] = [s.text for s in item["sentences"]]
        item["sent_count"] = len(item["sentences"])

    return pdf_pages

def chunk_sentences(pdf_pages: list[dict], chunk_size: int) -> list[dict]:
    """
    Chunk sentences into groups of a specified size.
    """
    for item in tqdm(pdf_pages, desc="Chunking sentences"):
        item["chunks"] = [item["sentences"][i:i + chunk_size] for i in range(0, len(item["sentences"]), chunk_size)]
        item["page_chunk_count"] = len(item["chunks"])

    return pdf_pages

def split_chunks_into_items(pages: list[dict], token_count_threshold: int=30) -> list[dict]:
    """
    Split each chunk in the pages into separate items.
    Each item will contain the chunk text and metadata from the original page.
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

def create_embedding_from_pdf(pdf_path: str,
        chunk_size:int=10,
        token_count_threshold:int=30,
        embedding_model_name: str="all-mpnet-base-v2") -> list[dict]:

    time_start = time.time()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    base, ext = os.path.splitext(pdf_path)
    fout_path = (
        f"{base}_chunksize{chunk_size}_tokenthresh{token_count_threshold}_embedder_{embedding_model_name}.csv"
    )

    if os.path.exists(fout_path):
        print(f"Embeddings file {fout_path} exist! Loading ...")
        #read the embeddings and text chunks data frame
        pdf_chunks_df = pd.read_csv(fout_path)

        # Convert embedding column back to np.array (it got converted to string when it got saved to CSV)
        pdf_chunks_df["embedding"] = pdf_chunks_df["embedding"].apply(lambda x: np.fromstring(x.strip("[]"), sep=" "))
       
        # create a ditionnary from the data frame
        pdf_chunks = pdf_chunks_df.to_dict(orient="records")
        print("Loading complete!")

    else:
        pdf_pages = open_and_read_pdf(pdf_path)

        pdf_pages = split_pages_per_sentence(pdf_pages)

        pdf_pages = chunk_sentences(pdf_pages, chunk_size=chunk_size)

        pdf_chunks = split_chunks_into_items(pdf_pages)

        print("Creating embedding arrays ...")
        embedder = SentenceTransformer(model_name_or_path=embedding_model_name, device=device)

        # test the embedder on a sample of 3 chunks
        chunk_text = [item['text'] for item in pdf_chunks]

        # Compute embeddings for the sample texts
        embeddings = embedder.encode(chunk_text, show_progress_bar=True, convert_to_tensor=True)

        for chunk_idx, item in tqdm(enumerate(pdf_chunks), desc="Adding embeddings to dictionnary"):
            item["embedding"] = np.array(embeddings[chunk_idx].cpu())
        
        pdf_chunks_df = pd.DataFrame(pdf_chunks)
        pdf_chunks_df.to_csv(fout_path, index=False)

    time_end = time.time()
    duration = time_end - time_start
    hours = int(duration // 3600)
    minutes = int((duration % 3600) // 60)
    seconds = duration % 60
    print(f"\n------- Time taken: {hours:02d}:{minutes:02d}:{seconds:06.3f}")

    return pdf_chunks


# --- Functions from 00-simple-local-rag-nidhal-part2.ipynb ---



def print_wrapped_text(text, width=100):
    """
    Print the given text wrapped to the specified width for better readability.
    Existing newlines are preserved.
    """
    for line in text.splitlines():
        print(textwrap.fill(line, width=width, replace_whitespace=False))

def get_top_k_results(query, embeddings, embedder, dict_chunks, top_k=5, verbose=True):
    """
    Retrieve the top_k most relevant chunks for a query using dot product similarity.
    """
    import time
    query_embedding = embedder.encode(query, convert_to_tensor=True)
    start_time = time.time()
    dot_product = util.dot_score(a=query_embedding, b=embeddings)[0]
    scores, indices = torch.topk(dot_product, k=top_k)
    end_time = time.time()
    if verbose:
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

def get_model_mem_size(model: torch.nn.Module):
    """
    Get how much memory a PyTorch model takes up.
    """
    param_size = sum([param.nelement() * param.element_size() for param in model.parameters()])
    buff_size = sum([buff.nelement() * buff.element_size() for buff in model.buffers()])
    return param_size + buff_size

def get_model_num_param(model: torch.nn.Module):
    """
    Get the number of parameters in a PyTorch model.
    """
    return sum([param.numel() for param in model.parameters()])

def prompt_augmenter(query: str, context_items: list, tokenizer=None) -> str:
    """
    Augments query with text-based context from context_items.
    """
    context = "- " + "\n- ".join([item["text"] for item in context_items])
    base_prompt = """Based on the following context items, please answer the query.
    Give yourself room to think by extracting relevant passages from the context before answering the query.
    Don't return the thinking, only return the answer.
    Make sure your answers are as explanatory as possible.
    Use the following examples as reference for the ideal answer style.
    \nExample 1:
    Query: What are the fat-soluble vitamins?
    Answer: The fat-soluble vitamins include Vitamin A, Vitamin D, Vitamin E, and Vitamin K. These vitamins are absorbed along with fats in the diet and can be stored in the body's fatty tissue and liver for later use. Vitamin A is important for vision, immune function, and skin health. Vitamin D plays a critical role in calcium absorption and bone health. Vitamin E acts as an antioxidant, protecting cells from damage. Vitamin K is essential for blood clotting and bone metabolism.
    \nExample 2:
    Query: What are the causes of type 2 diabetes?
    Answer: Type 2 diabetes is often associated with overnutrition, particularly the overconsumption of calories leading to obesity. Factors include a diet high in refined sugars and saturated fats, which can lead to insulin resistance, a condition where the body's cells do not respond effectively to insulin. Over time, the pancreas cannot produce enough insulin to manage blood sugar levels, resulting in type 2 diabetes. Additionally, excessive caloric intake without sufficient physical activity exacerbates the risk by promoting weight gain and fat accumulation, particularly around the abdomen, further contributing to insulin resistance.
    \nExample 3:
    Query: What is the importance of hydration for physical performance?
    Answer: Hydration is crucial for physical performance because water plays key roles in maintaining blood volume, regulating body temperature, and ensuring the transport of nutrients and oxygen to cells. Adequate hydration is essential for optimal muscle function, endurance, and recovery. Dehydration can lead to decreased performance, fatigue, and increased risk of heat-related illnesses, such as heat stroke. Drinking sufficient water before, during, and after exercise helps ensure peak physical performance and recovery.
    \nNow use the following context items to answer the user query:
    {context}
    \nRelevant passages: <extract relevant passages from the context here>
    User query: {query}
    Answer:"""
    base_prompt = base_prompt.format(context=context, query=query)
    if tokenizer is not None:
        dialogue_template = [
            {"role": "user", "content": base_prompt}
        ]
        prompt = tokenizer.apply_chat_template(conversation=dialogue_template,
                                               tokenize=False,
                                               add_generation_prompt=True)
        return prompt
    else:
        return base_prompt 