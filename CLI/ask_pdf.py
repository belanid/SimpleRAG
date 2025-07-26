#!/usr/bin/env python3
import argparse
from pipeline.pdf_ragger import AskPDF, print_wrapped_text

def main():
    parser = argparse.ArgumentParser(
        description="Interactive PDF Q&A using embeddings + LLM"
    )
    parser.add_argument("pdf_path", type=str, help="Path to the PDF file.")
    parser.add_argument(
        "--chunk_size", type=int, default=10,
        help="Number of sentences per chunk."
    )
    parser.add_argument(
        "--token_count_threshold", type=int, default=30,
        help="Minimum token count per chunk."
    )
    parser.add_argument(
        "--embedding_model_name", type=str, default="all-mpnet-base-v2",
        help="Sentence transformer model for embeddings."
    )
    parser.add_argument(
        "--llm_model_name", type=str, default="google/gemma-2-2b-it",
        help="Hugging Face LLM model."
    )
    parser.add_argument(
        "--max_new_tokens", type=int, default=50,
        help="Max tokens to generate in each answer."
    )
    parser.add_argument(
        "--top_k", type=int, default=5,
        help="Number of top relevant chunks to retrieve."
    )
    parser.add_argument(
        "--device", type=str, default="auto",
        help="Device to use: 'cpu', 'cuda', or 'auto'."
    )

    args = parser.parse_args()

    # Initialize the PDF Q&A tool
    tool = AskPDF(
        pdf_path=args.pdf_path,
        chunk_size=args.chunk_size,
        token_count_threshold=args.token_count_threshold,
        embedding_model_name=args.embedding_model_name,
        llm_model_name=args.llm_model_name,
        device=args.device,
        max_new_tokens=args.max_new_tokens,
        top_k=args.top_k
    )

    # Interactive loop
    print("\nPDF loaded. You can now ask questions. Type 'exit' or 'quit' to close.\n")
    while True:
        query = input(">> ")
        if query.strip().lower() in ("exit", "quit"):  
            print("Exiting. Goodbye!")
            break
        answer = tool.answer(query)
        print_wrapped_text(f"\n{answer}\n")

if __name__ == "__main__":
    main()