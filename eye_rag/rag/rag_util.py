import os
from uuid import uuid4

import tiktoken
from langchain_text_splitters import RecursiveCharacterTextSplitter
from tqdm.asyncio import tqdm


def load_markdown_files(md_folder_path):
    """Load all markdown files with original formatting preserved."""
    md_data = []

    # Recursively find all .md files
    for root, dirs, files in os.walk(md_folder_path):
        for file in files:
            if file.endswith('.md'):
                file_path = os.path.join(root, file)
                try:
                    # Read file with original encoding and formatting
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()  # Keep original content without modification

                    md_data.append({
                        'reference': file_path,
                        'text': content,  # Original content preserved
                        'filename': file
                    })
                    print(f"Loaded: {file}")

                except Exception as e:
                    print(f"Error loading {file_path}: {e}")
                    continue

    print(f"Loaded {len(md_data)} markdown files")
    return md_data


def _get_text_splitter():
    """Helper to get the text splitter optimized for markdown content."""
    tokenizer = tiktoken.get_encoding('cl100k_base')

    def tiktoken_len(text):
        tokens = tokenizer.encode(
            text,
            disallowed_special=()
        )
        return len(tokens)

    return RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100,
        length_function=tiktoken_len,
        separators=[
            "\n## ",
            "\n### ",
            "\n#### ",
            "\n##### ",
            "\n\n",
            "\n",
            ". ",
            "! ",
            "? ",
            "; ",
            ", ",
            " ",
            ""
        ],
        keep_separator=True
    )


def _split_and_format_chunks(medical_guide_data):
    """
    Splits text from guide_data into chunks and formats them.
    Optimized for markdown content structure.
    """
    text_splitter = _get_text_splitter()
    chunks_to_upsert = []

    for idx, record in enumerate(tqdm(medical_guide_data, desc="Processing markdown files")):
        texts = text_splitter.split_text(record['text'])
        chunks_to_upsert.extend([{
            'id': str(uuid4()),
            'chunk_text': texts[i],
            'chunk': i,
            'reference': record['reference'],
            'filename': record.get('filename', '')  # Include filename in metadata
        } for i in range(len(texts))])

    print(f"Processed {len(chunks_to_upsert)} chunks from markdown files.")
    return chunks_to_upsert
