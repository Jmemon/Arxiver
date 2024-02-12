
from datetime import datetime
from functools import reduce
from pathlib import Path
import sqlite3
from typing import List

from langchain.prompts import PromptTemplate
from langchain.docstore.document import Document
from langchain_community.embeddings.huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores.chroma import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

from load_dotenv import load_dotenv
from unstructured.partition.auto import partition_pdf, PartitionStrategy

from qa_chain.models import mistral7b


def get_pdf_chunks(paper_path: Path) -> List[str]:
    paper_elements = []
    for el in partition_pdf(str(paper_path), strategy=PartitionStrategy.HI_RES):
        if str(el).lower().strip() == 'references':
            break

        if len(str(el).split(' ')) < 7:  # exclude titles
            continue

        if str(el).lower().strip().startswith('figure'):  # exclude figure captions
            continue

        if str(el).lower().strip().startswith('table'):  # exlude table captions
            continue

        paper_elements.append(str(el))

    return paper_elements


def get_text_vectorstore(paper_paths: List[Path] | Path, vdb_name: str | None = None) -> Chroma | None:
    """
    Get vectorstore from a sequence of text documents.
    If the 
    """
    if isinstance(paper_paths, Path):
        paper_paths = [paper_paths]
    
    if vdb_name is None:
        vdb_name = datetime.now().strftime('%y-%m-%d_%H-%M-%S')

    with sqlite3.connect(str(Path(__file__).parent / 'vectorstores' / 'chroma.sqlite3')) as conn:
        cursor = conn.cursor()
        cursor.execute('SELECT name FROM collections')
        vdb_exists = vdb_name in [row[0] for row in cursor.fetchall()]

    if vdb_exists:
        vdb = Chroma(collection_name=vdb_name)

    else:
        vdb = Chroma.from_documents(
            [reduce(
                lambda x, y: x + y, 
                [Document(el) for el in get_pdf_chunks(paper_path)], 
                []) 
             for paper_path in paper_paths], 
            embedding=HuggingFaceEmbeddings())

    return vdb


"""
if __name__ == '__main__':
    load_dotenv()

    rag_chain = (
        {
            'context': get_text_vectorstore(
                            [Path(__file__).parent.parent / 'papers' / 'llm_apps' / 'mixtral_of_experts.pdf'],
                            'mixtral_of_experts') 
                        | format_docs, 
            'question': RunnablePassthrough()}
        | PromptTemplate(
                input_variables=['question', 'context'],
                template=''
                    '<s> [INST] You are an assistant for question-answering tasks. Use the following pieces of '
                    'retrieved context to answer the question. If you don\'t know the answer, just say that you '
                    'don\'t know. [/INST] </s> \n'
                    '[INST] Question: {question} \n'
                    'Context: {context} \n'
                    'Answer: [/INST]'
            )
        | mistral7b(n_ctx=2048, max_tokens=None)
        | StrOutputParser()
    )"""
