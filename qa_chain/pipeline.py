
from datetime import datetime
from dotenv import load_dotenv
from functools import reduce
import os
from pathlib import Path
import sqlite3
from typing import List

from langchain.prompts import PromptTemplate
from langchain.docstore.document import Document
from langchain_openai.llms import OpenAI
from langchain_community.embeddings.huggingface import HuggingFaceBgeEmbeddings
from langchain_community.vectorstores.chroma import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableSequence, RunnableParallel
from unstructured.partition.pdf import partition_pdf, PartitionStrategy

from utils import get_version
from qa_chain import vectorstore_dir


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
    """
    if isinstance(paper_paths, Path):
        paper_paths = [paper_paths]
    
    if vdb_name is None:
        vdb_name = datetime.now().strftime('%y-%m-%d_%H-%M-%S')

    chroma_sql_path = Path(__file__).parent.parent / 'vectorstores' / 'chroma.sqlite3'
    if not chroma_sql_path.exists():
        vdb_exists = False
    else:
        with sqlite3.connect(str(chroma_sql_path)) as conn:
            cursor = conn.cursor()
            cursor.execute('SELECT name FROM collections')
            vdb_exists = vdb_name in [row[0] for row in cursor.fetchall()]

    if vdb_exists:
        vdb = Chroma(
            collection_name=vdb_name, 
            embedding_function=HuggingFaceBgeEmbeddings(),
            persist_directory=str(vectorstore_dir))

    else:
        vdb = Chroma.from_documents(
                reduce(
                    lambda x, y: x + y, 
                    [[Document(el) for el in get_pdf_chunks(paper_path)] for paper_path in paper_paths], 
                    []
                ), 
                embedding=HuggingFaceBgeEmbeddings(),
                persist_directory=str(vectorstore_dir),
                collection_name=vdb_name)
        vdb.persist()

    return vdb


def get_retriever(papers: List[Path], vdb_name: str | None = None):
    return (
        get_text_vectorstore(papers, vdb_name).as_retriever(search_kwargs={'k': 6})
        | (lambda doc_list: '\n\n'.join([doc.page_content for doc in doc_list]))
    )


def get_prompt_template():
    return PromptTemplate(
        input_variables=['question', 'context'],
        template=''
            '<s> [INST] You are an assistant for question-answering tasks. Use the following pieces of '
            'retrieved context to answer the question. If the context doesn\'t seem relevant to the question, '
            'say that you don\'t know and that the question might not be relevant. [/INST] </s> \n'
            '[INST] Question: {question} \n'
            'Context: {context} \n'
            'Answer: [/INST]'
    )


def get_llm():
    return OpenAI(
        openai_api_key=os.environ.get('ANYSCALE_API_KEY'), 
        openai_api_base=os.environ.get('ANYSCALE_API_BASE'),
        model_name='mistralai/Mistral-7B-Instruct-v0.1',
        streaming=True
    )


def get_rag_chain(
        papers: List[Path], 
        vdb_name: str | None = None, 
        retriever_cbs: List | None = None,
        prompt_cbs: List | None = None,
        llm_cbs: List | None = None,
        chain_cbs: List | None = None
) -> RunnableSequence:
    if chain_cbs is None:
        chain_cbs = []
    
    for cbs in [retriever_cbs, prompt_cbs, llm_cbs, chain_cbs]:
        if cbs is None:
            cbs = []

    retriever = get_retriever(papers, vdb_name).with_config(
        {
            'callbacks': retriever_cbs,
            'metadata': {
                'sources': [paper.name for paper in papers]
            },
            'tags': ['Chroma', HuggingFaceBgeEmbeddings.__name__]
        }
    )

    prompt_templ = get_prompt_template().with_config({'callbacks': prompt_cbs})

    llm = get_llm().with_config({'callbacks': llm_cbs})

    return (
        {'context': retriever, 'question': RunnablePassthrough()}
        | prompt_templ
        | llm
        | StrOutputParser()
    ).with_config({
        'callbacks': chain_cbs,
        'metadata': {
            'version': get_version()
        }
    })
