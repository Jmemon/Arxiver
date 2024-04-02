
from datetime import datetime
from functools import reduce
from pathlib import Path
import sqlite3
from typing import List

from langchain.prompts import PromptTemplate
from langchain.docstore.document import Document
from langchain_community.embeddings.huggingface import HuggingFaceBgeEmbeddings
from langchain_community.vectorstores.chroma import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableSequence

from unstructured.partition.auto import partition_pdf, PartitionStrategy

from utils import get_version
from qa_chain import vectorstore_dir
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


def get_simple_rag_chain(papers: List[Path], vdb_name: str | None = None, callbacks: List | None = None) -> RunnableSequence:
    if callbacks is None:
        callbacks = []
    
    retriever = (
        get_text_vectorstore(papers, vdb_name).as_retriever(search_kwargs={'k': 6})
        | (lambda doc_list: '\n\n'.join([doc.page_content for doc in doc_list]))
    ).with_config({
            'metadata': {
                'sources': [paper.name for paper in papers]
            },
            'tags': ['Chroma', HuggingFaceBgeEmbeddings.__name__],
            'handlers': callbacks
        }
    )

    prompt_templ = PromptTemplate(
            input_variables=['question', 'context'],
            template=''
                '<s> [INST] You are an assistant for question-answering tasks. Use the following pieces of '
                'retrieved context to answer the question. If the context doesn\'t seem relevant to the question, '
                'say that you don\'t know and that the question might not be relevant. [/INST] </s> \n'
                '[INST] Question: {question} \n'
                'Context: {context} \n'
                'Answer: [/INST]'
    ).with_config({'handlers': callbacks})

    mistral = mistral7b(n_ctx=3072, max_tokens=None).with_config({'handlers': callbacks})

    return (
        {'context': retriever, 'question': RunnablePassthrough()}
        | prompt_templ
        | mistral
        | StrOutputParser().with_config({'handlers': callbacks})
    ).with_config({
        'metadata': {
            'version': get_version()
        },
        'tags': ['simple_rag_chain']
    })


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
