
from datetime import datetime
from pathlib import Path

from langchain import hub
from langchain.callbacks import FileCallbackHandler
from langchain.docstore.document import Document
from langchain.prompts import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores.chroma import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

from loguru import logger
from pypdf import PdfReader

from bot.models import mistral7b


"""
Implement rag pipeline to ask questions about the papers in papers/llm_app.

Start by just getting a pipeline to talk to one paper.
"""

def format_docs(docs):
    return '\n\n'.join(doc.page_content for doc in docs)


if __name__ == '__main__':
    dt = datetime.now().strftime('%Y-%m-$d_%H-%M-%S')
    logfile = Path(__file__).parent / 'logs' / f'{dt}.log'
    logger.add(logfile, colorize=True, enqueue=True)

    paper_dir = Path(__file__).parent / 'papers' / 'llm_apps'
    paper_path = paper_dir / 'mixtral_of_experts.pdf'

    doc = Document(page_content='\n\n'.join(
                        [page.extract_text() 
                         for page in PdfReader(str(paper_path)).pages]))
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=150, chunk_overlap=30)
    splits = text_splitter.split_documents([doc])
    vectorstore = Chroma.from_documents(documents=splits, embedding=HuggingFaceEmbeddings())

    retriever = vectorstore.as_retriever(search_kwargs={'k': 10})
    prompt = PromptTemplate(
        input_variables=['question', 'context'],
        template=''
            '<s> [INST] You are an assistant for question-answering tasks. Use the following pieces of '
            'retrieved context to answer the question. If you don\'t know the answer, just say that you '
            'don\'t know. [/INST] </s> \n'
            '[INST] Question: {question} \n'
            'Context: {context} \n'
            'Answer: [/INST]'
        )

    llm = mistral7b(callbacks=[FileCallbackHandler(str(logfile))])

    rag_chain = (
        {'context': retriever | format_docs, 'question': RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    test_queries = [
        'What benchmarks did the researchers evaluate the model on?',
        'How does the model compare to other models?',
        'How does mixtral compare to other models on the benchmarks it was evaluated on?',
        'Did certain experts become specialized in specific topics?',
        'Can you explain the architecture of the model?',
        'What is the architecture of mixtral?',
        'Can you explain what the sparse mixture of experts method is?',
        'Can you explain what sparse mixture of experts means?',
        'Can you explain what spare mixture of experts means?',
        'Can you explain spare mixture of experts?',
        'What is the defining characteristic of mixtral\'s architecture?'
    ]


    for query_num, query in enumerate(test_queries, start=1):
        print(f'({query_num}/{len(test_queries)}) {query} ')
        response = ''
        for chunk_num, chunk in enumerate(rag_chain.stream(query), start=1):
            response += chunk
            print(f'{chunk_num} toks', end='\r')
        print()
        
        logger.info(query)
        logger.info(response)
