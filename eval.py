
from datetime import datetime

from dotenv import load_dotenv

from langchain_core.runnables import RunnableLambda
import langsmith

from qa_chain.pipeline import get_rag_chain
from utils import ARXIVER_PATH, LOGS_PATH


# I genereted these with ChatGPT: "Can you generate 50 questions that could be asked of an arbitrary AI research paper?"
eval_queries = [
    'What specific AI technology or model does the paper focus on?',
    'What are the primary research objectives of the study?',
    'How does the paper contribute to the field of artificial intelligence?',
    'What datasets were used, and why were they chosen?',
    'How were the AI models trained and evaluated?',
    'What are the key findings and how do they advance the field?',
    'Were there any novel algorithms or techniques introduced?',
    'How does the study address bias and fairness in AI systems?',
    'What are the computational requirements for the proposed models?',
    'How scalable are the solutions proposed in the paper?',
    'What limitations are acknowledged by the authors?',
    'How does the research compare with existing state-of-the-art AI models?',
    'What ethical considerations are discussed in the context of AI deployment?',
    'How do the authors propose to mitigate potential risks associated with AI?',
    'What are the practical applications of the research findings?',
    'How does the paper address data privacy and security concerns?',
    'What future research directions do the authors suggest?',
    'How does the research impact specific industries or sectors?',
    'Were any open-source tools or frameworks developed as part of the research?',
    'How does the paper contribute to the understanding of AI explainability and interpretability?',
    'What statistical methods were used to analyze the results?',
    'How do the authors validate the robustness of their AI models?',
    'What challenges did the researchers face during the study?',
    'How do the findings affect the development of future AI systems?',
    'Were there any unexpected outcomes or discoveries?',
    'How does the study contribute to interdisciplinary research in AI?',
    'What are the key variables and metrics used in the evaluation?',
    'How do the authors discuss the generalizability of their findings?',
    'What are the key assumptions made in the study?',
    'How does the paper address potential societal impacts of AI technologies?',
    'What collaboration or contributions from other fields are highlighted?',
    'How do the authors foresee the evolution of the AI technologies discussed?',
    'What are the power consumption and environmental impacts of the proposed AI models?',
    'What are the key theoretical frameworks utilized in the study?',
    'How do the authors suggest overcoming the limitations identified?',
    'What are the main controversies or debates related to the paper\'s topic?',
    'How does the paper fit within the current AI research landscape?',
    'What methodologies are proposed for ensuring the reliability of AI systems?',
    'How does the research address the integration of AI with other technologies?',
    'What are the potential commercial implications of the research findings?',
    'How does the paper address the longevity and maintenance of AI systems?',
    'What are the contributions of the paper to AI education and literacy?',
    'How does the research facilitate the development of AI standards and best practices?',
    'What are the global implications of the research findings?',
    'How does the paper address the challenge of AI model reproducibility?',
    'What are the implications of the research for AI governance?',
    'How does the paper contribute to the ethical design and deployment of AI technologies?',
    'How do application architectures for LLMs differ for cloud-based versus edge computing environments?',
    'What are the critical considerations in designing application architectures for real-time LLM applications?',
    'Describe how containerization technologies like Docker can be used with LLMs to improve deployment flexibility.',
    'In what ways do microservices architectures benefit the scalability and maintainability of LLM applications?',
    'How can LLMs be integrated into existing legacy systems within enterprise architectures?',
    'What advancements in LLM architectures have enabled better handling of context and memory?',
    'How do different attention mechanisms in LLM architectures affect their language comprehension abilities?',
    'Discuss the impact of layer normalization techniques on the training stability of LLMs.',
    'Compare the efficiency of sparse versus dense architectures in LLMs.',
    'Explain how hybrid architectures combine the strengths of CNNs and RNNs with transformer models in LLMs.',
    'What role does hardware acceleration (e.g., GPUs, TPUs) play in optimizing LLM inference?',
    'Discuss the benefits and trade-offs of using distillation methods for LLM inference optimization.',
    'How does dynamic batching improve the efficiency of LLM inference?',
    'Describe the challenges and solutions in optimizing LLMs for low-resource languages.',
    'Explain the use of adaptive computation time (ACT) techniques in LLM inference optimization.',
    'How does the choice of knowledge base affect the performance of retrieval-augmented generation in LLMs?',
    'Discuss the mechanisms of incorporating real-time web search results into LLM responses.',
    'What are the challenges in ensuring the relevance and accuracy of retrieved information in RAG models?',
    'Compare the effectiveness of different retrieval strategies used in RAG for LLMs.',
    'Explain how RAG models balance the trade-off between retrieval latency and response quality.',
    'Describe the role of LLMs in developing conversational agents for customer service applications.',
    'How do reinforcement learning techniques enhance the capabilities of LLM-based agents?',
    'Discuss the ethical considerations in deploying autonomous agents powered by LLMs in public domains.',
    'What are the key challenges in training LLM-based agents for multi-agent systems?',
    'Explain the integration of LLMs with IoT devices in creating smart agents for home automation.',
    'How can prompt engineering be used to guide LLMs in generating creative content?',
    'What strategies exist for minimizing biases in LLM responses through careful prompt engineering?',
    'Discuss the role of zero-shot and few-shot learning in the context of prompt engineering for LLMs.',
    'How does prompt engineering affect the interpretability of LLM responses?',
    'Compare the effectiveness of manual versus automated prompt engineering techniques.',
]


if __name__ == '__main__':
    load_dotenv()

    chain = get_rag_chain(
        [ARXIVER_PATH / 'papers' / 'llm_apps' / 'mixtral_of_experts.pdf'], 
        'mixtral_of_experts',
    )

    client = langsmith.Client()
    chain_results = client.run_on_dataset(
        dataset_name="arxiver-mixtral-of-experts-no-output",
        llm_or_chain_factory=RunnableLambda(lambda dct: dct['input']) | chain,
        project_name=f"arxiver-{datetime.now().strftime('%y%m%d_%H%M%S')}",
        concurrency_level=5,
        verbose=True
    )