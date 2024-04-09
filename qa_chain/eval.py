
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Iterable, Callable, List

from langsmith import Client
from dominate.tags import div, style, script, br, strong
from dominate.document import document
from dominate.util import raw, text


# I genereted these with ChatGPT: "Can you generate 50 questions that could be asked of an arbitrary AI research paper?"
ai_eval_queries = [
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
    'How does the paper contribute to the ethical design and deployment of AI technologies?'
]
