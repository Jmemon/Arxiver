
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable, Callable

from langsmith import Client
from dominate.tags import div, style, script, br, strong
from dominate.document import document
from dominate.util import raw, text

"""
- rag chain configuration:
    - chain graph
    - vdb type, embedding model, list of data sources; eg: Chroma, HuggingFaceEmbeddings(), ['mixtral_of_experts.pdf']
    - prompt template
    - model, llamacpp args; eg: Mistral7b, n_ctx=2048, max_tokens=None, top_p=1, ...
"""


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


class RAGRunDebugDisplay:

    def attr_processing(name, value):
        if isinstance(value, datetime):
            return value.strftime("%y-%m-%d_%H-%M-%S-%f")
        elif isinstance(value, dict):
            return value
        else:
            return str(value)

    def __init__(
            self, 
            runs: Iterable, 
            tree_attrs: Iterable = ['id', 'name', 'start_time', 'end_time', 'inputs', 'outputs']
    ) -> None:
        """
        Assumes trace structure:
            RunnableSequence
                RunnableParallel<context,question>
                    RunnableSequence
                        Retriever
                        RunnableLambda
                    RunnablePassthrough
                LlamaCpp
                PromptTemplate
                StrOutputParser
        """
        
        self.run_dict = {str(run.id): run for run in runs}
        self.run_forest = []
        self.tree_attrs = tree_attrs

        num_parents = 0
        runs_added = 0
        while runs_added < len(self.run_dict):
            for run in self.run_dict.values():
                if len(run.parent_run_ids) == num_parents:
                    if num_parents == 0:
                        self.run_forest.append(
                            {
                                **({attr: self.attr_processing(getattr(run, attr)) for attr in tree_attrs}), 
                                **({'sub_runs': []})
                            }
                        )
                    else:
                        tmp_lst = self.run_forest
                        for parent_id in run.parent_run_ids:  # Get to the run's parent dict. Highest level parent to most immediate parent
                            for _dct in tmp_lst:
                                if _dct['id'] == str(parent_id):
                                    tmp_lst = _dct['sub_runs']
                                    break
                        tmp_lst.insert(
                            0, 
                            {
                                **({attr: self.attr_processing(getattr(run, attr)) for attr in tree_attrs}), 
                                **({'sub_runs': []})
                            }
                        )

                    runs_added += 1

            num_parents += 1

    def print(self):
        def print_forest(forest, top_level=True, prefix=''):
            for run in forest:
                print(f'{prefix}{run["name"]: <34}: {run["start_time"]} -- {run["end_time"]}')
                print_forest(run['sub_runs'], top_level=False, prefix=prefix + '  ')
                if top_level:
                    print()

        print_forest(self.run_forest)

    def get_text_from_tree(self, tree: dict):
        assert tree['name'] == 'RunnableSequence', print(tree['name'], tree['id'])
        assert len(tree['sub_runs']) > 2, print(tree['name'], len(tree['sub_runs']))  # make sure top-level

        query_str = tree['inputs']['input']

        for itm in tree['sub_runs']:
            if itm['name'] == 'PromptTemplate':
                prompt_str = itm['outputs']['kwargs']['text']

            if itm['name'] == 'StrOutputParser':
                response_str = itm['outputs']['output']

        if 'query_str' not in locals():
            raise ValueError('query_str not set')
    
        if 'prompt_str' not in locals():
            raise ValueError('prompt_str not set')

        if 'response_str' not in locals():
            raise ValueError('response_str not set')

        return query_str, prompt_str, response_str
    
    def traverse_tree(self, tree: dict, node_fxn: Callable):
        """
        
        """
        node_outs = []
        node_stack = [tree]

        while len(node_stack) > 0:
            node = node_stack.pop()
            node_outs.append(node_fxn(node))
            node_stack.extend(list(reversed(node['sub_runs'])))  # pop the children in the order they appear in the list

        return node_outs
    
    def tree_check(self, tree, structure):
        tree_nodes = self.traverse_tree(tree, lambda nd: nd['name'])
        structure_nodes = self.traverse_tree(structure, lambda nd: nd['name'])
        return all([tree_name == struct_name for tree_name, struct_name in zip(tree_nodes, structure_nodes)])

    def generate_html(self):
        """
        Expecting query_prompt_response_tuples to be a list of 3-tuples
        where each tuple is ({query}, {prompt}, {response})
        """
        doc = document(title='RAG Debug')

        query_prompt_response_tuples = [] 
        for tree in self.run_forest:
            try:
                query_prompt_response_tuples.append(self.get_text_from_tree(tree))
            except AssertionError:
                continue
            except ValueError:
                continue
        
        with doc.head:
            # Add your styles and scripts here
            style("""
            .content {
                display: none;
                margin-top: 10px;
            }
            .question {
                cursor: pointer;
                color: blue;
                margin-bottom: 5px;
            }
            """)
        
        with doc.body:
            content_div = div(id="content")
            for index, (query, prompt, response) in enumerate(query_prompt_response_tuples, start=1):
                with content_div:
                    div(query, _class='question')

                    with div(_class='content'):
                        strong('===== Prompt =====')
                        br()
                        [(text(itm), br()) for itm in prompt.split('\n')]

                    with div(_class='content'):
                        strong('===== Response =====')
                        br()
                        text(response)
                        br()
                        strong('====================')

            script(
                raw("""
                    function toggleVisibility(event) {
                        let nextElement = event.target.nextElementSibling;
                        nextElement.style.display = nextElement.style.display === 'block' ? 'none' : 'block';
            
                        let secondElement = nextElement.nextElementSibling;
                        if (secondElement && secondElement.classList.contains('content')) {
                            secondElement.style.display = secondElement.style.display === 'block' ? 'none' : 'block';
                        }
                    }

                    document.addEventListener('DOMContentLoaded', function() {
                        document.querySelectorAll('.question').forEach(function(question) {
                            question.addEventListener('click', toggleVisibility);
                        });
                    });
                """)
            )
        
        return str(doc)


if __name__ == '__main__':
    client = Client()
    #runs = client.list_runs(project_name='paper_chat')
    run_gen = client.list_runs(
        project_name='paper_chat', 
        filter='gt(start_time, "2024-02-14T00:00:00Z")'
    )
    
    debug = RAGRunDebugDisplay(run_gen)
    html_str = debug.generate_html()

    with open(str(Path(__file__).parent / 'tmp.html'), 'w') as f:
        f.write(html_str)
