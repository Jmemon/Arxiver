
from langchain.prompts.prompt import PromptTemplate


query_gen_prompt_templ = PromptTemplate(
    input_variables=['num_queries', 'topic'],
    template="""
        I want to get {num_queries} queries to search arXiv to learn about {topic}. These queries should be short space-separated lists of keywords that seem relevant to the topic.

        Can you help me put these queries together by engaging with me in a question-answer conversation following the format:
        ```
        AI: Our current list of queries contains <CurrentList>. One additional possible search query would be "<NewQuery>". Does this sound useful?

        User: <Yes|No|<Change>>
        ```
        If the user answers yes, append NewQuery to CurrentList. If they answer no, present the user with CurrentList and a new query. If they ask you to tweak it, tweak NewQuery according to Change and present the new version to them until they say yes or no.
        """
)
