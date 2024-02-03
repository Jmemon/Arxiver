
from langchain.prompts import ChatPromptTemplate


"""
Best one yet.
=====
parameters: num_queries=five, topic=language model inference optimization
response:


1. Efficient Inference
2. Neural Architecture Search
3. Knowledge Distillation
4. Low-Rank Approximation
5. Tensor Decomposition.
===== Could combine these together into search queries or something
parameters: num_queries=fifteen, topic=latent variable models in speech processing
response:


1. Latent Variable Model (LVMM)
2. Factor Analysis (FA)
3. Principal Component Analysis (PCA)
4. Independent Component Analysis (ICA)
5. Non-Negative Matrix Factorization (NMF)
6. Gaussian Mixture Model (GMM)
7. Hidden Markov Model (HMM)
8. Long Short-Term Memory (LSTM)
9. Recurrent Neural Networks (RNNs)
10. Convolutional Neural Networks (CNNs)
11. Autoencoder (AE)
12. Variational Autoencoder (VAE)
13. Generative Adversarial Network (GAN)
14. Deep Belief Network (DBN)
15. Restricted Boltzmann Machine (RBM)
===== Eh response. Some of these aren't latent variable models and no mention of speech processing
parameters: num_queries=fiftenn, topic=cosmic background radiation
response:


1. Cosmic Microwave Background (CMB)
2. Anisotropy
3. Temperature fluctuations
4. Redshift
5. Inflation theory
6. Quasars
7. Dark matter
8. Sunyaev-Zeldovich effect
9. CMB maps
10. Power spectrum
11. Cosmological principle
12. Large scale structure of the universe
13. Reionization
14. Galaxy clusters
15. Foreground emission.
===== This seems like a pretty solid response. I think most of these are related. Some might not be.
"""
keyword_list_prompt = ChatPromptTemplate.from_messages([
    ('human', 'Give me a numbered list of {num_queries} keywords related to {topic} to '
              'be used as search queries with no explanations.')
])



"""
I might like this one the most. The extra newlines at the start of responses is from the actual response, not an accident.

parameters: num_queries=ten, topic=cosmic background radiation
response:


1. Cosmic microwave background temperature
2. CMBR anisotropy
3. Cosmic background radiation redshift
4. CMBR power spectrum
5. Cosmic background radiation fluctuations
6. CMBR polarization
7. Cosmic background radiation and the early universe
8. CMBR measurements and observations
9. Cosmic background radiation and dark matter
10. CMBR and the theory of inflation
=====
parameters: num_queries=five, topic=llm inference
response:


1. LLM Inference Optimization Algorithms
2. Efficient Implementation Techniques for LLM Inference Optimization
3. Performance Comparison between Different LLM Inference Optimization Methods
4. Real-world Applications of LLM Inference Optimization in Machine Learning and AI
5. LLM Inference Optimization: Theoretical Frameworks and Research Directions.
===== This one's not as good.
parameters: num_queries=give, topic= language model inference optimization


1. Language model inference efficiency
2. Optimization techniques for language model inference
3. Parallel processing in language model inference optimization
4. GPU utilization in large scale language model inference optimization
5. Memory management strategies in language model inference optimization.
"""
keyword_query_list_prompt = ChatPromptTemplate.from_messages([
    ('human', 'Give me a numbered list of {num_queries} search queries about various aspects of {topic}, '
              'where each search should be an individual keyword related to the topic with no explanations.')
])


"""
This one gets pretty solid results with wizard-LM temperature=0.7. Just the list with nothing else.
Parameters: num_queries=three, topic=cosmic background radiation
Response:


    1. Cosmic microwave background temperature evolution
    2. Anisotropy in cosmic background radiation maps
    3. CMBR power spectrum analysis and redshift effects.
======
Parameters: num_queries=two, topic=cosmic background radiation


Sure, here are two search queries related to cosmic background radiation:

1. Cosmic Background Radiation temperature evolution
2. Anisotropy in Cosmic Background Radiation maps
===== Whenever I do num_queries=two it puts extra text in there
"""
query_list_prompt = ChatPromptTemplate.from_messages([
    ('human', 'Give me a numbered list of {num_queries} search queries about various aspects of {topic}, '
              'where each search should be keywords related to the topic with no explanations.')
])


query_few_shot_list_prompt = ChatPromptTemplate.from_messages([
    ('system', 'You are a concise speaker. You are an expert in the field under discussion.'),
    ('human', 'I want to get two queries to search for information about the use of latent variable modeling '
              'in speech processing. These queries should be short space-separated lists of keywords that are '
              'relevant to the topic.'),
    ('ai', 'Our list of queries contains ["bayesian networks speech recognition synthesis diarization", '
           '"variational autoencoder speech enhancement"].'
           'Does this look good?'),
    ('human', 'Can you change the first one?'),
    ('ai', 'Yes. Our list of queries contains ["deep learning latent representations speech", '
           '"variational autoencoder speech enhancement"]. Does this look good?'),
    ('human', 'yes'),
    ('ai', 'Great! Happy researching!'),
    ('human', 'I want to get {num_queries} queries to search for information about {topic}. These queries'
              ' should be short space-separated lists of keywords that are relevant to the topic.')
])


query_gen_prompt = ChatPromptTemplate.from_messages([
    ('system', 'You are a concise speaker. You follow exactly what is requested of you. '
               'You are an expert in the field under discussion.'),
    ('human', 'I want to get three queries to search for information about the use of latent variable modeling '
              'in speech processing. These queries should be short space-separated lists of keywords that are '
              'relevant to the topic.'),
    ('ai', 'Our current list of queries contains []. One additional possible search query would be '
           '"latent variable models speech processing". Does this sound useful?'),
    ('human', 'no'),

    ('ai', 'Our current list of queries contains []. One additional possible search query would be '
           '"deep learning latent representations speech". Does this sound useful?'),
    ('human', 'yes'),
    ('ai', 'Our current list of queries contains ["deep learning latent representations speech"]. '
           'One additional possible search query would be "variational autoencoder speech enhancement". '
           'Does this sound useful?'),
    ('human', 'If variational autoencoders are latent variable models, then yes.'),
    ('ai', 'Yes, variational autoencoders (VAEs) are indeed a type of latent variable model. Given this, '
           'I\'ll add "variational autoencoder speech enhancement" to our list.\n'
           '\n'
           'Our current list of queries now contains ["deep learning latent representations speech", '
           '"variational autoencoder speech enhancement"]. One additional possible search query would be'
           ' "bayesian networks speech synthesis". Does this sound useful?'),
    ('human', 'Can you include additional speech processing tasks to this query?'),
    ('ai', 'Our current list of queries contains ["deep learning latent representations speech", '
           '"variational autoencoder speech enhancement"]. One additional possible search query would be '
           '"bayesian networks speech recognition synthesis diarization". Does this sound useful?'),
    ('human', 'yes'),
    ('ai', 'Our current list of queries now contains ["deep learning latent representations speech", '
           '"variational autoencoder speech enhancement", "bayesian networks speech recognition synthesis diarization"].\n'
           '\n'
           'Would you like to continue refining or adding more queries, or does this list meet your needs for now?'),
    ('human', 'That\'s enough, thank you.'),
    ('ai', 'You\'re welcome! If you have any more questions or need further assistance, feel free to ask. Happy researching!'),
    ('human', 'I want to get {num_queries} queries to search for information about {topic}. These queries should'
              ' be short space-separated lists of keywords that are relevant to the topic.'),
])


query_gen_prompt_short = ChatPromptTemplate.from_messages([
    ('system', 'You are a concise speaker. You follow exactly what is requested of you. '
               'You are an expert in the field under discussion.'),
    ('human', 'I want to get two queries to search for information about the use of latent variable modeling '
              'in speech processing. These queries should be short space-separated lists of keywords that are '
              'relevant to the topic.'),
    ('ai', 'Our current list of queries contains []. One additional possible search query would be '
           '"latent variable models speech processing". Does this sound useful?'),
    ('human', 'no'),
    ('ai', 'Our current list of queries contains []. One additional possible search query would be '
           '"deep learning latent representations speech". Does this sound useful?'),
    ('human', 'yes'),
    ('ai', 'Our current list of queries now contains ["deep learning latent representations speech", '
           '"variational autoencoder speech enhancement"]. One additional possible search query would be'
           ' "bayesian networks speech synthesis". Does this sound useful?'),
    ('human', 'Can you include additional speech processing tasks to this query?'),
    ('ai', 'Our current list of queries contains ["deep learning latent representations speech", '
           '"variational autoencoder speech enhancement"]. One additional possible search query would be '
           '"bayesian networks speech recognition synthesis diarization". Does this sound useful?'),
    ('human', 'yes'),
    ('ai', 'Our current list of queries now contains ["deep learning latent representations speech", '
           '"variational autoencoder speech enhancement", "bayesian networks speech recognition synthesis diarization"].\n'
           '\n'
           'Would you like to continue refining or adding more queries, or does this list meet your needs for now?'),
    ('human', 'That\'s enough, thank you.'),
    ('ai', 'You\'re welcome! If you have any more questions or need further assistance, feel free to ask. Happy researching!'),
    ('human', 'I want to get {num_queries} queries to search for information about {topic}. These queries should'
              ' be short space-separated lists of keywords that are relevant to the topic.'),
])
