
Aiming to be a research tool that can spin up chatbots with strong knowledge of some user-specified research area. After some direction from the user, the tool will download a set of papers (currently from arxiv), index them, and run queries through a RAG pipeline.

Chatbots ideally will be runnable locally, although the pipeline is getting more resource-intensive. Currently experimenting with running it on the cloud. Hopefully down the line this can be changed.

Currently we use a naive RAG architecture consisting of HuggingFaceEmbeddings, ChromaDB, and Mistral7B, although we are currently improving on this architecture drawing inspiration from:
- [WalkingRAG](https://twitter.com/hrishioa/status/1745835962108985737)
- [RAPTOR](https://github.com/parthsarthi03/raptor)

Using python version 3.10.8.

Run
```
huggingface-cli download TheBloke/Mistral-7B-Instruct-v0.2-GGUF mistral-7b-instruct-v0.2.Q4_K_M.gguf --local-dir qa_chain/weights
```
from the project root dir to download the proper model weights.
