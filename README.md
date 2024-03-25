
Aiming to be a research tool that can spin up chatbots with strong knowledge of some user-specified research area. After some direction from the user, the tool will download a set of papers (currently from arxiv), index them, and run queries through a RAG pipeline.

Chatbots should be runnable locally.

Currently we use a naive RAG architecture consisting of HuggingFaceEmbeddings, ChromaDB, and Mistral7B, although we are currently improving on this architecture drawing inspiration from:
- [WalkingRAG](https://twitter.com/hrishioa/status/1745835962108985737)
- [RAPTOR](https://github.com/parthsarthi03/raptor)
