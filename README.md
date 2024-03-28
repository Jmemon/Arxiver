
Note: Full pipeline is on branch EvalTool. main is only a test app.

Aiming to be a research tool that can spin up chatbots with strong knowledge of some user-specified research area. After some direction from the user, the tool will download a set of papers (currently from arxiv), index them, and run queries through a RAG pipeline.

Chatbots ideally will be runnable locally, although the pipeline is getting more resource-intensive. Currently experimenting with running it on the cloud. Hopefully down the line this can be changed.

Currently we use a naive RAG architecture consisting of HuggingFaceEmbeddings, ChromaDB, and Mistral7B, although we are currently improving on this architecture drawing inspiration from:
- [WalkingRAG](https://twitter.com/hrishioa/status/1745835962108985737)
- [RAPTOR](https://github.com/parthsarthi03/raptor)

Using python version 3.10.8.

To run the app:
```
git clone https://github.com/Jmemon/Arxiver.git
cd Arxiver
docker build -t arxiver-app .
docker run -p <host-port>:5000 arxiver-app
```
Then navigate to `http://localhost:<host-port>/<endpoint>`

A warning:
Docker uses [hypervisor](https://developer.apple.com/documentation/hypervisor) to run containers on MacOS, which to my understanding does not allow GPU passthrough, so running this app on a Mac will be very slow. It's best to run it on linux. Docker says in the note at the top of [this](https://docs.docker.com/desktop/gpu/) page that it only supports GPUs for windows.
