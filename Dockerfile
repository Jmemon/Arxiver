# Use Python 3.10.8 as the base image
FROM python:3.10.8-slim

# Set the working directory in the Docker container
WORKDIR /Arxiver

# Copy all repo files into the container's working directory
COPY . /Arxiver/

RUN mkdir qa_chain/weights

# install a C++ compiler for llama-cpp-python
RUN apt-get update && \
    apt-get -y install g++ && \
    rm -rf /var/lib/apt/lists/*

# Install the Python dependencies from requirements.txt
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install -U "huggingface_hub[cli]"

RUN huggingface-cli download TheBloke/Mistral-7B-Instruct-v0.2-GGUF mistral-7b-instruct-v0.2.Q4_K_M.gguf --local-dir qa_chain/weights

# Expose the port the app runs on
EXPOSE 5000

CMD ["python", "chat.py"]
