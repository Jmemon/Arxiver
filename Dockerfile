# Use Python 3.10.8 as the base image
FROM python:3.10.8-slim

# Set the working directory in the Docker container
WORKDIR /Arxiver

# Copy all repo files into the container's working directory
COPY ./requirements.txt requirements.txt

# install a C++ compiler for llama-cpp-python
RUN apt-get update && \
    apt-get -y install g++ && \
    rm -rf /var/lib/apt/lists/*

# Install the Python dependencies from requirements.txt
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install -U "huggingface_hub[cli]"
RUN pip install pyopenssl --upgrade

RUN apt-get -y update
RUN apt-get -y upgrade
RUN apt-get -y install poppler-utils
RUN apt-get -y install tesseract-ocr
RUN apt-get -y install ffmpeg libsm6 libxext6

# install newer version of sqlite3. For some reason apt only can see 3.34.1 which is too old. I need >= 3.35.0.
RUN apt-get -y install build-essential wget
WORKDIR /root
RUN mkdir sqlite3 
WORKDIR /root/sqlite3
RUN wget https://www.sqlite.org/2024/sqlite-autoconf-3450200.tar.gz 
RUN tar xzvf sqlite-autoconf-3450200.tar.gz
WORKDIR /root/sqlite3/sqlite-autoconf-3450200
RUN export CFLAGS="-DSQLITE_ENABLE_FTS3 \
    -DSQLITE_ENABLE_FTS3_PARENTHESIS \
    -DSQLITE_ENABLE_FTS4 \
    -DSQLITE_ENABLE_FTS5 \
    -DSQLITE_ENABLE_JSON1 \
    -DSQLITE_ENABLE_LOAD_EXTENSION \
    -DSQLITE_ENABLE_RTREE \
    -DSQLITE_ENABLE_STAT4 \
    -DSQLITE_ENABLE_UPDATE_DELETE_LIMIT \
    -DSQLITE_SOUNDEX \
    -DSQLITE_TEMP_STORE=3 \
    -DSQLITE_USE_URI \
    -O2 \
    -fPIC"
    
RUN export PREFIX="/usr/local"
RUN LIBS="-lm" ./configure --disable-tcl --enable-shared --enable-tempstore=always --prefix="$PREFIX"
RUN make
RUN make install
RUN export PATH=/usr/local/lib:$PATH
RUN cp /root/sqlite3/sqlite-autoconf-3450200/.libs/libsqlite3.so.0 /usr/lib/x86_64-linux-gnu/libsqlite3.so.0

WORKDIR /Arxiver
COPY . /Arxiver/

#RUN mkdir qa_chain/weights
RUN mkdir -p qa_chain/weights
RUN huggingface-cli download TheBloke/Mistral-7B-Instruct-v0.2-GGUF mistral-7b-instruct-v0.2.Q4_K_M.gguf --local-dir qa_chain/weights

# Expose the port the app runs on
EXPOSE 5000

CMD ["python", "chat.py"]