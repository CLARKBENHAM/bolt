#Based on pytorch extension script dockerfile: https://github.com/pytorch/extension-script
FROM ubuntu:20.04

# so tzdata doesn't hang
ENV TZ=America/Los_Angeles
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone 

RUN apt-get update  -y \
  && apt-get install -y \
	git \
	cmake \
	vim \
	make \
	wget \
	gnupg \
	build-essential \
	software-properties-common \
	gdb \
	zip \
	libopencv-dev

# Install Miniconda
RUN wget https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh \
  && chmod +x miniconda.sh \
  && ./miniconda.sh -b -p ~/local/miniconda \
  && rm ./miniconda.sh

# Symlink the Miniconda activation script to /activate
RUN ln -s ~/local/miniconda/bin/activate /activate
# For conda 
ENV PATH=$PATH:/root/local/miniconda/bin/  
# Install PyTorch
RUN . /activate && \
	conda update -n base -c defaults conda --repodata-fn=repodata.json && \
	conda install python=3.8 && \
	conda install pytorch torchvision torchaudio cpuonly -c pytorch

# Download LibTorch # why used Pre-cxx11 ABI before?
RUN wget https://download.pytorch.org/libtorch/cpu/libtorch-cxx11-abi-shared-with-deps-2.0.1%2Bcpu.zip && \
		unzip libtorch-cxx11-abi-shared-with-deps-2.0.1+cpu.zip && \
		rm libtorch-cxx11-abi-shared-with-deps-2.0.1+cpu.zip 

###%% Things added to get Bolt working; based on: https://github.com/dblalock/bolt/blob/master/BUILD.md
RUN apt-get install \
	-y \
	apt-transport-https \
	apt-utils \
	build-essential \
#	clang-8 \
#	clang \
	curl \
	gnupg \
	libc++-dev \
	libeigen3-dev \
	python3-setuptools \
	swig \
	sudo \
	&& curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh | sudo bash \
	&& apt-get install  git-lfs
# Don't think the clang versions are needed

#Need to install Bazel for build
RUN curl -fsSL https://bazel.build/bazel-release.pub.gpg | gpg --dearmor >bazel-archive-keyring.gpg \
	&& sudo mv bazel-archive-keyring.gpg /usr/share/keyrings \
	&& echo "deb [arch=amd64 signed-by=/usr/share/keyrings/bazel-archive-keyring.gpg] https://storage.googleapis.com/bazel-apt stable jdk1.8" | sudo tee /etc/apt/sources.list.d/bazel.list \
	&& sudo apt update && sudo apt install bazel

#How to Git clone in Docker?
#TODO: should be my branch to clone; but then need git creds. For now just using on same volume as repos already exist on
RUN  git clone https://github.com/pytorch/extension-script.git 
# RUN git clone https://github.com/dblalock/bolt.git 

# (Is this actually needed, already apt-installed? Plus hardcoded in bolt)
# Untaring gives bad directory name by default, had to explicitly set to eigen-3.4.0
#RUN	curl -L https://gitlab.com/libeigen/eigen/-/archive/3.4.0/eigen-3.4.0.tar > eigen-3.4.0.tar \
# 	&& mkdir eigen-3.4.0 tar xf eigen-3.4.0.tar  -C eigen-3.4.0 --strip-components 1 \ 
# 	&& mkdir -p build_dir_eigen \
#	&& cd build_dir_eigen \
# 	&& cmake -DEIGEN_MAX_ALIGN_BYTES=32 ../eigen-3.4.0 \
# 	&& make install
RUN curl -L https://gitlab.com/libeigen/eigen/-/archive/3.3.8/eigen-3.3.8.tar > eigen-3.3.8.tar \
  && mkdir eigen-3.3.8 && tar xf eigen-3.3.8.tar  -C eigen-3.3.8 --strip-components 1 \ 
  && mkdir -p build_dir_eigen \
  && cd build_dir_eigen \
  && cmake ../eigen-3.3.8  \
	&& make install

#Bolt packages
WORKDIR /home/cbenham/bolt/
COPY requirements.txt /home/cbenham/bolt/requirements.txt 
RUN conda install --file requirements.txt 
#install kmc2 
RUN cd .. \
	&& git clone -b mneilly/cythonize https://github.com/mneilly/kmc2.git \
	&& cd kmc2 \
	&& . /activate \
	&& pip install .

COPY . /home/cbenham/bolt

##Build Code with cmake, nessisary? Main code is bad right now
##.bashrc activates conda for packages
#RUN . ~/.bashrc \
#	&& ./build.sh \
#	&& true
#	##last cpp tests are very slow; but did get a seg fault on  Ucr128 f32
#	#&& cd cpp/build-bolt \
#	#&& ./bolt amm* 

# Do we even need setup.py since we're using bazel?
# RUN  . /activate \
# 	&& echo $PATH  \
# 	&& python3 setup.py install \
# 	&& pytest tests/ 

# Check that it'll work
RUN . /activate && cd cpp && bazel build :mithral_wrapped 
# RUN . /activate && cd cpp && bazel build :main && bazel run :main 
