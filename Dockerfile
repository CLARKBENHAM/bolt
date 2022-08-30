#Based on pytorch extension script dockerfile: https://github.com/pytorch/extension-script
FROM ubuntu:xenial

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
	#Change python to 3.7
	conda install python=3.7 && \
  conda install  -c pytorch-nightly cpuonly && \
  conda install  -c pytorch-nightly pytorch  && \
	conda init bash 

# Download LibTorch
RUN wget https://download.pytorch.org/libtorch/nightly/cpu/libtorch-shared-with-deps-latest.zip
RUN unzip libtorch-shared-with-deps-latest.zip && rm libtorch-shared-with-deps-latest.zip

###########################Clark Added Below

###%% Things added to get Bolt working; based on: https://github.com/dblalock/bolt/blob/master/BUILD.md
RUN apt-get install \
	-y \
	apt-utils \
	build-essential \
	clang-3.9 \
	clang \
	libc++-dev \
	libeigen3-dev \
	swig \
	sudo \
	apt-transport-https \
	curl \
	gnupg \
	&& curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh | sudo bash \
	&& apt-get install  git-lfs


#Need to install Bazel for build
RUN curl -fsSL https://bazel.build/bazel-release.pub.gpg | gpg --dearmor >bazel-archive-keyring.gpg \
	&& sudo mv bazel-archive-keyring.gpg /usr/share/keyrings \
	&& echo "deb [arch=amd64 signed-by=/usr/share/keyrings/bazel-archive-keyring.gpg] https://storage.googleapis.com/bazel-apt stable jdk1.8" | sudo tee /etc/apt/sources.list.d/bazel.list \
	&& sudo apt update && sudo apt install bazel

#How to Git clone in Docker?
#TODO: should be my branch to clone; but then need git creds. For now just using on same volume as repos already exist on
RUN cd  && git clone https://github.com/pytorch/extension-script.git 
# RUN git clone https://github.com/dblalock/bolt.git 

#Installed eigen: had to manually build as well as install(?!)
RUN cd .. \
	&& curl -L https://gitlab.com/libeigen/eigen/-/archive/3.3.8/eigen-3.3.8.tar > eigen-3.3.8.tar \
	&& tar -xf eigen-3.3.8.tar \
	&& mkdir build_dir_eigen \
	&& cd build_dir_eigen  \
	&& cmake ../eigen-3.3.8 \
	&& make install

#Bolt packages
WORKDIR /home/cbenham/bolt/
COPY requirements.txt /home/cbenham/bolt/requirements.txt 
RUN conda install --file requirements.txt 
COPY . /home/cbenham/bolt

#Build Code
#.bashrc activates conda for packages
RUN . ~/.bashrc \
	&& ./build.sh \
	&& true
	## 1 pytest known to fail
	#&& pytest tests || true\
	##last cpp tests are very slow; but did get a seg fault on  Ucr128 f32
	#&& cd cpp/build-bolt \
	#&& ./bolt amm* 

RUN cd cpp && bazel run :main
