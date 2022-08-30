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
	## 1 test known to fail
	#&& pytest tests || true\
	##last cpp tests are very slow; but did get a seg fault on  Ucr128 f32
	#&& cd cpp/build-bolt \
	#&& ./bolt amm* 

RUN cd cpp && bazel run :main

#Manual setup
##Build python package, needs external kmc2
#RUN cd .. \
#	&& git clone -b mneilly/cythonize https://github.com/mneilly/kmc2.git \
#	&& cd kmc2 \
#	&& . ~/.bashrc \
#	&& cd ../bolt \
#	&& EIGEN_INCLUDE_DIR=/usr/include/eigen3 python setup.py install \
#	&& pytest tests/ 
##Note: 1 test currently fails
#
##Build C++
#RUN  cd cpp \
#  && mkdir build-bolt \
#  && cd build-bolt \
#  && cmake .. \
#  && make \
#  && ./bolt amm*

## Build C++: taken from ./build.sh
#RUN	git submodule update --init \
#	&& pip install -r requirements.txt \
#	&& pip install ./third_party/kmc2 \
#	&& . ~/.bashrc \
#	&& python setup.py install \
#	&& mkdir -p cpp/build-bolt \
#	&& cd cpp/build-bolt \
#	&& cmake .. \
#	&& make -j4
#
# RUN printf 'cmake_minimum_required(VERSION 3.3 FATAL_ERROR)\n\nproject(bolt CXX)\n\nfind_package(PkgConfig)\npkg_search_module(Eigen3 REQUIRED eigen3)\nfind_package(Eigen3 REQUIRED)\n\nset(sourceFiles\n  ${CMAKE_SOURCE_DIR}/src/quantize/bolt.cpp\n  ${CMAKE_SOURCE_DIR}/src/quantize/mithral.cpp\n  ${CMAKE_SOURCE_DIR}/src/utils/avx_utils.cpp\n  ${CMAKE_SOURCE_DIR}/test/main.cpp\n  ${CMAKE_SOURCE_DIR}/test/quantize\n  ${CMAKE_SOURCE_DIR}/test/test_avx_utils.cpp\n  ${CMAKE_SOURCE_DIR}/test/quantize/profile_amm.cpp\n  #${CMAKE_SOURCE_DIR}/test/quantize/profile_amm_old.cpp\n  ${CMAKE_SOURCE_DIR}/test/quantize/profile_bolt.cpp\n  ${CMAKE_SOURCE_DIR}/test/quantize/profile_encode.cpp\n  ${CMAKE_SOURCE_DIR}/test/quantize/profile_lut_creation.cpp\n  ${CMAKE_SOURCE_DIR}/test/quantize/profile_multicodebook.cpp\n  ${CMAKE_SOURCE_DIR}/test/quantize/profile_pq.cpp\n  ${CMAKE_SOURCE_DIR}/test/quantize/profile_scan.cpp\n  ${CMAKE_SOURCE_DIR}/test/quantize/test_bolt.cpp\n  ${CMAKE_SOURCE_DIR}/test/quantize/test_mithral.cpp\n  ${CMAKE_SOURCE_DIR}/test/quantize/test_multicodebook.cpp\n  )\n\nset(headerFiles\n  ${CMAKE_SOURCE_DIR}/src/include/public.hpp\n  ${CMAKE_SOURCE_DIR}/src/quantize/bolt.hpp\n  ${CMAKE_SOURCE_DIR}/src/quantize/mithral.hpp\n  ${CMAKE_SOURCE_DIR}/src/quantize/mithral_v1.hpp\n  ${CMAKE_SOURCE_DIR}/src/quantize/multi_codebook.hpp\n  ${CMAKE_SOURCE_DIR}/src/quantize/multisplit.hpp\n  ${CMAKE_SOURCE_DIR}/src/quantize/product_quantize.hpp\n  ${CMAKE_SOURCE_DIR}/src/utils/avx_utils.hpp\n  ${CMAKE_SOURCE_DIR}/src/utils/bit_ops.hpp\n  ${CMAKE_SOURCE_DIR}/src/utils/debug_utils.hpp\n  ${CMAKE_SOURCE_DIR}/src/utils/eigen_utils.hpp\n  ${CMAKE_SOURCE_DIR}/src/utils/memory.hpp\n  ${CMAKE_SOURCE_DIR}/src/utils/nn_utils.hpp\n  ${CMAKE_SOURCE_DIR}/src/utils/timing_utils.hpp\n  ${CMAKE_SOURCE_DIR}/test/external/catch.hpp\n  ${CMAKE_SOURCE_DIR}/test/quantize/amm_common.hpp\n  ${CMAKE_SOURCE_DIR}/test/quantize/profile_amm.hpp\n  ${CMAKE_SOURCE_DIR}/test/quantize/test_bolt.hpp\n  ${CMAKE_SOURCE_DIR}/test/testing_utils/testing_utils.hpp\n  )\n\nadd_executable(bolt ${sourceFiles} ${headerFiles})\n#add_library(bolt SHARED ${sourceFiles} ${headerFiles})\nset_target_properties(bolt PROPERTIES LINKER_LANGUAGE CXX)\ntarget_compile_definitions(bolt PRIVATE "-DBLAZE")\ntarget_link_libraries(bolt Eigen3::Eigen)\ntarget_include_directories(bolt PUBLIC ${CMAKE_SOURCE_DIR})\nset(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14 -march=native -fno-rtti -ffast-math")' > cpp/CMakeLists.txt
#RUN mkdir -p cpp/build-bolt \
#	&& cd cpp/build-bolt \
#	&& cmake -DCMAKE_PREFIX_PATH=/libtorch .. \
#	&& make -j4
