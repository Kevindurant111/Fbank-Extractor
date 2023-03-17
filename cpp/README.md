wget https://versaweb.dl.sourceforge.net/project/arma/armadillo-12.0.1.tar.xz
tar -xvf armadillo-12.0.1.tar.xz
sudo apt-get install -y liblapack-dev libblas-dev libopenblas-dev libsndfile1-dev libarmadillo-dev

apt下载的包版本较低，但是bug少，适配arm64；如果使用官网直接下载的包可能会发生类似malloc.c: No such file or directory的错误
