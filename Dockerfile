FROM tensorflow/tensorflow:1.13.1-gpu-py3-jupyter
ENV LANG C.UTF-8
ENV LC_ALL C.UTF-8

RUN rm /etc/bash.bashrc

WORKDIR /root
RUN pip freeze > requirements.txt
RUN add-apt-repository -y ppa:deadsnakes/ppa && apt-get update
RUN apt-get install -y python3.7 python3.7-dev
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.7 1
RUN python -m pip install --upgrade pip
RUN pip install -r requirements.txt

RUN pip install gensim nltk pandas subword-nmt tqdm

RUN apt-get install -y openssh-server git
RUN mkdir /var/run/sshd
RUN echo 'root:testdocker' | chpasswd
RUN sed -i 's/PermitRootLogin prohibit-password/PermitRootLogin yes/' /etc/ssh/sshd_config

# SSH login fix. Otherwise user is kicked off after login
RUN sed 's@session\s*required\s*pam_loginuid.so@session optional pam_loginuid.so@g' -i /etc/pam.d/sshd

ENV NOTVISIBLE "in users profile"
RUN echo "export VISIBLE=now" >> /etc/profile

RUN git clone https://github.com/facebookresearch/fastText.git
WORKDIR /root/fastText
RUN pip install .

WORKDIR /root
COPY . .

EXPOSE 22 8888
CMD ["bash", "-c", "/usr/sbin/sshd && jupyter notebook --ip 0.0.0.0 --no-browser --allow-root"]