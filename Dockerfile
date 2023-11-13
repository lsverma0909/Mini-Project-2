FROM anibali/pytorch:1.5.0-cuda10.2
USER root
ENV TZ=America/Los_Angeles
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone
RUN apt-get update
RUN apt-get install -y python3-pip python3-matplotlib python3-pandas python3-numpy python3-scipy ipython
RUN pip install jupyter scikit-learn ventmap

RUN mkdir data/
ADD data data/
RUN mkdir pva_dataset/
ADD pva_dataset pva_dataset/
ADD lab_b2.ipynb ./lab-b2.ipynb
ADD *.png ./
ADD dataset.py ./
ADD densenet.py ./
ADD cnn_lstm_net.py ./
RUN jupyter notebook --generate-config
EXPOSE 8888
ENTRYPOINT ["jupyter", "notebook", "--ip=0.0.0.0", "--allow-root"]
