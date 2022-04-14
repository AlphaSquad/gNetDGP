FROM python:3.7

RUN pip install click
RUN pip install torch==1.4.0+cpu -f https://download.pytorch.org/whl/torch_stable.html
RUN pip install torchvision==0.5.0+cpu -f https://download.pytorch.org/whl/torch_stable.html
RUN pip install scikit-image==0.17.2
RUN pip install scikit-learn==0.23.1
RUN pip install scipy==1.5.0
RUN apt-get -y update && apt-get install -y build-essential
RUN pip install torch-scatter==2.0.4 -f https://pytorch-geometric.com/whl/torch-1.4.0+cpu.html
RUN pip install torch-sparse==0.6.0 -f https://pytorch-geometric.com/whl/torch-1.4.0+cpu.html
RUN pip install torch-cluster==1.5.4 -f https://pytorch-geometric.com/whl/torch-1.4.0+cpu.html
RUN pip install torch-spline-conv==1.2.0 -f https://pytorch-geometric.com/whl/torch-1.4.0+cpu.html
RUN pip install torch-geometric==1.4.2
RUN pip install tqdm==4.46.1
RUN pip install optuna==1.5.0
RUN pip install colorama==0.4.3
RUN pip install colorlog==4.1.0
RUN pip install pandas==1.0.5
