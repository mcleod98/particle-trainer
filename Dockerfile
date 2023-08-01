FROM pytorch/pytorch

ENV WORKDIR=/opt

RUN pip install --upgrade pip
RUN pip install notebook
RUN pip install -U scikit-learn
RUN pip install pillow sqlalchemy pyyaml click
RUN pip install torchsummary

WORKDIR $WORKDIR

CMD ["jupyter", "notebook", "--port=8888", "--no-browser", "--ip=0.0.0.0", "--allow-root"]

#CMD ["python", "-u", "/opt/src/cli.py", "train"]