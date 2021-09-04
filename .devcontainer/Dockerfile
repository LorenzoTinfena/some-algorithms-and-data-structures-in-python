FROM python

RUN apt upgrade
RUN apt update

ADD ./.devcontainer/pip-requirements.txt /tmp/
RUN pip install -r /tmp/pip-requirements.txt
RUN rm /tmp/pip-requirements.txt