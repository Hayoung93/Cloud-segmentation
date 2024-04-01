FROM pytorch/pytorch:1.12.1-cuda11.3-cudnn8-devel

USER root

RUN apt update && apt install nano git tmux libgl1-mesa-glx libglib2.0-0 -y