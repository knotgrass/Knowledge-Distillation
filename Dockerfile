# image cơ sở
FROM python:3.8.13
FROM ubuntu:20.04

#author
LABEL author.name="VietAnh" \
author.email="imnova1212@gmail.com"

MAINTAINER VietAnh<imnova1212@gmail.com>

RUN apt-get update && \
  apt-get install -y nodejs nano vim
