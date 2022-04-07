# See http://bugs.python.org/issue19846
ENV LANG C.UTF-8
ARG PY_VER=38
ARG PYTHON=python3

RUN yum update -y && yum install -y \
    python${PY_VER} \
    python${PY_VER}-pip \
    which && \
    yum clean all

RUN ${PYTHON} -m pip --no-cache-dir install --upgrade \
    pip \
    setuptools

# Some TF tools expect a "python" binary
RUN ln -sf $(which ${PYTHON}) /usr/local/bin/python && \
    ln -sf $(which ${PYTHON}) /usr/local/bin/python3 && \
    ln -sf $(which ${PYTHON}) /usr/bin/python

# Installs the latest version by default.
ARG TF_WHEEL=tf_nightly-2.7.0.202142-cp36-cp36m-linux_x86_64.whl

COPY ./whls/${TF_WHEEL} /tmp/pip3/

RUN python -m pip install --no-cache-dir /tmp/pip3/${TF_WHEEL}


# fix keras-nightly and tf-estimator-nightly versions
RUN python -m pip uninstall -y \
      keras-nightly \
      tf-estimator-nightly && \
    python -m pip install \
      'keras-nightly==2.9.0.dev2022021708' \
      tf-estimator-nightly~=2.9.0.dev
