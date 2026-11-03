FROM python:3.13-slim
RUN pip install --no-cache-dir numpy==2.2.6
COPY worker.py /opt/wsp-worker.py
WORKDIR /work
USER 65534:65534
ENTRYPOINT ["python", "-u", "/opt/wsp-worker.py"]
