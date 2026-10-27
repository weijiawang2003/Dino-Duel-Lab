FROM python:3.12-slim
WORKDIR /app
COPY requirements-core.txt /app/
RUN pip install --no-cache-dir -r requirements-core.txt && \
    pip install --no-cache-dir torch==2.10.0 --index-url https://download.pytorch.org/whl/cpu
COPY . /app
ENTRYPOINT ["python", "run.py"]
CMD ["doctor"]

