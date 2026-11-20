FROM python:3.12-slim
WORKDIR /code
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY app app/
COPY config config/
ENV HOST=0.0.0.0 PORT=8000
EXPOSE 8000
CMD ["python", "app/web.py"]