FROM python:latest

# app directory
WORKDIR /app

# copy required files
COPY requirements.txt ./
COPY app.py ./

# install dependencies
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

EXPOSE 8501
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
