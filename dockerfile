FROM python:3.10-slim

WORKDIR /app
   
COPY requirements.txt .
COPY setup.sh .

RUN chmod +x setup.sh && ./setup.sh

COPY . . 

CMD ["python", "src/app.py"]    