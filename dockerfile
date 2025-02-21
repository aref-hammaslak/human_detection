FROM python:3.10-slim

WORKDIR /app
   
COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

RUN apt-get update && apt-get install -y libgl1-mesa-glx libglib2.0-0

COPY . . 

CMD ["python", "src/app.py"]    