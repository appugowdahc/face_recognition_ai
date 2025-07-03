FROM python:3.11-slim

RUN mkdir -p /root
# Set work directory
WORKDIR /root

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install tf-keras
RUN pip install tensorflow==2.15
RUN pip install deepface opencv-python scikit-learn imutils torch torchvision

# Copy application code
COPY . ./root   

# Run app with uvicorn
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
