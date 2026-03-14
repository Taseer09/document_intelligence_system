FROM python:3.10

WORKDIR /app

# 1. Copy ONLY the requirements file first to take advantage of Docker caching
COPY requirements.txt .

# 2. Install the massive AI libraries (Docker will save this step permanently!)
RUN pip install --no-cache-dir -r requirements.txt

# 3. NOW copy the rest of your actual Python code and PDFs
COPY . /app

# 4. Expose the API port
EXPOSE 8000

# 5. Boot up the engine
CMD ["uvicorn", "app.api:app", "--host", "0.0.0.0", "--port", "8000"]