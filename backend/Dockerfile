FROM python:3.10-slim

WORKDIR /code

# Copy the requirements file
COPY ./requirements.txt /code/requirements.txt

# Install dependencies (using CPU-only PyTorch to save massive space and build time)
RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu

# Copy the rest of the backend files AND the model file
# (We assume the model file is copied into the backend directory before pushing)
COPY . /code

# Expose port 7860 which is required by Hugging Face Spaces
EXPOSE 7860

# Run the FastAPI server
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860"]
