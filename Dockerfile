FROM python:3.11-slim

WORKDIR /app

# Install dependencies
RUN pip install --no-cache-dir numpy pillow scipy

# Copy solution script
COPY order.py ./

# Make script executable
RUN chmod +x order.py

ENTRYPOINT ["python3", "order.py"]
CMD ["/frames"]
