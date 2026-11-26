FROM python:3.12-slim

# LightGBM needs the OpenMP runtime.
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY vidyutrakshak ./vidyutrakshak
COPY static ./static
COPY scripts ./scripts

# Build data, train v1..v4 through the feedback loop, export the dashboard.
RUN python -m vidyutrakshak.pipeline --fresh

RUN useradd --create-home vr && chown -R vr /app
USER vr

EXPOSE 8000
HEALTHCHECK CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/api/health')"
CMD ["uvicorn", "vidyutrakshak.api:app", "--host", "0.0.0.0", "--port", "8000"]
