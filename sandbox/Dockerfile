FROM python:3.11-slim
RUN pip install --no-cache-dir pandas scikit-learn openpyxl
RUN useradd --create-home --uid 1000 sandbox
USER sandbox
WORKDIR /tmp
ENV MPLCONFIGDIR=/tmp PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1