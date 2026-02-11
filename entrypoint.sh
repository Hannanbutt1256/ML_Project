#!/bin/sh
set -e

echo "Running training..."
python train.py

echo "Starting Streamlit app..."
cd /app
streamlit run app.py --server.port=$PORT 