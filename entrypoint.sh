#!/bin/sh
set -e

# 1. Run the initial training to ensure the base model exists for Streamlit
echo "Starting initial training run..."
python train.py --model_type rf --n_estimators 10 --max_depth 5

# 2. Start Streamlit in the background
echo "Starting Streamlit app on port $PORT..."
# We use --server.address=0.0.0.0 for Docker so it's accessible from the host
cd /app
streamlit run app.py --server.port=$PORT --server.address=0.0.0.0 &

# 3. Initialize the W&B Sweep and capture the FULL AGENT PATH
echo "Initializing W&B Sweep..."
# Capturing the output to extract the full path (entity/project/id)
SWEEP_OUTPUT=$(wandb sweep sweep.yaml 2>&1)
echo "--- W&B SWEEP OUTPUT START ---"
echo "$SWEEP_OUTPUT"
echo "--- W&B SWEEP OUTPUT END ---"

# Extract the full string like "entity/project/id" from the recommended command line
# Example: "wandb: Run sweep agent with: wandb agent hannanbutt-dev-hannan/loan-approval-prediction/h21h6ppw"
AGENT_FULL_PATH=$(echo "$SWEEP_OUTPUT" | grep "Run sweep agent with:" | awk '{print $NF}')

if [ -z "$AGENT_FULL_PATH" ]; then
    echo "Error: Could not capture Sweep Agent path. Check the output above for errors."
    exit 1
fi

echo "Captured Agent Path: $AGENT_FULL_PATH"

# 4. Start the W&B Agent
echo "Starting W&B Agent for $AGENT_FULL_PATH..."
# --count 5 ensures the container doesn't run forever during this flow
wandb agent "$AGENT_FULL_PATH" --count 5

echo "Sweep agent finished. Streamlit remains active."
# Keep the script alive so the container doesn't exit immediately
wait
