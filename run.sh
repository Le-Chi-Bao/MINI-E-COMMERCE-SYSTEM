step 0: create conda env
conda create -n crawler-env python=3.9 -y
conda activate crawler-env
pip install -r requirements.txt

# Step 0: Start Redis
echo "Step 0: Starting Redis..."
docker-compose up -d redis
