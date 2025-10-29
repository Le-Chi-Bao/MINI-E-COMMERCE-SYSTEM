import subprocess
import time
from pathlib import Path
from datetime import datetime
import sys
import os

# Fix Unicode encoding issue on Windows
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')
    os.environ['PYTHONIOENCODING'] = 'utf-8'
class FeatureStoreManager:
    """Manage Feast feature store operations"""
    
    def __init__(self, repo_path="."):
        self.repo_path = Path(repo_path)
    
    def setup_redis(self):
        """Start Redis server for online store"""
        try:
            result = subprocess.run([
                "docker", "run", "-d", 
                "-p", "6379:6379",
                "--name", "feast-redis",
                "redis:alpine"
            ], capture_output=True, text=True)
            
            if result.returncode == 0:
                print("✅ Redis server started successfully")
                time.sleep(3)
                return True
            else:
                print("⚠️ Redis might already be running")
                return True
                
        except Exception as e:
            print(f"❌ Failed to start Redis: {e}")
            return False
    
    def apply_feature_store(self):
        """Apply Feast feature definitions"""
        try:
            result = subprocess.run(
                ["feast", "apply"], 
                cwd=self.repo_path,
                capture_output=True,
                text=True
            )
            
            if result.returncode == 0:
                print("✅ Feast feature store applied successfully!")
                return True
            else:
                print("❌ Error applying Feast feature store")
                if result.stderr:
                    print(result.stderr)
                return False
                
        except FileNotFoundError:
            print("❌ 'feast' command not found. Install with: pip install feast")
            return False
    
    def materialize_features(self, start_date="2024-01-01", end_date=None):
        """Materialize features to online store"""
        if end_date is None:
            end_date = datetime.now().strftime("%Y-%m-%d")
            
        try:
            result = subprocess.run([
                "feast", "materialize", start_date, end_date
            ], cwd=self.repo_path, capture_output=True, text=True)
            
            if result.returncode == 0:
                print("✅ Features materialized successfully!")
                return True
            else:
                print("❌ Error materializing features")
                return False
                
        except Exception as e:
            print(f"❌ Error: {e}")
            return False
    
    def setup_complete_pipeline(self):
        """Run complete feature store setup"""
        print("🚀 Setting up Feature Store Pipeline...")
        
        if self.setup_redis():
            if self.apply_feature_store():
                self.materialize_features()
                print("🎉 Feature Store setup completed!")
                return True
        
        return False

# Usage
if __name__ == "__main__":
    fs_manager = FeatureStoreManager()
    fs_manager.setup_complete_pipeline()