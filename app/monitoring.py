# 📁 app/monitoring.py (THÊM MỚI)
from prometheus_client import Counter, Histogram, generate_latest
from fastapi import Response
from fastapi.routing import APIRoute

# Metrics
PREDICTION_COUNT = Counter('predictions_total', 'Total predictions', ['model', 'status'])
PREDICTION_DURATION = Histogram('prediction_duration_seconds', 'Prediction latency')

class MonitoringRoute(APIRoute):
    def get_route_handler(self):
        original_route_handler = super().get_route_handler()
        
        async def custom_route_handler(request):
            with PREDICTION_DURATION.time():
                response = await original_route_handler(request)
            return response
        
        return custom_route_handler