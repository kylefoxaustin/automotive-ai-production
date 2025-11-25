class MetricsCollector:
    def __init__(self):
        self.metrics = []
    
    def add(self, metric):
        self.metrics.append(metric)
