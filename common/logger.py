import os
import json
from datetime import datetime


class Logger:
    def __init__(self, log_dir, log_file="log.json"):
        os.makedirs(log_dir, exist_ok=True)
        self.log_path = os.path.join(log_dir, log_file)
        self.logs = []

    def log(self, data):
        timestamp = str(datetime.now())
        data['timestamp'] = timestamp
        self.logs.append(data)
        print(json.dumps(data, indent=4, ensure_ascii=False))

    def save(self):
        with open(self.log_path, 'w', encoding='utf-8') as f:
            json.dump(self.logs, f, indent=4, ensure_ascii=False)
        print(f"Log saved to {self.log_path}")


