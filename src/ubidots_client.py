import os
import requests
from dotenv import load_dotenv

class UbidotsClient:
    def __init__(self):
        load_dotenv() # Load environment variables from .env file
        self.token = os.getenv("UBIDOTS_TOKEN")
        self.device_label = os.getenv("UBIDOTS_DEVICE_LABEL")
        self.base_url = "https://industrial.api.ubidots.com/api/v1.6/devices/"

        if not self.token:
            raise ValueError("UBIDOTS_TOKEN not found in .env file")
        if not self.device_label:
            raise ValueError("UBIDOTS_DEVICE_LABEL not found in .env file")

        self.headers = {
            "X-Auth-Token": self.token,
            "Content-Type": "application/json"
        }

    def publish_event(self, fall_event=None, help_signal=None, gesture=None, confidence=None):
        payload = {}
        if fall_event is not None:
            payload["fall_event"] = {"value": fall_event}
        if help_signal is not None:
            payload["help_signal"] = {"value": help_signal}
        if gesture is not None:
            payload["gesture"] = {"value": gesture}
        if confidence is not None:
            payload["confidence"] = {"value": confidence}

        if not payload:
            print("No data to publish.")
            return

        url = f"{self.base_url}{self.device_label}"

        try:
            response = requests.post(url, headers=self.headers, json=payload)
            response.raise_for_status()
            print(f"Data published to Ubidots: {payload}")
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"Error publishing data to Ubidots: {e}")
            return None

if __name__ == '__main__':
    # Example Usage (for testing purposes)
    # Make sure you have a .env file with UBIDOTS_TOKEN and UBIDOTS_DEVICE_LABEL
    try:
        ubidots_client = UbidotsClient()
        # Example: Send a fall event and a gesture
        ubidots_client.publish_event(fall_event=1, gesture="Peligro", confidence=0.95)
        ubidots_client.publish_event(help_signal=0, gesture="Todo_OK", confidence=0.8)
    except ValueError as e:
        print(f"Configuration Error: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
