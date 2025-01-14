from pprint import pprint
from graph import app

inputs = {"question": "who is albert einstein"}
for output in app.stream(inputs):
    for key, value in output.items():
        pprint(f"Node '{key}':")
    pprint("\n---\n")

pprint(value["generation"])