from bag_and_bow import bag
from predict import get_result
import requests
import json

word = "super super lange beschrijvng"
labels = ['EAN', 'GTIN',  'Title', 'Description', 'Short description', 'Medium description','Long description']

request = requests.post("http://localhost:8501/v1/models/test:predict", data=json.dumps({"instances":[bag(word)]}))
predictions = json.loads(request.text)

print(get_result(predictions['predictions'][0], labels))