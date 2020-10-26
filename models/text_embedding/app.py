from transformers import DistilBertTokenizer, DistilBertModel
from flask import request, Flask, jsonify
import json

app = Flask(__name__)

tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')#, output_hidden_states=True)
model = DistilBertModel.from_pretrained('distilbert-base-uncased')
model.eval()


def preprocess(text):
    '''
    Take input text and convert it into form needed for model
    '''
    return tokenizer(text, return_tensors='pt')

def get_text(request):
    json = request.get_json()
    return json['text']

@app.route('/embed', methods=['POST'])
def embed():
    if request.method == 'POST':
        text = get_text(request)
        inputs = preprocess(text)
        outputs = model(**inputs)
        '''
        The output is (Hidden Layer #, # tokens, # embedding dimension). In our case the only hidden
        layer is the last one anyways so we'd get something with size (1, # tokens, 768) and then 
        we just want to get the [CLS] token for purposes of getting the input text embedding
        '''
        hidden_layer_embedding = outputs[0]
        cls_embedding = hidden_layer_embedding[0][0]
        # convert the torch.Tensor to list so we can pass it
        data = {'embedding': cls_embedding.tolist()}
        return data

if __name__ == '__main__':
    app.run(port=5001)