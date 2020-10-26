from transformers import DistilBertTokenizer, DistilBertModel
import torch
from ts.torch_handler.base_handler import BaseHandler
import json



class MyHandler(BaseHandler):
    '''
    Custom handler for pytorch serve. This handler supports single requests.
    '''
    
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.initialized = False
    
    def initialize(self, ctx):
        self.manifest = ctx.manifest
        
        properties = ctx.system_properties
        model_dir = properties.get("model_dir")
#         self.device = torch.device("cuda:" + str(properties.get("gpu_id")) if torch.cuda.is_available() else "cpu")
        
        self.model = DistilBertModel.from_pretrained(model_dir)
        self.tokenizer = DistilBertTokenizer.from_pretrained(model_dir)
        
#         self.model.to(self.device)
        self.model.eval()
        
        logger.debug('Tranformer model from path {} loaded successfully'.format(model_dir))
        
        self.initialized = True        
    
    def preprocess(self, text):
        '''
        Take input text and convert it into form needed for model
        '''
        return tokenizer(text, return_tensors='pt')

    def get_text(self, request):
        json = request.get_json()
        return json['text']
    
    def embed(self, inputs):
        outputs = model(**inputs)
        hidden_layer_embedding = outputs[0]
        cls_embedding = hidden_layer_embedding[0][0]
        embedding = {'embedding': cls_embedding.tolist()}
        return embedding
    
_service = MyHandler()

def handle(request, context):
    try:
        if not _service.initialized:
            _service.intialize(context)
            
        if data is None:
            return None
        
        text = _service.get_text(request)
        inputs = _service.preprocess(text)
        embedding = _service.embed(inputs)
        
        return embedding
    except Exception as e:
        raise e
        






# def preprocess(text):
#     '''
#     Take input text and convert it into form needed for model
#     '''
#     return tokenizer(text, return_tensors='pt')

# def get_text(request):
#     json = request.get_json()
#     return json['text']

# @app.route('/embed', methods=['POST'])
# def embed():
#     if request.method == 'POST':
# #         print(request.data[0])
#         text = get_text(request)
#         inputs = preprocess(text)
#         outputs = model(**inputs)
#         '''
#         The output is (Hidden Layer #, # tokens, # embedding dimension). In our case the only hidden
#         layer is the last one anyways so we'd get something with size (1, # tokens, 768) and then 
#         we just want to get the [CLS] token for purposes of getting the input text embedding
#         '''
#         hidden_layer_embedding = outputs[0]
#         cls_embedding = hidden_layer_embedding[0][0]
#         # convert the torch.Tensor to list so we can pass it
#         data = {'embedding': cls_embedding.tolist()}
#         return data
    
# if __name__ == '__main__':
#     app.run(port=5001)