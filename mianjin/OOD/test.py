class MLP(BaseModel):
    
    def __init__(self, num_input_columns=104):
        super().__init__()
        # define submodules/layers here 
        # please fill in
        self.mlp=nn.Sequential(
            nn.Linear(104,128),
            nn.ReLU(),
            nn.Linear(128,64),
            nn.ReLU(),
            nn.Linear(64,1),
            nn.Sigmoid()
        )
    
    def forward(self, numerical_data, text_data): 
        # define model forward pass here (ignore text_data for now)
        # return predicted probability of label==1
        # please fill in
        output=self.mlp(numerical_data).squeeze()
        return output

class MLP_v2(BaseModel):  #base on the base model
    def __init__(self, num_input_columns=104):
        super().__init__()
        self.layers = torch.nn.Sequential(
            torch.nn.Linear(num_input_columns,256),
            torch.nn.BatchNorm1d(256),
            torch.nn.Dropout(0.1),
            torch.nn.ReLU(),
            torch.nn.Linear(256,128),
            torch.nn.BatchNorm1d(128),
            torch.nn.Dropout(0.1),
            torch.nn.ReLU(),
            torch.nn.Linear(128,1),
            torch.nn.Sigmoid()
        )
        
    def forward(self, numerical_data, text_data):
        return self.layers(numerical_data).squeeze()

class BusinessNameModel(BaseModel):
    def __init__(self, business_names):
        super().__init__()
        # define model
        # please fill in 
        '''
        先define embedding
        再define mlp
        '''
        self.token=SimpleTokenizer(business_names)
        self.embed_layer=nn.EmbeddingBag(num_embeddings=len(self.token),
                                      embedding_dim=32,padding_idx=0,mode='mean') #The default mode is "mean"

        self.mlp=nn.Sequential(
            nn.Linear(32,128),
            nn.BatchNorm1d(128),
            nn.Dropout(0.1),
            nn.ReLU(),
            nn.Linear(128,64),
            nn.BatchNorm1d(64),
            nn.Dropout(0.1),
            nn.ReLU(),
            nn.Linear(64,1),
            nn.Sigmoid()
        )

    
    def forward(self, numerical_data, text_data):
        # implement forward pass
        # Please fill in
        x=self.token(text_data)
        x=self.embed_layer(x)
        x=self.mlp(x)
        return x.squeeze()



def __init__(self, business_names):
        super().__init__()
        # define model
        self.tokenizer=SimpleTokenizer(business_names)
        self.embedding_layer=torch.nn.EmbeddingBag(
            num_embeddings=len(self.tokenizer),
            embedding_dim=32,
            padding_idx=0
        )

        self.text_layers = torch.nn.Sequential(
            torch.nn.Linear(32,256), #32 是embedding dim
            torch.nn.BatchNorm1d(256),
            torch.nn.Dropout(0.1),
            torch.nn.ReLU(),
            torch.nn.Linear(256,128),
            torch.nn.BatchNorm1d(128),
            torch.nn.Dropout(0.1),
            torch.nn.ReLU()
        )

        self.numerical_layers=torch.nn.Sequential(
            torch.nn.Linear(104,256),
            torch.nn.BatchNorm1d(256),
            torch.nn.Dropout(0.1),
            torch.nn.ReLU(),
            torch.nn.Linear(256,128),
            torch.nn.BatchNorm1d(128),
            torch.nn.Dropout(0.1),
            torch.nn.ReLU()
        )

        self.combined_layer=torch.nn.Sequential(
            torch.nn.Linear(256,128), #128+128=256
            torch.nn.BatchNorm1d(128),
            torch.nn.Dropout(0.1),
            torch.nn.ReLU(),
            torch.nn.Linear(128,1),
            torch.nn.Sigmoid()
        )
    
    def forward(self, numerical_data, text_data):
        # implement forward pass
        #text
        Xt=self.tokenizer(text_data)
        Xt=self.embedding_layer(Xt)
        Xt=self.text_layers(Xt)

        Xnum=self.numerical_layers(numerical_data)

        X=torch.cat([Xt,Xnum],dim=1)
        
        return self.combined_layer(X).squeeze()

