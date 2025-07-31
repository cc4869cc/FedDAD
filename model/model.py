import copy
import torch

from model.engine import Engine

class FedDAD(torch.nn.Module):
    def __init__(self, config):
        super(FedDAD, self).__init__()
        self.config = config
        self.num_items = config['num_items']
        self.num_users = config['num_users']
        self.latent_dim = config['latent_dim']

        self.user_personality = torch.nn.Embedding(num_embeddings=self.num_users, embedding_dim=self.latent_dim)
        self.user_commonality = torch.nn.Embedding(num_embeddings=self.num_users, embedding_dim=self.latent_dim)
        self.item_personality = torch.nn.Embedding(num_embeddings=self.num_items, embedding_dim=self.latent_dim)
        self.item_commonality = torch.nn.Embedding(num_embeddings=self.num_items, embedding_dim=self.latent_dim)

        self.logistic = torch.nn.Sigmoid()

    def setItemCommonality(self, item_commonality):
        self.item_commonality = copy.deepcopy(item_commonality)

    def setUserCommonality(self, user_commonality):
        self.user_commonality = copy.deepcopy(user_commonality)      

    def forward(self, user_indices, item_indices):
        
        user_personality = self.user_personality(user_indices)
        user_commonality = self.user_commonality(user_indices)
        item_personality = self.item_personality(item_indices)
        item_commonality = self.item_commonality(item_indices)

        user_embedding = user_personality + user_commonality
        item_embedding = item_personality + item_commonality

        logits = torch.sum(user_embedding * item_embedding, dim=1, keepdim=True)
        rating = self.logistic(logits)

        return rating, user_personality, user_commonality, item_personality, item_commonality

class FedDADEngine(Engine):
    """Engine for training & evaluating GMF model"""
    
    def __init__(self, config):
        self.model = FedDAD(config)
        if config['use_cuda'] is True:
            self.model.cuda()
        super(FedDADEngine, self).__init__(config)
        print(self.model)

