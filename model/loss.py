import torch

class Loss(torch.nn.Module):
    
    def __init__(self, config, lam, mu):
        super(Loss, self).__init__()
        self.config = config
        self.lam = lam
        self.mu = mu

        self.crit = torch.nn.BCELoss()
        self.independency = torch.nn.MSELoss()

        if self.config['regular'] == 'l2':
            self.reg = torch.nn.MSELoss()
        elif self.config['regular'] == 'l1':
            self.reg = torch.nn.L1Loss()
        else:
            self.reg = torch.nn.MSELoss()
    
    def forward(self, predictions, truth, user_personality, user_commonality, item_personality, item_commonality):

        if self.config['regular'] == 'l2':
            user_dummy_target = torch.zeros_like(user_commonality, requires_grad=False)
            item_dummy_target = torch.zeros_like(item_commonality, requires_grad=False)
            user_reg_term = self.reg(user_commonality, user_dummy_target)
            item_reg_term = self.reg(item_commonality, item_dummy_target)
            third = user_reg_term + item_reg_term
        elif self.config['regular'] == 'l1':
            user_dummy_target = torch.zeros_like(user_commonality, requires_grad=False)
            item_dummy_target = torch.zeros_like(item_commonality, requires_grad=False)
            user_reg_term = self.reg(user_commonality, user_dummy_target)
            item_reg_term = self.reg(item_commonality, item_dummy_target)
            third = user_reg_term + item_reg_term
        elif self.config['regular'] == 'none':
            self.config['mu'] = 0
            user_dummy_target = user_commonality
            item_dummy_target = item_commonality
            user_reg_term = self.reg(user_commonality, user_dummy_target)
            item_reg_term = self.reg(item_commonality, item_dummy_target)
            third = user_reg_term + item_reg_term
        elif self.config['regular'] == 'nuc':
            user_third = torch.norm(user_commonality, p='nuc')
            item_third = torch.norm(item_commonality, p='nuc')
            third = user_third + item_third
        elif self.config['regular'] == 'inf':
            user_third = torch.norm(user_commonality, p=float('inf'))
            item_third = torch.norm(item_commonality, p=float('inf'))
            third = user_third + item_third
        else:
            user_dummy_target = torch.zeros_like(user_commonality, requires_grad=False)
            item_dummy_target = torch.zeros_like(item_commonality, requires_grad=False)
            user_reg_term = self.reg(user_commonality, user_dummy_target)
            item_reg_term = self.reg(item_commonality, item_dummy_target)
            third = user_reg_term + item_reg_term

        loss = self.crit(predictions, truth) \
               - self.lam * self.independency(user_personality, user_commonality) \
               - self.lam * self.independency(item_personality, item_commonality) \
               + self.mu * third

        return loss
