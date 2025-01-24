from torch.nn import Module, Embedding, Parameter
import torch

class MF(torch.nn.Module):

    def __init__(self, n_users, n_subs, d=256, device='cuda', bias=None):
        """Initialize MF Class."""
        super(MF, self).__init__()

        self.n_users = n_users
        self.n_subs = n_subs
        self.d = d
        self.bias = bias

        self.user_emb = Embedding(self.n_users, self.d)
        self.sub_emb = Embedding(self.n_subs, self.d)
        
        torch.nn.init.normal_(self.user_emb.weight, 0, .1)
        torch.nn.init.normal_(self.sub_emb.weight, 0, .1)

        if bias == 'prior':

            self.u_emb_bias = Parameter(torch.rand(d, device=device) * .2 - .1) 
            self.s_emb_bias = Parameter(torch.rand(d, device=device) * .2 - .1)

        elif bias == 'unknowns':

            self.unknown_u = Parameter(torch.rand(d, device=device) * .2 - .1) 
            self.unknown_s = Parameter(torch.rand(d, device=device) * .2 - .1)

        elif bias == 'mf_google':
            self.user_bias = Embedding(self.n_users, 1)
            self.sub_bias = Embedding(self.n_subs, 1)
            self.global_bias = Parameter(torch.zeros(1))
            self.user_bias.weight.data.fill_(0.0)
            self.sub_bias.weight.data.fill_(0.0)
            self.global_bias.data.fill_(0.0)


        self.device = device
        self = self.to(device)

    def __str__(self) -> str:
        return f"MF (d={self.d}, bias {self.bias})"

    def forward(self, batch_data, test=False, training_phase = 0):
        """Trian the model.
        Args:
            batch_data: tuple consists of [users, subs], which must be LongTensor.
        """
        users, subs = batch_data[:,0], batch_data[:,1]
        u_emb = self.user_emb(users)
        s_emb = self.sub_emb(subs)

        if self.bias == 'mf_google':
            u_bias = self.user_bias(users)
            s_bias = self.sub_bias(subs)    

            scores = torch.sigmoid(
            torch.sum(torch.mul(u_emb, s_emb).squeeze(), dim=1)
            + u_bias.squeeze()
            + s_bias.squeeze()
            + self.global_bias
            )

            regularizer = (
            (u_emb ** 2).sum()
            + (s_emb ** 2).sum()
            + (u_bias ** 2).sum()
            + (s_bias ** 2).sum()
            ) / u_emb.size()[0]
        
        elif self.bias == 'prior':

            
            scores = torch.sigmoid(
                torch.sum(torch.mul(u_emb+self.u_emb_bias, s_emb+self.s_emb_bias).squeeze(), dim=1)
                # torch.sum(torch.mul(u_emb, s_emb+self.s_emb_bias).squeeze(), dim=1)
                # torch.sum(torch.mul(u_emb+self.u_emb_bias, s_emb).squeeze(), dim=1)
                # torch.sum(torch.mul(u_emb, self.s_emb_bias).squeeze(), dim=1)
                # torch.sum(torch.mul(s_emb, self.u_emb_bias).squeeze(), dim=1)
            )

            if training_phase == 0:
                regularizer = (
                    (u_emb ** 2).sum()
                    + (s_emb ** 2).sum()
                    + (self.u_emb_bias ** 2).sum()
                    + (self.s_emb_bias ** 2).sum()
                ) / users.size()[0]
            
            elif training_phase==1:

                regularizer = (
                    (u_emb ** 2).sum()
                    + (self.s_emb_bias ** 2).sum()
                ) / users.size()[0]

            elif training_phase==2:
                regularizer = (
                    (s_emb ** 2).sum()
                    + (self.u_emb_bias ** 2).sum()
                ) / users.size()[0]

        elif self.bias == 'unknowns':
            unknown_ratio=0.2

            if not test:
                use_unknown_u = torch.rand((u_emb.size()[0],1), device=self.device)
                use_unknown_s = torch.rand((s_emb.size()[0],1), device=self.device)

                u_aux = torch.add(u_emb*((use_unknown_u>unknown_ratio).float()),
                                    torch.matmul((use_unknown_u<=unknown_ratio).float(),self.unknown_u.unsqueeze(0)))
                s_aux = torch.add(s_emb*((use_unknown_s>unknown_ratio).float()),
                                    torch.matmul((use_unknown_s<=unknown_ratio).float(),self.unknown_s.unsqueeze(0)))

            else: 
                u_aux = u_emb
                s_aux = s_emb

            scores = torch.sigmoid(
                torch.sum(torch.mul(u_aux, s_aux).squeeze(), dim=1)
            )

            regularizer = (
                (u_aux ** 2).sum()
                + (s_aux ** 2).sum()
            ) / u_emb.size()[0]

        return scores, regularizer

    def predict(self, batch, **params):
        """Predcit result with the model.
        Return:
            scores (int, or list of int): predicted scores of these user-item pairs.
        """
        with torch.no_grad():
            scores, _ = self.forward(batch, **params)
        return scores

    