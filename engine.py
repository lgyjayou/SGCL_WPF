import util
import torch
import torch.optim as optim


class trainer():
    def __init__(self, device, model, scaler, method, lam1, lam2, lrate):
        self.device = device
        self.model = model
        self.model.to(device)
        self.scaler = scaler
        self.loss = util.masked_mae
        self.optimizer = optim.AdamW(self.model.parameters(), lr=lrate, weight_decay=1e-4)

        self.method = method
        self.lam1 = lam1
        self.lam2 = lam2

    def train(self, input, real_val):
        self.model.train()
        self.optimizer.zero_grad()
        
        if self.method == 'pure':
            output = self.model(input)
            output = output.squeeze(-1)
            predict = self.scaler.inverse_transform(output)
            s_loss = self.loss(predict, real_val)
            rmse = util.masked_rmse(predict, real_val).item()
            s_loss.backward()
            self.optimizer.step()
            return s_loss.item(), rmse, 0, 0, 0, 0
        
        elif self.method == 'contrastive_learning':
            output, node_u_loss, tc_loss = self.model(input)
            output = output.squeeze(-1)
            predict = self.scaler.inverse_transform(output)
            s_loss = self.loss(predict, real_val, 0.0)
            rmse = util.masked_rmse(predict, real_val, 0.0).item()
            loss = s_loss + self.lam1 * node_u_loss + self.lam2 * tc_loss
            loss.backward()
            self.optimizer.step()
            return loss.item(), rmse, s_loss.item(), node_u_loss.item(), tc_loss.item()
    
    def eval(self, input, real_val):
        self.model.eval()
        with torch.no_grad():
            output = self.model(input)
            output = output.squeeze(-1)
            predict = self.scaler.inverse_transform(output)
            loss = self.loss(predict, real_val, 0.0)
            rmse = util.masked_rmse(predict, real_val, 0.0).item()
            return loss.item(), rmse
