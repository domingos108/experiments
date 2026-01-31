import torch
from neuralforecast.models import KAN
import torch.optim as optim

class KANLBFGS(KAN):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Ativa otimização manual para suportar o closure do LBFGS
        self.automatic_optimization = False

    def configure_optimizers(self):
        return optim.LBFGS(
            self.parameters(),
            lr=self.learning_rate,
            history_size=10,
            line_search_fn="strong_wolfe"
        )

    def training_step(self, batch, batch_idx):
        opt = self.optimizers()

        def closure():
            opt.zero_grad()
            # O NeuralForecast armazena a função de perda em self.loss
            # self._get_batch_vars extrai as variáveis X, y, etc. do dicionário
            batch_vars = self._get_batch_vars(batch)
            insample_y, insample_mask, outsample_y, outsample_mask, hist_exog, futr_exog, stat_exog = batch_vars
            
            # Passada para frente
            outputs = self(batch)
            
            # Cálculo da perda (usando a máscara para ignorar zeros/paddings)
            loss = self.loss(outsample_y, outputs, outsample_mask)
            
            self.manual_backward(loss)
            return loss

        opt.step(closure=closure)
        
        # Log da perda opcional
        current_loss = closure()
        self.log("train_loss", current_loss, prog_bar=True)