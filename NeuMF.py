import torch
import torch.nn as nn


class NeuMF(nn.Module):
    def __init__(self, n_users, n_items, latent_mlp, latent_mf, layers):
        super().__init__()
        self.embedding_user_mlp = nn.Embedding(num_embeddings=n_users, embedding_dim=latent_mlp)
        self.embedding_item_mlp = nn.Embedding(num_embeddings=n_items, embedding_dim=latent_mlp)

        self.embedding_user_mf = nn.Embedding(num_embeddings=n_users, embedding_dim=latent_mf)
        self.embedding_item_mf = nn.Embedding(num_embeddings=n_items, embedding_dim=latent_mf)

        self.fc_layers = nn.ModuleList([])
        mlp_out = 0
        for in_dim, out_dim in layers:
            mlp_out = out_dim
            self.fc_layers.append(nn.Linear(in_dim, out_dim))

        self.affine_output = nn.Linear(latent_mf + mlp_out, 1)
        self.logistic = nn.Sigmoid()

    def forward(self, u, i):
        u_embedding_mlp = self.embedding_user_mlp(u)
        u_embedding_mf = self.embedding_user_mf(u)

        i_embedding_mlp = self.embedding_item_mlp(i)
        i_embedding_mf = self.embedding_item_mf(i)

        mlp_vector = torch.cat([u_embedding_mlp, i_embedding_mlp], dim=-1)
        mf_vector = torch.mul(u_embedding_mf, i_embedding_mf)

        for i in range(len(self.fc_layers)):
            mlp_vector = self.fc_layers[i](mlp_vector)
            mlp_vector = nn.GELU()(mlp_vector)

        vector = torch.cat([mlp_vector, mf_vector], dim=-1)
        logits = self.affine_output(vector)
        rating = self.logistic(logits)
        return rating




