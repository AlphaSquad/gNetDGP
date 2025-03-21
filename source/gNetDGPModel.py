import torch
import torch.nn.functional as F
from torch.nn import Linear
from torch_geometric.nn import GCNConv, SAGEConv, GraphConv


class gNetDGPModel(torch.nn.Module):
    def __init__(
            self,
            gene_feature_dim,
            pheno_feature_dim,
            fc_hidden_dim=2048,
            gene_net_hidden_dim=512,
            pheno_net_hidden_dim=512,
            mode='DGP'
    ):
        super(gNetDGPModel, self).__init__()
        self.mode = mode
        fc_gene_classification_hidden_dim = fc_hidden_dim

        self.gene_conv_0 = GraphConv(gene_feature_dim, gene_net_hidden_dim)
        self.gene_conv_1 = GraphConv(gene_net_hidden_dim, gene_net_hidden_dim)
        self.gene_conv_2 = GraphConv(gene_net_hidden_dim, gene_net_hidden_dim)
        self.pheno_conv_0 = GraphConv(pheno_feature_dim, pheno_net_hidden_dim)
        self.pheno_conv_1 = GraphConv(pheno_net_hidden_dim, pheno_net_hidden_dim)
        self.pheno_conv_2 = GraphConv(pheno_net_hidden_dim, pheno_net_hidden_dim)

        self.bn_gene_0 = torch.nn.BatchNorm1d(gene_net_hidden_dim)
        self.bn_gene_1 = torch.nn.BatchNorm1d(gene_net_hidden_dim)
        self.bn_gene_2 = torch.nn.BatchNorm1d(gene_net_hidden_dim)
        self.bn_pheno_0 = torch.nn.BatchNorm1d(pheno_net_hidden_dim)
        self.bn_pheno_1 = torch.nn.BatchNorm1d(pheno_net_hidden_dim)
        self.bn_pheno_2 = torch.nn.BatchNorm1d(pheno_net_hidden_dim)

        self.lin1 = Linear(gene_net_hidden_dim + pheno_net_hidden_dim, fc_hidden_dim)
        self.lin2 = torch.nn.Linear(fc_hidden_dim, fc_hidden_dim // 2)
        self.lin3 = torch.nn.Linear(fc_hidden_dim // 2, fc_hidden_dim // 4)
        self.lin4 = torch.nn.Linear(fc_hidden_dim // 4, 2)

        # Gene classification mode.
        self.lin_gc1 = Linear(gene_net_hidden_dim, fc_gene_classification_hidden_dim)
        self.lin_gc2 = torch.nn.Linear(fc_gene_classification_hidden_dim, fc_gene_classification_hidden_dim // 2)
        self.lin_gc3 = torch.nn.Linear(fc_gene_classification_hidden_dim // 2, fc_gene_classification_hidden_dim // 4)
        self.lin_gc4 = torch.nn.Linear(fc_gene_classification_hidden_dim // 4, 2)

    def forward(self, gene_net_data, pheno_net_data, batch_idx):
        gene_x, gene_edge_index = gene_net_data.x, gene_net_data.edge_index
        pheno_x, pheno_edge_index = pheno_net_data.x, pheno_net_data.edge_index

        gene_x_out_0 = self.bn_gene_0(
            F.leaky_relu(self.gene_conv_0(gene_x, gene_edge_index))
        )
        gene_x_out_1 = self.bn_gene_1(
            F.leaky_relu(self.gene_conv_1(gene_x_out_0, gene_edge_index))
        )
        gene_x_out_2 = self.bn_gene_2(
            F.leaky_relu(self.gene_conv_2(gene_x_out_1, gene_edge_index))
        )

        pheno_x_out_0 = self.bn_pheno_0(
            F.leaky_relu(self.pheno_conv_0(pheno_x, pheno_edge_index))
        )
        pheno_x_out_1 = self.bn_pheno_1(
            F.leaky_relu(self.pheno_conv_1(pheno_x_out_0, pheno_edge_index))
        )
        pheno_x_out_2 = self.bn_pheno_2(
            F.leaky_relu(self.pheno_conv_2(pheno_x_out_1, pheno_edge_index))
        )

        gene_x_out = 0.7 * gene_x_out_0 + 0.2 * gene_x_out_1 + 0.1 * gene_x_out_2
        pheno_x_out = 0.7 * pheno_x_out_0 + 0.2 * pheno_x_out_1 + 0.1 * pheno_x_out_2

        x_gene = gene_x_out[batch_idx[:, 0]]
        x_pheno = pheno_x_out[batch_idx[:, 1]]

        if self.mode == 'DGP':
            x = torch.cat((x_gene, x_pheno), dim=1)

            x = F.dropout(x, p=0.5, training=self.training)
            x = F.leaky_relu(self.lin1(x))
            x = F.dropout(x, p=0.4, training=self.training)
            x = F.leaky_relu(self.lin2(x))
            x = F.dropout(x, p=0.2, training=self.training)
            x = F.leaky_relu(self.lin3(x))
            x = self.lin4(x)
        else:  # Gene classification.
            x = F.dropout(x_gene, p=0.5, training=self.training)
            x = F.leaky_relu(self.lin_gc1(x))
            x = F.dropout(x, p=0.4, training=self.training)
            x = F.leaky_relu(self.lin_gc2(x))
            x = F.dropout(x, p=0.2, training=self.training)
            x = F.leaky_relu(self.lin_gc3(x))
            x = self.lin_gc4(x)

        return x

    def get_gene_features(self, gene_net_data, disease_net_data):
        gene_x, gene_edge_index = gene_net_data.x, gene_net_data.edge_index

        gene_x_out_0 = self.bn_gene_0(
            F.leaky_relu(self.gene_conv_0(gene_x, gene_edge_index))
        )
        gene_x_out_1 = self.bn_gene_1(
            F.leaky_relu(self.gene_conv_1(gene_x_out_0, gene_edge_index))
        )
        gene_x_out_2 = self.bn_gene_2(
            F.leaky_relu(self.gene_conv_2(gene_x_out_1, gene_edge_index))
        )

        gene_x_out = 0.7 * gene_x_out_0 + 0.2 * gene_x_out_1 + 0.1 * gene_x_out_2

        return gene_x_out

    def get_pheno_features(self, gene_net_data, pheno_net_data):
        pheno_x, pheno_edge_index = pheno_net_data.x, pheno_net_data.edge_index

        pheno_x_out_0 = self.bn_pheno_0(
            F.leaky_relu(self.pheno_conv_0(pheno_x, pheno_edge_index))
        )
        pheno_x_out_1 = self.bn_pheno_1(
            F.leaky_relu(self.pheno_conv_1(pheno_x_out_0, pheno_edge_index))
        )
        pheno_x_out_2 = self.bn_pheno_2(
            F.leaky_relu(self.pheno_conv_2(pheno_x_out_1, pheno_edge_index))
        )

        pheno_x_out = 0.7 * pheno_x_out_0 + 0.2 * pheno_x_out_1 + 0.1 * pheno_x_out_2

        return pheno_x_out
