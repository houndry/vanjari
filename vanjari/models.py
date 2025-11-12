import torch
from torch import nn
from hierarchicalsoftmax import HierarchicalSoftmaxLazyLinear, SoftmaxNode
from polytorch import split_tensor, total_size
import torch.nn.functional as F


class VanjariAttentionModel(nn.Module):
    def __init__(
        self, 
        classification_tree:SoftmaxNode, 
        features:int=5120, 
        intermediate_layers:int=0, 
        growth_factor:float=2.0, 
        dropout:float=0.0,
        attention_hidden_size:int=512,
    ):
        super().__init__()

        assert growth_factor > 0.0
        
        self.classification_tree = classification_tree
        modules = [nn.LazyLinear(out_features=features), nn.PReLU()]
        for _ in range(intermediate_layers):
            out_features = int(features * growth_factor + 0.5)
            modules += [nn.LazyLinear(out_features=out_features), nn.PReLU(), nn.Dropout(dropout)]
            features = out_features

        self.sequential = nn.Sequential(*modules)

        self.attention_layer = nn.Sequential(
            nn.Linear(out_features, attention_hidden_size),  # (batch_size, seq_length, hidden_size)
            nn.Tanh(),
            nn.Linear(attention_hidden_size, 1)  # (batch_size, seq_length, 1)
        )

        self.classifier = HierarchicalSoftmaxLazyLinear(root=classification_tree)
        self.model_dtype = next(self.sequential.parameters()).dtype

    def forward(self, x):        
        if self.model_dtype != x.dtype:
            x = x.to(dtype=self.model_dtype)

        # Hack for sanity
        # assert x.shape[-1] == 1024

        x = self.sequential(x)

        attention_scores = self.attention_layer(x)
        attention_weights = torch.softmax(attention_scores, dim=1)

        context_vector = torch.sum(attention_weights * x, dim=1)

        result = self.classifier(context_vector)

        return result
        

class ConvAttentionClassifier(nn.Module):
    def __init__(
        self,
        embedding_dim=8,
        cnn_layers=6,
        output_types=None,
        cnn_dims_start=64,
        kernel_size_maxpool=2,
        return_penultimate:bool=False,
        num_embeddings=5,  # i.e. the size of the vocab which is N, A, C, G, T
        kernel_size=3,
        factor=2,
        padding="same",
        padding_mode="zeros",
        dropout=0.5,
        final_bias=True,
        penultimate_dims: int = 1028,
        attention_hidden_size: int = 512,
    ):
        super().__init__()

        self.embedding_dim = embedding_dim
        self.cnn_layers = cnn_layers
        self.output_types = output_types
        self.kernel_size_maxpool = kernel_size_maxpool
        self.return_penultimate = return_penultimate

        self.num_embeddings = num_embeddings
        self.kernel_size = kernel_size
        self.factor = factor
        self.dropout = dropout

        self.embedding = nn.Embedding(
            num_embeddings=num_embeddings,
            embedding_dim=embedding_dim,
        )

        in_channels = embedding_dim
        out_channels = cnn_dims_start
        conv_layers = []
        for _ in range(cnn_layers):
            conv_layers += [
                nn.Conv1d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=kernel_size,
                    padding=padding,
                    padding_mode=padding_mode,
                ),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.MaxPool1d(kernel_size_maxpool),
            ]
            in_channels = out_channels
            out_channels = int(out_channels * factor)

        self.attention_layer = nn.Sequential(
            nn.Linear(in_channels, attention_hidden_size),  # (batch_size, seq_length, hidden_size)
            nn.Tanh(),
            nn.Linear(attention_hidden_size, 1)  # (batch_size, seq_length, 1)
        )

        self.conv = nn.Sequential(*conv_layers)

        current_dims = in_channels
        self.penultimate = nn.Linear(in_features=current_dims, out_features=penultimate_dims, bias=True)
        self.final = nn.Linear(in_features=penultimate_dims, out_features=total_size(self.output_types), bias=final_bias)

    def forward(self, x):
        # Convert to int because it may be simply a byte
        x = x.int()
        x = self.embedding(x)

        # Transpose seq_len with embedding dims to suit convention of pytorch CNNs (batch_size, input_size, seq_len)
        x = x.transpose(1, 2)
        x = self.conv(x)
        x = x.transpose(1, 2)

        # weighted sum with weights from attention layer
        attention_scores = self.attention_layer(x)
        attention_weights = torch.softmax(attention_scores, dim=1)
        context = torch.sum(attention_weights * x, dim=1)

        penultimate_result = self.penultimate(context)
        predictions = self.final(F.relu(penultimate_result))

        split_predictions = split_tensor(predictions, self.output_types, feature_axis=1)

        if self.return_penultimate:
            return split_predictions, penultimate_result
    
        return split_predictions

    def replace_output_types(self, output_types, final_bias:bool=None) -> None:
        device = next(self.parameters()).device
        self.output_types = output_types
        penultimate_dims = self.penultimate.out_features
        if final_bias is None:
            final_bias = self.final.bias
        
        self.final = nn.Linear(in_features=penultimate_dims, out_features=total_size(self.output_types), bias=final_bias, device=device)
        
