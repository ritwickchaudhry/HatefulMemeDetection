import torch
import torch.nn as nn
from mmf.common.registry import registry
from mmf.models.base_model import BaseModel
from mmf.utils.build import (
    build_classifier_layer,
    build_image_encoder,
    build_text_encoder,
)


# Register the model for MMF, "concat_bert_tutorial" key would be used to find the model
@registry.register_model("concat_bert_ud")
class ConcatBERTUD(BaseModel):
    # All models in MMF get first argument as config which contains all
    # of the information you stored in this model's config (hyperparameters)
    def __init__(self, config):
        # This is not needed in most cases as it just calling parent's init
        # with same parameters. But to explain how config is initialized we
        # have kept this
        super().__init__(config)
        self.build()

    # This classmethod tells MMF where to look for default config of this model
    @classmethod
    def config_path(cls):
        # Relative to user dir root
        return "configs/models/concat_bert_ud/defaults.yaml"

    # Each method need to define a build method where the model's modules
    # are actually build and assigned to the model
    def build(self):
        """
        Config's image_encoder attribute will be used to build an MMF image
        encoder. This config in yaml will look like:

        # "type" parameter specifies the type of encoder we are using here.
        # In this particular case, we are using resnet152
        type: resnet152
        # Parameters are passed to underlying encoder class by
        # build_image_encoder
        params:
            # Specifies whether to use a pretrained version
            pretrained: true
            # Pooling type, use max to use AdaptiveMaxPool2D
            pool_type: avg
            # Number of output features from the encoder, -1 for original
            # otherwise, supports between 1 to 9
            num_output_features: 1
        """
        self.vision_module = build_image_encoder(self.config.image_encoder)

        """
        For text encoder, configuration would look like:
        # Specifies the type of the langauge encoder, in this case mlp
        type: transformer
        # Parameter to the encoder are passed through build_text_encoder
        params:
            # BERT model type
            bert_model_name: bert-base-uncased
            hidden_size: 768
            # Number of BERT layers
            num_hidden_layers: 12
            # Number of attention heads in the BERT layers
            num_attention_heads: 12
        """
        self.language_module = build_text_encoder(self.config.text_encoder)

        self.attention = ImageDefinitionAttention(self.config.modal_hidden_size,
                                                    self.config.text_hidden_size,
                                                    self.config.defn_hidden_size)

        """
        For classifer, configuration would look like:
        # Specifies the type of the classifier, in this case mlp
        type: mlp
        # Parameter to the classifier passed through build_classifier_layer
        params:
            # Dimension of the tensor coming into the classifier
            # Visual feature dim + Language feature dim : 2048 + 768
            in_dim: 2816
            # Dimension of the tensor going out of the classifier
            out_dim: 2
            # Number of MLP layers in the classifier
            num_layers: 2
        """
        self.classifier = build_classifier_layer(self.config.classifier)

    # Each model in MMF gets a dict called sample_list which contains
    # all of the necessary information returned from the image
    def forward(self, sample_list):
        # Text input features will be in "input_ids" key
        text = sample_list["input_ids"]
        # Similarly, image input will be in "image" key
        image = sample_list["image"]

        # Get the text and image features from the encoders
        text_features = self.language_module(text)[1]
        image_features = self.vision_module(image)

        # import pdb; pdb.set_trace()
        num_defs = [len(defns) for defns in sample_list["definitions"]]
        defn_ids = [torch.stack([defn["input_ids"] for defn in defns], axis=0) if len(defns) > 0 else [] for defns in sample_list["definitions"]]
        defn_features = [self.language_module(defn_id.to(text.device))[1] if len(defn_id) > 0 else torch.zeros_like(text_features[0])[None, :] for defn_id in defn_ids]

        # Flatten the embeddings before concatenation
        image_features = torch.flatten(image_features, start_dim=1)
        text_features = torch.flatten(text_features, start_dim=1)
        defn_features = torch.cat([self.attention(image, defn_feats) for image, defn_feats in zip(image_features, defn_features)], axis=0)

        # Concatenate the features returned from two modality encoders
        combined = torch.cat([text_features, image_features, defn_features], dim=1)

        # Pass final tensor to classifier to get scores
        logits = self.classifier(combined)

        # For loss calculations (automatically done by MMF
        # as per the loss defined in the config),
        # we need to return a dict with "scores" key as logits
        output = {"scores": logits}

        # MMF will automatically calculate loss
        return output


class ImageDefinitionAttention(nn.Module):
    def __init__(self, image_dim, definition_dim, output_dim):
        super().__init__()
        self.attention = nn.MultiheadAttention(image_dim, 1, kdim=definition_dim, vdim=definition_dim)
        self.fc = nn.Linear(image_dim, output_dim)
    
    def forward(self, image, definitions):
        # Image - (I,) = (2048,)
        # Definitions - (L,D) = (N, 768)
        import pdb; pdb.set_trace()
        output,_ = self.attention(image[None,None,:], definitions[:,None,:], definitions[:,None,:]) # (1,1,2048)
        output = self.fc(output)
        output = torch.flatten(output, start_dim=1)
        return output # (1, output_dim)