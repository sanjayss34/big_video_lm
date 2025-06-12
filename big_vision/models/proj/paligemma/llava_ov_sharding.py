import mlxu
from scalax.sharding import (
    MeshShardingHelper, TreePathShardingRule, with_sharding_annotation, ShardingRule
)
from jax.sharding import PartitionSpec as PS
import numpy as np

class LlavaOneVisionShardingConfig(object):
    """Sharding config for llama model."""

    @staticmethod
    def get_default_config(updates=None, mesh_dims=None):
        config = mlxu.config_dict()
        config.mesh_dim = mesh_dims
        config.shard_model_along_sequence = True
        return mlxu.update_config_dict(config, updates)

    def __init__(self, config, img_scan=True, mesh_dims=None):
        self.config = self.get_default_config(config, mesh_dims)
        self._ring_attention_function = None
        self.img_scan = img_scan

    def get_mesh(self):
        axis_dims = self.config.mesh_dim
        if axis_dims.startswith('!'):
            # Allow splitting a physical mesh axis if needed
            mesh_axis_splitting = True
            axis_dims = axis_dims[1:]
        else:
            mesh_axis_splitting = False

        names = ('replica', 'fsdp', 'sequence', 'tensor')
        dims = [int(x) for x in axis_dims.split(',')]
        assert len(dims) == len(names)
        return MeshShardingHelper(dims, names, mesh_axis_splitting)

    def get_model_sharding_rule(self):
        """ Get the tree path based partition rule for LLaMA model. """
        if self.config.shard_model_along_sequence:
            model_all_gather_axis = ('fsdp', 'sequence')
        else:
            model_all_gather_axis = 'fsdp'
        img_all_gather_axis = 'fsdp'
        data_axis = ('replica', 'fsdp')
        img_encoder_block = [
          ('img/Transformer/encoderblock/MlpBlock_0/Dense_0/kernel', PS(None, img_all_gather_axis, None)),
          ('img/Transformer/encoderblock/MlpBlock_0/Dense_1/kernel', PS(None, None, img_all_gather_axis)), # model_all_gather_axis)),
          # ('img/Transformer/encoderblock/MultiHeadDotProductAttention_0/(key|query|value)/kernel', PS(None, model_all_gather_axis, None, None)),
          ('img/Transformer/encoderblock/MultiHeadDotProductAttention_0/(key|query|value)/kernel', PS(None, img_all_gather_axis, None)),
          # ('img/Transformer/encoderblock/MultiHeadDotProductAttention_0/out/kernel', PS(None, None, None, model_all_gather_axis)),
          ('img/Transformer/encoderblock/MultiHeadDotProductAttention_0/out/kernel', PS(None, None, img_all_gather_axis)),
        ]
        if not self.img_scan:
          img_encoder_block = [
            ('img/Transformer/encoderblock_.*/MlpBlock_0/Dense_0/kernel', PS(img_all_gather_axis, None)),
            ('img/Transformer/encoderblock_.*/MlpBlock_0/Dense_1/kernel', PS(None, img_all_gather_axis)), # model_all_gather_axis)),
            ('img/Transformer/encoderblock_.*/MultiHeadDotProductAttention_0/(key|query|value)/kernel', PS(img_all_gather_axis, None)),
            ('img/Transformer/encoderblock_.*/MultiHeadDotProductAttention_0/out/kernel', PS(None, img_all_gather_axis)),
          ]
        return TreePathShardingRule(
            # embeddings
            ('llm/model/embed_tokens/embedding', PS(model_all_gather_axis, 'tensor')),
            # atention
            ('llm/model/layers/.*/self_attn/(k_proj|q_proj|v_proj)/kernel', PS(model_all_gather_axis, 'tensor')),
            ('llm/model/layers/.*/self_attn/(k_proj|q_proj|v_proj)/bias', PS('tensor')),
            ('llm/layers/.*/attn/kv_einsum/w', PS(None, None, model_all_gather_axis, 'tensor')),
            ('llm/model/layers/.*/mlp/(gate_proj|up_proj)/kernel', PS(model_all_gather_axis, 'tensor')),
            ('llm/model/layers/.*/mlp/down_proj/kernel', PS('tensor', model_all_gather_axis)),
            ('llm/model/layers/.*/input_layernorm/kernel', PS('tensor')),
            ('llm/model/layers/.*/post_attention_layernorm/kernel', PS('tensor')),
            ('llm/model/layers/.*/self_attn/o_proj/kernel', PS(model_all_gather_axis, 'tensor')),
            ('llm/lm_head/kernel', PS('tensor', model_all_gather_axis)),
            ('llm/model/norm/kernel', PS('tensor')),
            *img_encoder_block,
            # ('img/pos_embedding', PS(None, 'sequence', None)),
            ('img/head1/kernel', PS(None, img_all_gather_axis)),
            ('img/head2/kernel', PS(img_all_gather_axis, None)),
            ('.*', PS(None)),
        )

    def get_intermediate_sharding_rules(self):
        return {
            'concatenated_image_text': PS(('replica', 'fsdp'), 'sequence'),
            'ffw_intermediate': PS(('replica', 'fsdp'), 'sequence', 'tensor'),
            'attention_kqv': PS(('replica', 'fsdp'), 'sequence'),
            'mask': PS(('replica', 'fsdp'), 'sequence'),
            'flattened_image': PS(('replica', 'fsdp', 'sequence'), None, None, None),
            'flattened_image_patches': PS(('replica', 'fsdp', 'sequence'), None, None),
        }

    def get_batch_sharding(self):
        return TreePathShardingRule(
          ('text', PS(('replica', 'fsdp'), None)),
          ('mask_image', PS(('replica', 'fsdp'), None)),
          ('image', PS(('replica', 'fsdp'), None, None, None, None)),
          ('mask_ar', PS(('replica', 'fsdp'), None)),
          ('mask_loss', PS(('replica', 'fsdp'), None)),
          # ('_.*', PS(None)),
          ('_mask', PS(('replica', 'fsdp'))),
          ('.*', PS(("replica", "fsdp"))),
        )

    def get_cache_sharding(self):
        return TreePathShardingRule(
          ('llm/model/layers/.*/self_attn/cached_key', PS(('replica', 'fsdp'), 'sequence', 'tensor')),
          ('llm/model/layers/.*/self_attn/cached_value', PS(('replica', 'fsdp'), 'sequence', 'tensor')),
          ('llm/model/layers/.*/self_attn/cache_index', PS(('replica', 'fsdp'))),
          ('llm/pre_logits_mask', PS(('replica', 'fsdp'), 'sequence')),
          ('llm/pre_logits', PS(('replica', 'fsdp'), 'sequence', 'tensor')),
          ('llm/full_position_ids', PS(('replica', 'fsdp'), 'sequence')),
          ('.*', PS(('replica', 'fsdp')))
        )

    def get_cache_template(self, batch_size, max_decode_len, train_state):
        return {
          'llm': {
            'seq_len': np.zeros((batch_size,), dtype=np.int32),
            'cache_begin': np.zeros((batch_size,), dtype=np.int32),
            'cache_end': np.zeros((batch_size,), dtype=np.int32),
            'mask': np.zeros((batch_size, 1, max_decode_len*2), dtype=np.int32),
            'pre_logits': np.zeros((batch_size, max_decode_len*2), dtype=np.float32),
            'pre_logits_mask': np.zeros((batch_size, max_decode_len*2), dtype=np.float32),
            'full_position_ids': np.zeros((batch_size, max_decode_len*2), dtype=np.float32),
            'model': {
              'layers': {
                l: {
                  'self_attn': {
                    'cached_key': np.zeros((batch_size, max_decode_len*2, 1024), dtype=np.float32),
                    'cached_value': np.zeros((batch_size, max_decode_len*2, 1024), dtype=np.float32),
                    'cache_index': np.zeros((batch_size,), dtype=np.int32)
                  },
                }
                for l in train_state['params']['llm']['model']['layers']
              },
            }
          }
        }

    def make_shard_and_gather_fns(self, pytree, sharding_rule):
        """
        Create pytree of sharding and gathering functions from sharding rule
        or a pytree of PartitionSpecs. This can be used to shard and gather
        a pytree of tensors.

        Args:
            pytree: The pytree to be sharded and gathered.
            sharding_rule: The sharding rule or partition specs for the pytree.

        Returns:
            A pair of pytrees of sharding and gathering functions, each with the
            same structure as the input pytree.
        """
        named_shardings = self.match_sharding_rule(sharding_rule, pytree)
        def make_shard_fn(partition_spec):
            jax_shard_function = jax.jit(
                lambda x: x,
                in_shardings=None,
                out_shardings=partition_spec
            )
            def shard_fn(tensor):
                return jax_shard_function(tensor).block_until_ready()
            return shard_fn

        def make_gather_fn(partition_spec):
            jax_gather_fn = jax.jit(
                lambda x: x,
                in_shardings=partition_spec,
                out_shardings=jax.sharding.NamedSharding(self.mesh, PartitionSpec()),
            )
            def gather_fn(tensor):
                return jax.device_get(jax_gather_fn(tensor))
            return gather_fn

        shard_fns = jax.tree_util.tree_map(make_shard_fn, named_shardings)
        gather_fns = jax.tree_util.tree_map(make_gather_fn, named_shardings)
        return shard_fns, gather_fns

    def match_sharding_rule(self, sharding_rules, pytree):
        """ Apply sharding rules to a pytree to get a pytree of PartitionSpecs.

        Args:
            sharding_rules: The sharding rules or partition specs for the pytree.
            pytree: The pytree to be sharded.

        Returns:
            A pytree of PartitionSpecs with the same structure as the input pytree.
        """
        def get_partition_spec(rule, pytree):
            if isinstance(rule, ShardingRule):
                return jax.tree_util.tree_map(
                    lambda x: jax.sharding.NamedSharding(self.mesh, x),
                    rule.apply(pytree)
                )
            else:
                return jax.tree_util.tree_map(
                    lambda x: jax.sharding.NamedSharding(self.mesh, rule),
                    pytree
                )
        def is_leaf(x):
            # Check if the node is None, a PartitionSpec or a ShardingRule
            return (
                x is None
                or isinstance(x, ShardingRule)
                or isinstance(x, PS)
            )

        return jax.tree_util.tree_map(
            get_partition_spec, sharding_rules, pytree, is_leaf=is_leaf
        )
