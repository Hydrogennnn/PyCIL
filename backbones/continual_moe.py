
import torch
import torch.nn as nn
from torch.distributions import Bernoulli, Dirichlet, Normal, kl_divergence
import math


EPSILON = 1e-8



class Adapter(nn.Module):
    def __init__(self,
                 d_model=None,
                 bottleneck=None,
                 dropout=0.0,
                 init_option="lora",
                 adapter_scalar="1.0",
                 adapter_layernorm_option="in"):
        super().__init__()
        self.n_embd = d_model if d_model is None else d_model
        self.down_size = bottleneck

        #_before
        self.adapter_layernorm_option = adapter_layernorm_option

        self.adapter_layer_norm_before = None
        if adapter_layernorm_option == "in" or adapter_layernorm_option == "out":
            self.adapter_layer_norm_before = nn.LayerNorm(self.n_embd)

        if adapter_scalar == "learnable_scalar":
            self.scale = nn.Parameter(torch.ones(1))
        else:
            self.scale = float(adapter_scalar)

        self.down_proj = nn.Linear(self.n_embd, self.down_size)
        self.non_linear_func = nn.ReLU()
        self.up_proj = nn.Linear(self.down_size, self.n_embd)

        self.dropout = dropout
        if init_option == "bert":
            raise NotImplementedError
        elif init_option == "lora":
            pass
            # with torch.no_grad():
            #     nn.init.kaiming_uniform_(self.down_proj.weight, a=math.sqrt(5))
            #     nn.init.zeros_(self.up_proj.weight)
            #     nn.init.zeros_(self.down_proj.bias)
            #     nn.init.zeros_(self.up_proj.bias)

    def forward(self, x, add_residual=True, residual=None):

        residual = x if residual is None else residual
        if self.adapter_layernorm_option == 'in': #  none
            x = self.adapter_layer_norm_before(x)

        down = self.down_proj(x)
        down = self.non_linear_func(down)
        # down = nn.functional.dropout(down, p=self.dropout, training=self.training)
        up = self.up_proj(down)
        up = nn.functional.dropout(up, p=self.dropout, training=self.training)
        up = up * self.scale

        if self.adapter_layernorm_option == 'out': #  none
            up = self.adapter_layer_norm_before(up)

        if add_residual:
            output = up + residual
        else:
            output = up
        return output
    
    
class SparseDispatcher(object):
    """Helper for implementing a mixture of experts.
    The purpose of this class is to create input minibatches for the
    experts and to combine the results of the experts to form a unified
    output tensor.
    There are two functions:
    dispatch - take an input Tensor and create input Tensors for each expert.
    combine - take output Tensors from each expert and form a combined output
      Tensor.  Outputs from different experts for the same batch element are
      summed together, weighted by the provided "gates".
    The class is initialized with a "gates" Tensor, which specifies which
    batch elements go to which experts, and the weights to use when combining
    the outputs.  Batch element b is sent to expert e iff gates[b, e] != 0.
    The inputs and outputs are all two-dimensional [batch, depth].
    Caller is responsible for collapsing additional dimensions prior to
    calling this class and reshaping the output to the original shape.
    See common_layers.reshape_like().
    Example use:
    gates: a float32 `Tensor` with shape `[batch_size, num_experts]`
    inputs: a float32 `Tensor` with shape `[batch_size, input_size]`
    experts: a list of length `num_experts` containing sub-networks.
    dispatcher = SparseDispatcher(num_experts, gates)
    expert_inputs = dispatcher.dispatch(inputs)
    expert_outputs = [experts[i](expert_inputs[i]) for i in range(num_experts)]
    outputs = dispatcher.combine(expert_outputs)
    The preceding code sets the output for a particular example b to:
    output[b] = Sum_i(gates[b, i] * experts[i](inputs[b]))
    This class takes advantage of sparsity in the gate matrix by including in the
    `Tensor`s for expert i only the batch elements for which `gates[b, i] > 0`.
    """

    def __init__(self, num_experts, gates):
        """Create a SparseDispatcher."""

        self._gates = gates # [B, num_experts]
        self._num_experts = num_experts

        sorted_experts, index_sorted_experts = torch.nonzero(gates).sort(0)
        # drop indices
        _, self._expert_index = sorted_experts.split(1, dim=1) # expert_idx: [N, 1]
        # get according batch index for each expert
        self._batch_index = torch.nonzero(gates)[index_sorted_experts[:, 1], 0] #[N,]
        # calculate num samples that each expert gets
        self._part_sizes = (gates > 0).sum(0).tolist() # [num_experts,]
        # expand gates to match with self._batch_index
        gates_exp = gates[self._batch_index.flatten()] # [N, num_experts]
        self._nonzero_gates = torch.gather(gates_exp, 1, self._expert_index) #[N,1]

    def dispatch(self, inp):
        """Create one input Tensor for each expert.
        The `Tensor` for a expert `i` contains the slices of `inp` corresponding
        to the batch elements `b` where `gates[b, i] > 0`.
        Args:
          inp: a `Tensor` of shape "[batch_size, <extra_input_dims>]`
        Returns:
          a list of `num_experts` `Tensor`s with shapes
            `[expert_batch_size_i, <extra_input_dims>]`.
        """

        # assigns samples to experts whose gate is nonzero

        inp_exp = inp[self._batch_index].squeeze(1)
        return torch.split(inp_exp, self._part_sizes, dim=0)

    def combine(self, expert_out, multiply_by_gates=True):
        """Sum together the expert output, weighted by the gates.
        The slice corresponding to a particular batch element `b` is computed
        as the sum over all experts `i` of the expert output, weighted by the
        corresponding gate values.  If `multiply_by_gates` is set to False, the
        gate values are ignored.
        Args:
          expert_out: a list of `num_experts` `Tensor`s, each with shape
            `[expert_batch_size_i, <extra_output_dims>]`.
          multiply_by_gates: a boolean
        Returns:
          a `Tensor` with shape `[batch_size, <extra_output_dims>]`.
        """
        # apply exp to expert outputs, so we are not longer in log space
        
        stitched = torch.cat(expert_out, 0) # [b, d]
        if multiply_by_gates:
            stitched = stitched.mul(self._nonzero_gates)  # 加权

        zeros = torch.zeros(self._gates.size(0), expert_out[-1].size(1), device=stitched.device)
        # combine samples that have been processed by the same k experts

        combined = zeros.index_add(0, self._batch_index, stitched.float())
        # add eps to all zero values in order to avoid nans when going back to log space
        # back to log space
        return combined

    def expert_to_gates(self):
        """Gate values corresponding to the examples in the per-expert `Tensor`s.
        Returns:
          a list of `num_experts` one-dimensional `Tensor`s with type `tf.float32`
              and shapes `[expert_batch_size_i]`
        """
        # split nonzero gates for each expert
        return torch.split(self._nonzero_gates, self._part_sizes, dim=0)

class Continual_MoE(nn.Module):
    def __init__(
        self,
        experts_num=16,
        top_k=2,
        d_model=768,
        ffn_num=64,
        dirichlet_min_alpha=1e-3,
        gumbel_tau=1.0,
    ) -> None:
        super().__init__()
        self.experts_num = experts_num
        self.register_buffer("mean", torch.tensor([0.0]))
        self.register_buffer("std", torch.tensor([1.0]))
        self.top_k = top_k
        self.d_model = d_model
        self.ffn_num = ffn_num
        self.dirichlet_min_alpha = dirichlet_min_alpha
        self.gumbel_tau = gumbel_tau
        # select_router models Bernoulli expert activation; alpha_router models
        # the Dirichlet concentration used to mix activated experts.
        self.select_router = nn.Linear(self.d_model, self.experts_num)
        self.alpha_router = nn.Linear(self.d_model, self.experts_num)
        self.adaptmlp_list = nn.ModuleList()
        self.noisy_gating = True
        self.softmax = nn.Softmax(1)
        self.softplus = nn.Softplus()
        self.last_alpha = None
        self.last_select_probs = None
        self.last_gates = None
        for _ in range(self.experts_num):  #
            self.adaptmlp = Adapter(d_model=self.d_model, dropout=0.2, bottleneck=self.ffn_num,
                                    init_option='lora',
                                    adapter_scalar=0.1,
                                    adapter_layernorm_option='out',
                                    )
            self.adaptmlp_list.append(self.adaptmlp)
        
    
    def get_gating(self, x):
        # Keep the old public API for visualization/KD callers. These are
        # Bernoulli logits, not final normalized MoE gates.
        return self.route_distribution(x)["select_logits"]

    def get_route_params(self, x):
        # Returned parameters define the full routing posterior q(z, w | x):
        # Bernoulli expert selection probabilities and Dirichlet mix weights.
        route = self.route_distribution(x)
        return route["alpha"], route["select_probs"]
    
    def get_load(self, x):
        gates = self.dirichlet_routing(x)
        return (gates > 0).float()
    
    def forward(self, x, return_load=False):
        gates = self.dirichlet_routing(x)
        load = (gates > 0).float()
        dispatcher = SparseDispatcher(self.experts_num, gates)
        expert_inputs = dispatcher.dispatch(x)  # list of [n_i, d_model]，n_i 为分配到第i个专家的样本数
        expert_outputs = [self.adaptmlp_list[i](expert_inputs[i].to(x), add_residual=True)
                              for i in range(self.experts_num)]
        
            # 过滤空专家（没有分配到样本的专家）
        expert_outputs = [
            out for out in expert_outputs if out.shape[0] > 0
        ]
        
            # 按 gates 权重合并回完整 batch
        outputs = dispatcher.combine(expert_outputs)
        # y: [batch_size, d]
        if return_load:
            return outputs, load
        else:
            return outputs

    def route_distribution(self, x):
        # alpha must stay strictly positive for a valid Dirichlet distribution.
        select_logits = self.select_router(x)
        select_probs = torch.sigmoid(select_logits)
        alpha = self.softplus(self.alpha_router(x)) + self.dirichlet_min_alpha
        return {
            "alpha": alpha,
            "select_logits": select_logits,
            "select_probs": select_probs,
        }

    def _relaxed_bernoulli_sample(self, logits):
        # Differentiable Bernoulli relaxation via logistic/Gumbel noise.
        eps = torch.finfo(logits.dtype).eps
        uniform = torch.rand_like(logits).clamp_(eps, 1.0 - eps)
        logistic_noise = torch.log(uniform) - torch.log1p(-uniform)
        return torch.sigmoid((logits + logistic_noise) / self.gumbel_tau)

    def _straight_through_top_k(self, scores):
        # Sparse forward pass with dense gradients: the dispatcher sees exact
        # zeros outside top-k, while gradients still flow through relaxed scores.
        top_indices = torch.topk(scores, min(self.top_k, self.experts_num), dim=1).indices
        hard_selection = torch.zeros_like(scores).scatter(1, top_indices, 1.0)
        return hard_selection + scores - scores.detach()

    def dirichlet_routing(self, x):
        route = self.route_distribution(x)
        alpha = route["alpha"]
        select_probs = route["select_probs"]
        if self.training:
            # Sample a relaxed Bernoulli score, then use a straight-through top-k
            # mask. This keeps the actual MoE computation sparse and avoids
            # sending every token to every expert.
            relaxed_selection = self._relaxed_bernoulli_sample(route["select_logits"])
            selection = self._straight_through_top_k(relaxed_selection)
            weights = Dirichlet(alpha).rsample()
        else:
            # At inference, use deterministic top-k selection and the Dirichlet
            # mean to avoid sampling variance.
            top_indices = torch.topk(select_probs, min(self.top_k, self.experts_num), dim=1).indices
            selection = torch.zeros_like(select_probs).scatter(1, top_indices, 1.0)
            weights = alpha / alpha.sum(dim=1, keepdim=True).clamp_min(EPSILON)

        # Final gates combine "which experts are active" with "how much each
        # active expert contributes", then renormalize per token.
        gates = selection * weights
        gates = gates / gates.sum(dim=1, keepdim=True).clamp_min(EPSILON)
        self.last_alpha = alpha
        self.last_select_probs = select_probs
        self.last_gates = gates
        return gates

    def dirichlet_prior_loss(self, x, prior_alpha=None, prior_select_probs=None):
        # KL(q_current || q_prior). In continual learning, q_prior is usually
        # the previous task model's routing posterior for old-class samples.
        route = self.route_distribution(x)
        alpha = route["alpha"]
        select_probs = route["select_probs"]
        if prior_alpha is None:
            prior_alpha = torch.ones_like(alpha)
        prior_alpha = prior_alpha.clamp_min(self.dirichlet_min_alpha)
        alpha = alpha.clamp_min(self.dirichlet_min_alpha)
        # Match the Dirichlet expert-weight distribution.
        dir_kl = kl_divergence(Dirichlet(alpha), Dirichlet(prior_alpha)).mean()

        if prior_select_probs is None:
            target_active = min(self.top_k, self.experts_num)
            prior_select_probs = torch.full_like(select_probs, target_active / self.experts_num)
        prior_select_probs = prior_select_probs.clamp(1e-5, 1.0 - 1e-5)
        select_probs = select_probs.clamp(1e-5, 1.0 - 1e-5)
        # Match the Bernoulli expert-activation distribution.
        bern_kl = kl_divergence(Bernoulli(probs=select_probs), Bernoulli(probs=prior_select_probs)).mean()
        return dir_kl + bern_kl, {
            "dirichlet_kl": dir_kl,
            "bernoulli_kl": bern_kl,
        }

    def sparsity_loss(self, x):
        # Keep the expected number of activated experts near top_k. This
        # prevents the relaxed Bernoulli path from activating every expert.
        select_probs = self.route_distribution(x)["select_probs"]
        target_active = min(self.top_k, self.experts_num)
        expected_active = select_probs.sum(dim=1)
        return (expected_active - target_active).pow(2).mean()

    
    def noisy_top_k_gating(self, x, train, w_gate, w_noise, noise_epsilon=1e-2, return_logits=False):
        """Noisy top-k gating.
          See paper: https://arxiv.org/abs/1701.06538.
          Args:
            x: input Tensor with shape [batch_size, input_size]
            train: a boolean - we only add noise at training time.
            noise_epsilon: a float
          Returns:
            gates: a Tensor with shape [batch_size, num_experts]
            load: a Tensor with shape [num_experts]
        """

        clean_logits = x @ w_gate.to(x)
        if self.noisy_gating and train:
            raw_noise_stddev = x @ w_noise.to(x)
            noise_stddev = ((self.softplus(raw_noise_stddev) + noise_epsilon))
            noisy_logits = clean_logits + (torch.randn_like(clean_logits) * noise_stddev)
            logits = noisy_logits
        else:
            logits = clean_logits
        
        if return_logits:
            return logits
        # calculate topk + 1 that will be needed for the noisy gates
        top_logits, top_indices = logits.topk(min(self.top_k + 1, self.experts_num), dim=1)
        top_k_logits = top_logits[:, :self.top_k]
        top_k_indices = top_indices[:, :self.top_k]
        top_k_gates = self.softmax(top_k_logits)
        zeros = torch.zeros_like(logits)
        gates = zeros.scatter(1, top_k_indices, top_k_gates)
        if self.noisy_gating and self.top_k < self.experts_num and train:  # 目前未用上
            load = (self._prob_in_top_k(clean_logits, noisy_logits, noise_stddev, top_logits)).sum(0)
        else:
            load = self._gates_to_load(gates)
        return gates, load

    def _gates_to_load(self, gates):
        """Compute the true load per expert, given the gates.
        The load is the number of examples for which the corresponding gate is >0.
        Args:
        gates: a `Tensor` of shape [batch_size, n]
        Returns:
        a float32 `Tensor` of shape [n]
        """
        return (gates > 0).float()
    
    
    def _prob_in_top_k(self, clean_values, noisy_values, noise_stddev, noisy_top_values):
        """Helper function to NoisyTopKGating.
        Computes the probability that value is in top k, given different random noise.
        This gives us a way of backpropagating from a loss that balances the number
        of times each expert is in the top k experts per example.
        In the case of no noise, pass in None for noise_stddev, and the result will
        not be differentiable.
        Args:
        clean_values: a `Tensor` of shape [batch, n].
        noisy_values: a `Tensor` of shape [batch, n].  Equal to clean values plus
          normally distributed noise with standard deviation noise_stddev.
        noise_stddev: a `Tensor` of shape [batch, n], or None
        noisy_top_values: a `Tensor` of shape [batch, m].
           "values" Output of tf.top_k(noisy_top_values, m).  m >= k+1
        Returns:
        a `Tensor` of shape [batch, n].
        """
        # print('1231',clean_values)  # 全nan
        batch = clean_values.size(0)
        m = noisy_top_values.size(1)
        top_values_flat = noisy_top_values.flatten()

        threshold_positions_if_in = torch.arange(batch, device=clean_values.device) * m + self.top_k
        threshold_if_in = torch.unsqueeze(torch.gather(top_values_flat, 0, threshold_positions_if_in), 1)
        is_in = torch.gt(noisy_values, threshold_if_in)
        threshold_positions_if_out = threshold_positions_if_in - 1
        threshold_if_out = torch.unsqueeze(torch.gather(top_values_flat, 0, threshold_positions_if_out), 1)
        # is each value currently in the top k.
        normal = Normal(self.mean, self.std)
        #

        prob_if_in = normal.cdf((clean_values - threshold_if_in)/noise_stddev)
        prob_if_out = normal.cdf((clean_values - threshold_if_out)/noise_stddev)
        prob = torch.where(is_in, prob_if_in, prob_if_out)
        return prob
        
