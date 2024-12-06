from typing import Tuple
import torch


def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor):
    """
    Helper function to reshape frequency tensor to have the same shape as the target tensor 'x'
    for the purpose of broadcasting the frequency tensor during element-wise operations.

    Args:
        freqs_cis (torch.Tensor): Frequency tensor to be reshaped.
        x (torch.Tensor): Target tensor for broadcasting compatibility.

    Returns:
        torch.Tensor: Reshaped frequency tensor.

    Raises:
        AssertionError: If the frequency tensor doesn't match the expected shape.
        AssertionError: If the target tensor 'x' doesn't have the expected number of dimensions.
    """
    ndim = x.ndim
    assert 0 <= 1 < ndim
    assert freqs_cis.shape == (x.shape[1], x.shape[-1])
    shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
    return freqs_cis.view(shape)

def apply_rotary_emb(
    query: torch.Tensor,
    key: torch.Tensor,
    head_dim: int,
    max_seq_len: int,
    theta: float = 10000.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Apply rotary embeddings to input tensors using the given frequency tensor.

    This function applies rotary embeddings to the given query and key tensors. The rotation to each token
    embedding is a function of that token's position in the sequence, head_dim, and theta.
    The input tensors are reshaped as complex numbers to simplify your implementation.

    Args:
        query (torch.Tensor): Query tensor to apply rotary embeddings.
                              Shape: (batch_size, seqlen, n_local_heads, self.head_dim)
        key (torch.Tensor): Key tensor to apply rotary embeddings.
                              Shape: (batch_size, seqlen, n_local_kv_heads, self.head_dim)
        head_dim (int): Dimension of each attention head.
        max_seq_len (int): Maximum sequence length supported by model.
    Returns:
        Tuple[torch.Tensor, torch.Tensor]: Tuple of modified query tensor and key tensor with rotary embeddings.
    """

    _, seqlen, _, _ = query.shape
    device = query.device

    query_real, query_imag = query.float().reshape(query.shape[:-1] + (-1, 2)).unbind(-1)
    key_real, key_imag = key.float().reshape(key.shape[:-1] + (-1, 2)).unbind(-1)
    # todo
    # Please refer to slide 22 in https://phontron.com/class/anlp2024/assets/slides/anlp-05-transformers.pdf
    # and Section 3 in https://arxiv.org/abs/2104.09864.

    # reshape xq and xk to match the complex representation
    #print("query", query)
    #print("query_real", query_real)
    #print("query_imag", query_imag)

    #print("key", key)
    #print("key_real", key_real)
    #print("key_imag", key_imag)
    
    # This separates each query/key vector into its odd and even indices (assuming *one-indexing*).
    # query_real contains q_1, q_3, q_5, ... and query_imag contains q_2, q_4, q_6, ...

    # First, compute the trigonometric values in the second and fourth columns in
    # slide 22 (linked above).
    
    # Create a frequency tensor
    freqs = torch.arange(0, head_dim // 2, device=device, dtype=torch.float32)
    freqs = (theta ** (-2 * freqs / head_dim)).unsqueeze(dim=0)  # shape: (1, head_dim // 2)
    #print("freqs", freqs)

    # Create a position tensor
    positions = torch.arange(0, max_seq_len, device=device, dtype=torch.float32).unsqueeze(dim=1)  # shape: (max_seq_len, 1)
    #print("positions", positions)

    # Compute the argument for the sin and cos functions
    angles = positions * freqs  # shape: (max_seq_len, head_dim // 2)
    #print("angles", angles)

    # Compute the sin and cos values
    sin_values = torch.sin(angles)
    cos_values = torch.cos(angles)
    #print("sin_values", sin_values)
    #print("cos_values", cos_values)

    # Combine the sin and cos values into a complex number tensor
    freqs_cis = torch.stack((cos_values, sin_values), dim=-1)  # shape: (max_seq_len, head_dim // 2, 2)
    #print("before1", freqs_cis)
    freqs_cis = freqs_cis[:seqlen, :head_dim//2]  # シーケンス長とヘッド次元に合わせて切り取る
    #print("before2", freqs_cis)
    freqs_cis = freqs_cis.view(seqlen, head_dim)  # (seqlen, head_dim) の形状に変形
    #print("freqs_cis_before_reshape", freqs_cis)

    # Reshape the frequency tensor for broadcasting
    freqs_cis = reshape_for_broadcast(freqs_cis, query)
    #print("freqs_cis_after_reshape", freqs_cis)

    # Compute the rotary position embeddings for the query tensor
    #query_rot = (query_real * freqs_cis[..., 0] - query_imag * freqs_cis[..., 1]).type_as(query)
    query_rot = (query_real[0][0] * freqs_cis[..., 0] - query_imag[0][0] * freqs_cis[..., 1]).type_as(query)
    query_rot = torch.stack((query_rot, query_real[0][1] * freqs_cis[..., 2] - query_imag[0][1] * freqs_cis[..., 3]), dim=-1)
    #print("freqs_cis[..., 0]", freqs_cis[..., 0])
    #print("freqs_cis[..., 1]", freqs_cis[..., 1])
    #print("query_rot_1", query_rot)
    query_rot2 = (query_imag[0][0] * freqs_cis[..., 0] + query_real[0][0] * freqs_cis[..., 1]).type_as(query)
    query_rot2 = torch.stack((query_rot2, query_imag[0][1] * freqs_cis[..., 2] + query_real[0][1] * freqs_cis[..., 3]), dim=-1)
    #print("query_rot_2", query_rot2)
    query_rot = torch.stack((query_rot, query_rot2), dim=-1)
    query_out = query_rot.view(*query.shape[:-1], -1)

    # Compute the rotary position embeddings for the key tensor
    key_rot = (key_real[0][0] * freqs_cis[..., 0] - key_imag[0][0] * freqs_cis[..., 1]).type_as(key)
    key_rot = torch.stack((key_rot, key_real[0][1] * freqs_cis[..., 2] - key_imag[0][1] * freqs_cis[..., 3]), dim=-1)
    #print("freqs_cis[..., 0]", freqs_cis[..., 0])
    #print("freqs_cis[..., 1]", freqs_cis[..., 1])
    #print("key_rot_1", key_rot)
    key_rot2 = (key_imag[0][0] * freqs_cis[..., 0] + key_real[0][0] * freqs_cis[..., 1]).type_as(key)
    key_rot2 = torch.stack((key_rot2, key_imag[0][1] * freqs_cis[..., 2] + key_real[0][1] * freqs_cis[..., 3]), dim=-1)
    #print("key_rot_2", key_rot2)
    key_rot = torch.stack((key_rot, key_rot2), dim=-1)
    key_out = key_rot.view(*key.shape[:-1], -1)

    # Return the rotary position embeddings for the query and key tensors
    return query_out, key_out

    