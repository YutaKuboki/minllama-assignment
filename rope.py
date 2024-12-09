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

    #実数部と虚数部に分解
    query_real, query_imag = query.float().reshape(query.shape[:-1] + (-1, 2)).unbind(-1)
    key_real, key_imag = key.float().reshape(key.shape[:-1] + (-1, 2)).unbind(-1)
    # This separates each query/key vector into its odd and even indices (assuming *one-indexing*).
    # query_real contains q_1, q_3, q_5, ... and query_imag contains q_2, q_4, q_6, ...

    # todo
    # Please refer to slide 22 in https://phontron.com/class/anlp2024/assets/slides/anlp-05-transformers.pdf
    # and Section 3 in https://arxiv.org/abs/2104.09864.

    # reshape xq and xk to match the complex representation

    # First, compute the trigonometric values in the second and fourth columns in
    # slide 22 (linked above).
    
    # freqs : frequency tensor ： theta 0 から theta (head_dim // 2) まで格納
    freqs = torch.arange(0, head_dim // 2, device=device, dtype=torch.float32)
    freqs = (theta ** (-2 * freqs / head_dim)).unsqueeze(dim=0)  # shape: (1, head_dim // 2)

    # positions : position tensor : 0 から max_seq_len - 1　までの数字を格納
    positions = torch.arange(0, seqlen, device=device, dtype=torch.float32).unsqueeze(dim=1)  # shape: (seq_len, 1)

    # angles : [[0 * theta 0, 0 * theta 1, ...], [1 * theta 0, 1 * theta 1, ...], ..., [(max_seq_len - 1) * theta 0, (max_seq_len - 1) * theta 1, ...]]
    angles = positions * freqs  # shape: (seq_len, head_dim // 2)

    # angles の全ての値を sin, cos に変換
    sin_values = torch.sin(angles)
    cos_values = torch.cos(angles)
    
    ###ここまでは合っていそう###

    '''
    freqs_cis = torch.stack((cos_values, sin_values), dim=-1)  # shape: (max_seq_len, head_dim // 2, 2)
    freqs_cis = freqs_cis[:seqlen, :head_dim//2]  # シーケンス長とヘッド次元に合わせて切り取る
    freqs_cis = freqs_cis.view(seqlen, head_dim)  # (seqlen, head_dim) の形状に変形
    # この時点で freqs_cis = [[cos(0 * theta 0), sin(0 * theta 0), cos(0 * theta 1), sin(0 * theta 1),...], ..., [cos((max_seq_len - 1) * theta0), ...]]

    # Reshape the frequency tensor for broadcasting
    freqs_cis = reshape_for_broadcast(freqs_cis, query)
    

    # Compute the rotary position embeddings for the query tensor
    #query_rot = (query_real * freqs_cis[..., 0] - query_imag * freqs_cis[..., 1]).type_as(query)
    query_rot = (query_real[0][0] * freqs_cis[..., 0] - query_imag[0][0] * freqs_cis[..., 1]).type_as(query)
    # slide の数式の一行目

    query_rot = torch.stack((query_rot, query_real[0][1] * freqs_cis[..., 2] - query_imag[0][1] * freqs_cis[..., 3]), dim=-1)
    # slide の数式の三行目を加える

    query_rot2 = (query_imag[0][0] * freqs_cis[..., 0] + query_real[0][0] * freqs_cis[..., 1]).type_as(query)
    # slide の数式の二行目

    query_rot2 = torch.stack((query_rot2, query_imag[0][1] * freqs_cis[..., 2] + query_real[0][1] * freqs_cis[..., 3]), dim=-1)
    # slide の数式の四行目を加える

    query_rot = torch.stack((query_rot, query_rot2), dim=-1)
    # 重ね合わせる

    query_out = query_rot.view(*query.shape[:-1], -1)

    # Compute the rotary position embeddings for the key tensor
    key_rot = (key_real[0][0] * freqs_cis[..., 0] - key_imag[0][0] * freqs_cis[..., 1]).type_as(key)
    key_rot = torch.stack((key_rot, key_real[0][1] * freqs_cis[..., 2] - key_imag[0][1] * freqs_cis[..., 3]), dim=-1)
    key_rot2 = (key_imag[0][0] * freqs_cis[..., 0] + key_real[0][0] * freqs_cis[..., 1]).type_as(key)
    key_rot2 = torch.stack((key_rot2, key_imag[0][1] * freqs_cis[..., 2] + key_real[0][1] * freqs_cis[..., 3]), dim=-1)
    key_rot = torch.stack((key_rot, key_rot2), dim=-1)
    key_out = key_rot.view(*key.shape[:-1], -1)
    '''
    
    #一緒にするとよく分からなくなってしまうので別々にしておく
    #queryのshape: (batch_size, seqlen, n_local_heads, self.head_dim)
    #keyのshape: (batch_size, seqlen, n_local_kv_heads, self.head_dim)
    #よってfreqs_???: (seqlen, head_dim/2) -> (1, seqlen, 1, head_dim/2) #1にしておけば次元が一致するまで繰り返される
    freqs_cos = cos_values.view(1, seqlen, 1, cos_values.shape[-1])
    freqs_sin = sin_values.view(1, seqlen, 1, sin_values.shape[-1])
    
    #回転行列を実部と虚部からなる列ベクトルにかけて回転させた後の実部と虚部を求める
    #おそらく下みたいな計算をたくさんする???はず
    #(cos -sin       (real
    #            x   
    # sin  cos)       imag) 
    
    query_out_real = query_real * freqs_cos - query_imag * freqs_sin
    # slide の数式の奇数行

    query_out_imag = query_real * freqs_sin + query_imag * freqs_cos
    # slide の数式の偶数行

    key_out_real = key_real * freqs_cos - key_imag * freqs_sin
    key_out_imag = key_real * freqs_sin + key_imag * freqs_cos
    
    query_out = torch.stack([query_out_real, query_out_imag], dim=-1)
    # 奇数行と偶数行を重ね合わせる

    query_out = query_out.view(*query.shape[:-1], -1)
    
    key_out = torch.stack([key_out_real, key_out_imag], dim=-1)
    key_out = key_out.view(*key.shape[:-1], -1)

    # Return the rotary position embeddings for the query and key tensors
    return query_out, key_out