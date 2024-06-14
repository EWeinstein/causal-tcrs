import pandas as pd
import numpy as np
import pytest
import torch

from CausalReceptors.layers import one_hot, KmerEmbed, SeqEmbed


def test_one_hot():
    data = torch.zeros(2, 3, 4, dtype=torch.long)
    data[0, 0] = torch.tensor([1, 2, -1, -1])
    data[1, 2] = torch.tensor([2, 1, 0, -1])

    torch.set_default_dtype(torch.float32)

    data_onehot = one_hot(data, 3)
    check_data_onehot = torch.zeros((2, 3, 4, 3))
    check_data_onehot[0, 0, 0, 1] = 1
    check_data_onehot[0, 0, 1, 2] = 1
    check_data_onehot[1, 2, 0, 2] = 1
    check_data_onehot[1, 2, 1, 1] = 1
    check_data_onehot[1, 2, 2, 0] = 1
    check_data_onehot[0, 1:, :, 0] = 1
    check_data_onehot[1, :2, :, 0] = 1

    assert torch.allclose(data_onehot, check_data_onehot)


@pytest.mark.parametrize("customize", [False, True])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("cuda", [False, True])
def test_KmerFeatures(customize, dtype, cuda):
    if cuda:
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    N = 4
    M = 2
    seq_length = 5
    alphabet = 'AB*'
    kmer_length = 3
    if customize:
        kmers = torch.zeros(list(map(int, (len(alphabet)-1) * np.ones(kmer_length))) + [kmer_length],
                            dtype=torch.long, device=device)
        for j in range(kmer_length):
            x = torch.arange(len(alphabet)-1, device=device)
            for jp in range(kmer_length):
                if j != jp:
                    x = x.unsqueeze(jp)
            kmers[..., j] += x
        custom_kmers = kmers.reshape([-1, kmer_length])[:-1]
        nkmers = (len(alphabet)-1) ** kmer_length - 1
    else:
        custom_kmers = None
        nkmers = (len(alphabet)-1) ** kmer_length
    kmerfeaturizer = KmerEmbed(seq_length, alphabet, kmer_length, custom_kmers=custom_kmers, dtype=dtype, cuda=cuda)

    seqs = torch.zeros((N, M, seq_length), dtype=torch.long)
    # Kmers: AAB, ABA, BAA
    seqs[0, 0, 2] = 1
    # Kmers: ABA, BAA, AAA
    seqs[0, 1, 1] = 1
    # Kmers: AAA
    seqs[1, 0, 3] = 2
    seqs[1, 0, 4] = -1
    # One-hot encode.
    rep = one_hot(seqs, len(alphabet)).to(dtype=dtype, device=device)

    # Kmer encode.
    tst_kmers = kmerfeaturizer(rep)

    # Check:                       AAA AAB ABA ABB BAA BAB BBA BBB
    chk_kmer00 = torch.tensor([  0,  1,  1,  0,  1,  0,  0,  0], dtype=dtype, device=device)
    assert torch.allclose(tst_kmers[0, 0], chk_kmer00[:nkmers])
    # Check:                       AAA AAB ABA ABB BAA BAB BBA BBB
    chk_kmer01 = torch.tensor([  1,  0,  1,  0,  1,  0,  0,  0], dtype=dtype, device=device)
    assert torch.allclose(tst_kmers[0, 1], chk_kmer01[:nkmers])
    # Check:                       AAA AAB ABA ABB BAA BAB BBA BBB
    chk_kmer10 = torch.tensor([  1,  0,  0,  0,  0,  0,  0,  0], dtype=dtype, device=device)
    assert torch.allclose(tst_kmers[1, 0], chk_kmer10[:nkmers])
    # Check:                       AAA AAB ABA ABB BAA BAB BBA BBB
    chk_kmer11 = torch.tensor([  3,  0,  0,  0,  0,  0,  0,  0], dtype=dtype, device=device)
    assert torch.allclose(tst_kmers[1, 1], chk_kmer11[:nkmers])
