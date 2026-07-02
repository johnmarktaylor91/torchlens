# FAITHFUL PORT of trangptm/DeepCare @ master (original framework: Theano/Lasagne-era raw Theano)
# https://raw.githubusercontent.com/trangptm/DeepCare/master/code/lstm_layer.py
# https://raw.githubusercontent.com/trangptm/DeepCare/master/code/readm.py
#
# Pham, Tran, Phung, Venkatesh 2016/2017 "Predicting healthcare trajectories from medical
# records: A deep learning approach" (J. Biomed. Inform.) / "DeepCare: A Deep Dynamic Memory
# Model for Predictive Medicine". Original repo is pure Theano (`theano.scan`, `theano.tensor`)
# with no PyTorch anywhere -- transcribed faithfully here (no usable code to vendor/run/install).
#
# The real architecture (readm.py:build_model + lstm_layer.py:lstm_layer), transcribed
# mechanism-for-mechanism:
#   1. Two admission "bag-of-words" embedding lookups (diagnosis codes `emb_dia`, medication/
#      procedure codes `emb_pm`), each an embedding-table lookup averaged (mean-pooled) over the
#      words of every admission via a mask (`options['embed']=='mean'` branch of build_model).
#   2. A custom time-aware LSTM cell (`lstm_layer.lstm_layer`'s `_step` function): standard
#      input/output/cell gates PLUS two admission-specific modulations copied verbatim from the
#      real math:
#        - the input gate `i` is divided by `method_` (admission-type multiplier, 1.0=emergency
#          2.0=other in the real data prep) -- "irregular time" input decay;
#        - the forget gate `f` gets an extra `pre_f = pm_masked @ W_Pf` term coming from the
#          previous step's procedure/medication embedding gated by the admission mask
#          (`pm_ = m_[1,:,None] * xf_`), i.e. procedure/medication history modulates forgetting;
#        - the output gate `o` gets an extra `pre_o = pm @ W_Po` term from the CURRENT step's
#          admission mask-gated pm embedding;
#        - both f and o also receive a decayed-time embedding `pre_t = time_features @ W_Z`
#          where `time_features = [t/60, (t/180)^2, (t/365)^3]` -- the multi-scale time decay
#          that gives DeepCare its "irregular time" property.
#      All four gate pre-activations, the recurrent term `h_ @ W_lstm_U`, and the input term
#      `x_ @ W_lstm_W` (note: real code re-uses variable name `x_` for the *diagnosis* embedding
#      fed into this step, i.e. `emb_dia`) are exactly as in `_step`.
#   3. Multi-scale weighted temporal pooling over the LSTM hidden-state sequence: three
#      different weighted averages of the per-timestep hidden states (`hidd_0`/`hidd_1`/
#      `hidd_2`, using the mask channels `x_mask[:,3]`/`x_mask[:,4]`/no extra mask respectively,
#      all divided by the same base `weight = mask0 / (method + log(time_since/30 + 1))`),
#      concatenated into one `3*dim_prj`-wide vector -- this is the "multiscale pooling" the
#      DeepCare paper describes for producing a single patient-trajectory summary from the LSTM.
#   4. A 2-layer MLP classifier head (`U1`/`b1` + sigmoid, `U2`/`b2` + softmax) over the pooled
#      vector, exactly `init_top_params`'s shapes.
#
# Dropout branches (`drin`/`drfeat`/`drhid` in `options['reg']`) are training-time-only
# regularization toggles in the real code (gated by a `use_noise` scalar theano.shared that is
# 0 at eval/prediction time in `f_pred`) -- they are omitted here since this module targets eval
# forward-pass tracing (real `f_pred` theano.function already excludes noise), not the dropout
# masking implementation detail. No new architectural mechanism is added or removed.

import math

import torch
import torch.nn as nn


class DeepCareLSTMCell(nn.Module):
    """Time-aware, medication-modulated LSTM cell -- transcribed from lstm_layer.py:_step."""

    def __init__(self, dim_emb: int, dim_prj: int):
        super().__init__()
        self.dim_prj = dim_prj
        self.dim_emb = dim_emb

        # Real names: lstm_W (x_ -> gates), lstm_U (h_ -> gates), lstm_b (gate bias)
        self.lstm_W = nn.Parameter(torch.empty(dim_emb, 4 * dim_prj))
        self.lstm_U = nn.Parameter(torch.empty(dim_prj, 4 * dim_prj))
        self.lstm_b = nn.Parameter(torch.zeros(4 * dim_prj))

        # Real names: lstm_Pf (forget-gate pm modulation), lstm_Po (output-gate pm modulation)
        self.lstm_Pf = nn.Parameter(torch.empty(dim_emb, dim_prj))
        self.lstm_Po = nn.Parameter(torch.empty(dim_emb, dim_prj))

        # Real name: lstm_Z (multi-scale time-decay embedding), dim_time=3 fixed by build_model
        self.lstm_Z = nn.Parameter(torch.empty(3, dim_prj))

        self._reset_parameters()

    def _reset_parameters(self):
        for p in (self.lstm_W, self.lstm_U, self.lstm_Pf, self.lstm_Po, self.lstm_Z):
            nn.init.xavier_uniform_(p)

    def _slice(self, gates: torch.Tensor, n: int) -> torch.Tensor:
        return gates[..., n * self.dim_prj : (n + 1) * self.dim_prj]

    def forward(
        self,
        emb_dia: torch.Tensor,  # (n_steps, n_samples, dim_emb) diagnosis embedding sequence
        emb_pm: torch.Tensor,  # (n_steps, n_samples, dim_emb) medication/procedure embedding
        x_mask: torch.Tensor,  # (n_steps, 2, n_samples) mask channels [visit_mask, pm_mask]
        time: torch.Tensor,  # (n_steps, n_samples) time-since-previous-admission (days)
        method: torch.Tensor,  # (n_steps, n_samples) admission-type multiplier
    ) -> torch.Tensor:
        n_steps, n_samples, _ = emb_dia.shape
        h = emb_dia.new_zeros(n_samples, self.dim_prj)
        c = emb_dia.new_zeros(n_samples, self.dim_prj)
        pm_prev = emb_dia.new_zeros(n_samples, self.dim_emb)

        outputs = []
        for t in range(n_steps):
            x_ = emb_dia[t]
            xf_ = emb_pm[t]
            m_ = x_mask[t]  # (2, n_samples)
            time_ = time[t]  # (n_samples,)
            method_ = method[t]  # (n_samples,)

            preact = x_ @ self.lstm_W + h @ self.lstm_U + self.lstm_b

            pm = m_[1][:, None] * xf_

            pre_f = pm_prev @ self.lstm_Pf
            pre_o = pm @ self.lstm_Po
            time_feat = torch.stack(
                [time_ / 60.0, (time_ / 180.0) ** 2, (time_ / 365.0) ** 3], dim=-1
            )
            pre_t = time_feat @ self.lstm_Z

            i = torch.sigmoid(self._slice(preact, 0)) * (1.0 / method_[:, None])
            f = torch.sigmoid(self._slice(preact, 1) + pre_f + pre_t)
            o = torch.sigmoid(self._slice(preact, 2) + pre_o)
            c_tilde = torch.tanh(self._slice(preact, 3))

            c_new = f * c + i * c_tilde
            c = m_[0][:, None] * c_new + (1.0 - m_[0])[:, None] * c

            h_new = o * torch.tanh(c)
            h = m_[0][:, None] * h_new + (1.0 - m_[0])[:, None] * h

            outputs.append(h)
            pm_prev = pm

        return torch.stack(outputs, dim=0)  # (n_steps, n_samples, dim_prj)


class DeepCare(nn.Module):
    """Full DeepCare readmission-prediction model -- transcribed from readm.py:build_model."""

    def __init__(
        self,
        vocab_size: int = 64,
        dim_emb: int = 10,
        dim_prj: int = 20,
        dim_hid: int = 20,
        dim_y: int = 2,
    ):
        super().__init__()
        self.dim_emb = dim_emb
        self.dim_prj = dim_prj

        # Real name: Wemb (shared diagnosis + medication/procedure code embedding table)
        self.Wemb = nn.Embedding(vocab_size, dim_emb)

        self.lstm = DeepCareLSTMCell(dim_emb, dim_prj)

        # Real names: U1/b1 (sigmoid hidden layer), U2/b2 (softmax output layer)
        self.U1 = nn.Linear(3 * dim_prj, dim_hid)
        self.U2 = nn.Linear(dim_hid, dim_y)

    def forward(
        self,
        diag_codes: torch.Tensor,  # (n_steps, n_samples, n_words_dia) int64 code ids per visit
        pm_codes: torch.Tensor,  # (n_steps, n_samples, n_words_pm) int64 code ids per visit
        diag_word_mask: torch.Tensor,  # (n_steps, n_samples, n_words_dia)
        pm_word_mask: torch.Tensor,  # (n_steps, n_samples, n_words_pm)
        x_mask: torch.Tensor,  # (n_steps, 5, n_samples) visit-level mask channels
        time0: torch.Tensor,  # (n_steps, n_samples) inter-visit gap (days)
        time1: torch.Tensor,  # (n_steps, n_samples) time-to-present (days)
        method: torch.Tensor,  # (n_steps, n_samples) admission-type multiplier
    ) -> torch.Tensor:
        # Mean-pool embeddings over the admission's code words (options['embed']=='mean' branch)
        emb_dia_words = self.Wemb(diag_codes)
        emb_dia = (emb_dia_words * diag_word_mask[..., None]).sum(dim=2) / diag_word_mask.sum(
            dim=2
        )[..., None].clamp_min(1e-8)

        emb_pm_words = self.Wemb(pm_codes)
        emb_pm = (emb_pm_words * pm_word_mask[..., None]).sum(dim=2) / pm_word_mask.sum(dim=2)[
            ..., None
        ].clamp_min(1e-8)

        lstm_mask = torch.stack([x_mask[:, 0], x_mask[:, 1]], dim=1)  # (n_steps, 2, n_samples)
        proj = self.lstm(emb_dia, emb_pm, lstm_mask, time0, method)  # (n_steps, n_samples, dim_prj)

        weight = x_mask[:, 0] / (method + torch.log(time1 / 30.0 + 1))
        weight0 = weight * x_mask[:, 3]
        weight1 = weight * x_mask[:, 4]
        weight2 = weight

        def _pool(w: torch.Tensor) -> torch.Tensor:
            return (proj * w[:, :, None]).sum(dim=0) / w.sum(dim=0)[:, None].clamp_min(1e-8)

        hidd_0 = _pool(weight0)
        hidd_1 = _pool(weight1)
        hidd_2 = _pool(weight2)
        hidd = torch.cat([hidd_0, hidd_1, hidd_2], dim=1)

        hid1 = torch.sigmoid(self.U1(hidd))
        pred = torch.softmax(self.U2(hid1), dim=-1)
        return pred


def build_deepcare():
    torch.manual_seed(0)
    return DeepCare(vocab_size=64, dim_emb=10, dim_prj=20, dim_hid=20, dim_y=2)


def example_input_deepcare():
    torch.manual_seed(0)
    n_steps, n_samples, n_words = 6, 3, 4
    diag_codes = torch.randint(1, 64, (n_steps, n_samples, n_words))
    pm_codes = torch.randint(1, 64, (n_steps, n_samples, n_words))
    diag_word_mask = torch.ones(n_steps, n_samples, n_words)
    pm_word_mask = torch.ones(n_steps, n_samples, n_words)
    x_mask = torch.ones(n_steps, 5, n_samples)
    time0 = torch.rand(n_steps, n_samples) * 30.0
    time1 = torch.rand(n_steps, n_samples) * 100.0 + 1.0
    method = torch.randint(1, 3, (n_steps, n_samples)).float()
    return (diag_codes, pm_codes, diag_word_mask, pm_word_mask, x_mask, time0, time1, method)


MENAGERIE_ZOO = "ported-pytorch"
MENAGERIE_ENTRIES = [
    ("DeepCare", "build_deepcare", "example_input_deepcare", 2017, "ported"),
]
