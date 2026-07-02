# SOURCE: vendored from https://github.com/lilstrawberry/MIKT @ master
#   Vendored files:
#     - model.py  (the MIKT nn.Module, verbatim architecture)
#     - glo.py    (tiny get/set global-dict helper the real repo uses to pass the
#       `pro2skill` adjacency matrix, `state_d`, and `max_seq` into the model at
#       construction/forward time -- reproduced here as module-level state set by
#       build_mikt() instead of a bare global dict, since torchlens traces a single
#       instantiated model rather than re-running the repo's run.py driver script)
#
# MIKT ("Interpretable Knowledge Tracing with Multiscale State Representation",
# https://github.com/lilstrawberry/MIKT) is a knowledge-tracing model: problem and
# skill embeddings are fused via a problem<->skill cross-attention ("skill_attn")
# plus a difficulty-modulated Rasch-style correction; per-timestep it maintains BOTH
# a per-skill state bank (`skill_state`, one vector per skill) and a single global
# "all_state" summary vector, each with its own learned forget gate (`skill_forget`,
# `all_forget`) conditioned on elapsed time-gap embeddings; a soft "predict_attn" gate
# blends the skill-specific and global states before a final ability/difficulty
# sigmoid-difference readout produces the per-step correctness probability. The
# `contrast_loss` return value is 0 in the current released code path (referenced but
# never populated in forward()), so the second output is a constant zero scalar here
# exactly as in the real repo -- not synthesized.

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

MENAGERIE_ZOO = "vendored-pytorch"


# ---- glo.py: tiny global key/value store the real repo uses to pass side data
# (pro2skill adjacency, state_d, max_seq) into the model without threading them
# through every call. Reproduced verbatim in behavior; state is set by build_mikt()
# below instead of a separate run.py driver.
_glo_dict = {}


def glo_set_value(key, value):
    _glo_dict[key] = value


def glo_get_value(key, defValue=None):
    return _glo_dict[key]


# ---- model.py: MIKT, verbatim architecture ----


class MIKT(nn.Module):
    def __init__(self, skill_max, pro_max, embed, p):
        super(MIKT, self).__init__()

        self.skill_max = skill_max
        self.pro_max = pro_max

        self.pro_embed = nn.Parameter(torch.rand(pro_max, embed))
        nn.init.xavier_uniform_(self.pro_embed)

        self.skill_embed = nn.Parameter(torch.rand(skill_max, embed))
        nn.init.xavier_uniform_(self.skill_embed)

        self.var = nn.Parameter(torch.rand(pro_max, embed))
        self.change = nn.Parameter(torch.rand(pro_max, 1))

        d = embed
        state_d = glo_get_value("state_d")
        max_seq = glo_get_value("max_seq")

        self.pos_embed = nn.Parameter(torch.rand(max_seq, embed))
        nn.init.xavier_uniform_(self.pos_embed)

        state_embed = state_d

        self.skill_state = nn.Parameter(torch.rand(skill_max, state_embed))
        self.time_state = nn.Parameter(torch.rand(max_seq, state_embed))
        self.all_state = nn.Parameter(torch.rand(1, state_embed))

        self.all_forget = nn.Sequential(
            nn.Linear(2 * state_embed, state_embed),
            nn.ReLU(),
            nn.Linear(state_embed, state_embed),
            nn.Sigmoid(),
        )

        self.ans_embed = nn.Embedding(2, d)
        self.lstm = nn.LSTM(2 * d, d, batch_first=True)

        self.now_obtain = nn.Sequential(
            nn.Linear(d, state_embed), nn.Tanh(), nn.Linear(state_embed, state_embed), nn.Tanh()
        )

        self.pro_diff_embed = nn.Parameter(torch.rand(pro_max, d))
        self.pro_diff = nn.Embedding(pro_max, 1)

        self.pro_linear = nn.Linear(d, d)
        self.skill_linear = nn.Linear(d, d)
        self.pro_change = nn.Linear(d, d)

        self.pro_guess = nn.Embedding(pro_max, 1)
        self.pro_divide = nn.Embedding(pro_max, 1)

        self.skill_state = nn.Parameter(torch.rand(skill_max, state_embed))

        self.pro_ability = nn.Sequential(nn.Linear(3 * d, d), nn.ReLU(), nn.Linear(d, 1))

        self.obtain1_linear = nn.Linear(d, d)
        self.obtain2_linear = nn.Linear(d, d)

        self.pro_diff_judge = nn.Linear(d, 1)

        self.all_obtain = nn.Linear(d, d)

        self.skill_forget = nn.Sequential(
            nn.Linear(3 * d, d), nn.ReLU(), nn.Dropout(p), nn.Linear(d, d)
        )

        self.do_attn = nn.Sequential(nn.Linear(2 * d, d), nn.ReLU(), nn.Dropout(p), nn.Linear(d, 1))

        self.predict_attn = nn.Linear(3 * d, d)

        self.dropout = nn.Dropout(p=p)

        for m in self.modules():
            if isinstance(m, nn.Linear) or isinstance(m, nn.Embedding):
                nn.init.xavier_uniform_(m.weight)

    def forward(self, last_problem, last_ans, next_problem, next_ans):
        device = last_problem.device

        seq = last_problem.shape[1]
        batch = last_problem.shape[0]

        pro2skill = glo_get_value("pro2skill")

        pro_embed = self.pro_embed
        skill_embed = self.skill_embed

        skill_mean = torch.matmul(pro2skill, skill_embed) / (
            torch.sum(pro2skill, dim=-1, keepdims=True) + 1e-8
        )  # pro d

        pro_idx = torch.arange(self.pro_max).to(device)
        pro_diff_embed = F.embedding(pro_idx, self.pro_diff_embed)  # pro_max d  # noqa: F841

        pro_diff = torch.sigmoid(self.pro_diff(pro_idx))  # pro_max 1

        q_pro = self.pro_linear(pro_embed)
        q_skill = self.skill_linear(self.skill_embed)
        attn = torch.matmul(q_pro, q_skill.transpose(-1, -2)) / math.sqrt(q_pro.shape[-1])
        attn = torch.masked_fill(attn, pro2skill == 0, -1e9)
        attn = torch.softmax(attn, dim=-1)
        skill_attn = torch.matmul(attn, skill_embed)

        now_embed = skill_attn + pro_diff * self.pro_change(skill_mean)

        pro_embed = self.dropout(now_embed)

        # pro_diff = torch.sigmoid(self.pro_diff_judge(pro_embed))  # pro_max 1
        contrast_loss = 0

        next_origin_q = F.embedding(next_problem, pro_embed)  # noqa: F841

        # pro_diff = torch.sigmoid(self.pro_diff(pro_embed)) # pro_max 1

        last_pro_rasch = F.embedding(last_problem, pro_embed)
        next_pro_rasch = F.embedding(next_problem, pro_embed)
        next_pro_diff = F.embedding(next_problem, pro_diff)  # noqa: F841
        next_pro_guess = torch.sigmoid(self.pro_guess(next_problem))
        next_pro_divide = self.pro_divide(next_problem)

        next_X = next_pro_rasch + self.ans_embed(next_ans)
        X = last_pro_rasch + self.ans_embed(last_ans)  # noqa: F841

        last_all_time = torch.ones((batch)).to(device).long()

        time_embed = self.time_state  # seq d
        all_gap_embed = F.embedding(last_all_time, time_embed)  # batch d

        res_p = []

        last_skill_time = torch.zeros((batch, self.skill_max)).to(device).long()  # batch skill

        skill_state = self.skill_state.unsqueeze(0).repeat(batch, 1, 1)  # batch skill d

        all_state = self.all_state.repeat(batch, 1)  # batch d
        batch_indices = torch.arange(batch).to(device)  # noqa: F841

        for now_step in range(seq):
            now_pro = next_problem[:, now_step]  # batch
            now_pro2skill = F.embedding(now_pro, pro2skill).unsqueeze(1)  # batch 1 skill

            now_pro_embed = next_pro_rasch[:, now_step]  # batch d

            f1 = now_pro_embed.unsqueeze(1)  # batch 1 d
            f2 = skill_state  # batch skill d

            skill_time_gap = now_step - now_pro2skill.squeeze(1) * last_skill_time  # batch skill
            skill_time_gap_embed = F.embedding(skill_time_gap.long(), time_embed)  # batch skill d

            now_all_state = all_state  # batch d

            forget_now_all_state = now_all_state * self.all_forget(
                self.dropout(torch.cat([now_all_state, all_gap_embed], dim=-1))
            )

            effect_all_state = forget_now_all_state.unsqueeze(1).repeat(1, f2.shape[1], 1)

            skill_forget = torch.sigmoid(
                self.skill_forget(
                    self.dropout(
                        torch.cat([skill_state, skill_time_gap_embed, effect_all_state], dim=-1)
                    )
                )
            )
            skill_forget = torch.masked_fill(skill_forget, now_pro2skill.transpose(-1, -2) == 0, 1)
            skill_state = skill_state * skill_forget

            # now_pro_skill_attn = self.do_attn(torch.cat([f1+f2,f2], dim=-1)) # batch skill 1
            now_pro_skill_attn = (
                torch.matmul(f1, skill_state.transpose(-1, -2)) / f1.shape[-1]
            )  # batch 1 skill

            now_pro_skill_attn = torch.masked_fill(now_pro_skill_attn, now_pro2skill == 0, -1e9)

            now_pro_skill_attn = torch.softmax(now_pro_skill_attn, dim=-1)  # batch 1 skill

            now_need_state = torch.matmul(now_pro_skill_attn, skill_state).squeeze(1)  # batch d

            all_attn = torch.sigmoid(
                self.predict_attn(
                    self.dropout(
                        torch.cat([now_need_state, forget_now_all_state, now_pro_embed], dim=-1)
                    )
                )
            )

            now_need_state = torch.cat(
                [(1 - all_attn) * now_need_state, all_attn * forget_now_all_state], dim=-1
            )

            last_skill_time = torch.masked_fill(
                last_skill_time, now_pro2skill.squeeze(1) == 1, now_step
            )

            now_ability = torch.sigmoid(
                self.pro_ability(torch.cat([now_need_state, now_pro_embed], dim=-1))
            )  # batch 1
            now_guess = next_pro_guess[:, now_step]  # batch 1  # noqa: F841
            now_divide = next_pro_divide[:, now_step]  # batch 1  # noqa: F841
            now_diff = F.embedding(now_pro, pro_diff)  # batch 1

            now_output = torch.sigmoid(5 * (now_ability - now_diff))

            now_output = now_output.squeeze(-1)

            res_p.append(now_output)

            now_X = next_X[:, now_step]  # batch d

            all_state = forget_now_all_state + torch.tanh(
                self.all_obtain(self.dropout(now_X))
            ).squeeze(1)

            to_get = torch.tanh(self.now_obtain(self.dropout(now_X))).unsqueeze(1)  # batch 1 d

            f1 = to_get  # batch 1 d
            f2 = skill_state  # batch skill d

            now_pro_skill_attn = (
                torch.matmul(f1, f2.transpose(-1, -2)) / f1.shape[-1]
            )  # batch 1 skill
            now_pro_skill_attn = torch.masked_fill(now_pro_skill_attn, now_pro2skill == 0, -1e9)
            now_pro_skill_attn = torch.softmax(now_pro_skill_attn, dim=-1)  # batch 1 skill

            now_get = torch.matmul(now_pro_skill_attn.transpose(-1, -2), to_get)

            skill_state = skill_state + now_get
            # skill_state

        P = torch.vstack(res_p).T

        return P, contrast_loss


# ---- tiny build/example (architecture unmodified from the real repo) ----


def build_mikt():
    """Tiny MIKT for tracing. Architecture is unmodified from the real repo.
    Sets the glo-style state (pro2skill adjacency, state_d, max_seq) the real
    run.py driver script normally sets via glo.set_value() before constructing
    the model, since MIKT.__init__ reads those at construction time."""
    skill_max, pro_max, embed, state_d, max_seq = 6, 10, 8, 8, 12
    pro2skill = (torch.rand(pro_max, skill_max) > 0.6).float()
    # guarantee every problem maps to at least one skill (avoids an all-zero
    # softmax row, which the real repo's data pipeline also guarantees by
    # construction from the ques_skill mapping file)
    pro2skill[:, 0] = 1.0
    glo_set_value("state_d", state_d)
    glo_set_value("max_seq", max_seq)
    glo_set_value("pro2skill", pro2skill)
    model = MIKT(skill_max=skill_max, pro_max=pro_max, embed=embed, p=0.0)
    model.eval()
    return model


def example_input_mikt():
    batch, seq = 2, 5
    pro_max = 10
    last_problem = torch.randint(0, pro_max, (batch, seq))
    last_ans = torch.randint(0, 2, (batch, seq))
    next_problem = torch.randint(0, pro_max, (batch, seq))
    next_ans = torch.randint(0, 2, (batch, seq))
    return (last_problem, last_ans, next_problem, next_ans)


MENAGERIE_ENTRIES = [
    ("MIKT", build_mikt, example_input_mikt, 2023, "vendored-pytorch"),
]
