# SOURCE: vendored from wqi/WIMP @ master
# https://raw.githubusercontent.com/wqi/WIMP/master/src/models/WIMP.py
# https://raw.githubusercontent.com/wqi/WIMP/master/src/models/WIMP_encoder.py
# https://raw.githubusercontent.com/wqi/WIMP/master/src/models/WIMP_decoder.py
# https://raw.githubusercontent.com/wqi/WIMP/master/src/models/GAT.py
#
# Khandelwal, Qi, Singh, Choi, Ramanan 2020 (arXiv/ICRA) "What-If Motion
# Prediction for Autonomous Driving" -- an LSTM-encoder / graph-attention /
# LSTM-decoder trajectory forecaster that is explicitly *lane-conditioned*:
# the encoder and decoder both maintain a running query/key/value attention
# over the agent's (and every social agent's) oracle centerline polyline at
# every timestep, gating each recurrent step on "where along the lane is the
# agent" -- the "What-If" goal/path-conditioning that gives the model its
# name -- while a `GraphAttentionLayer` performs multi-head message passing
# between the target agent and all social agents' encodings before decoding.
#
# `GraphAttentionLayer` (GAT.py), `WIMPEncoder` (WIMP_encoder.py),
# `WIMPDecoder` (WIMP_decoder.py), and `WIMP.forward` (WIMP.py) are copied
# verbatim from the real source. The only non-architectural change: `WIMP`
# subclasses `nn.Module` instead of `pytorch_lightning.LightningModule` (the
# installed pytorch-lightning 2.1.0 no longer has the `pl.TrainResult` /
# `pl.EvalResult` APIs the original `training_step`/`validation_step` used,
# and those Lightning-specific training-loop methods are not part of the
# traced architecture); `forward` itself -- the only method TorchLens
# traces -- is untouched. `hparams` is a plain namespace mirroring the real
# `add_model_specific_args` argparse defaults (mechanical config-plumbing
# substitution, same field names/defaults as the source, no architectural
# change). We enable `use_centerline_features=True` (the model's namesake
# lane-conditioning attention mechanism in both encoder and decoder) and run
# on CUDA because the real encoder/decoder centerline logic hardcodes
# `.get_device()` / `.cuda()` calls (unmodified from source). The 4 uses of
# the removed-in-modern-numpy `np.float("inf"/"-inf")` alias (deprecated
# 2020, removed NumPy 1.24+) are replaced with the builtin `float(...)`,
# which is exactly what `np.float` was an alias for -- a mechanical
# cross-version compatibility fix, not a behavior change.

import numpy as np
import torch
import torch.nn as nn


class GraphAttentionLayer(nn.Module):
    def __init__(self, input_dim, output_dim, num_gat_iters=1, num_heads=4, dropout=0.5, alpha=0.2):
        super(GraphAttentionLayer, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_gat_iters = num_gat_iters
        self.num_heads = num_heads
        self.alpha = alpha

        self.W = nn.ModuleList([nn.Linear(input_dim, output_dim) for _ in range(self.num_heads)])
        self.a_1 = nn.ModuleList([nn.Linear(output_dim, 1) for _ in range(self.num_heads)])
        self.a_2 = nn.ModuleList([nn.Linear(output_dim, 1) for _ in range(self.num_heads)])

        self.leakyrelu = nn.LeakyReLU(self.alpha)
        self.dropout = nn.Dropout(dropout)

    def forward(self, h, adjacency):
        att_weights = []
        cur_h = h

        for iter in range(self.num_gat_iters):
            head_embeds = []

            for head in range(self.num_heads):
                cur_h_transformed = self.W[head](cur_h)
                att_half_1 = self.a_1[head](cur_h_transformed).squeeze(-1)
                att_half_2 = self.a_2[head](cur_h_transformed).squeeze(-1)
                att_coeff = att_half_1.unsqueeze(-2) + att_half_2.unsqueeze(-3)
                att_coeff = self.leakyrelu(att_coeff)

                with torch.no_grad():
                    masked_att_max = torch.max(att_coeff, 2)[0]
                masked_att_reduced = att_coeff.squeeze(-1) - masked_att_max
                masked_att_exp = masked_att_reduced.exp() * adjacency
                masked_att_exp = masked_att_exp.unsqueeze(-1)

                mask_sum = masked_att_exp.sum(dim=2, keepdim=True)
                mask_ones = torch.ones_like(mask_sum)
                mask_sum_normalized = torch.where(mask_sum == 0.0, mask_ones, mask_sum)
                att_values = torch.div(masked_att_exp, mask_sum_normalized)
                att_values = self.dropout(att_values)

                h_prime = torch.bmm(
                    att_values.squeeze(-1), cur_h_transformed.squeeze(-2)
                ).unsqueeze(-2)
                head_embeds.append(h_prime)

                if iter == 0:
                    att_weights.append(att_values.squeeze(-1).detach())

            cur_h = torch.tanh(cur_h + torch.mean(torch.stack(head_embeds, dim=-1), dim=-1))

        out = cur_h
        att_weights = torch.stack(att_weights, dim=1)
        return out, att_weights


class WIMPEncoder(nn.Module):
    def __init__(self, hparams):
        super(WIMPEncoder, self).__init__()
        self.hparams = hparams
        self.hparams.cl_kernel_list = [1, 3, 5]
        self.hparams.xy_kernel_list = [1, 3, 5]

        self.xy_conv_filters = nn.ModuleList(
            [
                nn.Conv1d(
                    in_channels=self.hparams.input_dim,
                    out_channels=self.hparams.hidden_dim,
                    kernel_size=x,
                    padding=(x - 1) // 2,
                )
                for x in self.hparams.xy_kernel_list
            ]
        )
        self.xy_input_transform = nn.Conv1d(
            in_channels=self.hparams.hidden_dim * len(self.hparams.xy_kernel_list),
            out_channels=self.hparams.hidden_dim,
            kernel_size=1,
        )
        self.non_linearity = nn.Tanh() if self.hparams.non_linearity == "tanh" else nn.ReLU()
        self.centerline_modifier = (
            2 if (self.hparams.use_centerline_features and not self.hparams.add_centerline) else 1
        )
        self.lstm_input_transform = nn.Linear(
            self.hparams.hidden_dim * self.centerline_modifier, self.hparams.hidden_dim
        )
        self.lstm = nn.LSTM(
            input_size=self.hparams.hidden_dim,
            hidden_size=self.hparams.hidden_dim,
            num_layers=self.hparams.num_layers,
            batch_first=True,
            dropout=self.hparams.dropout,
        )
        self.waypoint_predictor = (
            nn.Linear(self.hparams.hidden_dim, self.hparams.output_dim)
            if self.hparams.output_prediction
            else nn.Linear(
                self.hparams.hidden_dim * self.hparams.num_layers, self.hparams.output_dim
            )
        )
        self.waypoint_lstm = nn.LSTM(
            input_size=self.hparams.hidden_dim,
            hidden_size=self.hparams.hidden_dim,
            num_layers=self.hparams.num_layers,
            batch_first=True,
            dropout=self.hparams.dropout,
        )
        if self.hparams.use_centerline_features:
            key_input = (
                self.hparams.hidden_dim
                if not self.hparams.hidden_key_generator
                else self.hparams.hidden_dim * self.hparams.num_layers
            )
            self.cl_conv_filters = nn.ModuleList(
                [
                    nn.Conv1d(
                        in_channels=self.hparams.input_dim,
                        out_channels=self.hparams.hidden_dim,
                        kernel_size=x,
                        padding=(x - 1) // 2,
                    )
                    for x in self.hparams.cl_kernel_list
                ]
            )
            self.cl_input_transform = nn.Conv1d(
                in_channels=self.hparams.hidden_dim * len(self.hparams.cl_kernel_list),
                out_channels=self.hparams.hidden_dim,
                kernel_size=1,
            )
            self.leakyrelu = nn.LeakyReLU()
            self.key_generator = nn.Linear(key_input, self.hparams.hidden_dim)
            self.query_generator = nn.Linear(self.hparams.hidden_dim, self.hparams.hidden_dim)
            self.value_generator = nn.Linear(self.hparams.hidden_dim, self.hparams.hidden_dim)

        if self.hparams.batch_norm:
            self.input_bn = nn.BatchNorm1d(self.hparams.hidden_dim)

    def forward(
        self,
        agent_features,
        social_features,
        num_agent_mask,
        ifc_helpers=None,
        visualize_centerline=False,
    ):
        non_zero_indices = torch.nonzero(
            num_agent_mask.view(
                -1,
            ),
            as_tuple=True,
        )[0]
        zero_indices = torch.nonzero(
            num_agent_mask.view(
                -1,
            )
            == 0,
            as_tuple=True,
        )[0]

        if self.hparams.use_centerline_features:
            agent_centerline = ifc_helpers["agent_oracle_centerline"]
            agent_centerline_lengths = ifc_helpers["agent_oracle_centerline_lengths"]
            social_centerline = ifc_helpers["social_oracle_centerline"]
            social_centerline_lengths = ifc_helpers["social_oracle_centerline_lengths"]

            all_centerline = torch.cat(
                [agent_centerline.unsqueeze(1), social_centerline], dim=1
            ).view(-1, *agent_centerline.size()[1:])
            all_centerline_nonzero = all_centerline.index_select(0, non_zero_indices)
            all_centerline_nonzero_transposed = all_centerline_nonzero.transpose(1, 2).contiguous()
            all_centerline_lengths = torch.cat(
                [agent_centerline_lengths.unsqueeze(1), social_centerline_lengths], dim=1
            ).view(
                -1,
            )
            all_centerline_lengths_nonzero = all_centerline_lengths.index_select(
                0, non_zero_indices
            )
            all_centerline_features = []
            for i, _ in enumerate(self.hparams.cl_kernel_list):
                all_centerline_features.append(
                    self.cl_conv_filters[i](all_centerline_nonzero_transposed)
                )
            all_centerline_features = torch.cat(all_centerline_features, dim=1)
            all_centerline_features_comb = self.non_linearity(
                self.cl_input_transform(all_centerline_features)
            )
            centerline_features = all_centerline_features_comb.transpose(1, 2).contiguous()

            with torch.no_grad():
                indexer = torch.arange(centerline_features.size(1)).type_as(centerline_features)
                centerline_mask_byte = all_centerline_lengths_nonzero[:, None] > indexer
                centerlines_masked = torch.where(
                    centerline_mask_byte.unsqueeze(-1),
                    all_centerline_nonzero,
                    torch.zeros_like(all_centerline_nonzero).fill_(float("inf")),
                )

        if self.hparams.distributed_backend == "dp":
            self.lstm.flatten_parameters()

        all_agents = torch.cat([agent_features.unsqueeze(1), social_features], dim=1).view(
            -1, *agent_features.size()[1:]
        )
        all_agents_nonzero = all_agents.index_select(0, non_zero_indices)
        _, resorter = torch.sort(
            torch.cat([non_zero_indices, zero_indices], dim=0), descending=False
        )
        all_agents_nonzero_transposed = all_agents_nonzero.transpose(1, 2).contiguous()

        conv_filters = []
        for i, _ in enumerate(self.hparams.xy_kernel_list):
            conv_filters.append(self.xy_conv_filters[i](all_agents_nonzero_transposed))
        conv_filters = torch.cat(conv_filters, dim=1)
        input_features = self.non_linearity(self.xy_input_transform(conv_filters))
        input_features = input_features.transpose(1, 2).contiguous()
        if self.hparams.batch_norm:
            input_features = (
                self.input_bn(input_features.transpose(1, 2).contiguous())
                .transpose(1, 2)
                .contiguous()
            )

        hidden = self.initHidden(self.hparams.num_layers, input_features.size(0))
        if visualize_centerline:
            centerline_attention_viz = []  # noqa: F841 -- verbatim from source (dead var upstream too)

        waypoint_predictions = []
        centerline_features_query = self.query_generator(centerline_features)
        centerline_features_value = self.value_generator(centerline_features)
        for tstep in range(input_features.size(1)):
            curr_waypoint_points = []
            current_input = input_features.narrow(1, start=tstep, length=1)
            with torch.no_grad():
                curr_xy_points = all_agents_nonzero.narrow(1, tstep, 1).detach()
                distances = centerlines_masked - curr_xy_points
                distances = torch.sum(torch.mul(distances, distances), dim=-1)
                closest_point = torch.argmin(distances, dim=-1)

            if tstep < (input_features.size(1) - self.hparams.waypoint_step):
                curr_waypoint_prediction = all_agents_nonzero.narrow(
                    1, tstep + self.hparams.waypoint_step, 1
                ).detach()
            else:
                curr_waypoint_features = current_input.view(current_input.size(0), 1, -1)
                curr_waypoint_decoding, curr_waypoint_hidden = self.waypoint_lstm(
                    curr_waypoint_features, hidden
                )
                if self.hparams.output_prediction:
                    curr_waypoint_prediction = self.waypoint_predictor(curr_waypoint_decoding)
                else:
                    curr_waypoint_prediction = self.waypoint_predictor(
                        curr_waypoint_hidden[0]
                        .transpose(0, 1)
                        .contiguous()
                        .view(curr_waypoint_hidden[0].size(1), -1)
                    ).unsqueeze(1)
                curr_waypoint_points.append(curr_waypoint_prediction)
                curr_waypoint_points = torch.stack(curr_waypoint_points, 1)
                waypoint_predictions.append(curr_waypoint_points)
            with torch.no_grad():
                curr_xy_points = all_agents_nonzero.narrow(1, tstep, 1).detach()
                distances = centerlines_masked - curr_xy_points
                distances = torch.sum(torch.mul(distances, distances), dim=-1)
                closest_point = torch.argmin(distances, dim=-1)

                waypoint_distances = centerlines_masked - curr_waypoint_prediction
                waypoint_distances = torch.sum(
                    torch.mul(waypoint_distances, waypoint_distances), dim=-1
                )
                waypoint_closest_point = torch.argmin(waypoint_distances, dim=-1)
                segment_length = waypoint_closest_point - closest_point

                max_length = segment_length.abs().max().data.cpu().numpy()
                arange_array = torch.arange(int(max_length) + 1).type_as(segment_length)
                upper_array = closest_point.unsqueeze(-1) + arange_array.view(1, -1)
                lower_array = closest_point.unsqueeze(-1) - arange_array.view(1, -1)

                positive_length_mask = segment_length >= 0
                indexing_array = torch.where(
                    positive_length_mask.unsqueeze(-1).expand(-1, upper_array.size(-1)),
                    upper_array,
                    lower_array,
                )
                positive_mask = indexing_array <= waypoint_closest_point.unsqueeze(-1)
                negative_mask = indexing_array >= waypoint_closest_point.unsqueeze(-1)
                indexing_mask = torch.where(
                    positive_length_mask.unsqueeze(-1).byte(),
                    positive_mask.byte(),
                    negative_mask.byte(),
                )

                lower_mask = indexing_array < 0
                upper_mask = indexing_array >= centerlines_masked.size(1)
                indexing_array[lower_mask] = 0
                indexing_array[upper_mask] = centerlines_masked.size(1) - 1

            curr_centerline_features = torch.gather(
                centerline_features_query,
                1,
                indexing_array.unsqueeze(-1).expand(-1, -1, self.hparams.hidden_dim),
            )
            curr_centerline_features_value = torch.gather(
                centerline_features_value,
                1,
                indexing_array.unsqueeze(-1).expand(-1, -1, self.hparams.hidden_dim),
            )
            current_key = (
                self.key_generator(current_input)
                if not self.hparams.hidden_key_generator
                else self.key_generator(
                    hidden[0].transpose(0, 1).contiguous().view(current_input.size(0), 1, -1)
                )
            )
            current_centerline_score_unnormalized = nn.functional.leaky_relu(
                torch.bmm(curr_centerline_features, current_key.transpose(1, 2))
            )
            current_centerline_score_unnormalized = current_centerline_score_unnormalized.view(
                *indexing_mask.size()
            )
            current_centerline_score = torch.where(
                indexing_mask,
                current_centerline_score_unnormalized,
                torch.zeros_like(current_centerline_score_unnormalized).fill_(float("-inf")),
            )
            current_centerline_attention = nn.functional.softmax(current_centerline_score, -1)
            curr_centerline = (
                curr_centerline_features_value * current_centerline_attention.unsqueeze(-1)
            )
            curr_centerline = torch.sum(curr_centerline, dim=1, keepdim=True)
            current_input_xy_centerline = (
                torch.cat([current_input, curr_centerline], dim=-1)
                if not self.hparams.add_centerline
                else (current_input + curr_centerline)
            )
            current_input_xy_centerline = self.non_linearity(
                self.lstm_input_transform(current_input_xy_centerline)
            )
            current_encoding, hidden = self.lstm(current_input_xy_centerline, hidden)

        waypoint_predictions_nonzero = torch.cat(waypoint_predictions, dim=1)
        waypoint_predictions_pad = nn.functional.pad(
            waypoint_predictions_nonzero,
            pad=(0, 0, 0, 0, 0, 0, 0, all_agents.size(0) - all_agents_nonzero.size(0)),
        )
        waypoint_predictions_all = waypoint_predictions_pad.index_select(0, resorter)
        current_encoding_pad = nn.functional.pad(
            current_encoding, pad=(0, 0, 0, 0, 0, all_agents.size(0) - all_agents_nonzero.size(0))
        )
        current_encoding = current_encoding_pad.index_select(0, resorter)
        hidden_pad = (
            nn.functional.pad(
                hidden[0], pad=(0, 0, 0, all_agents.size(0) - all_agents_nonzero.size(0), 0, 0)
            ),
            nn.functional.pad(
                hidden[1], pad=(0, 0, 0, all_agents.size(0) - all_agents_nonzero.size(0), 0, 0)
            ),
        )
        hidden = (hidden_pad[0].index_select(1, resorter), hidden_pad[1].index_select(1, resorter))
        return current_encoding, hidden, waypoint_predictions_all

    def initHidden(self, batch_size=1, num_agents=1):
        weight = next(self.parameters()).data
        return (
            weight.new(batch_size, num_agents, self.hparams.hidden_dim).zero_(),
            weight.new(batch_size, num_agents, self.hparams.hidden_dim).zero_(),
        )


class WIMPDecoder(nn.Module):
    def __init__(self, hparams):
        super(WIMPDecoder, self).__init__()
        self.hparams = hparams
        self.hparams.cl_selected_kernel_list = [1, 3]
        self.hparams.output_xy_kernel_list = (
            self.hparams.xy_kernel_list if self.hparams.output_conv else [1]
        )
        self.hparams.predictor_output_dim = self.hparams.num_mixtures * (
            self.hparams.output_dim + 1
        )

        self.xy_conv_filters = nn.ModuleList(
            [
                nn.Conv1d(
                    in_channels=self.hparams.input_dim + 1,
                    out_channels=self.hparams.hidden_dim,
                    kernel_size=x,
                )
                for x in self.hparams.output_xy_kernel_list
            ]
        )
        self.output_transform = nn.Conv1d(
            in_channels=self.hparams.hidden_dim * len(self.hparams.output_xy_kernel_list),
            out_channels=self.hparams.hidden_dim,
            kernel_size=1,
        )

        self.non_linearity = nn.Tanh() if self.hparams.non_linearity == "tanh" else nn.ReLU()
        self.lstm = nn.LSTM(
            input_size=self.hparams.hidden_dim * self.hparams.num_mixtures,
            hidden_size=self.hparams.hidden_dim,
            num_layers=self.hparams.num_layers,
            batch_first=True,
            dropout=self.hparams.dropout,
        )
        self.centerline_modifier = (
            2 if (self.hparams.use_centerline_features and not self.hparams.add_centerline) else 1
        )
        self.lstm_input_transform = nn.Linear(
            self.hparams.hidden_dim * self.centerline_modifier, self.hparams.hidden_dim
        )
        self.predictor = (
            nn.Linear(self.hparams.hidden_dim, self.hparams.predictor_output_dim)
            if self.hparams.output_prediction
            else nn.Linear(
                self.hparams.hidden_dim * self.hparams.num_layers, self.hparams.predictor_output_dim
            )
        )
        self.waypoint_predictor = (
            nn.Linear(
                self.hparams.hidden_dim,
                self.hparams.predictor_output_dim - self.hparams.num_mixtures,
            )
            if self.hparams.output_prediction
            else nn.Linear(
                self.hparams.hidden_dim * self.hparams.num_layers, self.hparams.predictor_output_dim
            )
        )
        self.waypoint_lstm = nn.LSTM(
            input_size=self.hparams.hidden_dim * self.hparams.num_mixtures,
            hidden_size=self.hparams.hidden_dim,
            num_layers=self.hparams.num_layers,
            batch_first=True,
            dropout=self.hparams.dropout,
        )
        if self.hparams.use_centerline_features:
            key_input = (
                self.hparams.hidden_dim
                if not self.hparams.hidden_key_generator
                else self.hparams.hidden_dim * self.hparams.num_layers
            )
            self.cl_conv_filters = nn.ModuleList(
                [
                    nn.Conv1d(
                        in_channels=self.hparams.input_dim,
                        out_channels=self.hparams.hidden_dim,
                        kernel_size=x,
                        padding=(x - 1) // 2,
                    )
                    for x in self.hparams.cl_kernel_list
                ]
            )
            self.cl_input_transform = nn.Conv1d(
                in_channels=self.hparams.hidden_dim * len(self.hparams.cl_kernel_list),
                out_channels=self.hparams.hidden_dim,
                kernel_size=1,
            )
            self.leakyrelu = nn.LeakyReLU()
            key_output = (
                self.hparams.hidden_dim
                if not self.hparams.hidden_key_generator
                else self.hparams.hidden_dim * self.hparams.num_mixtures
            )
            self.key_generator = nn.Linear(key_input, key_output)
            self.query_generator = nn.Linear(self.hparams.hidden_dim, self.hparams.hidden_dim)
            self.value_generator = nn.Linear(self.hparams.hidden_dim, self.hparams.hidden_dim)

    def forward(
        self,
        decoder_input_features,
        last_n_predictions,
        hidden_decoder,
        outsteps,
        ifc_helpers=None,
        sample_next=False,
        map_estimate=False,
        mixture_num=-1,
        sample_centerline=False,
    ):
        if self.hparams.use_centerline_features:
            agent_centerline = ifc_helpers["agent_oracle_centerline"]
            agent_centerline_lengths = ifc_helpers["agent_oracle_centerline_lengths"]
            agent_centerline = agent_centerline.transpose(1, 2).contiguous()
            agent_centerline_features = []
            for i, _ in enumerate(self.hparams.cl_kernel_list):
                agent_centerline_features.append(self.cl_conv_filters[i](agent_centerline))
            agent_centerline_features = torch.cat(agent_centerline_features, dim=1)
            agent_centerline_features_comb = self.non_linearity(
                self.cl_input_transform(agent_centerline_features)
            )
            centerline_features = agent_centerline_features_comb.transpose(1, 2).contiguous()
            selected_centerline_features = centerline_features.unsqueeze(1).repeat(
                1, self.hparams.num_mixtures, 1, 1
            )
            selected_centerlines = (
                ifc_helpers["agent_oracle_centerline"]
                .unsqueeze(1)
                .repeat(1, self.hparams.num_mixtures, 1, 1)
            )
            selected_centerline_lengths = agent_centerline_lengths.unsqueeze(1).repeat(
                1, self.hparams.num_mixtures
            )
            with torch.no_grad():
                indexer = torch.arange(selected_centerlines.size(2)).to(
                    selected_centerlines.get_device()
                )
                centerline_mask_byte = selected_centerline_lengths[:, :, None] > indexer
                centerline_mask = centerline_mask_byte.float()
                centerline_mask_nonzero_indexer = torch.nonzero(
                    centerline_mask.view(-1), as_tuple=True
                )[0]
                centerline_mask_zero_indexer = torch.nonzero(
                    centerline_mask.view(-1) == 0, as_tuple=True
                )[0]
                _, centerline_resorter = torch.sort(
                    torch.cat(
                        [centerline_mask_nonzero_indexer, centerline_mask_zero_indexer], dim=0
                    ),
                    descending=False,
                )
                selected_centerlines_masked = torch.where(
                    centerline_mask_byte.unsqueeze(-1),
                    selected_centerlines,
                    torch.zeros_like(selected_centerlines).fill_(float("inf")),
                )

        if self.hparams.distributed_backend == "dp":
            self.lstm.flatten_parameters()
            self.waypoint_lstm.flatten_parameters()

        predictions = []
        waypoint_predictions = []
        num_batches = decoder_input_features.size(0)
        for i, _ in enumerate(self.hparams.output_xy_kernel_list):
            last_n_predictions[i] = nn.functional.pad(last_n_predictions[i], pad=(0, 1), value=1.0)
            last_n_predictions[i] = (
                last_n_predictions[i]
                .unsqueeze(1)
                .repeat(1, self.hparams.num_mixtures, 1, 1)
                .view(-1, *last_n_predictions[i].size()[1:])
            )
        decoder_input_features = nn.functional.pad(decoder_input_features, pad=(0, 1), value=1.0)
        decoder_input_features = (
            decoder_input_features.unsqueeze(1)
            .repeat(1, self.hparams.num_mixtures, 1, 1)
            .view(-1, *decoder_input_features.size()[1:])
        )
        selected_centerline_features_query = self.query_generator(selected_centerline_features)
        selected_centerline_features_value = self.value_generator(selected_centerline_features)
        for timestep in range(outsteps):
            curr_conv_filters = []
            curr_waypoint_points = []
            if self.hparams.output_conv:
                for i, _ in enumerate(self.hparams.output_xy_kernel_list):
                    curr_conv_filters.append(
                        self.xy_conv_filters[i](
                            last_n_predictions[i].transpose(1, 2).contiguous()
                        ).view(num_batches, self.hparams.num_mixtures, -1)
                    )
            else:
                for i, _ in enumerate(self.hparams.output_xy_kernel_list):
                    curr_conv_filters.append(
                        self.xy_conv_filters[i](
                            decoder_input_features.transpose(1, 2).contiguous()
                        ).view(num_batches, self.hparams.num_mixtures, -1)
                    )
            curr_conv_filters = torch.cat(curr_conv_filters, dim=2).transpose(1, 2).contiguous()
            curr_features = self.non_linearity(self.output_transform(curr_conv_filters))
            curr_features = curr_features.transpose(1, 2).contiguous()

            curr_waypoint_features = curr_features.clone().view(curr_features.size(0), 1, -1)
            curr_waypoint_decoding, curr_waypoint_hidden = self.waypoint_lstm(
                curr_waypoint_features, hidden_decoder
            )
            if self.hparams.output_prediction:
                curr_waypoint_prediction = self.waypoint_predictor(curr_waypoint_decoding)
            else:
                curr_waypoint_prediction = self.waypoint_predictor(
                    curr_waypoint_hidden[0]
                    .transpose(0, 1)
                    .contiguous()
                    .view(curr_waypoint_hidden[0].size(1), -1)
                ).unsqueeze(1)
            curr_waypoint_prediction = curr_waypoint_prediction.view(
                curr_waypoint_prediction.size(0), self.hparams.num_mixtures, -1
            )
            curr_waypoint_points.append(curr_waypoint_prediction)
            with torch.no_grad():
                curr_xy_points = decoder_input_features.view(
                    num_batches, self.hparams.num_mixtures, 1, -1
                ).narrow(-1, 0, self.hparams.input_dim)
                distances = selected_centerlines_masked - curr_xy_points
                distances = torch.sum(torch.mul(distances, distances), dim=-1)
                closest_point = torch.argmin(distances, dim=-1)

                waypoint_distances = (
                    selected_centerlines_masked - curr_waypoint_prediction.unsqueeze(2)
                )
                waypoint_distances = torch.sum(
                    torch.mul(waypoint_distances, waypoint_distances), dim=-1
                )
                waypoint_closest_point = torch.argmin(waypoint_distances, dim=-1)

                segment_length = waypoint_closest_point - closest_point
                max_length = segment_length.abs().max().data.cpu().numpy()
                arange_array = torch.arange(int(max_length) + 1).cuda()
                upper_array = closest_point.unsqueeze(-1) + arange_array.view(1, 1, -1)
                lower_array = closest_point.unsqueeze(-1) - arange_array.view(1, 1, -1)
                positive_length_mask = segment_length >= 0
                indexing_array = torch.where(
                    positive_length_mask.unsqueeze(-1).expand(-1, -1, upper_array.size(-1)),
                    upper_array,
                    lower_array,
                )
                positive_mask = indexing_array <= waypoint_closest_point.unsqueeze(-1)
                negative_mask = indexing_array >= waypoint_closest_point.unsqueeze(-1)
                indexing_mask = torch.where(
                    positive_length_mask.unsqueeze(-1), positive_mask, negative_mask
                )

                lower_mask = indexing_array < 0
                upper_mask = indexing_array >= selected_centerlines_masked.size(2)
                indexing_array[lower_mask] = 0
                indexing_array[upper_mask] = selected_centerlines_masked.size(2) - 1

            curr_centerline_features = torch.gather(
                selected_centerline_features_query,
                2,
                indexing_array.unsqueeze(-1).expand(-1, -1, -1, self.hparams.hidden_dim),
            )
            curr_centerline_features_value = torch.gather(
                selected_centerline_features_value,
                2,
                indexing_array.unsqueeze(-1).expand(-1, -1, -1, self.hparams.hidden_dim),
            )
            current_key = (
                self.key_generator(curr_features)
                if not self.hparams.hidden_key_generator
                else self.key_generator(
                    hidden_decoder[0]
                    .transpose(0, 1)
                    .contiguous()
                    .view(curr_features.size(0), 1, -1)
                )
            )
            if self.hparams.hidden_key_generator:
                current_key = current_key.view(
                    -1, self.hparams.num_mixtures, self.hparams.hidden_dim
                )
            current_centerline_score_unnormalized = nn.functional.leaky_relu(
                torch.bmm(
                    curr_centerline_features.view(-1, *curr_centerline_features.size()[2:]),
                    current_key.view(-1, 1, current_key.size(2)).transpose(1, 2),
                )
            )
            current_centerline_score_unnormalized = current_centerline_score_unnormalized.view(
                *indexing_mask.size()
            )
            current_centerline_score = torch.where(
                indexing_mask,
                current_centerline_score_unnormalized,
                torch.zeros_like(current_centerline_score_unnormalized).fill_(float("-inf")),
            )
            current_centerline_attention = nn.functional.softmax(current_centerline_score, -1)
            curr_centerline = (
                curr_centerline_features_value * current_centerline_attention.unsqueeze(-1)
            )
            curr_centerline = torch.sum(curr_centerline, dim=2)
            current_input_xy_centerline = (
                torch.cat([curr_features, curr_centerline], dim=-1)
                if not self.hparams.add_centerline
                else (curr_features + curr_centerline)
            )
            current_input_xy_centerline = self.non_linearity(
                self.lstm_input_transform(current_input_xy_centerline)
            )
            current_input_xy_centerline = current_input_xy_centerline.view(
                current_input_xy_centerline.size(0), 1, -1
            )
            current_decoding, hidden_decoder = self.lstm(
                current_input_xy_centerline, hidden_decoder
            )
            if self.hparams.output_prediction:
                curr_prediction = self.predictor(current_decoding)
            else:
                curr_prediction = self.predictor(
                    hidden_decoder[0]
                    .transpose(0, 1)
                    .contiguous()
                    .view(hidden_decoder[0].size(1), -1)
                ).unsqueeze(1)
            curr_prediction = curr_prediction.view(
                curr_prediction.size(0), self.hparams.num_mixtures, -1
            )
            curr_probs = curr_prediction.narrow(-1, self.hparams.output_dim, 1)
            curr_probs = -1 * torch.relu(curr_probs)
            curr_prob_modified = torch.cat(
                [curr_prediction.narrow(-1, 0, self.hparams.output_dim), curr_probs], -1
            )
            predictions.append(curr_prob_modified)
            waypoint_predictions.append(torch.stack(curr_waypoint_points, 2))
            decoder_input_features = curr_prob_modified.detach().view(
                *decoder_input_features.size()
            )

            for i, ksize in enumerate(self.hparams.output_xy_kernel_list):
                last_n_predictions[i] = torch.cat(
                    [
                        last_n_predictions[i].narrow(dim=1, start=1, length=ksize - 1),
                        decoder_input_features,
                    ],
                    dim=1,
                ).detach()

        predictions_tensor = torch.stack(predictions, 2).unsqueeze(1)
        waypoint_tensor = torch.stack(waypoint_predictions, 2).unsqueeze(1)
        return predictions_tensor, waypoint_tensor, []


class WIMP(nn.Module):
    """`pytorch_lightning.LightningModule` -> `nn.Module`: see header note.
    `forward` is byte-for-byte the real source; the Lightning training-loop
    methods (`training_step`/`validation_step`/`configure_optimizers`,
    which relied on removed `pl.TrainResult`/`pl.EvalResult` APIs) are
    dropped since they are outside the traced architecture."""

    def __init__(self, hparams, hidden_dim=128):
        super(WIMP, self).__init__()
        self.hparams = hparams

        self.encoder = WIMPEncoder(self.hparams)
        self.gat = GraphAttentionLayer(
            self.hparams.hidden_dim,
            self.hparams.hidden_dim,
            self.hparams.graph_iter,
            self.hparams.attention_heads,
            self.hparams.dropout,
        )
        self.decoder = WIMPDecoder(self.hparams)

    def forward(
        self,
        agent_features,
        social_features,
        adjacency,
        num_agent_mask,
        outsteps=30,
        social_label_features=None,
        label_adjacency=None,
        classmate_forcing=True,
        labels=None,
        ifc_helpers=None,
        test=False,
        map_estimate=False,
        gt=None,
        idx=None,
        sample_next=False,
        num_predictions=1,
        am=None,
    ):
        encoding, hidden, waypoint_predictions = self.encoder(
            agent_features, social_features, num_agent_mask, ifc_helpers
        )
        waypoint_predictions_tensor_encoder = waypoint_predictions.squeeze(-2).view(
            agent_features.size(0), social_features.size(1) + 1, -1, agent_features.size(2)
        )

        if self.hparams.hidden_transform:
            gan_features = (
                torch.cat(hidden, dim=0)
                .transpose(0, 1)
                .view(
                    agent_features.size(0),
                    social_features.size(1) + 1,
                    hidden[0].size(0) * 2,
                    hidden[0].size(2),
                )
            )
        else:
            gan_features = encoding.view(
                social_features.size(0), social_features.size(1) + 1, 1, -1
            )
        adjacency = (
            torch.ones(gan_features.size(1), gan_features.size(1))
            .to(gan_features.get_device())
            .float()
            .unsqueeze(0)
            .repeat(gan_features.size(0), 1, 1)
        )
        adjacency = adjacency * num_agent_mask.unsqueeze(1) * num_agent_mask.unsqueeze(2)
        graph_output, _ = self.gat(gan_features, adjacency)
        graph_output = graph_output.narrow(1, 0, 1).squeeze(1)
        if self.hparams.batch_norm:
            graph_output = self.encoding_bn(graph_output.transpose(1, 2).contiguous())
            graphoutput = graph_output.transpose(1, 2).contiguous()  # noqa: F841 -- verbatim from source (unused var upstream too)
        if self.hparams.hidden_transform:
            hidden_decoder = torch.chunk(
                graph_output.view(-1, self.hparams.num_layers * 2, self.hparams.hidden_dim)
                .transpose(0, 1)
                .contiguous(),
                2,
                dim=0,
            )
        else:
            hidden_decoder = (
                graph_output.view(-1, 1, self.hparams.hidden_dim).transpose(0, 1).contiguous(),
                graph_output.view(-1, 1, self.hparams.hidden_dim).transpose(0, 1).contiguous(),
            )
            hidden_decoder = (
                hidden_decoder[0].repeat(self.hparams.num_layers, 1, 1),
                hidden_decoder[1].repeat(self.hparams.num_layers, 1, 1),
            )

        decoder_input_features = agent_features.narrow(
            dim=1, start=agent_features.size(1) - 1, length=1
        )
        last_n_predictions = []
        for i in self.hparams.xy_kernel_list:
            last_n_predictions.append(
                agent_features.narrow(dim=1, start=agent_features.size(1) - i, length=i)
            )
        prediction_tensor, waypoints_prediction_tensor, prediction_stats = self.decoder(
            decoder_input_features,
            last_n_predictions,
            hidden_decoder,
            outsteps,
            ifc_helpers=ifc_helpers,
            sample_next=sample_next,
            map_estimate=map_estimate,
        )
        return (
            prediction_tensor,
            [waypoints_prediction_tensor, waypoint_predictions_tensor_encoder],
            prediction_stats,
        )


class Hparams:
    """Plain namespace mirroring `WIMP.add_model_specific_args` argparse
    defaults (mechanical config-plumbing substitution for the CLI parser,
    same field names/defaults as the source)."""

    def __init__(self):
        self.hidden_dim = 16
        self.input_dim = 2
        self.output_dim = 2
        self.graph_iter = 1
        self.attention_heads = 2
        self.num_layers = 2
        self.hidden_transform = False
        self.use_centerline_features = True
        self.num_mixtures = 1
        self.output_prediction = True
        self.output_conv = False
        self.non_linearity = "tanh"
        self.batch_norm = False
        self.hidden_key_generator = False
        self.add_centerline = False
        self.waypoint_step = 2
        self.segment_CL = False
        self.segment_CL_Encoder = False
        self.segment_CL_Encoder_Gaussian = False
        self.segment_CL_Prob = False
        self.segment_CL_Encoder_Prob = False
        self.segment_CL_Encoder_Gaussian_Prob = False
        self.segment_CL_Gaussian_Prob = False
        self.lr = 0.0001
        self.weight_decay = 0.0
        self.dropout = 0.0
        self.k_value_threshold = 5
        self.k_values = [6, 5, 4, 3, 2, 1]
        self.gradient_clipping = False
        self.scheduler_step_size = [30, 60, 90, 120, 150]
        self.wta = False
        self.predict_delta = False
        self.distributed_backend = None


def build_wimp():
    hparams = Hparams()
    model = WIMP(hparams)
    if torch.cuda.is_available():
        model = model.cuda()
    model.eval()
    return model


def example_input_wimp():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch = 2
    n_social = 2
    n_timesteps = 8
    cl_len = 10

    agent_features = torch.randn(batch, n_timesteps, 2, device=device)
    social_features = torch.randn(batch, n_social, n_timesteps, 2, device=device)
    adjacency = torch.ones(batch, n_social + 1, n_social + 1, device=device)
    num_agent_mask = torch.ones(batch, n_social + 1, device=device)

    agent_oracle_centerline = torch.randn(batch, cl_len, 2, device=device)
    agent_oracle_centerline_lengths = torch.full(
        (batch,), cl_len, dtype=torch.float32, device=device
    )
    social_oracle_centerline = torch.randn(batch, n_social, cl_len, 2, device=device)
    social_oracle_centerline_lengths = torch.full(
        (batch, n_social), cl_len, dtype=torch.float32, device=device
    )

    ifc_helpers = {
        "agent_oracle_centerline": agent_oracle_centerline,
        "agent_oracle_centerline_lengths": agent_oracle_centerline_lengths,
        "social_oracle_centerline": social_oracle_centerline,
        "social_oracle_centerline_lengths": social_oracle_centerline_lengths,
    }

    return (
        agent_features,
        social_features,
        adjacency,
        num_agent_mask,
        4,
        None,
        None,
        True,
        None,
        ifc_helpers,
    )


MENAGERIE_ZOO = "vendored-pytorch"
MENAGERIE_ENTRIES = [
    ("WIMP", "build_wimp", "example_input_wimp", 2020, "vendored"),
]
