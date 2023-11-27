# Copyright 2020 The Q2 Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

#!/bin/bash

# CUDA_VISIBLE_DEVICES=2 python pipeline/Q2_metric.py \
#       --infile third_party/data/CICERO/CICERO_gen_dh_best.csv \
#       --outfile results/CICERO_gen_dh_best/CICERO_gen_dh_best_out.csv \
#       --save_steps

# CUDA_VISIBLE_DEVICES=2 python pipeline/Q2_metric.py \
#       --infile third_party/data/CICERO/CICERO_gen_dh_baseline.csv \
#       --outfile results/CICERO_gen_dh_baseline/CICERO_gen_dh_baseline_out.csv \
#       --save_steps
      
# CUDA_VISIBLE_DEVICES=2 python pipeline/Q2_metric.py \
#       --infile third_party/data/CICERO/CICERO_gold_dh.csv \
#       --outfile results/CICERO_gold_dh/CICERO_gold_dh_out.csv \
#       --save_steps
      
CUDA_VISIBLE_DEVICES=2 python pipeline/Q2_metric.py \
      --infile third_party/data/CICERO/CICERO_target_dh.csv \
      --outfile results/CICERO_target_dh/CICERO_target_dh_out.csv \
      --save_steps