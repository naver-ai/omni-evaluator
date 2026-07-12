# OmniEvaluator
# Copyright (c) 2026-present NAVER Cloud Corp.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

prompt_en = """## Task
You are given an instruction, a dialogue, and two candidate responses for comparison.
The dialogue may include multimodal elements such as images, videos, or audio clips.  

1. Choose the response (A or B) that better satisfies the given instruction, and assign an integer score between **1 and 5** to each response.  
2. The final **choice** should align with the higher overall rating.
3. The output format should be exactly as follows: 
Rate:
{score_format}
Choice: [A|B]
5. Do **not** include any additional explanations, introductions, or closing remarks.

## Instruction
Refer to the following user utterance or dialogue to generate an appropriate response.  
If the dialogue includes multimodal content such as images, videos, or audio, please consider them as part of the context.  
{instruction}

## Dialogue:
{dialogue}

## Candidates to compare:
A: {response_a}

B: {response_b}

## Result:
"""

prompt_ko = """## Task
하나의 **지시문(instruction)**, **대화(dialogue)**, 그리고 비교할 **두 개의 후보 응답(candidate A/B)**이 주어집니다.  
대화에는 이미지, 비디오, 오디오 등 **멀티모달 요소**가 포함될 수 있습니다.  

1. 주어진 지시문과 평가기준에 따라 두 후보 중 더 적절한 응답(A 또는 B)을 선택하고, 각 응답에 대해 **1~5 사이의 정수 점수**를 부여하세요.  
2. 최종 선택(Choice)은 점수 결과에 따라 **더 높은 총점**을 받은 후보로 해야 합니다.  
3. 출력 형식은 아래 예시와 정확히 동일해야 합니다. 
Rate:
{score_format}
Choice: [A|B]
4. 모든 답변은 한국어로 작성하고, 그 외의 **추가 설명, 서론, 결론 문장**은 작성하지 마세요.

## Instruction
아래 사용자의 발화 혹은 대화를 참고하여 적절한 응답을 생성해야 합니다.  
대화에 이미지, 비디오, 오디오 등의 데이터가 포함되어 있다면 함께 고려해 주세요.  
{instruction}

### Dialogue:
{dialogue}

### Candidates to compare:
A: {response_a}

B: {response_b}

## Result:
"""