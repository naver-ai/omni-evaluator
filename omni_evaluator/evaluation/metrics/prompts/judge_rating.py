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
You are given an *instruction*, a *dialogue*, a *desired response*, and a *candidate response*.  
The dialogue may include multimodal elements such as images, videos, or audio clips.

1. Evaluate the *candidate response* based on the given *instruction*, and assign an **integer score from 1 to 5**.  
2. Use desired response as a reference to evaluate *the candidate response* according to the *instruction*.
3. The output format must be **exactly** as follows: 
Rate:
{score_format}
4. Do **not** include any additional explanations, reasoning, introductions, or closing statements.

## Instruction
Refer to the following user utterance or dialogue to generate an appropriate response.  
If the dialogue includes multimodal content (e.g., images, videos, or audio), consider these as part of the context.  
{instruction}

## Dialogue
{dialogue}

## Candidate Response
{response}

## Result
"""

prompt_ko = """## Task
당신에게는 *지시문(instruction)*, *대화(dialogue)*, *모범 응답(desired response)*, 그리고 *평가 대상 응답(candidate response)*이 주어집니다.  
대화에는 이미지, 동영상, 오디오 등과 같은 멀티모달 요소가 포함될 수 있습니다.

1. 주어진 *지시문(instruction)*을 기반으로 *평가 대상 응답(candidate response)*을 평가하고, **1점부터 5점 사이의 정수 점수**를 부여하세요.  
2. 점수를 부여하기 전, 판단 근거를 주어진 형식에 따라 남기고, 해당 근거에 맞는 점수를 부여해주세요.
3. 이 응답을 기준으로 *지시문(instruction)*을 참조하여 *평가 대상 응답(candidate response)*의 점수를 메겨주세요.
4. 출력 형식은 반드시 다음과 같아야 합니다.
Reason:
{reason_format}
Rate:
{score_format}
6. 평가 이유와 평가 점수 외에 **추가적인 설명, 근거, 도입부, 마무리 문장**을 포함하지 마세요.

## Instruction
다음의 사용자 발화나 대화를 참고하여 적절한 응답을 생성한다고 가정하세요.  
대화에 이미지, 동영상, 오디오 등 멀티모달 콘텐츠가 포함된 경우, 이를 문맥의 일부로 고려해야 합니다.  
{instruction}

## Dialogue
{dialogue}

## Candidate Response
{response}

## Result
"""

prompt_example_en = """## Task
You are given an *instruction*, a *dialogue*, a *desired response*, and a *candidate response*.  
The dialogue may include multimodal elements such as images, videos, or audio clips.

1. Evaluate the *candidate response* based on the given *instruction*, and assign an **integer score from 1 to 5**.  
2. The *desired response* represents an ideal example that fully satisfies both the *instruction*, corresponding to a **score of 5**.
3. Use desired response as a reference to evaluate *the candidate response* according to the *instruction*.
4. The output format must be **exactly** as follows: 
Rate:
{score_format}
5. Do **not** include any additional explanations, reasoning, introductions, or closing statements.

## Instruction
Refer to the following user utterance or dialogue to generate an appropriate response.  
If the dialogue includes multimodal content (e.g., images, videos, or audio), consider these as part of the context.  
{instruction}

## Dialogue
{dialogue}

## Desired Response
{example}

## Candidate Response
{response}

## Result
"""

prompt_reference_ko = """## Task
당신에게는 *지시문(instruction)*, *대화(dialogue)*, *모범 응답(desired response)*, 그리고 *평가 대상 응답(candidate response)*이 주어집니다.  
대화에는 이미지, 동영상, 오디오 등과 같은 멀티모달 요소가 포함될 수 있습니다.

1. 주어진 *지시문(instruction)*을 기반으로 *평가 대상 응답(candidate response)*을 평가하고, **1점부터 5점 사이의 정수 점수**를 부여하세요.  
2. 점수를 부여하기 전, 판단 근거를 주어진 형식에 따라 남기고, 해당 근거에 맞는 점수를 부여해주세요.
3. *모범 응답(desired response)*은 *지시문(instruction)*의 기준을 모두 완벽하게 충족하는 이상적인 예시로, **5점에 해당하는 응답**입니다.
4. 이 응답을 기준으로 *지시문(instruction)*을 참조하여 *평가 대상 응답(candidate response)*의 점수를 메겨주세요.
5. 출력 형식은 반드시 다음과 같아야 합니다.
Reason:
{reason_format}
Rate:
{score_format}
6. 평가 이유와 평가 점수 외에 **추가적인 설명, 근거, 도입부, 마무리 문장**을 포함하지 마세요.

## Instruction
다음의 사용자 발화나 대화를 참고하여 적절한 응답을 생성한다고 가정하세요.  
대화에 이미지, 동영상, 오디오 등 멀티모달 콘텐츠가 포함된 경우, 이를 문맥의 일부로 고려해야 합니다.  
{instruction}

## Dialogue
{dialogue}

## Desired Response
{example}

## Candidate Response
{response}

## Result
"""