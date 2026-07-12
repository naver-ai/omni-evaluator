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

def process_docs(dataset):
    """Process KMMLU-Pro dataset - generate prompts with dynamic options."""
    def _process_doc(doc):
        options = doc.get('options', [])
        solution = int(doc.get('solution', 1))
        
        # Generate options prompt dynamically
        option_labels = ['A', 'B', 'C', 'D', 'E']
        options_text = '\n'.join([f"{label}) {opt}" for label, opt in zip(option_labels[:len(options)], options)])
        
        prompt = f"다음 객관식 문제에 답하세요. 응답의 마지막 줄은 다음 형식이어야 합니다: '정답: {'/'.join(option_labels[:len(options)])}' (예: '정답: A').\n\n{doc.get('question', '').strip()}\n{options_text}"
        
        # Get target answer
        target = option_labels[solution - 1] if solution <= len(options) else 'A'
        
        return {
            **doc,
            'prompt': prompt,
            'target': target,
        }
    
    return dataset.map(_process_doc)


def process_results(doc, results):
    """Basic exact match for KMMLU-Pro."""
    return {
        "exact_match": int(results[0].strip() == doc['target']),
    }
