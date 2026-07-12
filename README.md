[한국어](./README.ko.md)

<div align="center">

# OmniEvaluator

**Run 2,000+ LLM/VLM benchmarks — across text, image, video, and audio — with a single command.**

OmniEvaluator wires together **4 inference backends** and **5 evaluation frameworks** behind one CLI, so you don't have to learn (or reimplement) each framework separately. Existing evaluators stay interoperable through a shared intermediate schema, and every run emits a provenance-rich artifact that captures the full configuration needed to reproduce it.

[![Demo](https://img.shields.io/badge/Live%20Demo-omni--evaluator.info-blue)](https://omni-evaluator.info)
[![Video](https://img.shields.io/badge/Demo%20Video-GitHub-black)](https://github.com/naver-ai/omni-evaluator/tree/main/demo)

</div>

---

## Overview

Evaluating a model today usually means juggling several incompatible benchmark suites, each with its own data format, runner, and quirks. OmniEvaluator removes that friction.

- **Omni-modality** Text, image, video, and audio benchmarks — 2,000+ tasks in total ([full catalog](docs/TASKS.md)).
- **Framework compatibility** Pick an inference backend (`huggingface`, `vllm`, `sglang`, or an api client such as `api/openai`, `api/gemini`, `api/claude`) and an evaluation engine (`builtin`, `lmms_eval`, `lm_eval_harness`, `vlm_eval_kit`) — any combination runs from the same command shape.
- **Code reusability** Existing evaluators are reused as-is and made interoperable through a shared intermediate schema.
- **Reproducibility** Each run writes a self-describing artifact with the exact configuration, so any result can be reproduced later.

---

## Supported Tasks

OmniEvaluator offers **2,820+ benchmarks** across four evaluation engines and five modalities. Counts are **based on base benchmarks and exclude variant counts**. For example, evaluation-protocol variants (`_zeroshot`, `_cot`, `_n_shot`, `_generative`, `_fewshot`, `_direct`, …) and video-frame variants (`_8frames`, `_64frames`, `_128frames`) collapse into a single entry. Conversely, subset markers that identify a different benchmark (`_pro`, `_redux`, `_hard`, `_diamond`, `_extended`, `_main`, `_mini`, `_v2`) are preserved as independent entries.

### By evaluation engine × modality

| Evaluation Engine | Text | Image | Audio | Video | **Total** |
|-------------------|-----:|------:|------:|------:|----------:|
| `builtin` | 5 | 13 | 25 | — | **43** |
| `lmms_eval` | — | 401 | 5 | 10 | **416** |
| `lm_eval_harness` | 1,986 | — | — | — | **1,986** |
| `vlm_eval_kit` | — | 368 | — | 7 | **375** |
| **All engines** | **1,991** | **782** | **30** | **17** | **2,820** |


```bash
# List supported benchmarks
python run.py list --tasks --evaluation_engine="<engine_name>"
```

### Per-engine full task lists

<details>
<summary><b><code>builtin</code></b> — 43 benchmarks • Vision, document understanding, math, multi-image, audio, video, Korean benchmarks</summary>

`charxiv_descriptive_validation`, `charxiv_reasoning_validation`, `clotho_aqa_test`, `cochlscene_test`
`common_voice_15_en_test`, `covost2_en2zh_test`, `covost2_zh2en_test`, `cruxeval_o_test`
`fleurs_en2ko_test`, `fleurs_en2zh_test`, `fleurs_en_test`, `fleurs_ko2en_test`
`fleurs_ko_test`, `fleurs_zh2en_test`, `fleurs_zh_test`, `gtzan_test`
`haerae_vision_dev`, `hike_test`, `hle_test`, `hmmt_nov_2025_test`
`imo_answer_bench_test`, `kdtcbench_test`, `kmmbench_dev`, `kmmstar_validation`
`kseed_test`, `librispeech_test_clean`, `librispeech_test_other`, `m3cot_test`
`meld_emotion_test`, `meld_sentiment_test`, `mmau_pro_test`, `mmau_test`
`mmau_test_mini`, `mmmu_validation`, `mmvet_test`, `mmvet_v2_test`
`mu_chomusic_test`, `omni_bench_test`, `polymath_en_test`, `vlms_are_biased_test`
`vocal_sound_test`, `voice_bench_test`, `zebralogic_mc_test`

</details>

<details>
<summary><b><code>lmms_eval</code></b> — 416 benchmarks • Multimodal benchmarks ([lmms-eval](https://github.com/EvolvingLMMs-Lab/lmms-eval))</summary>

`3dsrbench`, `ConBench`, `FALCONBench_mcq`, `FALCONBench_oq`
`VisualPuzzles`, `abench_dev`, `activitynetqa`, `ai2_arc`
`ai2d`, `ai2d_lite`, `aime24_agg8_reasoning`, `aime24_aldata_al_agg8`
`aime24_figures`, `aime24_nofigures`, `aime25_agg8_reasoning`, `aime25_aldata_al_agg8`
`aime25_nofigures`, `aime_2024_agg8`, `aime_2024_rebase`, `aime_figures`
`aime_nofigures`, `aime_reasoning`, `air_bench_chat`, `air_bench_foundation`
`alpaca_audio`, `amber_g`, `ami`, `anet_qa`
`arc_agi_1`, `arc_agi_2`, `arc_challenge`, `arc_easy`
`auxsolidmath`, `auxsolidmath_easy`, `auxsolidmath_hard`, `av_asr`
`av_odyssey`, `av_speakerbench_audio`, `av_speakerbench_audiovisual`, `av_speakerbench_visual`
`babyvision`, `background_test`, `blink`, `browsecomp`
`camera_test`, `camerabench_vqa`, `capability`, `captionqa`
`chain_of_thought`, `chartqa`, `chartqa_lite`, `chartqapro`
`charxiv`, `cinepile`, `clotho_aqa`, `clotho_asqa_test_v2`
`cmmmu`, `cn_college_listen_mcq_test`, `coco2014_cap`, `coco2017_cap`
`coco_cap`, `coco_caption`, `coco_caption2017`, `coco_karpathy`
`common_voice_15`, `corecognition`, `countbench`, `countbenchqa`
`countix`, `covost2`, `csbench`, `cuva`
`cv_bench`, `cvrr`, `d170_cn`, `d170_en`
`dc100_en`, `dc200_cn`, `detailcaps`, `detailed_test`
`docvqa`, `dream_tts_mcq_test`, `dtcbench`, `dude`
`dynamath_reasoning`, `egoplan`, `egosch_a`, `egoschema`
`egotempo`, `egothink`, `embspatial`, `emma`
`emma-mini`, `erqa`, `europal_asr`, `ferret`
`fgqa_test`, `fleurs`, `flickr30k`, `fsc147`
`funqa`, `gedit_bench`, `geometry3k`, `gigaspeech`
`gpqa`, `gpqa_diamond`, `gpqa_extended`, `gpqa_main`
`gqa`, `gqa-ru`, `gqa_lite`, `groundingme`
`gsm8k`, `hallusion_bench_image`, `hd_epic_vqa`, `hellaswag`
`hipho`, `hrbench`, `hrbench4k`, `hrbench8k`
`iconqa`, `ifeval`, `ii-bench`, `illusionbench`
`illusionvqa`, `imgedit`, `infovqa`
`jailbreak`, `jmmmu`, `jmmmu_pro`, `k12`
`kris_bench`, `lemonade`, `librispeech`, `live_bench`
`livexiv_tqa`, `livexiv_tqa_v2`, `livexiv_tqa_v3`, `livexiv_vqa`
`livexiv_vqa_v2`, `livexiv_vqa_v3`, `llava_bench_coco`, `llava_in_the_wild`
`llava_interleave_bench`, `llava_wilder_small`, `logicvista_reasoning`, `longtimescope`
`longvideobench_no_visual`, `longvideobench_random_choice`, `longvideobench_test_i`, `longvideobench_test_v`
`longvideobench_val_i`, `longvideobench_val_v`, `longvt`, `lsdbench`
`lvbench`, `main_object_test`, `mantis`, `math_word_problems`
`mathcanvas`, `mathkangaroo`, `mathverse`, `mathvision_reason_test`
`mathvision_reason_testmini`, `mathvision_test`, `mathvision_testmini`, `mathvista`
`medqa`, `megabench`, `mia_bench`, `mindcube_full`
`mindcube_tiny`, `minerva`, `mirb`, `mix_evals_audio2text`
`mix_evals_audio2text_hard`, `mix_evals_image2text`, `mix_evals_image2text_hard`, `mix_evals_video2text`
`mix_evals_video2text_hard`, `mle_bench`, `mlvu_dev`, `mlvu_test`
`mmaad_base`, `mmaad_instruction`, `mmaad_option`, `mmar`
`mmau`, `mmbench`, `mme`, `mmerealworld`
`mmerealworld_lite`, `mmiasd_base`, `mmiasd_instruction`, `mmiasd_option`
`mmie`, `mmivqd_base`, `mmivqd_instruction`, `mmivqd_option`
`mmlongbench`, `mmlu`, `mmlu_pro`, `mmmu`
`mmmu_pro`, `mmrefine`, `mmsearch`, `mmsi_bench`
`mmsi_video`, `mmstar`, `mmsu`, `mmt`
`mmupd`, `mmvet`, `mmvetv2`, `mmvp`
`mmvu_val`, `mmworld`, `motionbench`, `motionbench_full`
`moviechat_breakpoint`, `moviechat_global`, `mtvqa`, `muchomusic`
`muirbench`, `multidocvqa`, `multimodal_rewardbench`, `multiple_choice`
`mvbench`, `naturalbench`, `naverclip`, `neptune`
`neptune_full_i`, `neptune_full_v`, `nextqa`, `nocaps`
`o3`, `ocrbench`, `ocrbench_v2`, `officeqa`
`ok_vqa`, `olympiadbench`, `omni_bench`, `omnidocbench`
`omnispatial_test`, `open_asr_ami`, `open_asr_common_voice`, `open_asr_earnings22`
`open_asr_gigaspeech`, `open_asr_librispeech_test_clean`, `open_asr_librispeech_test_other`, `open_asr_spgispeech`
`open_asr_tedlium`, `open_asr_voxpopuli`, `openai_math`, `openasr`
`openhermes`, `openslr_librispeech`, `osi_bench`, `osworld_g`
`ovo_backward`, `ovo_forward`, `ovo_realtime`, `ovobench`
`ovr_kinetics`, `p3`, `paibench_u`, `people_speech_val`
`perceptiontest_test_mc`, `perceptiontest_test_mcppl`, `perceptiontest_val_mc`, `perceptiontest_val_mcppl`
`phyx`, `phyx_mini_mc`, `phyx_mini_oe`, `pixmo_count`
`pointbench`, `pope`, `pope_full`, `prismm_bench_identification`
`prismm_bench_pair_match`, `prismm_bench_remedy`, `public_eval_gemini3_family`, `public_eval_gpt5_family`
`public_eval_qwen3_5_family`, `public_eval_seed2_family`, `qbench2_dev`, `qbench_dev`
`qbenchs_dev`, `rcap_test`, `rdcap_test`, `realunify`
`realworldqa`, `redteam`, `refcoco`, `refcoco+`
`refcocog`, `refspatial`, `repcount`, `rewardbench`
`rtloc_test`, `saco_gold`, `safety`, `scibench`
`scienceqa`, `scienceqa_full`, `scivideobench`, `screenspot`
`screenspot_pro`, `screenspot_v2`, `seedbench`, `seedbench_lite`
`seephys`, `self_consistency`, `sgqa_test`, `short_test`
`simplevqa`, `site_bench_image`, `site_bench_video`, `snsbench`
`song_describer`, `sparbench`, `sparbench_tiny`, `spatial457`
`spatialtreebench`, `spatialviz_full`, `ssv2`, `stare_2d_text_instruct`
`stare_2d_va`, `stare_3d_text_instruct`, `stare_3d_va`, `stare_folding_nets`
`stare_full`, `stare_perspective`, `stare_tangram_puzzle`, `stare_temporal`
`step2_audio_paralinguistic`, `structeditbench`, `stvqa`, `super_gpqa`
`synthdog`, `tau2_bench_telecom`, `tedlium_dev_test`, `tedlium_long_form`
`tempcompass`, `temporal_grounding_charades`, `temporalbench`, `textcaps`
`textvqa`, `timescope`, `tomato`, `tvbench`
`ueval`, `uni_mmmu`, `vatex`, `vcr_wiki_en_easy`
`vcr_wiki_en_hard`, `vcr_wiki_zh_easy`, `vcr_wiki_zh_hard`, `vending_bench2`
`vggsound`, `vibe_eval`, `video_dc499`, `video_mmmu`
`video_qa`, `videochatgpt`, `videoevalpro`, `videomathqa_mbin`
`videomathqa_mcq`, `videomme`, `videott_all`, `videott_correctly_led_oe`
`videott_no_leading_oe`, `videott_paraphrase_oe`, `videott_single_mc`, `videott_wrongly_led_oe`
`viewspatial`, `vinoground`, `visres_bench`, `visual_reasoning_collection`
`visualwebbench_action_ground`, `visualwebbench_action_prediction`, `visualwebbench_element_ground`, `visualwebbench_element_ocr`
`visualwebbench_heading_ocr`, `visualwebbench_web_caption`, `visualwebbench_webqa`, `visulogic`
`vitatecs`, `viverbench`, `vizwiz_vqa`, `vl_rewardbench`
`vlms_are_biased`, `vlmsareblind`, `vlmsareblind_lite`, `vmcbench`
`vocalsound_test`, `vocalsound_val`, `voicebench`, `voxpopuli`
`vpct`, `vqav2`, `vsibench`, `vstar_bench`
`wavcaps`, `websrc`, `wemath_testmini_reasoning`, `wenet_speech`
`where2place`, `wildvision`, `worldqa`, `worldsense`
`worldvqa`, `xlrs-lite`, `youcook2_val`, `zerobench`

</details>

<details>
<summary><b><code>lm_eval_harness</code></b> — 1,986 benchmarks • Text-only benchmarks ([lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness))</summary>

`20_newsgroups`, `AraDiCE`, `RC_tasks`, `aclue`
`acp_app_bool`, `acp_app_gen`, `acp_app_mcq`, `acp_areach_bool`
`acp_areach_gen`, `acp_areach_mcq`, `acp_bench`, `acp_bench_hard`
`acp_bool_cot_2shot`, `acp_gen_2shot`, `acp_just_bool`, `acp_just_gen`
`acp_just_mcq`, `acp_land_bool`, `acp_land_gen`, `acp_land_mcq`
`acp_mcq_cot_2shot`, `acp_nexta_gen`, `acp_prog_bool`, `acp_prog_gen`
`acp_prog_mcq`, `acp_reach_bool`, `acp_reach_gen`, `acp_reach_mcq`
`acp_val_bool`, `acp_val_gen`, `acp_val_mcq`, `adr`
`advanced_ai_risk`, `aexams`, `african_flores`, `african_ntrex`
`afridiacritics_bbj_prompt_1`, `afridiacritics_bbj_prompt_2`, `afridiacritics_bbj_prompt_3`, `afridiacritics_bbj_prompt_4`
`afridiacritics_bbj_prompt_5`, `afridiacritics_fon_prompt_1`, `afridiacritics_fon_prompt_2`, `afridiacritics_fon_prompt_3`
`afridiacritics_fon_prompt_4`, `afridiacritics_fon_prompt_5`, `afridiacritics_ibo_prompt_1`, `afridiacritics_ibo_prompt_2`
`afridiacritics_ibo_prompt_3`, `afridiacritics_ibo_prompt_4`, `afridiacritics_ibo_prompt_5`, `afridiacritics_wol_prompt_1`
`afridiacritics_wol_prompt_2`, `afridiacritics_wol_prompt_3`, `afridiacritics_wol_prompt_4`, `afridiacritics_wol_prompt_5`
`afridiacritics_yor_prompt_1`, `afridiacritics_yor_prompt_2`, `afridiacritics_yor_prompt_3`, `afridiacritics_yor_prompt_4`
`afridiacritics_yor_prompt_5`, `afrimgsm-irokobench`, `afrimgsm_amh_prompt_1`, `afrimgsm_amh_prompt_2`
`afrimgsm_amh_prompt_3`, `afrimgsm_amh_prompt_4`, `afrimgsm_amh_prompt_5`, `afrimgsm_cot-irokobench`
`afrimgsm_cot_amh_prompt_1`, `afrimgsm_cot_amh_prompt_2`, `afrimgsm_cot_amh_prompt_3`, `afrimgsm_cot_amh_prompt_4`
`afrimgsm_cot_amh_prompt_5`, `afrimgsm_cot_eng_prompt_1`, `afrimgsm_cot_eng_prompt_2`, `afrimgsm_cot_eng_prompt_3`
`afrimgsm_cot_eng_prompt_4`, `afrimgsm_cot_eng_prompt_5`, `afrimgsm_cot_ewe_prompt_1`, `afrimgsm_cot_ewe_prompt_2`
`afrimgsm_cot_ewe_prompt_3`, `afrimgsm_cot_ewe_prompt_4`, `afrimgsm_cot_ewe_prompt_5`, `afrimgsm_cot_fra_prompt_1`
`afrimgsm_cot_fra_prompt_2`, `afrimgsm_cot_fra_prompt_3`, `afrimgsm_cot_fra_prompt_4`, `afrimgsm_cot_fra_prompt_5`
`afrimgsm_cot_hau_prompt_1`, `afrimgsm_cot_hau_prompt_2`, `afrimgsm_cot_hau_prompt_3`, `afrimgsm_cot_hau_prompt_4`
`afrimgsm_cot_hau_prompt_5`, `afrimgsm_cot_ibo_prompt_1`, `afrimgsm_cot_ibo_prompt_2`, `afrimgsm_cot_ibo_prompt_3`
`afrimgsm_cot_ibo_prompt_4`, `afrimgsm_cot_ibo_prompt_5`, `afrimgsm_cot_kin_prompt_1`, `afrimgsm_cot_kin_prompt_2`
`afrimgsm_cot_kin_prompt_3`, `afrimgsm_cot_kin_prompt_4`, `afrimgsm_cot_kin_prompt_5`, `afrimgsm_cot_lin_prompt_1`
`afrimgsm_cot_lin_prompt_2`, `afrimgsm_cot_lin_prompt_3`, `afrimgsm_cot_lin_prompt_4`, `afrimgsm_cot_lin_prompt_5`
`afrimgsm_cot_lug_prompt_1`, `afrimgsm_cot_lug_prompt_2`, `afrimgsm_cot_lug_prompt_3`, `afrimgsm_cot_lug_prompt_4`
`afrimgsm_cot_lug_prompt_5`, `afrimgsm_cot_orm_prompt_1`, `afrimgsm_cot_orm_prompt_2`, `afrimgsm_cot_orm_prompt_3`
`afrimgsm_cot_orm_prompt_4`, `afrimgsm_cot_orm_prompt_5`, `afrimgsm_cot_sna_prompt_1`, `afrimgsm_cot_sna_prompt_2`
`afrimgsm_cot_sna_prompt_3`, `afrimgsm_cot_sna_prompt_4`, `afrimgsm_cot_sna_prompt_5`, `afrimgsm_cot_sot_prompt_1`
`afrimgsm_cot_sot_prompt_2`, `afrimgsm_cot_sot_prompt_3`, `afrimgsm_cot_sot_prompt_4`, `afrimgsm_cot_sot_prompt_5`
`afrimgsm_cot_swa_prompt_1`, `afrimgsm_cot_swa_prompt_2`, `afrimgsm_cot_swa_prompt_3`, `afrimgsm_cot_swa_prompt_4`
`afrimgsm_cot_swa_prompt_5`, `afrimgsm_cot_tasks`, `afrimgsm_cot_translate_amh_prompt_1`, `afrimgsm_cot_translate_amh_prompt_2`
`afrimgsm_cot_translate_amh_prompt_3`, `afrimgsm_cot_translate_amh_prompt_4`, `afrimgsm_cot_translate_amh_prompt_5`, `afrimgsm_cot_translate_ewe_prompt_1`
`afrimgsm_cot_translate_ewe_prompt_2`, `afrimgsm_cot_translate_ewe_prompt_3`, `afrimgsm_cot_translate_ewe_prompt_4`, `afrimgsm_cot_translate_ewe_prompt_5`
`afrimgsm_cot_translate_fra_prompt_1`, `afrimgsm_cot_translate_fra_prompt_2`, `afrimgsm_cot_translate_fra_prompt_3`, `afrimgsm_cot_translate_fra_prompt_4`
`afrimgsm_cot_translate_fra_prompt_5`, `afrimgsm_cot_translate_hau_prompt_1`, `afrimgsm_cot_translate_hau_prompt_2`, `afrimgsm_cot_translate_hau_prompt_3`
`afrimgsm_cot_translate_hau_prompt_4`, `afrimgsm_cot_translate_hau_prompt_5`, `afrimgsm_cot_translate_ibo_prompt_1`, `afrimgsm_cot_translate_ibo_prompt_2`
`afrimgsm_cot_translate_ibo_prompt_3`, `afrimgsm_cot_translate_ibo_prompt_4`, `afrimgsm_cot_translate_ibo_prompt_5`, `afrimgsm_cot_translate_kin_prompt_1`
`afrimgsm_cot_translate_kin_prompt_2`, `afrimgsm_cot_translate_kin_prompt_3`, `afrimgsm_cot_translate_kin_prompt_4`, `afrimgsm_cot_translate_kin_prompt_5`
`afrimgsm_cot_translate_lin_prompt_1`, `afrimgsm_cot_translate_lin_prompt_2`, `afrimgsm_cot_translate_lin_prompt_3`, `afrimgsm_cot_translate_lin_prompt_4`
`afrimgsm_cot_translate_lin_prompt_5`, `afrimgsm_cot_translate_lug_prompt_1`, `afrimgsm_cot_translate_lug_prompt_2`, `afrimgsm_cot_translate_lug_prompt_3`
`afrimgsm_cot_translate_lug_prompt_4`, `afrimgsm_cot_translate_lug_prompt_5`, `afrimgsm_cot_translate_orm_prompt_1`, `afrimgsm_cot_translate_orm_prompt_2`
`afrimgsm_cot_translate_orm_prompt_3`, `afrimgsm_cot_translate_orm_prompt_4`, `afrimgsm_cot_translate_orm_prompt_5`, `afrimgsm_cot_translate_sna_prompt_1`
`afrimgsm_cot_translate_sna_prompt_2`, `afrimgsm_cot_translate_sna_prompt_3`, `afrimgsm_cot_translate_sna_prompt_4`, `afrimgsm_cot_translate_sna_prompt_5`
`afrimgsm_cot_translate_sot_prompt_1`, `afrimgsm_cot_translate_sot_prompt_2`, `afrimgsm_cot_translate_sot_prompt_3`, `afrimgsm_cot_translate_sot_prompt_4`
`afrimgsm_cot_translate_sot_prompt_5`, `afrimgsm_cot_translate_swa_prompt_1`, `afrimgsm_cot_translate_swa_prompt_2`, `afrimgsm_cot_translate_swa_prompt_3`
`afrimgsm_cot_translate_swa_prompt_4`, `afrimgsm_cot_translate_swa_prompt_5`, `afrimgsm_cot_translate_twi_prompt_1`, `afrimgsm_cot_translate_twi_prompt_2`
`afrimgsm_cot_translate_twi_prompt_3`, `afrimgsm_cot_translate_twi_prompt_4`, `afrimgsm_cot_translate_twi_prompt_5`, `afrimgsm_cot_translate_vai_prompt_1`
`afrimgsm_cot_translate_vai_prompt_2`, `afrimgsm_cot_translate_vai_prompt_3`, `afrimgsm_cot_translate_vai_prompt_4`, `afrimgsm_cot_translate_vai_prompt_5`
`afrimgsm_cot_translate_wol_prompt_1`, `afrimgsm_cot_translate_wol_prompt_2`, `afrimgsm_cot_translate_wol_prompt_3`, `afrimgsm_cot_translate_wol_prompt_4`
`afrimgsm_cot_translate_wol_prompt_5`, `afrimgsm_cot_translate_xho_prompt_1`, `afrimgsm_cot_translate_xho_prompt_2`, `afrimgsm_cot_translate_xho_prompt_3`
`afrimgsm_cot_translate_xho_prompt_4`, `afrimgsm_cot_translate_xho_prompt_5`, `afrimgsm_cot_translate_yor_prompt_1`, `afrimgsm_cot_translate_yor_prompt_2`
`afrimgsm_cot_translate_yor_prompt_3`, `afrimgsm_cot_translate_yor_prompt_4`, `afrimgsm_cot_translate_yor_prompt_5`, `afrimgsm_cot_translate_zul_prompt_1`
`afrimgsm_cot_translate_zul_prompt_2`, `afrimgsm_cot_translate_zul_prompt_3`, `afrimgsm_cot_translate_zul_prompt_4`, `afrimgsm_cot_translate_zul_prompt_5`
`afrimgsm_cot_twi_prompt_1`, `afrimgsm_cot_twi_prompt_2`, `afrimgsm_cot_twi_prompt_3`, `afrimgsm_cot_twi_prompt_4`
`afrimgsm_cot_twi_prompt_5`, `afrimgsm_cot_vai_prompt_1`, `afrimgsm_cot_vai_prompt_2`, `afrimgsm_cot_vai_prompt_3`
`afrimgsm_cot_vai_prompt_4`, `afrimgsm_cot_vai_prompt_5`, `afrimgsm_cot_wol_prompt_1`, `afrimgsm_cot_wol_prompt_2`
`afrimgsm_cot_wol_prompt_3`, `afrimgsm_cot_wol_prompt_4`, `afrimgsm_cot_wol_prompt_5`, `afrimgsm_cot_xho_prompt_1`
`afrimgsm_cot_xho_prompt_2`, `afrimgsm_cot_xho_prompt_3`, `afrimgsm_cot_xho_prompt_4`, `afrimgsm_cot_xho_prompt_5`
`afrimgsm_cot_yor_prompt_1`, `afrimgsm_cot_yor_prompt_2`, `afrimgsm_cot_yor_prompt_3`, `afrimgsm_cot_yor_prompt_4`
`afrimgsm_cot_yor_prompt_5`, `afrimgsm_cot_zul_prompt_1`, `afrimgsm_cot_zul_prompt_2`, `afrimgsm_cot_zul_prompt_3`
`afrimgsm_cot_zul_prompt_4`, `afrimgsm_cot_zul_prompt_5`, `afrimgsm_eng_prompt_1`, `afrimgsm_eng_prompt_2`
`afrimgsm_eng_prompt_3`, `afrimgsm_eng_prompt_4`, `afrimgsm_eng_prompt_5`, `afrimgsm_ewe_prompt_1`
`afrimgsm_ewe_prompt_2`, `afrimgsm_ewe_prompt_3`, `afrimgsm_ewe_prompt_4`, `afrimgsm_ewe_prompt_5`
`afrimgsm_fra_prompt_1`, `afrimgsm_fra_prompt_2`, `afrimgsm_fra_prompt_3`, `afrimgsm_fra_prompt_4`
`afrimgsm_fra_prompt_5`, `afrimgsm_hau_prompt_1`, `afrimgsm_hau_prompt_2`, `afrimgsm_hau_prompt_3`
`afrimgsm_hau_prompt_4`, `afrimgsm_hau_prompt_5`, `afrimgsm_ibo_prompt_1`, `afrimgsm_ibo_prompt_2`
`afrimgsm_ibo_prompt_3`, `afrimgsm_ibo_prompt_4`, `afrimgsm_ibo_prompt_5`, `afrimgsm_kin_prompt_1`
`afrimgsm_kin_prompt_2`, `afrimgsm_kin_prompt_3`, `afrimgsm_kin_prompt_4`, `afrimgsm_kin_prompt_5`
`afrimgsm_lin_prompt_1`, `afrimgsm_lin_prompt_2`, `afrimgsm_lin_prompt_3`, `afrimgsm_lin_prompt_4`
`afrimgsm_lin_prompt_5`, `afrimgsm_lug_prompt_1`, `afrimgsm_lug_prompt_2`, `afrimgsm_lug_prompt_3`
`afrimgsm_lug_prompt_4`, `afrimgsm_lug_prompt_5`, `afrimgsm_orm_prompt_1`, `afrimgsm_orm_prompt_2`
`afrimgsm_orm_prompt_3`, `afrimgsm_orm_prompt_4`, `afrimgsm_orm_prompt_5`, `afrimgsm_sna_prompt_1`
`afrimgsm_sna_prompt_2`, `afrimgsm_sna_prompt_3`, `afrimgsm_sna_prompt_4`, `afrimgsm_sna_prompt_5`
`afrimgsm_sot_prompt_1`, `afrimgsm_sot_prompt_2`, `afrimgsm_sot_prompt_3`, `afrimgsm_sot_prompt_4`
`afrimgsm_sot_prompt_5`, `afrimgsm_swa_prompt_1`, `afrimgsm_swa_prompt_2`, `afrimgsm_swa_prompt_3`
`afrimgsm_swa_prompt_4`, `afrimgsm_swa_prompt_5`, `afrimgsm_tasks`, `afrimgsm_translate_amh_prompt_1`
`afrimgsm_translate_amh_prompt_2`, `afrimgsm_translate_amh_prompt_3`, `afrimgsm_translate_amh_prompt_4`, `afrimgsm_translate_amh_prompt_5`
`afrimgsm_translate_ewe_prompt_1`, `afrimgsm_translate_ewe_prompt_2`, `afrimgsm_translate_ewe_prompt_3`, `afrimgsm_translate_ewe_prompt_4`
`afrimgsm_translate_ewe_prompt_5`, `afrimgsm_translate_fra_prompt_1`, `afrimgsm_translate_fra_prompt_2`, `afrimgsm_translate_fra_prompt_3`
`afrimgsm_translate_fra_prompt_4`, `afrimgsm_translate_fra_prompt_5`, `afrimgsm_translate_hau_prompt_1`, `afrimgsm_translate_hau_prompt_2`
`afrimgsm_translate_hau_prompt_3`, `afrimgsm_translate_hau_prompt_4`, `afrimgsm_translate_hau_prompt_5`, `afrimgsm_translate_ibo_prompt_1`
`afrimgsm_translate_ibo_prompt_2`, `afrimgsm_translate_ibo_prompt_3`, `afrimgsm_translate_ibo_prompt_4`, `afrimgsm_translate_ibo_prompt_5`
`afrimgsm_translate_kin_prompt_1`, `afrimgsm_translate_kin_prompt_2`, `afrimgsm_translate_kin_prompt_3`, `afrimgsm_translate_kin_prompt_4`
`afrimgsm_translate_kin_prompt_5`, `afrimgsm_translate_lin_prompt_1`, `afrimgsm_translate_lin_prompt_2`, `afrimgsm_translate_lin_prompt_3`
`afrimgsm_translate_lin_prompt_4`, `afrimgsm_translate_lin_prompt_5`, `afrimgsm_translate_lug_prompt_1`, `afrimgsm_translate_lug_prompt_2`
`afrimgsm_translate_lug_prompt_3`, `afrimgsm_translate_lug_prompt_4`, `afrimgsm_translate_lug_prompt_5`, `afrimgsm_translate_orm_prompt_1`
`afrimgsm_translate_orm_prompt_2`, `afrimgsm_translate_orm_prompt_3`, `afrimgsm_translate_orm_prompt_4`, `afrimgsm_translate_orm_prompt_5`
`afrimgsm_translate_sna_prompt_1`, `afrimgsm_translate_sna_prompt_2`, `afrimgsm_translate_sna_prompt_3`, `afrimgsm_translate_sna_prompt_4`
`afrimgsm_translate_sna_prompt_5`, `afrimgsm_translate_sot_prompt_1`, `afrimgsm_translate_sot_prompt_2`, `afrimgsm_translate_sot_prompt_3`
`afrimgsm_translate_sot_prompt_4`, `afrimgsm_translate_sot_prompt_5`, `afrimgsm_translate_swa_prompt_1`, `afrimgsm_translate_swa_prompt_2`
`afrimgsm_translate_swa_prompt_3`, `afrimgsm_translate_swa_prompt_4`, `afrimgsm_translate_swa_prompt_5`, `afrimgsm_translate_twi_prompt_1`
`afrimgsm_translate_twi_prompt_2`, `afrimgsm_translate_twi_prompt_3`, `afrimgsm_translate_twi_prompt_4`, `afrimgsm_translate_twi_prompt_5`
`afrimgsm_translate_wol_prompt_1`, `afrimgsm_translate_wol_prompt_2`, `afrimgsm_translate_wol_prompt_3`, `afrimgsm_translate_wol_prompt_4`
`afrimgsm_translate_wol_prompt_5`, `afrimgsm_translate_xho_prompt_1`, `afrimgsm_translate_xho_prompt_2`, `afrimgsm_translate_xho_prompt_3`
`afrimgsm_translate_xho_prompt_4`, `afrimgsm_translate_xho_prompt_5`, `afrimgsm_translate_yor_prompt_1`, `afrimgsm_translate_yor_prompt_2`
`afrimgsm_translate_yor_prompt_3`, `afrimgsm_translate_yor_prompt_4`, `afrimgsm_translate_yor_prompt_5`, `afrimgsm_translate_zul_prompt_1`
`afrimgsm_translate_zul_prompt_2`, `afrimgsm_translate_zul_prompt_3`, `afrimgsm_translate_zul_prompt_4`, `afrimgsm_translate_zul_prompt_5`
`afrimgsm_tt-irokobench`, `afrimgsm_tt_cot-irokobench`, `afrimgsm_tt_cot_tasks`, `afrimgsm_tt_tasks`
`afrimgsm_twi_prompt_1`, `afrimgsm_twi_prompt_2`, `afrimgsm_twi_prompt_3`, `afrimgsm_twi_prompt_4`
`afrimgsm_twi_prompt_5`, `afrimgsm_vai_prompt_1`, `afrimgsm_vai_prompt_2`, `afrimgsm_vai_prompt_3`
`afrimgsm_vai_prompt_4`, `afrimgsm_vai_prompt_5`, `afrimgsm_wol_prompt_1`, `afrimgsm_wol_prompt_2`
`afrimgsm_wol_prompt_3`, `afrimgsm_wol_prompt_4`, `afrimgsm_wol_prompt_5`, `afrimgsm_xho_prompt_1`
`afrimgsm_xho_prompt_2`, `afrimgsm_xho_prompt_3`, `afrimgsm_xho_prompt_4`, `afrimgsm_xho_prompt_5`
`afrimgsm_yor_prompt_1`, `afrimgsm_yor_prompt_2`, `afrimgsm_yor_prompt_3`, `afrimgsm_yor_prompt_4`
`afrimgsm_yor_prompt_5`, `afrimgsm_zul_prompt_1`, `afrimgsm_zul_prompt_2`, `afrimgsm_zul_prompt_3`
`afrimgsm_zul_prompt_4`, `afrimgsm_zul_prompt_5`, `afrimmlu-irokobench`, `afrimmlu_direct_amh_prompt_1`
`afrimmlu_direct_amh_prompt_2`, `afrimmlu_direct_amh_prompt_3`, `afrimmlu_direct_amh_prompt_4`, `afrimmlu_direct_amh_prompt_5`
`afrimmlu_direct_eng_prompt_1`, `afrimmlu_direct_eng_prompt_2`, `afrimmlu_direct_eng_prompt_3`, `afrimmlu_direct_eng_prompt_4`
`afrimmlu_direct_eng_prompt_5`, `afrimmlu_direct_ewe_prompt_1`, `afrimmlu_direct_ewe_prompt_2`, `afrimmlu_direct_ewe_prompt_3`
`afrimmlu_direct_ewe_prompt_4`, `afrimmlu_direct_ewe_prompt_5`, `afrimmlu_direct_fra_prompt_1`, `afrimmlu_direct_fra_prompt_2`
`afrimmlu_direct_fra_prompt_3`, `afrimmlu_direct_fra_prompt_4`, `afrimmlu_direct_fra_prompt_5`, `afrimmlu_direct_hau_prompt_1`
`afrimmlu_direct_hau_prompt_2`, `afrimmlu_direct_hau_prompt_3`, `afrimmlu_direct_hau_prompt_4`, `afrimmlu_direct_hau_prompt_5`
`afrimmlu_direct_ibo_prompt_1`, `afrimmlu_direct_ibo_prompt_2`, `afrimmlu_direct_ibo_prompt_3`, `afrimmlu_direct_ibo_prompt_4`
`afrimmlu_direct_ibo_prompt_5`, `afrimmlu_direct_kin_prompt_1`, `afrimmlu_direct_kin_prompt_2`, `afrimmlu_direct_kin_prompt_3`
`afrimmlu_direct_kin_prompt_4`, `afrimmlu_direct_kin_prompt_5`, `afrimmlu_direct_lin_prompt_1`, `afrimmlu_direct_lin_prompt_2`
`afrimmlu_direct_lin_prompt_3`, `afrimmlu_direct_lin_prompt_4`, `afrimmlu_direct_lin_prompt_5`, `afrimmlu_direct_lug_prompt_1`
`afrimmlu_direct_lug_prompt_2`, `afrimmlu_direct_lug_prompt_3`, `afrimmlu_direct_lug_prompt_4`, `afrimmlu_direct_lug_prompt_5`
`afrimmlu_direct_orm_prompt_1`, `afrimmlu_direct_orm_prompt_2`, `afrimmlu_direct_orm_prompt_3`, `afrimmlu_direct_orm_prompt_4`
`afrimmlu_direct_orm_prompt_5`, `afrimmlu_direct_sna_prompt_1`, `afrimmlu_direct_sna_prompt_2`, `afrimmlu_direct_sna_prompt_3`
`afrimmlu_direct_sna_prompt_4`, `afrimmlu_direct_sna_prompt_5`, `afrimmlu_direct_sot_prompt_1`, `afrimmlu_direct_sot_prompt_2`
`afrimmlu_direct_sot_prompt_3`, `afrimmlu_direct_sot_prompt_4`, `afrimmlu_direct_sot_prompt_5`, `afrimmlu_direct_swa_prompt_1`
`afrimmlu_direct_swa_prompt_2`, `afrimmlu_direct_swa_prompt_3`, `afrimmlu_direct_swa_prompt_4`, `afrimmlu_direct_swa_prompt_5`
`afrimmlu_direct_twi_prompt_1`, `afrimmlu_direct_twi_prompt_2`, `afrimmlu_direct_twi_prompt_3`, `afrimmlu_direct_twi_prompt_4`
`afrimmlu_direct_twi_prompt_5`, `afrimmlu_direct_wol_prompt_1`, `afrimmlu_direct_wol_prompt_2`, `afrimmlu_direct_wol_prompt_3`
`afrimmlu_direct_wol_prompt_4`, `afrimmlu_direct_wol_prompt_5`, `afrimmlu_direct_xho_prompt_1`, `afrimmlu_direct_xho_prompt_2`
`afrimmlu_direct_xho_prompt_3`, `afrimmlu_direct_xho_prompt_4`, `afrimmlu_direct_xho_prompt_5`, `afrimmlu_direct_yor_prompt_1`
`afrimmlu_direct_yor_prompt_2`, `afrimmlu_direct_yor_prompt_3`, `afrimmlu_direct_yor_prompt_4`, `afrimmlu_direct_yor_prompt_5`
`afrimmlu_direct_zul_prompt_1`, `afrimmlu_direct_zul_prompt_2`, `afrimmlu_direct_zul_prompt_3`, `afrimmlu_direct_zul_prompt_4`
`afrimmlu_direct_zul_prompt_5`, `afrimmlu_tasks`, `afrimmlu_translate_amh_prompt_1`, `afrimmlu_translate_amh_prompt_2`
`afrimmlu_translate_amh_prompt_3`, `afrimmlu_translate_amh_prompt_4`, `afrimmlu_translate_amh_prompt_5`, `afrimmlu_translate_ewe_prompt_1`
`afrimmlu_translate_ewe_prompt_2`, `afrimmlu_translate_ewe_prompt_3`, `afrimmlu_translate_ewe_prompt_4`, `afrimmlu_translate_ewe_prompt_5`
`afrimmlu_translate_fra_prompt_1`, `afrimmlu_translate_fra_prompt_2`, `afrimmlu_translate_fra_prompt_3`, `afrimmlu_translate_fra_prompt_4`
`afrimmlu_translate_fra_prompt_5`, `afrimmlu_translate_hau_prompt_1`, `afrimmlu_translate_hau_prompt_2`, `afrimmlu_translate_hau_prompt_3`
`afrimmlu_translate_hau_prompt_4`, `afrimmlu_translate_hau_prompt_5`, `afrimmlu_translate_ibo_prompt_1`, `afrimmlu_translate_ibo_prompt_2`
`afrimmlu_translate_ibo_prompt_3`, `afrimmlu_translate_ibo_prompt_4`, `afrimmlu_translate_ibo_prompt_5`, `afrimmlu_translate_kin_prompt_1`
`afrimmlu_translate_kin_prompt_2`, `afrimmlu_translate_kin_prompt_3`, `afrimmlu_translate_kin_prompt_4`, `afrimmlu_translate_kin_prompt_5`
`afrimmlu_translate_lin_prompt_1`, `afrimmlu_translate_lin_prompt_2`, `afrimmlu_translate_lin_prompt_3`, `afrimmlu_translate_lin_prompt_4`
`afrimmlu_translate_lin_prompt_5`, `afrimmlu_translate_lug_prompt_1`, `afrimmlu_translate_lug_prompt_2`, `afrimmlu_translate_lug_prompt_3`
`afrimmlu_translate_lug_prompt_4`, `afrimmlu_translate_lug_prompt_5`, `afrimmlu_translate_orm_prompt_1`, `afrimmlu_translate_orm_prompt_2`
`afrimmlu_translate_orm_prompt_3`, `afrimmlu_translate_orm_prompt_4`, `afrimmlu_translate_orm_prompt_5`, `afrimmlu_translate_sna_prompt_1`
`afrimmlu_translate_sna_prompt_2`, `afrimmlu_translate_sna_prompt_3`, `afrimmlu_translate_sna_prompt_4`, `afrimmlu_translate_sna_prompt_5`
`afrimmlu_translate_sot_prompt_1`, `afrimmlu_translate_sot_prompt_2`, `afrimmlu_translate_sot_prompt_3`, `afrimmlu_translate_sot_prompt_4`
`afrimmlu_translate_sot_prompt_5`, `afrimmlu_translate_swa_prompt_1`, `afrimmlu_translate_swa_prompt_2`, `afrimmlu_translate_swa_prompt_3`
`afrimmlu_translate_swa_prompt_4`, `afrimmlu_translate_swa_prompt_5`, `afrimmlu_translate_twi_prompt_1`, `afrimmlu_translate_twi_prompt_2`
`afrimmlu_translate_twi_prompt_3`, `afrimmlu_translate_twi_prompt_4`, `afrimmlu_translate_twi_prompt_5`, `afrimmlu_translate_wol_prompt_1`
`afrimmlu_translate_wol_prompt_2`, `afrimmlu_translate_wol_prompt_3`, `afrimmlu_translate_wol_prompt_4`, `afrimmlu_translate_wol_prompt_5`
`afrimmlu_translate_xho_prompt_1`, `afrimmlu_translate_xho_prompt_2`, `afrimmlu_translate_xho_prompt_3`, `afrimmlu_translate_xho_prompt_4`
`afrimmlu_translate_xho_prompt_5`, `afrimmlu_translate_yor_prompt_1`, `afrimmlu_translate_yor_prompt_2`, `afrimmlu_translate_yor_prompt_3`
`afrimmlu_translate_yor_prompt_4`, `afrimmlu_translate_yor_prompt_5`, `afrimmlu_translate_zul_prompt_1`, `afrimmlu_translate_zul_prompt_2`
`afrimmlu_translate_zul_prompt_3`, `afrimmlu_translate_zul_prompt_4`, `afrimmlu_translate_zul_prompt_5`, `afrimmlu_tt-irokobench`
`afrimmlu_tt_tasks`, `afriqa`, `afrisent_prompt_2`, `afrisenti`
`afrixnli`, `afrixnli-irokobench`, `afrobench`, `afrobench_lite`
`ag_news`, `agieval`, `ai2_arc`, `aime`
`aime24`, `aime25`, `anagrams1`, `anagrams2`
`anli`, `arab_culture`, `arabic_exams`, `arabic_leaderboard_acva`
`arabic_leaderboard_alghafa`, `arabic_leaderboard_arabic_exams`, `arabic_leaderboard_arabic_mmlu`, `arabic_leaderboard_arabic_mt_arc_challenge`
`arabic_leaderboard_arabic_mt_arc_easy`, `arabic_leaderboard_arabic_mt_boolq`, `arabic_leaderboard_arabic_mt_copa`, `arabic_leaderboard_arabic_mt_hellaswag`
`arabic_leaderboard_arabic_mt_mmlu`, `arabic_leaderboard_arabic_mt_openbook_qa`, `arabic_leaderboard_arabic_mt_piqa`, `arabic_leaderboard_arabic_mt_race`
`arabic_leaderboard_arabic_mt_sciq`, `arabic_leaderboard_arabic_mt_toxigen`, `arabic_leaderboard_complete`, `arabic_leaderboard_light`
`arabic_mt_arc_challenge`, `arabic_mt_arc_easy`, `arabic_mt_boolq`, `arabic_mt_copa`
`arabic_mt_hellaswag`, `arabic_mt_mmlu`, `arabic_mt_openbook_qa`, `arabic_mt_piqa`
`arabic_mt_race`, `arabic_mt_sciq`, `arabic_mt_toxigen`, `arabicmmlu`
`arc_ar`, `arc_bn`, `arc_ca`, `arc_ca_easy`
`arc_challenge`, `arc_da`, `arc_de`, `arc_easy`
`arc_es`, `arc_eu`, `arc_eu_easy`, `arc_fr`
`arc_gu`, `arc_hi`, `arc_hr`, `arc_hu`
`arc_hy`, `arc_id`, `arc_it`, `arc_kn`
`arc_ml`, `arc_mr`, `arc_multilingual`, `arc_ne`
`arc_nl`, `arc_pt`, `arc_ro`, `arc_ru`
`arc_sk`, `arc_sr`, `arc_sv`, `arc_ta`
`arc_te`, `arc_uk`, `arc_vi`, `arc_zh`
`argument_topic`, `arithmetic`, `asdiv`, `ask_gec`
`assin_entailment`, `assin_paraphrase`, `atis`, `babi`
`babilong`, `bangla_commonsenseqa`, `bangla_mmlu`, `banking77`
`basque-glue`, `basque_bench`, `bbh`, `bbq`
`bear`, `bec2016eu`, `belebele`, `bertaqa`
`bhs__basque__DO__S_DO_V_AUX`, `bhs__basque__DO__S_IO_DO_V_AUX`, `bhs__basque__IO__IO_S_V_AUX`, `bhs__basque__IO__S_IO_DO_V_AUX`
`bhs__basque__S__IO_S_V_AUX`, `bhs__basque__S__S_DO_V_AUX`, `bhs__basque__S__S_IO_DO_V_AUX`, `bhs__basque__S__S_V_AUX`
`bhs__hindi__S_O_V`, `bhs__hindi__S_PossPRN_O_V`, `bhs__hindi__S_PossPRN_PossN_O_V`, `bhs__hindi__S_ne_O_V`
`bhs__hindi__S_ne_PossPRN_O_V`, `bhs__hindi__S_ne_PossPRN_PossN_O_V`, `bhs__swahili__N_of_Poss_D_AP_V_ni_AN`, `bhs__swahili__N_of_Poss_D_AP_ni_AN`
`bhs__swahili__N_of_Poss_D_A_V`, `bhs__swahili__N_of_Poss_D_A_V1_V2`, `bhs__swahili__N_of_Poss_D_V`, `bhs__swahili__N_of_Poss_D_ni_A`
`bhs__swahili__N_of_Poss_V`, `bhs__swahili__N_of_Poss_ni_A`, `bhs_basque`, `bhs_hindi`
`bhs_swahili`, `bhtc_v2`, `bigbench_abstract_narrative_understanding_generate_until`, `bigbench_abstract_narrative_understanding_multiple_choice`
`bigbench_anachronisms_generate_until`, `bigbench_anachronisms_multiple_choice`, `bigbench_analogical_similarity_generate_until`, `bigbench_analogical_similarity_multiple_choice`
`bigbench_analytic_entailment_generate_until`, `bigbench_analytic_entailment_multiple_choice`, `bigbench_arithmetic_generate_until`, `bigbench_arithmetic_multiple_choice`
`bigbench_ascii_word_recognition_generate_until`, `bigbench_authorship_verification_generate_until`, `bigbench_authorship_verification_multiple_choice`, `bigbench_auto_categorization_generate_until`
`bigbench_auto_debugging_generate_until`, `bigbench_bbq_lite_json_generate_until`, `bigbench_bbq_lite_json_multiple_choice`, `bigbench_bridging_anaphora_resolution_barqa_generate_until`
`bigbench_causal_judgment_generate_until`, `bigbench_causal_judgment_multiple_choice`, `bigbench_cause_and_effect_generate_until`, `bigbench_cause_and_effect_multiple_choice`
`bigbench_checkmate_in_one_generate_until`, `bigbench_checkmate_in_one_multiple_choice`, `bigbench_chess_state_tracking_generate_until`, `bigbench_chinese_remainder_theorem_generate_until`
`bigbench_cifar10_classification_generate_until`, `bigbench_cifar10_classification_multiple_choice`, `bigbench_code_line_description_generate_until`, `bigbench_code_line_description_multiple_choice`
`bigbench_codenames_generate_until`, `bigbench_color_generate_until`, `bigbench_color_multiple_choice`, `bigbench_common_morpheme_generate_until`
`bigbench_common_morpheme_multiple_choice`, `bigbench_conceptual_combinations_generate_until`, `bigbench_conceptual_combinations_multiple_choice`, `bigbench_conlang_translation_generate_until`
`bigbench_contextual_parametric_knowledge_conflicts_generate_until`, `bigbench_contextual_parametric_knowledge_conflicts_multiple_choice`, `bigbench_crash_blossom_generate_until`, `bigbench_crash_blossom_multiple_choice`
`bigbench_crass_ai_generate_until`, `bigbench_crass_ai_multiple_choice`, `bigbench_cryobiology_spanish_generate_until`, `bigbench_cryobiology_spanish_multiple_choice`
`bigbench_cryptonite_generate_until`, `bigbench_cs_algorithms_generate_until`, `bigbench_cs_algorithms_multiple_choice`, `bigbench_dark_humor_detection_generate_until`
`bigbench_dark_humor_detection_multiple_choice`, `bigbench_date_understanding_generate_until`, `bigbench_date_understanding_multiple_choice`, `bigbench_disambiguation_qa_generate_until`
`bigbench_disambiguation_qa_multiple_choice`, `bigbench_discourse_marker_prediction_generate_until`, `bigbench_discourse_marker_prediction_multiple_choice`, `bigbench_disfl_qa_generate_until`
`bigbench_dyck_languages_generate_until`, `bigbench_dyck_languages_multiple_choice`, `bigbench_elementary_math_qa_generate_until`, `bigbench_elementary_math_qa_multiple_choice`
`bigbench_emoji_movie_generate_until`, `bigbench_emoji_movie_multiple_choice`, `bigbench_emojis_emotion_prediction_generate_until`, `bigbench_emojis_emotion_prediction_multiple_choice`
`bigbench_empirical_judgments_generate_until`, `bigbench_empirical_judgments_multiple_choice`, `bigbench_english_proverbs_generate_until`, `bigbench_english_proverbs_multiple_choice`
`bigbench_english_russian_proverbs_generate_until`, `bigbench_english_russian_proverbs_multiple_choice`, `bigbench_entailed_polarity_generate_until`, `bigbench_entailed_polarity_hindi_generate_until`
`bigbench_entailed_polarity_hindi_multiple_choice`, `bigbench_entailed_polarity_multiple_choice`, `bigbench_epistemic_reasoning_generate_until`, `bigbench_epistemic_reasoning_multiple_choice`
`bigbench_evaluating_information_essentiality_generate_until`, `bigbench_evaluating_information_essentiality_multiple_choice`, `bigbench_fact_checker_generate_until`, `bigbench_fact_checker_multiple_choice`
`bigbench_fantasy_reasoning_generate_until`, `bigbench_fantasy_reasoning_multiple_choice`, `bigbench_few_shot_nlg_generate_until`, `bigbench_figure_of_speech_detection_generate_until`
`bigbench_figure_of_speech_detection_multiple_choice`, `bigbench_formal_fallacies_syllogisms_negation_generate_until`, `bigbench_formal_fallacies_syllogisms_negation_multiple_choice`, `bigbench_gem_generate_until`
`bigbench_gender_inclusive_sentences_german_generate_until`, `bigbench_general_knowledge_generate_until`, `bigbench_general_knowledge_multiple_choice`, `bigbench_generate_until`
`bigbench_geometric_shapes_generate_until`, `bigbench_geometric_shapes_multiple_choice`, `bigbench_goal_step_wikihow_generate_until`, `bigbench_goal_step_wikihow_multiple_choice`
`bigbench_gre_reading_comprehension_generate_until`, `bigbench_gre_reading_comprehension_multiple_choice`, `bigbench_hhh_alignment_generate_until`, `bigbench_hhh_alignment_multiple_choice`
`bigbench_hindi_question_answering_generate_until`, `bigbench_hindu_knowledge_generate_until`, `bigbench_hindu_knowledge_multiple_choice`, `bigbench_hinglish_toxicity_generate_until`
`bigbench_hinglish_toxicity_multiple_choice`, `bigbench_human_organs_senses_generate_until`, `bigbench_human_organs_senses_multiple_choice`, `bigbench_hyperbaton_generate_until`
`bigbench_hyperbaton_multiple_choice`, `bigbench_identify_math_theorems_generate_until`, `bigbench_identify_math_theorems_multiple_choice`, `bigbench_identify_odd_metaphor_generate_until`
`bigbench_identify_odd_metaphor_multiple_choice`, `bigbench_implicatures_generate_until`, `bigbench_implicatures_multiple_choice`, `bigbench_implicit_relations_generate_until`
`bigbench_implicit_relations_multiple_choice`, `bigbench_intent_recognition_generate_until`, `bigbench_intent_recognition_multiple_choice`, `bigbench_international_phonetic_alphabet_nli_generate_until`
`bigbench_international_phonetic_alphabet_nli_multiple_choice`, `bigbench_international_phonetic_alphabet_transliterate_generate_until`, `bigbench_intersect_geometry_generate_until`, `bigbench_intersect_geometry_multiple_choice`
`bigbench_irony_identification_generate_until`, `bigbench_irony_identification_multiple_choice`, `bigbench_kanji_ascii_generate_until`, `bigbench_kanji_ascii_multiple_choice`
`bigbench_kannada_generate_until`, `bigbench_kannada_multiple_choice`, `bigbench_key_value_maps_generate_until`, `bigbench_key_value_maps_multiple_choice`
`bigbench_known_unknowns_generate_until`, `bigbench_known_unknowns_multiple_choice`, `bigbench_language_games_generate_until`, `bigbench_language_identification_generate_until`
`bigbench_language_identification_multiple_choice`, `bigbench_linguistic_mappings_generate_until`, `bigbench_linguistics_puzzles_generate_until`, `bigbench_list_functions_generate_until`
`bigbench_logic_grid_puzzle_generate_until`, `bigbench_logic_grid_puzzle_multiple_choice`, `bigbench_logical_args_generate_until`, `bigbench_logical_args_multiple_choice`
`bigbench_logical_deduction_generate_until`, `bigbench_logical_deduction_multiple_choice`, `bigbench_logical_fallacy_detection_generate_until`, `bigbench_logical_fallacy_detection_multiple_choice`
`bigbench_logical_sequence_generate_until`, `bigbench_logical_sequence_multiple_choice`, `bigbench_mathematical_induction_generate_until`, `bigbench_mathematical_induction_multiple_choice`
`bigbench_matrixshapes_generate_until`, `bigbench_metaphor_boolean_generate_until`, `bigbench_metaphor_boolean_multiple_choice`, `bigbench_metaphor_understanding_generate_until`
`bigbench_metaphor_understanding_multiple_choice`, `bigbench_minute_mysteries_qa_generate_until`, `bigbench_misconceptions_generate_until`, `bigbench_misconceptions_multiple_choice`
`bigbench_misconceptions_russian_generate_until`, `bigbench_misconceptions_russian_multiple_choice`, `bigbench_mnist_ascii_generate_until`, `bigbench_mnist_ascii_multiple_choice`
`bigbench_modified_arithmetic_generate_until`, `bigbench_moral_permissibility_generate_until`, `bigbench_moral_permissibility_multiple_choice`, `bigbench_movie_dialog_same_or_different_generate_until`
`bigbench_movie_dialog_same_or_different_multiple_choice`, `bigbench_movie_recommendation_generate_until`, `bigbench_movie_recommendation_multiple_choice`, `bigbench_mult_data_wrangling_generate_until`
`bigbench_multiemo_generate_until`, `bigbench_multiemo_multiple_choice`, `bigbench_multiple_choice_a`, `bigbench_multiple_choice_b`
`bigbench_natural_instructions_generate_until`, `bigbench_navigate_generate_until`, `bigbench_navigate_multiple_choice`, `bigbench_nonsense_words_grammar_generate_until`
`bigbench_nonsense_words_grammar_multiple_choice`, `bigbench_novel_concepts_generate_until`, `bigbench_novel_concepts_multiple_choice`, `bigbench_object_counting_generate_until`
`bigbench_odd_one_out_generate_until`, `bigbench_odd_one_out_multiple_choice`, `bigbench_operators_generate_until`, `bigbench_paragraph_segmentation_generate_until`
`bigbench_parsinlu_qa_generate_until`, `bigbench_parsinlu_qa_multiple_choice`, `bigbench_parsinlu_reading_comprehension_generate_until`, `bigbench_penguins_in_a_table_generate_until`
`bigbench_penguins_in_a_table_multiple_choice`, `bigbench_periodic_elements_generate_until`, `bigbench_periodic_elements_multiple_choice`, `bigbench_persian_idioms_generate_until`
`bigbench_persian_idioms_multiple_choice`, `bigbench_phrase_relatedness_generate_until`, `bigbench_phrase_relatedness_multiple_choice`, `bigbench_physical_intuition_generate_until`
`bigbench_physical_intuition_multiple_choice`, `bigbench_physics_generate_until`, `bigbench_physics_multiple_choice`, `bigbench_physics_questions_generate_until`
`bigbench_play_dialog_same_or_different_generate_until`, `bigbench_play_dialog_same_or_different_multiple_choice`, `bigbench_polish_sequence_labeling_generate_until`, `bigbench_presuppositions_as_nli_generate_until`
`bigbench_presuppositions_as_nli_multiple_choice`, `bigbench_qa_wikidata_generate_until`, `bigbench_question_selection_generate_until`, `bigbench_question_selection_multiple_choice`
`bigbench_real_or_fake_text_generate_until`, `bigbench_real_or_fake_text_multiple_choice`, `bigbench_reasoning_about_colored_objects_generate_until`, `bigbench_reasoning_about_colored_objects_multiple_choice`
`bigbench_repeat_copy_logic_generate_until`, `bigbench_rephrase_generate_until`, `bigbench_riddle_sense_generate_until`, `bigbench_riddle_sense_multiple_choice`
`bigbench_ruin_names_generate_until`, `bigbench_ruin_names_multiple_choice`, `bigbench_salient_translation_error_detection_generate_until`, `bigbench_salient_translation_error_detection_multiple_choice`
`bigbench_scientific_press_release_generate_until`, `bigbench_semantic_parsing_in_context_sparc_generate_until`, `bigbench_semantic_parsing_spider_generate_until`, `bigbench_sentence_ambiguity_generate_until`
`bigbench_sentence_ambiguity_multiple_choice`, `bigbench_similarities_abstraction_generate_until`, `bigbench_similarities_abstraction_multiple_choice`, `bigbench_simp_turing_concept_generate_until`
`bigbench_simple_arithmetic_json_generate_until`, `bigbench_simple_arithmetic_json_multiple_choice_generate_until`, `bigbench_simple_arithmetic_json_subtasks_generate_until`, `bigbench_simple_arithmetic_multiple_targets_json_generate_until`
`bigbench_simple_ethical_questions_generate_until`, `bigbench_simple_ethical_questions_multiple_choice`, `bigbench_simple_text_editing_generate_until`, `bigbench_snarks_generate_until`
`bigbench_snarks_multiple_choice`, `bigbench_social_iqa_generate_until`, `bigbench_social_iqa_multiple_choice`, `bigbench_social_support_generate_until`
`bigbench_social_support_multiple_choice`, `bigbench_sports_understanding_generate_until`, `bigbench_sports_understanding_multiple_choice`, `bigbench_strange_stories_generate_until`
`bigbench_strange_stories_multiple_choice`, `bigbench_strategyqa_generate_until`, `bigbench_strategyqa_multiple_choice`, `bigbench_sufficient_information_generate_until`
`bigbench_suicide_risk_generate_until`, `bigbench_suicide_risk_multiple_choice`, `bigbench_swahili_english_proverbs_generate_until`, `bigbench_swahili_english_proverbs_multiple_choice`
`bigbench_swedish_to_german_proverbs_generate_until`, `bigbench_swedish_to_german_proverbs_multiple_choice`, `bigbench_symbol_interpretation_generate_until`, `bigbench_symbol_interpretation_multiple_choice`
`bigbench_temporal_sequences_generate_until`, `bigbench_temporal_sequences_multiple_choice`, `bigbench_tense_generate_until`, `bigbench_timedial_generate_until`
`bigbench_timedial_multiple_choice`, `bigbench_topical_chat_generate_until`, `bigbench_tracking_shuffled_objects_generate_until`, `bigbench_tracking_shuffled_objects_multiple_choice`
`bigbench_understanding_fables_generate_until`, `bigbench_understanding_fables_multiple_choice`, `bigbench_undo_permutation_generate_until`, `bigbench_undo_permutation_multiple_choice`
`bigbench_unit_conversion_generate_until`, `bigbench_unit_conversion_multiple_choice`, `bigbench_unit_interpretation_generate_until`, `bigbench_unit_interpretation_multiple_choice`
`bigbench_unnatural_in_context_learning_generate_until`, `bigbench_vitaminc_fact_verification_generate_until`, `bigbench_vitaminc_fact_verification_multiple_choice`, `bigbench_what_is_the_tao_generate_until`
`bigbench_what_is_the_tao_multiple_choice`, `bigbench_which_wiki_edit_generate_until`, `bigbench_which_wiki_edit_multiple_choice`, `bigbench_winowhy_generate_until`
`bigbench_winowhy_multiple_choice`, `bigbench_word_sorting_generate_until`, `bigbench_word_unscrambling_generate_until`, `blimp`
`boolq`, `boolq-seq2seq`, `boolqa_bn`, `c4`
`cabbq`, `cabreu`, `careqa_en`, `careqa_es`
`careqa_open`, `catalan_bench`, `catalanqa`, `catcola`
`cb`, `celep1`, `celep2`, `ceval-valid`
`chain_of_thought`, `chartqa`, `claim_stance_topic`, `click`
`cmmlu`, `cnn_dailymail`, `cocoteros_es`, `cocoteros_va`
`code2text`, `coedit_gec`, `cola`, `common_voice_en`
`commonsense_qa`, `copa`, `copal_id`, `coqa`
`coqcat`, `crows_pairs`, `csatqa`, `cycle_letters`
`darija_sentiment`, `darija_summarization`, `darija_translation`, `darija_transliteration`
`darijahellaswag`, `darijammlu`, `dbpedia_14`, `discrim_eval_explicit`
`discrim_eval_implicit`, `doc_vqa`, `drop`, `egyhellaswag`
`egymmlu`, `epec_koref_bin`, `eq_bench`, `eqbench_ca`
`eqbench_es`, `esbbq`, `escola`, `ethics_cm`
`ethics_deontology`, `ethics_justice`, `ethics_utilitarianism`, `ethics_virtue`
`ethos_binary`, `eus_exams_es`, `eus_exams_eu`, `eus_proficiency`
`eus_reading`, `eus_trivia`, `evalita-mp`, `evalita-sp_sum_task_fp-small_p1`
`evalita-sp_sum_task_fp-small_p2`, `evalita-sp_sum_task_fp_p1`, `evalita-sp_sum_task_fp_p2`, `fda`
`financial_tweets`, `flan_held_in`, `flan_held_out`, `fld_default`
`fld_logical_formula_default`, `fld_logical_formula_star`, `fld_star`, `flores`
`freebase`, `french_bench`, `galcola`, `galician_bench`
`glianorex`, `global_mmlu_ar`, `global_mmlu_bn`, `global_mmlu_de`
`global_mmlu_en`, `global_mmlu_es`, `global_mmlu_fr`, `global_mmlu_full_am`
`global_mmlu_full_ar`, `global_mmlu_full_bn`, `global_mmlu_full_cs`, `global_mmlu_full_de`
`global_mmlu_full_el`, `global_mmlu_full_en`, `global_mmlu_full_es`, `global_mmlu_full_fa`
`global_mmlu_full_fil`, `global_mmlu_full_fr`, `global_mmlu_full_ha`, `global_mmlu_full_he`
`global_mmlu_full_hi`, `global_mmlu_full_id`, `global_mmlu_full_ig`, `global_mmlu_full_it`
`global_mmlu_full_ja`, `global_mmlu_full_ko`, `global_mmlu_full_ky`, `global_mmlu_full_lt`
`global_mmlu_full_mg`, `global_mmlu_full_ms`, `global_mmlu_full_ne`, `global_mmlu_full_nl`
`global_mmlu_full_ny`, `global_mmlu_full_pl`, `global_mmlu_full_pt`, `global_mmlu_full_ro`
`global_mmlu_full_ru`, `global_mmlu_full_si`, `global_mmlu_full_sn`, `global_mmlu_full_so`
`global_mmlu_full_sr`, `global_mmlu_full_sv`, `global_mmlu_full_sw`, `global_mmlu_full_te`
`global_mmlu_full_tr`, `global_mmlu_full_uk`, `global_mmlu_full_vi`, `global_mmlu_full_yo`
`global_mmlu_full_zh`, `global_mmlu_generative_ar`, `global_mmlu_generative_bn`, `global_mmlu_generative_de`
`global_mmlu_generative_en`, `global_mmlu_generative_es`, `global_mmlu_generative_fr`, `global_mmlu_generative_hi`
`global_mmlu_generative_id`, `global_mmlu_generative_it`, `global_mmlu_generative_ja`, `global_mmlu_generative_ko`
`global_mmlu_generative_pt`, `global_mmlu_generative_sw`, `global_mmlu_generative_yo`, `global_mmlu_generative_zh`
`global_mmlu_hi`, `global_mmlu_id`, `global_mmlu_it`, `global_mmlu_ja`
`global_mmlu_ko`, `global_mmlu_pt`, `global_mmlu_sw`, `global_mmlu_yo`
`global_mmlu_zh`, `global_piqa_completions`, `global_piqa_prompted`, `glue`
`gpqa`, `gpqa_diamond`, `gpqa_extended`, `gpqa_main`
`gpt3_translation_benchmarks`, `graphwalks`, `groundcocoa`, `gsm8k`
`gsm_plus`, `gsm_plus_mini`, `haerae`, `headqa`
`hellaswag`, `hendrycks_ethics`, `hendrycks_math`, `hendrycks_math500`
`histoires_morales`, `hrm8k`, `humaneval`, `icelandic_winogrande`
`ifbench`, `ifeval`, `include_base_44_albanian`, `include_base_44_arabic`
`include_base_44_armenian`, `include_base_44_azerbaijani`, `include_base_44_basque`, `include_base_44_belarusian`
`include_base_44_bengali`, `include_base_44_bulgarian`, `include_base_44_chinese`, `include_base_44_croatian`
`include_base_44_dutch`, `include_base_44_estonian`, `include_base_44_finnish`, `include_base_44_french`
`include_base_44_georgian`, `include_base_44_german`, `include_base_44_greek`, `include_base_44_hebrew`
`include_base_44_hindi`, `include_base_44_hungarian`, `include_base_44_indonesian`, `include_base_44_italian`
`include_base_44_japanese`, `include_base_44_kazakh`, `include_base_44_korean`, `include_base_44_lithuanian`
`include_base_44_malay`, `include_base_44_malayalam`, `include_base_44_nepali`, `include_base_44_north macedonian`
`include_base_44_persian`, `include_base_44_polish`, `include_base_44_portuguese`, `include_base_44_russian`
`include_base_44_serbian`, `include_base_44_spanish`, `include_base_44_tagalog`, `include_base_44_tamil`
`include_base_44_telugu`, `include_base_44_turkish`, `include_base_44_ukrainian`, `include_base_44_urdu`
`include_base_44_uzbek`, `include_base_44_vietnamese`, `injongointent`, `inverse_scaling_hindsight_neglect_10shot`
`inverse_scaling_into_the_unknown`, `inverse_scaling_mc`, `inverse_scaling_memo_trap`, `inverse_scaling_modus_tollens`
`inverse_scaling_neqa`, `inverse_scaling_pattern_matching_suppression`, `inverse_scaling_quote_repetition`, `inverse_scaling_redefine_math`
`inverse_scaling_repetitive_algebra`, `inverse_scaling_sig_figs`, `inverse_scaling_winobias_antistereotype`, `iwslt2017`
`iwslt2017-ar-en`, `iwslt2017-en-ar`, `ja_leaderboard_jaqket_v2`, `ja_leaderboard_jcommonsenseqa`
`ja_leaderboard_jnli`, `ja_leaderboard_jsquad`, `ja_leaderboard_marc_ja`, `ja_leaderboard_mgsm`
`ja_leaderboard_xlsum`, `ja_leaderboard_xwinograd`, `japanese_leaderboard`, `jsonschema_bench`
`jsonschema_bench_easy`, `jsonschema_bench_hard`, `kbl`, `kmmlu`
`kmmlu_hard`, `kmmlu_pro`, `kobalt`, `kobest`
`kobigbench`, `kormedmcqa`, `lambada`, `law_stack_exchange`
`leaderboard`, `ledgar`, `libra`, `librusec_history`
`librusec_mhqa`, `lingoly`, `livecodebench`, `llama`
`lm_syneval`, `logieval`, `logiqa`, `logiqa2`
`long_context_multiq`, `longbench`, `longbench2`, `longcxt`
`m_mmlu`, `mafand`, `masakhaner`, `masakhanews`
`masakhapos`, `mastermind`, `mastermind_easy`, `mastermind_hard`
`math`, `mathqa`, `matreshka_names`, `matreshka_yes_no`
`mbpp`, `mc_taco`, `med_concepts_qa`, `med_prescriptions`
`med_prescriptions_easy`, `med_prescriptions_hard`, `med_text_classification`, `med_text_classification_easy`
`med_text_classification_hard`, `meddialog`, `medical_abstracts`, `mediqa_qa2019`
`medmcqa`, `medqa_4options`, `medtext`, `mela`
`meqsum`, `metabench`, `mgsm`, `mimic_repsum`
`minerva_math`, `minerva_math500`, `mlqa_ar_ar`, `mlqa_ar_de`
`mlqa_ar_en`, `mlqa_ar_es`, `mlqa_ar_hi`, `mlqa_ar_vi`
`mlqa_ar_zh`, `mlqa_de_ar`, `mlqa_de_de`, `mlqa_de_en`
`mlqa_de_es`, `mlqa_de_hi`, `mlqa_de_vi`, `mlqa_de_zh`
`mlqa_en_ar`, `mlqa_en_de`, `mlqa_en_en`, `mlqa_en_es`
`mlqa_en_hi`, `mlqa_en_vi`, `mlqa_en_zh`, `mlqa_es_ar`
`mlqa_es_de`, `mlqa_es_en`, `mlqa_es_es`, `mlqa_es_hi`
`mlqa_es_vi`, `mlqa_es_zh`, `mlqa_hi_ar`, `mlqa_hi_de`
`mlqa_hi_en`, `mlqa_hi_es`, `mlqa_hi_hi`, `mlqa_hi_vi`
`mlqa_hi_zh`, `mlqa_vi_ar`, `mlqa_vi_de`, `mlqa_vi_en`
`mlqa_vi_es`, `mlqa_vi_hi`, `mlqa_vi_vi`, `mlqa_vi_zh`
`mlqa_zh_ar`, `mlqa_zh_de`, `mlqa_zh_en`, `mlqa_zh_es`
`mlqa_zh_hi`, `mlqa_zh_vi`, `mlqa_zh_zh`, `mmlu`
`mmlu_pro`, `mmlu_redux`, `mmlusr`, `mmmlu`
`mmmu_val`, `mnli`, `moral_stories`, `mrpc`
`mts_dialog`, `multiblimp`, `multimedqa`, `multiple_choice`
`multirc`, `mutual`, `naijarc`, `ncb`
`niah_multikey_1`, `niah_multikey_2`, `niah_multikey_3`, `niah_multiquery`
`niah_multivalue`, `niah_single_1`, `niah_single_2`, `niah_single_3`
`nollysenti`, `non_greedy_robustness_agieval_aqua_rat`, `non_greedy_robustness_agieval_logiqa_en`, `non_greedy_robustness_agieval_lsat_ar`
`non_greedy_robustness_agieval_lsat_lr`, `non_greedy_robustness_agieval_lsat_rc`, `non_greedy_robustness_agieval_sat_en`, `non_greedy_robustness_agieval_sat_math`
`non_greedy_robustness_math_algebra`, `non_greedy_robustness_math_counting_and_prob`, `non_greedy_robustness_math_geometry`, `non_greedy_robustness_math_intermediate_algebra`
`non_greedy_robustness_math_num_theory`, `non_greedy_robustness_math_prealgebra`, `non_greedy_robustness_math_precalc`, `noor`
`norbelebele`, `norcommonsenseqa_nno`, `norcommonsenseqa_nob`, `norec_document`
`norec_sentence`, `noridiom_nno`, `noridiom_nob`, `noropenbookqa_nno`
`noropenbookqa_nob`, `norquad`, `norrewrite_instruct`, `norsumm_nno`
`norsumm_nob`, `norsummarize_instruct`, `nortruthfulqa_gen_nno`, `nortruthfulqa_gen_nob`
`nortruthfulqa_mc_nno`, `nortruthfulqa_mc_nob`, `noticia`, `nq_open`
`nrk_quiz_qa_nno`, `nrk_quiz_qa_nob`, `ntrex_afr-eng`, `ntrex_afr_Latn-eng_Latn_prompt_1`
`ntrex_afr_Latn-eng_Latn_prompt_2`, `ntrex_afr_Latn-eng_Latn_prompt_3`, `ntrex_amh_Ethi-eng_Latn_prompt_1`, `ntrex_amh_Ethi-eng_Latn_prompt_2`
`ntrex_amh_Ethi-eng_Latn_prompt_3`, `ntrex_arb_Arab-eng_Latn_prompt_1`, `ntrex_arb_Arab-eng_Latn_prompt_2`, `ntrex_arb_Arab-eng_Latn_prompt_3`
`ntrex_bem_Latn-eng_Latn_prompt_1`, `ntrex_bem_Latn-eng_Latn_prompt_2`, `ntrex_bem_Latn-eng_Latn_prompt_3`, `ntrex_eng-afr`
`ntrex_eng_Latn-afr_Latn_prompt_1`, `ntrex_eng_Latn-afr_Latn_prompt_2`, `ntrex_eng_Latn-afr_Latn_prompt_3`, `ntrex_eng_Latn-amh_Ethi_prompt_1`
`ntrex_eng_Latn-amh_Ethi_prompt_2`, `ntrex_eng_Latn-amh_Ethi_prompt_3`, `ntrex_eng_Latn-arb_Arab_prompt_1`, `ntrex_eng_Latn-arb_Arab_prompt_2`
`ntrex_eng_Latn-arb_Arab_prompt_3`, `ntrex_eng_Latn-bem_Latn_prompt_1`, `ntrex_eng_Latn-bem_Latn_prompt_2`, `ntrex_eng_Latn-bem_Latn_prompt_3`
`ntrex_eng_Latn-ewe_Latn_prompt_1`, `ntrex_eng_Latn-ewe_Latn_prompt_2`, `ntrex_eng_Latn-ewe_Latn_prompt_3`, `ntrex_eng_Latn-fra_Latn_prompt_1`
`ntrex_eng_Latn-fra_Latn_prompt_2`, `ntrex_eng_Latn-fra_Latn_prompt_3`, `ntrex_eng_Latn-hau_Latn_prompt_1`, `ntrex_eng_Latn-hau_Latn_prompt_2`
`ntrex_eng_Latn-hau_Latn_prompt_3`, `ntrex_eng_Latn-ibo_Latn_prompt_1`, `ntrex_eng_Latn-ibo_Latn_prompt_2`, `ntrex_eng_Latn-ibo_Latn_prompt_3`
`ntrex_eng_Latn-kin_Latn_prompt_1`, `ntrex_eng_Latn-kin_Latn_prompt_2`, `ntrex_eng_Latn-kin_Latn_prompt_3`, `ntrex_eng_Latn-mey_Arab_prompt_1`
`ntrex_eng_Latn-mey_Arab_prompt_2`, `ntrex_eng_Latn-mey_Arab_prompt_3`, `ntrex_eng_Latn-mlg_Latn_prompt_1`, `ntrex_eng_Latn-mlg_Latn_prompt_2`
`ntrex_eng_Latn-mlg_Latn_prompt_3`, `ntrex_eng_Latn-msa_Latn_prompt_1`, `ntrex_eng_Latn-msa_Latn_prompt_2`, `ntrex_eng_Latn-msa_Latn_prompt_3`
`ntrex_eng_Latn-nde_Latn_prompt_1`, `ntrex_eng_Latn-nde_Latn_prompt_2`, `ntrex_eng_Latn-nde_Latn_prompt_3`, `ntrex_eng_Latn-nso_Latn_prompt_1`
`ntrex_eng_Latn-nso_Latn_prompt_2`, `ntrex_eng_Latn-nso_Latn_prompt_3`, `ntrex_eng_Latn-nya_Latn_prompt_1`, `ntrex_eng_Latn-nya_Latn_prompt_2`
`ntrex_eng_Latn-nya_Latn_prompt_3`, `ntrex_eng_Latn-orm_Ethi_prompt_1`, `ntrex_eng_Latn-orm_Ethi_prompt_2`, `ntrex_eng_Latn-orm_Ethi_prompt_3`
`ntrex_eng_Latn-shi_Arab_prompt_1`, `ntrex_eng_Latn-shi_Arab_prompt_2`, `ntrex_eng_Latn-shi_Arab_prompt_3`, `ntrex_eng_Latn-sna_Latn_prompt_1`
`ntrex_eng_Latn-sna_Latn_prompt_2`, `ntrex_eng_Latn-sna_Latn_prompt_3`, `ntrex_eng_Latn-som_Latn_prompt_1`, `ntrex_eng_Latn-som_Latn_prompt_2`
`ntrex_eng_Latn-som_Latn_prompt_3`, `ntrex_eng_Latn-ssw_Latn_prompt_1`, `ntrex_eng_Latn-ssw_Latn_prompt_2`, `ntrex_eng_Latn-ssw_Latn_prompt_3`
`ntrex_eng_Latn-swa_Latn_prompt_1`, `ntrex_eng_Latn-swa_Latn_prompt_2`, `ntrex_eng_Latn-swa_Latn_prompt_3`, `ntrex_eng_Latn-tam_Taml_prompt_1`
`ntrex_eng_Latn-tam_Taml_prompt_2`, `ntrex_eng_Latn-tam_Taml_prompt_3`, `ntrex_eng_Latn-tel_Telu_prompt_1`, `ntrex_eng_Latn-tel_Telu_prompt_2`
`ntrex_eng_Latn-tel_Telu_prompt_3`, `ntrex_eng_Latn-tir_Ethi_prompt_1`, `ntrex_eng_Latn-tir_Ethi_prompt_2`, `ntrex_eng_Latn-tir_Ethi_prompt_3`
`ntrex_eng_Latn-ton_Latn_prompt_1`, `ntrex_eng_Latn-ton_Latn_prompt_2`, `ntrex_eng_Latn-ton_Latn_prompt_3`, `ntrex_eng_Latn-tsn_Latn_prompt_1`
`ntrex_eng_Latn-tsn_Latn_prompt_2`, `ntrex_eng_Latn-tsn_Latn_prompt_3`, `ntrex_eng_Latn-urd_Arab_prompt_1`, `ntrex_eng_Latn-urd_Arab_prompt_2`
`ntrex_eng_Latn-urd_Arab_prompt_3`, `ntrex_eng_Latn-ven_Latn_prompt_1`, `ntrex_eng_Latn-ven_Latn_prompt_2`, `ntrex_eng_Latn-ven_Latn_prompt_3`
`ntrex_eng_Latn-wol_Latn_prompt_1`, `ntrex_eng_Latn-wol_Latn_prompt_2`, `ntrex_eng_Latn-wol_Latn_prompt_3`, `ntrex_eng_Latn-xho_Latn_prompt_1`
`ntrex_eng_Latn-xho_Latn_prompt_2`, `ntrex_eng_Latn-xho_Latn_prompt_3`, `ntrex_eng_Latn-yor_Latn_prompt_1`, `ntrex_eng_Latn-yor_Latn_prompt_2`
`ntrex_eng_Latn-yor_Latn_prompt_3`, `ntrex_eng_Latn-zul_Latn_prompt_1`, `ntrex_eng_Latn-zul_Latn_prompt_2`, `ntrex_eng_Latn-zul_Latn_prompt_3`
`ntrex_ewe_Latn-eng_Latn_prompt_1`, `ntrex_ewe_Latn-eng_Latn_prompt_2`, `ntrex_ewe_Latn-eng_Latn_prompt_3`, `ntrex_fra_Latn-eng_Latn_prompt_1`
`ntrex_fra_Latn-eng_Latn_prompt_2`, `ntrex_fra_Latn-eng_Latn_prompt_3`, `ntrex_hau_Latn-eng_Latn_prompt_1`, `ntrex_hau_Latn-eng_Latn_prompt_2`
`ntrex_hau_Latn-eng_Latn_prompt_3`, `ntrex_ibo_Latn-eng_Latn_prompt_1`, `ntrex_ibo_Latn-eng_Latn_prompt_2`, `ntrex_ibo_Latn-eng_Latn_prompt_3`
`ntrex_kin_Latn-eng_Latn_prompt_1`, `ntrex_kin_Latn-eng_Latn_prompt_2`, `ntrex_kin_Latn-eng_Latn_prompt_3`, `ntrex_mey_Arab-eng_Latn_prompt_1`
`ntrex_mey_Arab-eng_Latn_prompt_2`, `ntrex_mey_Arab-eng_Latn_prompt_3`, `ntrex_mlg_Latn-eng_Latn_prompt_1`, `ntrex_mlg_Latn-eng_Latn_prompt_2`
`ntrex_mlg_Latn-eng_Latn_prompt_3`, `ntrex_msa_Latn-eng_Latn_prompt_1`, `ntrex_msa_Latn-eng_Latn_prompt_2`, `ntrex_msa_Latn-eng_Latn_prompt_3`
`ntrex_nde_Latn-eng_Latn_prompt_1`, `ntrex_nde_Latn-eng_Latn_prompt_2`, `ntrex_nde_Latn-eng_Latn_prompt_3`, `ntrex_nso_Latn-eng_Latn_prompt_1`
`ntrex_nso_Latn-eng_Latn_prompt_2`, `ntrex_nso_Latn-eng_Latn_prompt_3`, `ntrex_nya_Latn-eng_Latn_prompt_1`, `ntrex_nya_Latn-eng_Latn_prompt_2`
`ntrex_nya_Latn-eng_Latn_prompt_3`, `ntrex_orm_Ethi-eng_Latn_prompt_1`, `ntrex_orm_Ethi-eng_Latn_prompt_2`, `ntrex_orm_Ethi-eng_Latn_prompt_3`
`ntrex_shi_Arab-eng_Latn_prompt_1`, `ntrex_shi_Arab-eng_Latn_prompt_2`, `ntrex_shi_Arab-eng_Latn_prompt_3`, `ntrex_sna_Latn-eng_Latn_prompt_1`
`ntrex_sna_Latn-eng_Latn_prompt_2`, `ntrex_sna_Latn-eng_Latn_prompt_3`, `ntrex_som_Latn-eng_Latn_prompt_1`, `ntrex_som_Latn-eng_Latn_prompt_2`
`ntrex_som_Latn-eng_Latn_prompt_3`, `ntrex_ssw_Latn-eng_Latn_prompt_1`, `ntrex_ssw_Latn-eng_Latn_prompt_2`, `ntrex_ssw_Latn-eng_Latn_prompt_3`
`ntrex_swa_Latn-eng_Latn_prompt_1`, `ntrex_swa_Latn-eng_Latn_prompt_2`, `ntrex_swa_Latn-eng_Latn_prompt_3`, `ntrex_tam_Taml-eng_Latn_prompt_1`
`ntrex_tam_Taml-eng_Latn_prompt_2`, `ntrex_tam_Taml-eng_Latn_prompt_3`, `ntrex_tasks`, `ntrex_tel_Telu-eng_Latn_prompt_1`
`ntrex_tel_Telu-eng_Latn_prompt_2`, `ntrex_tel_Telu-eng_Latn_prompt_3`, `ntrex_tir_Ethi-eng_Latn_prompt_1`, `ntrex_tir_Ethi-eng_Latn_prompt_2`
`ntrex_tir_Ethi-eng_Latn_prompt_3`, `ntrex_ton_Latn-eng_Latn_prompt_1`, `ntrex_ton_Latn-eng_Latn_prompt_2`, `ntrex_ton_Latn-eng_Latn_prompt_3`
`ntrex_tsn_Latn-eng_Latn_prompt_1`, `ntrex_tsn_Latn-eng_Latn_prompt_2`, `ntrex_tsn_Latn-eng_Latn_prompt_3`, `ntrex_urd_Arab-eng_Latn_prompt_1`
`ntrex_urd_Arab-eng_Latn_prompt_2`, `ntrex_urd_Arab-eng_Latn_prompt_3`, `ntrex_ven_Latn-eng_Latn_prompt_1`, `ntrex_ven_Latn-eng_Latn_prompt_2`
`ntrex_ven_Latn-eng_Latn_prompt_3`, `ntrex_wol_Latn-eng_Latn_prompt_1`, `ntrex_wol_Latn-eng_Latn_prompt_2`, `ntrex_wol_Latn-eng_Latn_prompt_3`
`ntrex_xho_Latn-eng_Latn_prompt_1`, `ntrex_xho_Latn-eng_Latn_prompt_2`, `ntrex_xho_Latn-eng_Latn_prompt_3`, `ntrex_yor_Latn-eng_Latn_prompt_1`
`ntrex_yor_Latn-eng_Latn_prompt_2`, `ntrex_yor_Latn-eng_Latn_prompt_3`, `ntrex_zul_Latn-eng_Latn_prompt_1`, `ntrex_zul_Latn-eng_Latn_prompt_2`
`ntrex_zul_Latn-eng_Latn_prompt_3`, `nug`, `olaph`, `openai_mmlu`
`openbookqa`, `openllm`, `option_order_robustness_agieval_aqua_rat`, `option_order_robustness_agieval_logiqa_en`
`option_order_robustness_agieval_lsat_ar`, `option_order_robustness_agieval_lsat_lr`, `option_order_robustness_agieval_lsat_rc`, `option_order_robustness_agieval_sat_en`
`option_order_robustness_agieval_sat_math`, `paloma`, `parafraseja`, `parafrases_gl`
`passkey`, `paws_ca`, `paws_de`, `paws_en`
`paws_es`, `paws_eu`, `paws_fr`, `paws_gl`
`paws_ja`, `paws_ko`, `paws_zh`, `pawsx`
`persona`, `phrases_ca-va`, `phrases_es`, `phrases_es-va`
`phrases_va`, `phrases_va-ca`, `phrases_va-es`, `pile_10k`
`pile_arxiv`, `pile_bookcorpus2`, `pile_books3`, `pile_dm-mathematics`
`pile_enron`, `pile_europarl`, `pile_freelaw`, `pile_github`
`pile_gutenberg`, `pile_hackernews`, `pile_nih-exporter`, `pile_opensubtitles`
`pile_openwebtext2`, `pile_philpapers`, `pile_pile-cc`, `pile_pubmed-abstracts`
`pile_pubmed-central`, `pile_stackexchange`, `pile_ubuntu-irc`, `pile_uspto`
`pile_wikipedia`, `pile_youtubesubtitles`, `piqa`, `pisa`
`polemo2`, `portuguese_bench`, `prompt_robustness_agieval_aqua_rat`, `prompt_robustness_agieval_logiqa_en`
`prompt_robustness_agieval_lsat_ar`, `prompt_robustness_agieval_lsat_lr`, `prompt_robustness_agieval_lsat_rc`, `prompt_robustness_agieval_sat_en`
`prompt_robustness_agieval_sat_math`, `prompt_robustness_math_algebra`, `prompt_robustness_math_counting_and_prob`, `prompt_robustness_math_geometry`
`prompt_robustness_math_intermediate_algebra`, `prompt_robustness_math_num_theory`, `prompt_robustness_math_prealgebra`, `prompt_robustness_math_precalc`
`prost`, `pubmedqa`, `pythia`, `qa4mre`
`qasper`, `qnli`, `qnlieu`, `qqp`
`race`, `random_insertion`, `realtoxicityprompts`, `record`
`reversed_words`, `rte`, `ru_2wikimultihopqa`, `ru_babilong_qa1`
`ru_babilong_qa2`, `ru_babilong_qa3`, `ru_babilong_qa4`, `ru_babilong_qa5`
`ru_gsm100`, `ru_qasper`, `ru_quality`, `ru_sci_abstract_retrieval`
`ru_sci_passage_count`, `ruler`, `salt`, `sciknoweval_mcqa`
`sciq`, `score_non_greedy_robustness_agieval`, `score_non_greedy_robustness_math`, `score_non_greedy_robustness_mmlu_pro`
`score_option_order_robustness_agieval`, `score_option_order_robustness_mmlu_pro`, `score_prompt_robustness_agieval`, `score_prompt_robustness_math`
`score_prompt_robustness_mmlu_pro`, `score_robustness`, `scrolls_contractnli`, `scrolls_govreport`
`scrolls_narrativeqa`, `scrolls_qasper`, `scrolls_qmsum`, `scrolls_quality`
`scrolls_summscreenfd`, `self_consistency`, `sglue_rte`, `sib`
`simple_cooccurrence_bias`, `siqa_ca`, `slr_bench_all`, `slr_bench_basic`
`slr_bench_easy`, `slr_bench_group`, `slr_bench_hard`, `slr_bench_medium`
`social_bias`, `social_iqa`, `spanish_bench`, `squad_completion`
`squadv2`, `sst2`, `storycloze`, `stsb`
`summarization_gl`, `super-glue-lm-eval-v1`, `super-glue-lm-eval-v1-seq2seq`, `super-glue-t5-prompt`
`super_glue-boolq-t5-prompt`, `super_glue-cb-t5-prompt`, `super_glue-copa-t5-prompt`, `super_glue-multirc-t5-prompt`
`super_glue-record-t5-prompt`, `super_glue-rte-t5-prompt`, `super_glue-wic-t5-prompt`, `super_glue-wsc-t5-prompt`
`swag`, `swde`, `sycophancy`, `t0_eval`
`tatoeba_eng_nno`, `tatoeba_eng_nob`, `tatoeba_nno_eng`, `tatoeba_nob_eng`
`teca`, `tinyArc`, `tinyBenchmarks`, `tinyGSM8k`
`tinyHellaswag`, `tinyMMLU`, `tinyTruthfulQA`, `tinyWinogrande`
`tmlu`, `tmmluplus`, `toxigen`, `translation`
`transliteration_all`, `transliteration_ar_dr`, `transliteration_dr_ar`, `trasnlation_all_doda`
`trasnlation_all_flores`, `trasnlation_all_madar`, `trasnlation_all_seed`, `trasnlation_dr_en_doda`
`trasnlation_dr_en_flores`, `trasnlation_dr_en_seed`, `trasnlation_dr_fr_doda`, `trasnlation_dr_fr_flores`
`trasnlation_dr_msa_doda`, `trasnlation_dr_msa_flores`, `trasnlation_dr_msa_madar`, `trasnlation_en_dr_doda`
`trasnlation_en_dr_flores`, `trasnlation_en_dr_seed`, `trasnlation_fr_dr_doda`, `trasnlation_fr_dr_flores`
`trasnlation_msa_dr_doda`, `trasnlation_msa_dr_flores`, `trasnlation_msa_dr_madar`, `triviaqa`
`truthfulqa`, `truthfulqa-multi`, `turblimp_anaphor_agreement`, `turblimp_argument_structure_ditransitive`
`turblimp_argument_structure_transitive`, `turblimp_binding`, `turblimp_core`, `turblimp_determiners`
`turblimp_ellipsis`, `turblimp_irregular_forms`, `turblimp_island_effects`, `turblimp_nominalization`
`turblimp_npi_licensing`, `turblimp_passives`, `turblimp_quantifiers`, `turblimp_relative_clauses`
`turblimp_scrambling`, `turblimp_subject_agreement`, `turblimp_suspended_affixation`, `turkishmmlu`
`uhura-arc-easy_am_prompt_1`, `uhura-arc-easy_am_prompt_2`, `uhura-arc-easy_am_prompt_3`, `uhura-arc-easy_am_prompt_4`
`uhura-arc-easy_am_prompt_5`, `uhura-arc-easy_en_prompt_1`, `uhura-arc-easy_en_prompt_2`, `uhura-arc-easy_en_prompt_3`
`uhura-arc-easy_en_prompt_4`, `uhura-arc-easy_en_prompt_5`, `uhura-arc-easy_ha_prompt_1`, `uhura-arc-easy_ha_prompt_2`
`uhura-arc-easy_ha_prompt_3`, `uhura-arc-easy_ha_prompt_4`, `uhura-arc-easy_ha_prompt_5`, `uhura-arc-easy_nso_prompt_1`
`uhura-arc-easy_nso_prompt_2`, `uhura-arc-easy_nso_prompt_3`, `uhura-arc-easy_nso_prompt_4`, `uhura-arc-easy_nso_prompt_5`
`uhura-arc-easy_sw_prompt_1`, `uhura-arc-easy_sw_prompt_2`, `uhura-arc-easy_sw_prompt_3`, `uhura-arc-easy_sw_prompt_4`
`uhura-arc-easy_sw_prompt_5`, `uhura-arc-easy_yo_prompt_1`, `uhura-arc-easy_yo_prompt_2`, `uhura-arc-easy_yo_prompt_3`
`uhura-arc-easy_yo_prompt_4`, `uhura-arc-easy_yo_prompt_5`, `uhura-arc-easy_zu_prompt_1`, `uhura-arc-easy_zu_prompt_2`
`uhura-arc-easy_zu_prompt_3`, `uhura-arc-easy_zu_prompt_4`, `uhura-arc-easy_zu_prompt_5`, `uhura_arc_easy`
`uleval`, `ulqa`, `ulut`, `unfair_tos`
`unscramble`, `uyghur_language`, `uyghur_literature`, `uyghur_llm`
`vaxx_stance`, `wag`, `webqs`, `wic`
`wiceu`, `wikitext`, `winogender`, `winogrande`
`wmdp`, `wmt-ro-en-t5-prompt`, `wmt14`, `wmt14-en-fr`
`wmt14-fr-en`, `wmt16`, `wmt16-de-en`, `wmt16-en-de`
`wmt16-en-ro`, `wmt16-ro-en`, `wnli`, `wsc`
`wsc273`, `wsm`, `wub`, `wum`
`xcopa`, `xlsum_amharic_prompt_1`, `xlsum_amharic_prompt_2`, `xlsum_amharic_prompt_3`
`xlsum_arabic_prompt_1`, `xlsum_arabic_prompt_2`, `xlsum_arabic_prompt_3`, `xlsum_es`
`xlsum_hausa_prompt_1`, `xlsum_hausa_prompt_2`, `xlsum_hausa_prompt_3`, `xlsum_igbo_prompt_1`
`xlsum_igbo_prompt_2`, `xlsum_igbo_prompt_3`, `xlsum_kirundi_prompt_1`, `xlsum_kirundi_prompt_2`
`xlsum_kirundi_prompt_3`, `xlsum_oromo_prompt_1`, `xlsum_oromo_prompt_2`, `xlsum_oromo_prompt_3`
`xlsum_pidgin_prompt_1`, `xlsum_pidgin_prompt_2`, `xlsum_pidgin_prompt_3`, `xlsum_prompt_1`
`xlsum_prompt_2`, `xlsum_prompt_3`, `xlsum_somali_prompt_1`, `xlsum_somali_prompt_2`
`xlsum_somali_prompt_3`, `xlsum_swahili_prompt_1`, `xlsum_swahili_prompt_2`, `xlsum_swahili_prompt_3`
`xlsum_tasks`, `xlsum_telugu_prompt_1`, `xlsum_telugu_prompt_2`, `xlsum_telugu_prompt_3`
`xlsum_tigrinya_prompt_1`, `xlsum_tigrinya_prompt_2`, `xlsum_tigrinya_prompt_3`, `xlsum_yoruba_prompt_1`
`xlsum_yoruba_prompt_2`, `xlsum_yoruba_prompt_3`, `xlum`, `xnli`
`xquad`, `xstorycloze`, `xsum`, `xwinograd`
`yahoo_answers_topics`, `zhoblimp`

</details>

<details>
<summary><b><code>vlm_eval_kit</code></b> — 375 benchmarks • Vision-language benchmarks ([VLMEvalKit](https://github.com/open-compass/VLMEvalKit))</summary>

`3DSRBench`, `A-Bench_TEST`, `A-Bench_VAL`, `A-OKVQA`
`A4Bench`, `AI2D_MINI`, `AI2D_TEST`, `AMBER`
`APhO_2025`, `AesBench_TEST`, `AesBench_VAL`, `Asclepius`
`AyaVisionBench`, `B`, `BLINK`, `BMMR`
`BMMR_mini`, `C`, `CCBench`, `CCOCR`
`CG-AV-Counting_32frame`, `CG-AV-Counting_64frame`, `CG-Bench_MCQ_Grounding`, `CG-Bench_MCQ_Grounding_Mini`
`CG-Bench_OpenEnded`, `CG-Bench_OpenEnded_Mini`, `CGAVCounting`, `CGBench_MCQ_Grounding_16frame_subs_subt_ft`
`CGBench_MCQ_Grounding_32frame_subs`, `CGBench_MCQ_Grounding_Mini_8frame_subs_subt`, `CGBench_OpenEnded_16frame_subs_subt_ft`, `CGBench_OpenEnded_8frame`
`CGBench_OpenEnded_Mini_8frame_subs_subt_ft`, `CMMMU_VAL`, `CMMU_MCQ`, `COCO_VAL`
`CRPE_EXIST`, `CRPE_RELATION`, `CV-Bench-2D`, `CV-Bench-3D`
`CVQA_EN`, `CVQA_LOC`, `CharXiv_descriptive_val`, `CharXiv_reasoning_val`
`ChartCap`, `ChartMimic_v1`, `ChartMimic_v2`, `ChartMuseum_dev`
`ChartMuseum_test`, `ChartQAPro`, `ChartQA_TEST`, `ChartX`
`CoreCognition`, `CountBenchQA`, `Creation_MMBench`, `DA-2K`
`DREAM-1K`, `DSRBench`, `DUDE`, `DUDE_MINI`
`Design2Code`, `Detailed_Difference`, `DocVQA_TEST`, `DocVQA_VAL`
`DynaMath`, `EMMA`, `ERIQ`, `ERQA`
`EgoExoBench_64frame`, `EgoExoBench_MCQ`, `EmbSpatialBench`, `EuPhO_2024`
`EuPhO_2025`, `F_MA_2024`, `F_MA_2025`, `FoxBench`
`GMAI-MMBench_TEST`, `GMAI-MMBench_VAL`, `GOBench`, `GQA_TestDev_Balanced`
`GSM8K-V`, `GroundingME`, `HRBench4K`, `HRBench8K`
`HallusionBench`, `HiPhO_ALL`, `IPhO_2024`, `IPhO_2025`
`InfoVQA_TEST`, `InfoVQA_VAL`, `Instance_Comparison`, `K-DTCBench`
`LEGO`, `LENS-CN-QA`, `LENS-CN-QA_MINI`, `LLaVABench`
`LiveMMBench_Creation`, `LiveMMBench_Infographic`, `LiveMMBench_Perception`, `LiveMMBench_Reasoning`
`LogicVista`, `LongVideoBench`, `M4Bench`, `MATBench`
`MEGABench`, `MIA-Bench`, `MLLMGuard_DS`, `MLVU`
`MM-HELIX`, `MM-IFEval`, `MM-Math`, `MMAlignBench`
`MMBench`, `MMBench-Video`, `MMCR`, `MMDU`
`MME`, `MME-RealWorld`, `MME-RealWorld-CN`, `MME-RealWorld-Lite`
`MME-Reasoning`, `MMESCI_EN`, `MMESCI_ES`, `MMESCI_FR`
`MMESCI_JA`, `MMESCI_VisionOnly`, `MMESCI_ZH`, `MMGenBench-Domain`
`MMGenBench-Test`, `MMLongBench_DOC`, `MMMB`, `MMMU_DEV_VAL`
`MMMU_Pro_10c`, `MMMU_Pro_V`, `MMMU_TEST`, `MMReason_testmini`
`MMSIBench_circular`, `MMSIBench_wo_circular`, `MMSIVideoBench`, `MMSci_DEV_Captioning_image_only`
`MMSci_DEV_Captioning_with_abs`, `MMSci_DEV_MCQ`, `MMStar`, `MMStar_MINI`
`MMT-Bench_ALL`, `MMT-Bench_VAL`, `MMVMBench`, `MMVP`
`MMVet`, `MMVet_Hard`, `MM_NIAH_TEST`, `MM_NIAH_VAL`
`MOAT`, `MSEarthMCQ`, `MTL_MMBench_DEV`, `MTVQA_TEST`
`MUIRBench`, `MVBench`, `MVTamperBench`, `MVTamperBenchEnd`
`MVTamperBenchStart`, `MVU-Eval`, `MaCBench`, `MathCanvas-Bench`
`MathVerse_MINI`, `MathVision`, `MathVision_MINI`, `MathVista_MINI`
`MedXpertQA_MM_test`, `MedqbenchCaption`, `MedqbenchMCQ`, `MedqbenchPairedDescription_dev`
`MedqbenchPairedDescription_test`, `MicroBench`, `MicroVQA`, `MindCubeBench_raw_qa`
`MindCubeBench_tiny_raw_qa`, `MovieChat1k`, `NBPhO_2024`, `NBPhO_2025`
`NaturalBenchDataset`, `OCRBench`, `OCRBench_MINI`, `OCRBench_v2`
`OCRVQA_TEST`, `OCRVQA_TESTCORE`, `OCR_Reasoning`, `OMTGBench`
`OST`, `OSWorld_G`, `OceanOCRBench`, `OlympiadBench`
`Omni3DBench`, `OmniDocBench`, `OmniEarth-Bench`, `OmniMedVQA`
`OmniSpatialBench`, `POPE`, `PanMechanics_2024`, `PanMechanics_2025`
`PanPhO_2024`, `PanPhO_2025`, `PathMMU_TEST`, `PathMMU_VAL`
`PathVQA_TEST`, `PathVQA_VAL`, `PhyX_MC`, `PhyX_OE`
`PhyX_mini_MC`, `PhyX_mini_OE`, `Physics`, `PlotQA`
`PuzzleVQA`, `Q-Bench1_TEST`, `Q-Bench1_VAL`, `QBench_Video`
`QSpatial_plus`, `QSpatial_scannet`, `R-Bench-Dis`, `R-Bench-Ref`
`RealWorldQA`, `ReasonMap-Plus`, `RefCOCO`, `RefSpatial`
`RefSpatial-Bench`, `RefSpatial-Bench-Location`, `RefSpatial-Bench-Placement`, `RefSpatial-Bench-Unseen`
`RefSpatial-Location`, `RefSpatial-Placement`, `RefSpatial-Unseen`, `RoboSpatialHome`
`SArena_MINI`, `SCAM`, `SEEDBench2`, `SEEDBench_IMG`
`SFE`, `SFE-zh`, `SLIDEVQA`, `SLIDEVQA_MINI`
`SPBench-MV`, `SPBench-SI`, `SSI_Bench`, `STI-Bench`
`ScienceQA_TEST`, `ScienceQA_VAL`, `ScreenSpot`, `ScreenSpot_Pro`
`ScreenSpot_v2`, `SeePhys`, `SimpleVQA`, `SiteBenchImage`
`SiteBenchVideo`, `SparBench`, `SparBench_tiny`, `Spatial457`
`SpatialEval`, `SpatialVizBench`, `Spatial_Perception`, `StareBench`
`State_Comparison`, `State_Invariance`, `StaticEmbodiedBench`, `T`
`TableVQABench`, `TallyQA`, `TaskMeAnything_v1_imageqa_random`, `TempCompass`
`TextVQA_VAL`, `TopViewRS`, `TreeBench`, `UniSVG`
`V`, `V2P-Bench`, `V2PBench_128frame_nopack`, `V2PBench_16frame_nopack`
`V2PBench_1fps_nopack`, `V2PBench_2frame_nopack`, `V2PBench_64frame_nopack`, `V2PBench_8frame_nopack`
`VBGD`, `VCR-Bench`, `VCRBench_16frame_nopack`, `VCRBench_1fps_nopack`
`VCRBench_32frame_nopack`, `VCRBench_64frame_nopack`, `VCRBench_8frame_nopack`, `VCR_EN_EASY_100`
`VCR_EN_EASY_500`, `VCR_EN_EASY_ALL`, `VCR_EN_HARD_100`, `VCR_EN_HARD_500`
`VCR_EN_HARD_ALL`, `VCR_ZH_EASY_100`, `VCR_ZH_EASY_500`, `VCR_ZH_EASY_ALL`
`VCR_ZH_HARD_100`, `VCR_ZH_HARD_500`, `VCR_ZH_HARD_ALL`, `VDC`
`VGRPBench`, `VL-RewardBench`, `VLADBench`, `VLM2Bench`
`VLMBias`, `VLMBlind`, `VLRMBench`, `VMCBench_DEV`
`VMCBench_TEST`, `VSI-Bench`, `VSI-Bench-Debiased`, `VSR-zeroshot`
`VStarBench`, `VenusBench-GD`, `Video-MME`, `Video-TT`
`VideoMMMU`, `Video_Holmes`, `Video_MMLU_CAP`, `Video_MMLU_QA`
`Video_TT_16frame`, `Video_TT_32frame`, `Video_TT_64frame`, `ViewSpatialBench`
`VisFactor`, `VisOnlyQA-VLMEvalKit`, `VisuLogic`, `VisualPuzzles`
`VizWiz`, `VsiSuperCount_10mins`, `VsiSuperCount_120mins`, `VsiSuperCount_30mins`
`VsiSuperCount_60mins`, `VsiSuperRecall_10mins`, `VsiSuperRecall_120mins`, `VsiSuperRecall_240mins`
`VsiSuperRecall_30mins`, `VsiSuperRecall_60mins`, `WeMath`, `WildDoc`
`WildVision`, `WorldMedQA-V`, `WorldSense`, `WorldVQA`
`XLRS-Bench-lite`, `ZEROBench`, `atomic_dataset`, `c`
`e`, `electro_dataset`, `h`, `hle`
`mechanics_dataset`, `moviechat1k_breakpoint_8frame`, `moviechat1k_global_14frame`, `moviechat1k_global_8frame_limit0.01`
`n`, `npmm`, `olmOCRBench`, `optics_dataset`
`quantum_dataset`, `statistics_dataset`, `tdbench_cs_depth`, `tdbench_cs_height`
`tdbench_cs_integrity`, `tdbench_cs_zoom`, `tdbench_grounding_rot0`, `tdbench_grounding_rot180`
`tdbench_grounding_rot270`, `tdbench_grounding_rot90`, `tdbench_rot0`, `tdbench_rot180`
`tdbench_rot270`, `tdbench_rot90`, `vlms_are_biased_main`

</details>

---
## 📦 Installation

### 🔧 Prerequisites

Tested on Ubuntu with Python **3.10.2**. 
A CUDA GPU is required only for local model inference (the `huggingface` backend); the `vllm`/`sglang`/`api/*` backends run without a GPU.

```bash
# ffmpeg is needed for audio/video benchmarks.
# Java 11 is required for the pycocoevalcap SPICE / SPIDER metrics.
# Avoid newer JREs (Java 17+/21): their module-access tightening breaks the
# FST serialization library bundled with the SPICE jars.
sudo apt-get update && sudo apt-get install -y ffmpeg openjdk-11-jre-headless
sudo update-alternatives --set java /usr/lib/jvm/java-11-openjdk-amd64/bin/java
```

### 🤖 Install — For AI Agents

If you're an AI coding agent (e.g. Claude Code) working in this repo, you can use the **`setup-env`** skill at [`.claude/skills/setup-env`](.claude/skills/setup-env/SKILL.md) instead of running install commands by hand.
Invoke it with `/setup-env` (or with trigger phrases like *"install env"*, *"setup environment"*).

The skill:

- Creates isolated per-group venvs (`.omni_<group>`, `uv`, Python 3.10.2) so mutually-exclusive extras never clash.
- Reads `pyproject.toml` as the source of truth for pinned commits, and prefers submodule-first installs over `pip+git`.
- Drives selection interactively (one engine/model group at a time) and runs an `import` verification at the end.

> First install the system prerequisites ([Prerequisites](#-prerequisites)) — the skill manages the Python environment only.

### 🧑 Install — For Humans

If you're setting up the environment manually, we strongly recommend [`uv`](https://github.com/astral-sh/uv) for fast, reproducible installs.

**1. Create and activate a virtual environment**

```bash
# install uv (if you don't have it), then create the env
uv venv ~/.omni --python=3.10.2
source ~/.omni/bin/activate
```

**2. Clone and install**

```bash
git clone https://github.com/naver-ai/omni-evaluator.git
cd omni-evaluator
uv pip install -e ".[lmms_eval,lm_eval]" --cache-dir=/tmp
uv pip install flash-attn --no-build-isolation  # optional; requires CUDA compute capability >= 8.0
python -m nltk.downloader all
```

**3. (Optional) Install extras** — install extras for the specific evaluation engines and models you plan to use. See [Optional Dependencies](#-optional-dependencies).

That's it — head to the [Quick Start](#-quick-start). 
If you hit a snag during install, check [Troubleshooting](#troubleshooting).

### 🧩 Optional Dependencies

Install extras according to the evaluation engine and model you intend to use. Each extras group pins different dependency versions, so **installing multiple extras at once may cause version conflicts** — when in doubt, reinstall with a single group before running.

#### Evaluation Engines

| Extra | Package | Repository | Commit | Verified |
|-------|---------|------------|--------|----------|
| `lmms_eval` | lmms_eval | [EvolvingLMMs-Lab/lmms-eval](https://github.com/EvolvingLMMs-Lab/lmms-eval) | `540724a` | 2026.03.14 |
| `lm_eval` | lm_eval | [EleutherAI/lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness) | `d800e04` | 2026.03.14 |
| `vlmeval` | vlmeval | [open-compass/VLMEvalKit](https://github.com/open-compass/VLMEvalKit) | `0080421` | 2026.03.14 |

```bash
pip install -e ".[lmms_eval]"               # single install
pip install -e ".[lmms_eval,lm_eval]"       # multiple install
```

> **Note:** Installing some engines via `pip install -e` (pip+git) can drop files — e.g. `vlmeval` is missing `__init__.py` in some upstream modules, which causes import errors. When that happens, clone the submodule yourself at the commit pinned in `pyproject.toml` and install from the local path:

```bash
# Example: lmms_eval — clone under submodules/ at the pinned hash, then install editable
git clone https://github.com/EvolvingLMMs-Lab/lmms-eval.git submodules/lmms-eval
cd submodules/lmms-eval
git checkout 540724a5250672b03dc6a6b4b38cff39d5868eb6   # rev pinned in pyproject.toml
uv pip install -e . --cache-dir=/tmp
```

#### Model-specific Dependencies

Additional packages required only when running inference on specific models with the `huggingface` backend. 
Not needed for `vllm`, `sglang`, or `api/*`.

| Extra | Package (Version) | Repository / Source | Commit | Verified | Inference Engine / Model |
|-------|-------------------|---------------------|--------|----------|--------------------------|
| `deepseek_vl` | deepseek_vl | [deepseek-ai/DeepSeek-VL](https://github.com/deepseek-ai/DeepSeek-VL) | `681bffb` | 2026.02.04 | `huggingface` / `deepseek_vl` |
| `emu3` | emu3 | [baaivision/Emu3](https://github.com/baaivision/Emu3) | `9d0ae34` | 2025.12.25 | `huggingface` / `emu3` |
| `janus` | janus | [deepseek-ai/Janus](https://github.com/deepseek-ai/Janus) | `1daa72f` | 2025.12.25 | `huggingface` / `janus`, `janus_pro` |
| `minicpmo` | minicpmo (0.1.2), minicpmo-utils[all], onnx, onnxruntime, hyperpyyaml | PyPI | - | - | `huggingface` / `mini_cpm_o` |
| `voxtral` | mistral-common[audio] (>=1.5.0) | PyPI | - | - | `huggingface` / `voxtral` |

```bash
pip install -e ".[janus]"                   # single install
pip install -e ".[janus,emu3]"              # multiple install
```

<details>
<summary><b>Model-specific notes</b> (mini_cpm_o, voxtral)</summary>

- **mini_cpm_o**: Even after installing the `minicpmo` extras, the following packages must be pinned to exact versions:
  ```bash
  uv pip install peft==0.17.1 transformers==4.51.0 vector_quantize_pytorch vocos
  ```
- **voxtral**: Install `mistral-common[audio]` (`from mistral_common.audio import Audio`).

</details>

<details>
<summary><b>Submodule packages</b> (CharXiv, tau-bench, MultimodalOCR, …)</summary>

#### Custom-Packaged Submodules

Three evaluation-related submodules (`CharXiv`, `Tar`, `VoiceBench`) are research codebases without Python packaging metadata. Custom `setup.py` wrappers are provided in `submodules/_packaging/` to make them pip-installable.

| Submodule | Package | Purpose | Processed in |
|-----------|---------|---------|--------------|
| `CharXiv` | `charxiv` | Chart understanding evaluation for multimodal LLMs | `evaluation/builtin/tasks/charxiv_*`, `omni_bench_test` |
| `Tar` | `ta_tok` | Text-aligned visual tokenizer (TaTok) | `modules/image_generation/ta_tok/` |
| `VoiceBench` | `voice_bench` | Voice assistant evaluation benchmark | `evaluation/builtin/tasks/voice_bench_test` |

```bash
git submodule update --init submodules/CharXiv
cp -r submodules/_packaging/CharXiv/* submodules/CharXiv/
pip install submodules/CharXiv
```

Repeat for `Tar`, `VoiceBench`.

> **Install caveats** (the wrappers leave `install_requires` empty so the install never downgrades
> the project's pinned stack — `transformers>=4.57`, `datasets>=4.0`; install with `--no-deps`):
> - **CharXiv** uses bare intra-package imports (`from constants import ...`). The `charxiv_*` task
>   shims this at runtime (`sys.modules["constants"] = charxiv.constants`), so it works as-is once
>   installed — no manual fix needed.
> - **VoiceBench**'s evaluators need a few *lightweight* import-time deps the wrapper omits — install
>   `qa_metrics` and the `contractions` chain (`contractions textsearch pyahocorasick anyascii`) with
>   `--no-deps`. Do **not** install its full `requirements.txt` (`transformers==4.47` / `datasets==3.0`
>   conflict with the modern venv); the eval path uses an OpenAI API judge, not local whisper/litgpt.

#### Self-Packaged Submodules

These submodules ship their own packaging metadata (`setup.py` or `pyproject.toml`) and can be installed directly.

| Submodule | Package | Purpose | Task | Processed in |
|-----------|---------|---------|------|--------------|
| `tau-bench` | `tau_bench` | Tool-agent user benchmark | `tau_bench_test` | `evaluation/builtin/tasks/tau_bench_test` |
| `tau2-bench` | `tau2` | Tool-agent user benchmark v2 | `tau2_bench_test` | `evaluation/builtin/tasks/tau2_bench_test`, `live_code_bench_test` |

```bash
git submodule update --init submodules/tau-bench submodules/tau2-bench
pip install submodules/tau-bench submodules/tau2-bench
```

#### Data Submodules

These submodules provide evaluation data only and do not require installation.

| Submodule | Purpose | Task | Processed in |
|-----------|---------|------|--------------|
| `MultimodalOCR` | OCRBench / OCRBench v2 evaluation data | `ocr_bench_test`, `ocr_bench_v2_test` | `evaluation/builtin/tasks/ocr_bench_test`, `ocr_bench_v2_test` |

```bash
git submodule update --init submodules/MultimodalOCR
```

</details>

---

## 🚀 Quick Start

The Quick Start runs one image-text benchmark end-to-end. 
It evaluates **Qwen2.5-Omni-3B** on the local **`huggingface`** backend against `textvqa_val` from the **`lmms_eval`** engine. 
A CUDA GPU and a one-time model download (~7 GB) are required; the run uses `--debug` mode (3 samples), so it finishes within a couple of minutes.

**1. Configure `.env`.** Copy the template (git-ignored, auto-loaded at runtime via `dotenv`), then fill in only what your run needs:

```bash
cp .env.example .env
```

```bash
# ── HuggingFace — required for the Quick Start below (huggingface backend) ──
HF_TOKEN="hf_..."                # required by the huggingface engine; any non-empty value works for public models
HF_HOME="/mnt/tmp/hf"            # point large model / dataset downloads to a volume with free space —
HF_HUB_CACHE="${HF_HOME}/hub"    # keeps the home / root volume from filling up

# ── Inference-engine API keys — only for the api/* and vllm backends ──
OPENAI_API_KEY="sk-..."          # api/openai
ANTHROPIC_API_KEY="sk-ant-..."   # api/anthropic
GOOGLE_API_KEY="..."             # api/google
VLLM_API_KEY="..."               # vllm

# ── S3 (S3ClientArgs) — only when builtin benchmarks pull their data from S3 ──
S3_BUCKET_NAME="..."   S3_ACCESS_KEY="..."   S3_SECRET_KEY="..."   S3_ENDPOINT_URL="..."
```

> Additional submodule / external-library variables (VLMEvalKit, lmms-eval, tau-bench, experiment tracking, proxy, …) live in [`.env.example`](.env.example) under **SECTION 2** — copy what you need.

**2. Run one benchmark:**

```bash
CUDA_VISIBLE_DEVICES=0 python run.py evaluate \
    --inference_engine="huggingface" \
    --model_name_or_path="Qwen/Qwen2.5-Omni-3B" \
    --evaluation_engine="lmms_eval" \
    --benchmarks="textvqa_val" \
    --exp_name="quickstart" \
    --output_dirpath="/mnt/tmp/omni_output" \
    --torch_dtype="bfloat16" \
    --debug \ # Only 3 samples are used
    --do_async \
    --verbose
```

**3. Expected output.** You'll see a one-time model download, then inference and scoring (abridged):

```
INFO  Set `exp_name`: debug__quickstart__huggingface
INFO  Set `output_dirpath`: /mnt/tmp/omni_output/debug__quickstart__huggingface/checkpoint-none/lmms_eval
INFO  Execute inference
...   (one-time model download, then inference on 3 samples — --debug)
INFO  Saved output: /mnt/tmp/omni_output/debug__quickstart__huggingface/checkpoint-none/lmms_eval/output/textvqa_val__generation.json

# EvaluationOutput [textvqa_val] (huggingface/lmms_eval)
- evaluation_method        : generation
- num_samples              : 3.0000
- coverage_inference       : 1.0000
- coverage_evaluation      : 1.0000
- metrics (overall)        :
	- exact_match    : 1.0000
	- exact_match_stderr: 0.0000
```

Results are written under your output directory, organized as:

```
<output_dir>/<exp_name>/<version>/<evaluation_engine>/output/<benchmark>__<method>.json
```

> **No GPU?** Swap in an API backend instead — e.g. `--inference_engine="api/openai" --api_name="gpt-4o"` with `--evaluation_engine="builtin" --benchmarks="mmbench_en_dev"` (set `OPENAI_API_KEY` in `.env`). This needs no GPU and no model download (see the [Engine-specific Guides](#cli-reference)).

---

## CLI Reference

Everything runs through a single entry point — `python run.py` — and the entire CLI is covered by just **two subcommands**:

```
python run.py [-h] {list,evaluate} ...
```

| Subcommand | What it's for |
|------------|---------------|
| [`list`](#list--query-available-engines-and-tasks) | **Discover** what you can run — the available inference / evaluation engines and their tasks |
| [`evaluate`](#evaluate--run-evaluation) | **Run** an evaluation end-to-end (identical to `python evaluate.py`) |

The usual flow is two steps: `list` to find an engine and task, then `evaluate` to run it. Running with no arguments prints help.

### `list` — Query available engines and tasks

```bash
# List inference engines
python run.py list --inference_engines
# → ['huggingface', 'vllm', 'sglang', 'api/openai', 'api/anthropic', 'api/google']

# List evaluation engines
python run.py list --evaluation_engines
# → ['builtin', 'lmms_eval', 'lm_eval_harness', 'vlm_eval_kit']

# List tasks for a specific evaluation engine
python run.py list --tasks --evaluation_engine="builtin"
```

| Flag | Description |
|------|-------------|
| `--inference_engines` | Print available inference engines |
| `--evaluation_engines` | Print available evaluation engines |
| `--tasks` | Print tasks for an engine. Must be used with `--evaluation_engine` |
| `--evaluation_engine` | Engine to query: `builtin`, `lmms_eval`, `lm_eval_harness`, `vlm_eval_kit` |

> `--tasks` requires `--evaluation_engine`. Querying `lmms_eval`, `lm_eval_harness`, or `vlm_eval_kit` requires the corresponding optional dependency to be installed.

### `evaluate` — Run evaluation

`python run.py evaluate` behaves identically to `python evaluate.py`. For engine-specific arguments and example commands, see the [Engine-specific Guides](#engine-specific-guides) below.

**Common arguments**

| Argument | Default | Description |
|----------|---------|-------------|
| `--inference_engine` | **(required)** | Inference backend: `huggingface`, `vllm`, `sglang`, `api/openai`, `api/anthropic`, `api/google` |
| `--evaluation_engine` | `builtin` | Evaluation framework: `builtin`, `lmms_eval`, `lm_eval_harness`, `vlm_eval_kit` |
| `--exp_name` | **(required)** | Experiment name used for the result directory and display |
| `--benchmarks` | (engine default) | Comma-separated list of benchmark names |
| `--do_async` | `false` | Enable asynchronous parallel requests |
| `--resume` | `false` | Skip benchmarks that already have saved results |
| `--skip_inference` | `false` | Skip inference and reuse existing results |
| `--reasoning` | `false` | Enable reasoning mode |
| `--reasoning_effort` | `None` | Reasoning effort for OpenAI o-series: `low`, `medium`, `high` |
| `--thinking_budget` | `None` | Thinking token budget for Anthropic/Google thinking models |
| `--debug` | `false` | Debug mode with reduced dataset |
| `--seed` | `None` | Random seed (vLLM, SGLang) |

<details>
<summary><b>Generation options</b> (temperature, top_p, max_new_tokens, …)</summary>

Generation options are shared across engines via `GenerationOptionArgs`. Each engine uses only the subset it supports — unsupported options are silently dropped. See each engine's README for per-engine support.

| Argument | Default | Description |
|----------|---------|-------------|
| `--temperature` | `None` | Sampling temperature |
| `--top_p` | `None` | Top-p (nucleus) sampling threshold |
| `--top_k` | `None` | Top-k filtering (not supported by OpenAI) |
| `--num_beams` | `None` | Beam search width (HuggingFace only) |
| `--max_new_tokens` | `None` | Maximum new tokens to generate |
| `--repetition_penalty` | `None` | Repetition penalty; maps to `frequency_penalty` for OpenAI/Google |
| `--length_penalty` | `None` | Length penalty (HuggingFace only) |
| `--stop_words` | `None` | Comma-separated stop sequences |
| `--frequency_penalty` | `None` | Frequency penalty (OpenAI, Google) |
| `--presence_penalty` | `None` | Presence penalty (OpenAI, Google) |
| `--n` | `None` | Number of independent output sequences (vLLM, SGLang, OpenAI) |
| `--logprobs` | `None` | Number of log-probability tokens per step (vLLM, SGLang, OpenAI) |
| `--top_logprobs` | `None` | Top token log probabilities to return (OpenAI only) |
| `--do_sample` | `None` | Enable sampling; `None` uses model default (HuggingFace only) |

</details>

### Engine-specific Guides

For detailed arguments, required environment variables, and **example run commands**, see each engine's README:

| Engine Type | Engine | README |
|-------------|--------|--------|
| Inference | `huggingface` | [`omni_evaluator/inference/huggingface/`](omni_evaluator/inference/huggingface/README.md) |
| Inference | `vllm` | [`omni_evaluator/inference/vllm/`](omni_evaluator/inference/vllm/README.md) |
| Inference | `api/openai`, `api/anthropic`, `api/google` | [`omni_evaluator/inference/api/`](omni_evaluator/inference/api/README.md) |
| Evaluation | `builtin` | [`omni_evaluator/evaluation/builtin/`](omni_evaluator/evaluation/builtin/README.md) |
| Evaluation | `lmms_eval` | [`omni_evaluator/evaluation/lmms_eval/`](omni_evaluator/evaluation/lmms_eval/README.md) |
| Evaluation | `lm_eval_harness` | [`omni_evaluator/evaluation/lm_eval_harness/`](omni_evaluator/evaluation/lm_eval_harness/README.md) |
| Evaluation | `vlm_eval_kit` | [`omni_evaluator/evaluation/vlm_eval_kit/`](omni_evaluator/evaluation/vlm_eval_kit/README.md) |
<!-- | Inference | `sglang` | [`omni_evaluator/inference/sglang/`](omni_evaluator/inference/sglang/README.md) | -->

---

## Usage

### Local Evaluation

Run evaluations locally with `python run.py evaluate` or the equivalent `python evaluate.py`. Run with `--help` to see every option.

**Useful flags**

- `--resume` — when running multiple benchmarks, skip those that already have saved results.
- `--skip_inference` — reuse existing inference results and re-run only the evaluation (e.g. with different eval parameters).
- **Leaderboard submission** — for benchmarks requiring submission, submission files are generated automatically ([submission benchmarks](./omni_evaluator/submission/leaderboard.py)).
- Video benchmarks in `lmms_eval` and `vlm_eval_kit` may take several minutes or more to load datasets.

### Remote Evaluation

`launch_server.py` is a FastAPI-based job-queue server for submitting, querying, and deleting evaluation jobs over HTTP.

<details>
<summary><b>Start the server & API endpoints</b></summary>

**Start the server**

```bash
python launch_server.py \
    --host 0.0.0.0 \
    --port 8080 \
    --base "python evaluate.py" \
    --max_concurrent 1 \
    --log_dir "./logs"
```

| Argument | Description | Default |
|----------|-------------|---------|
| `--host` | Server bind address | `0.0.0.0` |
| `--port` | Server port | `8080` |
| `--base` | Base command to execute | `python evaluate.py` |
| `--max_concurrent` | Maximum number of concurrent jobs | `1` |
| `--log_dir` | Directory for job logs | `./logs` |

**`POST /add_job` — Add a job**

```bash
# Pass arguments as a string
curl -X POST http://localhost:8080/add_job \
    -H "Content-Type: application/json" \
    -d '{"arguments": "--task=docvqa --model_path=/path/to/model"}'

# Pass arguments as a dict
curl -X POST http://localhost:8080/add_job \
    -H "Content-Type: application/json" \
    -d '{"arguments": {"task": "docvqa", "model_path": "/path/to/model"}}'
```

**`POST /get_state` — Query job status**

```bash
curl -X POST http://localhost:8080/get_state \
    -H "Content-Type: application/json" \
    -d '{"pid": "<job_pid>"}'
```

Status: `pending` → `inprogress` → `completed` / `failed` / `terminated`

**`POST /remove_job` — Remove/terminate a job**

```bash
curl -X POST http://localhost:8080/remove_job \
    -H "Content-Type: application/json" \
    -d '{"pid": "<job_pid>"}'
```

**`GET /jobs` — List all jobs**

```bash
curl http://localhost:8080/jobs
```

</details>

---

## Troubleshooting

Common setup and runtime issues, grouped by where they occur. 
Expand a group to see fixes.

<details>
<summary><b>Inference engines</b> (flash-attn, mini_cpm_o, voxtral)</summary>

**`flash-attn` fails to build or import**
Cause: GPU compute capability below 8.0. flash-attn is optional.
Fix: install only on a GPU with compute capability ≥ 8.0:
```bash
uv pip install flash-attn --no-build-isolation
```

**`mini_cpm_o` doesn't work even after installing the `minicpmo` extras**
Cause: a few packages aren't version-pinned by the extras group.
Fix: pin them explicitly:
```bash
uv pip install peft==0.17.1 transformers==4.51.0 vector_quantize_pytorch vocos
```

**`voxtral` audio import fails** (`from mistral_common.audio import Audio`)
Cause: the `mistral-common` audio extra is missing.
Fix:
```bash
pip install "mistral-common[audio]"
```

</details>

<details>
<summary><b>Evaluation engines & metrics</b> (Java, CoreNLP, COMET, VLMEvalKit)</summary>

**SPICE / SPIDER metrics crash on Java 17 or 21**
Cause: newer JREs tighten module access, which breaks the FST serialization library bundled with the SPICE jars.
Fix: use Java 11:
```bash
sudo apt-get install -y openjdk-11-jre-headless
sudo update-alternatives --set java /usr/lib/jvm/java-11-openjdk-amd64/bin/java
```

**`pycocoevalcap` fails to download Stanford CoreNLP** (`urllib.error.ContentTooShortError`)
Cause: the runtime auto-download is unreliable.
Fix: pre-fetch the two jars from Maven Central and place them under `<pycocoevalcap>/spice/lib/`:
```bash
SPICE_LIB="$(python -c 'import pycocoevalcap, os; print(os.path.join(os.path.dirname(pycocoevalcap.__file__), "spice", "lib"))')"
curl -L --retry 5 -o "$SPICE_LIB/stanford-corenlp-3.6.0.jar" \
    https://repo1.maven.org/maven2/edu/stanford/nlp/stanford-corenlp/3.6.0/stanford-corenlp-3.6.0.jar
curl -L --retry 5 -o "$SPICE_LIB/stanford-corenlp-3.6.0-models.jar" \
    https://repo1.maven.org/maven2/edu/stanford/nlp/stanford-corenlp/3.6.0/stanford-corenlp-3.6.0-models.jar
```

**`unbabel-comet` is rejected by the resolver** (uv / modern pip)
Cause: COMET hard-pins `transformers<5`, `torchmetrics<0.11`, etc., which conflict with the project's modern stack. The runtime works fine despite those metadata pins.
Fix: install without dependencies:
```bash
uv pip install --no-deps --cache-dir=/tmp unbabel-comet
```
COMET is only needed for `fleurs_en2ko_test`, `fleurs_en2zh_test`, `fleurs_ko2en_test`, `fleurs_zh2en_test` — skip it otherwise.

**`vlmeval` import errors after a pip+git install**
Cause: some upstream modules are missing `__init__.py` files.
Fix: clone the repository manually and install from the local path:
```bash
cd submodules
git clone https://github.com/open-compass/VLMEvalKit.git
cd VLMEvalKit && git checkout 0080421
uv pip install -e . --cache-dir=/tmp
```

</details>

<details>
<summary><b>Other / general install</b> (extras conflicts, broken extras)</summary>

**Version conflicts after installing multiple extras at once**
Cause: each extras group pins different dependency versions.
Fix: reinstall with a single extras group before running an evaluation:
```bash
pip install -e ".[lmms_eval]"   # one group at a time
```

**Extras dependencies are broken or incomplete**
Fix: install the base package first, then install each evaluation engine manually from `submodules/`:
```bash
cd submodules
git clone https://github.com/EvolvingLMMs-Lab/lmms-eval.git
cd lmms-eval
git checkout 540724a
uv pip install -e . --cache-dir=/tmp
```

</details>

---

## License

```
OmniEvaluator
Copyright (c) 2026-present NAVER Cloud Corp.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
implied. See the License for the specific language governing
permissions and limitations under the License.
```

See [LICENSE](./LICENSE) and [NOTICE](./NOTICE).
