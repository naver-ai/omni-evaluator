# CLIcK Generative

This is the **generative variant** of the CLIcK benchmark, where models generate free-form answers instead of choosing from multiple choice options.

## Overview

Based on the CLIcK dataset: A Benchmark Dataset of Cultural and Linguistic Intelligence in Korean.

## Dataset Information

### Paper

Title: `CLIcK: A Benchmark Dataset of Cultural and Linguistic Intelligence in Korean`

Abstract: `Despite the rapid development of large language models (LLMs) for the Korean language, there remains an obvious lack of benchmark datasets that test the requisite Korean cultural and linguistic knowledge. Because many existing Korean benchmark datasets are derived from the English counterparts through translation, they often overlook the different cultural contexts. For the few benchmark datasets that are sourced from Korean data capturing cultural knowledge, only narrow tasks such as bias and hate speech detection are offered. To address this gap, we introduce a benchmark of Cultural and Linguistic Intelligence in Korean (CLIcK), a dataset comprising 1,995 QA pairs. CLIcK sources its data from official Korean exams and textbooks, partitioning the questions into eleven categories under the two main categories of language and culture. For each instance in CLIcK, we provide fine-grained annotation of which cultural and linguistic knowledge is required to answer the question correctly. Using CLIcK, we test 13 language models to assess their performance. Our evaluation uncovers insights into their performances across the categories, as well as the diverse factors affecting their comprehension. CLIcK offers the first large-scale comprehensive Korean-centric analysis of LLMs' proficiency in Korean culture and language.`

Homepage: https://huggingface.co/datasets/EunsuKim/CLIcK


### Citation

```
@misc{kim2024click,
      title={CLIcK: A Benchmark Dataset of Cultural and Linguistic Intelligence in Korean},
      author={Eunsu Kim and Juyoung Suk and Philhoon Oh and Haneul Yoo and James Thorne and Alice Oh},
      year={2024},
      eprint={2403.06412},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```

### Groups, Tags, and Tasks

#### Groups

* `click_generative`: All 11 categories of the CLIcK dataset
* `click_lang_generative`: "Language" category of the CLIcK dataset, consisting of 3 subcategories
* `click_cul_generative`: "Culture" category of the CLIcK dataset, consisting of 8 subcategories

#### Tasks

* Three tasks under `click_lang_generative`:
    * `click_lang_text_generative`
    * `click_lang_grammar_generative`
    * `click_lang_function_generative`

* Eight tasks under `click_cul_generative`:
    * `click_cul_society_generative`
    * `click_cul_tradition_generative`
    * `click_cul_politics_generative`
    * `click_cul_economy_generative`
    * `click_cul_law_generative`
    * `click_cul_history_generative`
    * `click_cul_geography_generative`
    * `click_cul_kpop_generative`

