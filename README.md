# Beyond Surface Fluency: Domain-Robust AI-Generated Text Detection via Semantic-Syntactic Discrepancy

## background

The rapid advancement of large language models (LLMs)—such as GPT-5 and Qwen3—has made machine-generated text almost indistinguishable from human writing. This capability, however, can be exploited for malicious purposes, including fake news fabrication and dissemination, academic misconduct, and social media manipulation. Hence, developing reliable detectors for LLM-generated text has become a crucial and urgent task.

![shouye](.\images\shouye.png)

we analyze human-written and LLM-generated texts by extracting their sequential features and dependency tree information, computing the token-level distance to quantify the discrepancy between semantic and syntactic views. As illustrated in Figure 1a, this discrepancy captures the divergence between surface-level semantic fluency and deep syntactic organization. Figure 1b further shows that human-written texts exhibit a wider spread of higher discrepancy values, whereas LLM-generated texts display denser lower discrepancy, indicating a stronger coupling between sequence and structure in machine-generated content.

## method

![framework](.\images\framework.png)

We propose SSD, which detects AI-generated text by explicitly modeling the discrepancy between semantic coherence and syntactic organization. As illustrated in Figure, SSD first encodes the input text from two complementary perspectives. The resulting representations are then orthogonalized to reduce redundancy, and their token-level divergence is computed to guide an adaptive fusion process. The final representation is used for classification. 

