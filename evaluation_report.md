1. Task and Dataset

The goal of this project is to fine‑tune a Phi‑3 family model for instruction‑style question answering on a small, custom text dataset. The dataset consists of user prompts paired with target responses, preprocessed into tokenized sequences suitable for causal language modeling. The data is split into three subsets: a training set used to update the model, a validation set used during development, and a held‑out test set used for final evaluation (all stored under data/processed/train, val, and test).​

Due to hardware constraints and time limits, the dataset size is modest (on the order of hundreds to a few thousand examples rather than millions), which means the results should be interpreted as a proof‑of‑concept rather than a production‑ready model. This smaller scale is also why intrinsic metrics like loss and perplexity are emphasized over large‑scale benchmark scores.​

2. Model and Training

The base model used is a Phi‑3 variant loaded via AutoModelForCausalLM and AutoTokenizer from the Hugging Face Transformers library (for example, microsoft/phi-3-mini-4k-instruct). Training is implemented with SFTTrainer from the TRL library and standard TrainingArguments, using CPU‑only execution and automatic device mapping to fit the model into limited memory. Key hyperparameters include a small per‑device batch size, gradient accumulation to simulate a larger effective batch, a learning rate in the 1e‑4 range, and 1–2 training epochs, chosen to keep runtime and memory usage manageable.​

Instead of full‑precision, multi‑GPU training, the project relies on efficient loading and offloading of model shards to CPU and disk, and disables mid‑epoch checkpointing to avoid MemoryError during save operations. Some functionality typical of a LoRA setup (saving separate adapter weights) could not be fully realized in this environment, so the main training artifact is a fine‑tuned checkpoint directory under outputs/phi-3 together with the original base model and tokenizer configuration.​

3. Quantitative Results

For intrinsic evaluation, the model is treated as a causal language model and evaluated on the processed test set using average cross‑entropy loss and perplexity. The current evaluation script (src/run_eval.py) reports, for the base model on the tokenized test sequences, a test loss of approximately 17.3651 and a corresponding perplexity of about 3.48×10⁷, indicating that the base model is extremely uncertain on this dataset and struggles to predict the next tokens accurately.​

Because of ongoing issues with saving and reloading a complete fine‑tuned checkpoint on CPU‑only hardware, the comparison between base and fine‑tuned models is limited. In practice, this means the report focuses more on base‑model behavior plus qualitative inspection of generated outputs rather than a clean “before vs after” table with large improvements on a specific task metric. In a future iteration with GPU access or smaller models, this evaluation could be extended to include side‑by‑side scores (e.g., perplexity reduction or task‑specific accuracy/ROUGE).​

4. Qualitative Examples

Qualitative analysis is carried out by manually prompting both the base model and the fine‑tuned checkpoint (when successfully loaded) with a small set of validation/test prompts, and observing differences in fluency and adherence to the target style. Typical prompts include short instruction‑like questions similar to the training data (for example, explanations, step‑by‑step answers, or domain‑specific descriptions). The base model tends to give generic or off‑topic answers on such prompts, reflecting its general‑purpose pretraining rather than specialization to this dataset.​

When the fine‑tuned model loads correctly, its outputs are observed to be more on‑topic and to follow the intended answer format more closely, especially for patterns that appeared frequently in the training set. However, because the dataset is small and training is constrained, the model still shows variability and occasional hallucinations, and the qualitative improvements are not yet consistent enough to claim strong task generalization. These observations highlight the value of fine‑tuning but also the need for more data and stable saving/loading of the fine‑tuned weights.​

5. Monitoring and Logs

Training was initially configured to use Weights & Biases for experiment tracking, but this was later disabled to simplify the setup and avoid interactive login prompts in the training script. Instead, progress is monitored through the console logs produced by the trainer, including step‑level loss values and warnings about memory usage and checkpoint saving. These logs can be redirected into a file or ingested by TensorBoard or similar tools if needed.​

Given the CPU‑only environment and repeated MemoryError during automatic checkpoint saves, the training configuration was adjusted to use save_strategy="no" and to rely on a final save_model call at the end of the run, which reduced logging richness but made the pipeline more robust. For submission, a compressed archive of the available logs and configuration (config/config.yaml, training script, and evaluation script) provides a reproducible record of the experiment settings and observed behavior.​

6. Challenges and Future Work

The main challenges in this project were hardware and tooling constraints: running a large Phi‑3 model without GPU support and with limited RAM led to long training times, offloading to disk, and failures when saving full sharded checkpoints using safetensors. These issues forced compromises such as disabling mid‑epoch saving and focusing on smaller datasets and simple metrics. Additional friction came from version mismatches and API changes (for example, SFTTrainer argument compatibility, evaluate package naming conflicts), and from aligning evaluation scripts with the exact structure of the preprocessed datasets.​

Future work should prioritize moving to a GPU‑enabled environment or using a smaller base model, enabling a proper LoRA/QLoRA setup where only adapter weights are trained and saved (reducing memory footprint and simplifying deployment). With more stable hardware and saving infrastructure, the project can expand to larger datasets, richer task‑specific metrics (e.g., ROUGE/accuracy on labeled tasks), and more extensive qualitative analysis, ultimately producing a fine‑tuned model with clearly demonstrated improvements over the base Phi‑3 checkpoint.​

| Model        | Test loss | Test perplexity    |
|--------------|-----------|--------------------|
| Base Phi‑3   | 17.3651   | 34,799,920.36      |
| Fine‑tuned   | …         | …                  |
