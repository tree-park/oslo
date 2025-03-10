from torch.utils.data import DataLoader
from oslo.torch.distributed import ParallelContext
from oslo.transformers.tasks.data_summarization import (
    ProcessorForSummarization,
    DataCollatorForSummarization,
)
from tests.transformers.tasks.test_data_base import TestDataBinarization
from datasets import load_dataset, DatasetDict

try:
    from transformers import T5ForConditionalGeneration
except ImportError:
    print("You have to install 'transformers' to test data_summarization.py")


class TestDataSequenceClassification(TestDataBinarization):
    def __init__(self, model_name, parallel_context=None, label_pad_token_id=-100):
        self.processor = ProcessorForSummarization(model_name, max_length=128)
        self.data_collator = DataCollatorForSummarization(
            self.processor, label_pad_token_id=label_pad_token_id
        )
        self.sp_data_collator = DataCollatorForSummarization(
            self.processor,
            parallel_context=parallel_context,
            label_pad_token_id=label_pad_token_id,
        )
        self.model_name = model_name
        self.tokenizer = self.processor._tokenizer
        self.parallel_context = parallel_context

    def __call__(
        self,
        max_length,
        dataset,
        batch_size=1024,
        batch_check_num_sample=2,
        batch_check_tokens=False,
        must_be_equal_to_max_length=False,
        model=None,
    ):
        self.processor._chunk_size = max_length
        self.processor._max_length = max_length
        self.data_collator.model = model
        if self.sp_data_collator:
            self.sp_data_collator.model = model
        print(
            "---------- Test Start ----------",
            f"Model: {self.model_name}",
            f"Max Length: {max_length}",
            f"Batch size: {batch_size}",
            sep="\n",
        )
        processed_dataset = dataset.map(
            self.processor, batched=True, remove_columns=dataset["train"].column_names
        )

        if self.data_collator.tokenizer.pad_token is None:
            self.data_collator.tokenizer.add_special_tokens({"pad_token": "<|pad|>"})
            self.tokenizer = self.data_collator.tokenizer
            print("pad_token is set.")

        dataloader = DataLoader(
            processed_dataset["train"],
            batch_size,
            shuffle=True,
            collate_fn=self.data_collator,
        )

        batch = next(iter(dataloader))
        self._batch_check(
            batch, num_samples=batch_check_num_sample, check_token=batch_check_tokens
        )

        self._length_check(
            dataloader,
            "input_ids",
            max_length,
            must_be_equal_to_max_length=must_be_equal_to_max_length,
        )

        if self.parallel_context is not None:
            self._test_sp_collator(processed_dataset, batch_size)

        print("---------- Test Pass ----------\n")


if "__main__" == __name__:
    dataset = load_dataset("xglue", "nc")
    dataset = dataset.rename_column("news_body", "text")
    dataset = dataset.rename_column("news_title", "labels")
    dataset = DatasetDict(
        {
            "train": dataset["train"],
            "validation": dataset["validation.en"],
            "test": dataset["test.en"],
        }
    )
    dataset["train"] = dataset["train"].shard(20, 1)
    dataset["validation"] = dataset["validation"].shard(10, 1)
    dataset["test"] = dataset["test"].shard(10, 1)

    # gpt2_test = TestDataSequenceClassification("gpt2")
    # gpt2_test(512, dataset, 1024, 3)
    # gpt2_test(24, dataset, 1024)

    # bert_test = TestDataSequenceClassification("bert-base-cased")
    # bert_test(512, dataset, 64, 3)
    # bert_test(16, dataset, 16)

    # roberta_test = TestDataSequenceClassification("roberta-base")
    # roberta_test(130, dataset, 512, 4)
    # roberta_test(40, dataset, 512)

    # albert_test = TestDataSequenceClassification("albert-base-v2")
    # albert_test(24, dataset, 128, 4)
    # albert_test(32, dataset, 128, 4)

    # bart_test = TestDataSequenceClassification("facebook/bart-base")
    # bart_test(32, dataset, 32, 3)
    # bart_test(64, dataset, 32)

    # t5_test = TestDataSequenceClassification("t5-small")
    # t5_test(
    #     512,
    #     dataset,
    #     1024,
    #     3,
    #     model=T5ForConditionalGeneration.from_pretrained("t5-small"),
    # )
    # t5_test(128, dataset, 1024)

    parallel_context = ParallelContext.from_torch(sequence_parallel_size=4)
    bert_sp_test = TestDataSequenceClassification(
        "t5-small", parallel_context, label_pad_token_id=0
    )
    bert_sp_test(
        253, dataset, 1024, model=T5ForConditionalGeneration.from_pretrained("t5-small")
    )
