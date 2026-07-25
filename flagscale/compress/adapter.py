from typing import Any

from llmcompressor import oneshot
from transformers import PreTrainedModel, PreTrainedTokenizer

from flagscale.logger import logger


class LLMCompressorAdapter:
    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: Optional[PreTrainedTokenizer] = None,
        dataset: Optional[Any] = None,
        output_dir: str = "./outputs",
        num_calibration_steps: int = 512,
        **kwargs,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.dataset = dataset
        self.output_dir = output_dir
        self.num_calibration_steps = num_calibration_steps

        self.algo = kwargs.get("algo", {})
        self.scheme = kwargs.get("scheme", "W8A16")
        self.targets = kwargs.get("targets", ["Linear"])
        self.ignore = kwargs.get("ignore", [])

        self.is_mix_precision = (self.scheme == "mix_precision_search") or (
            isinstance(self.algo, str) and self.algo == "mix_precision"
        )

    def _prepare_recipe(self):
        from llmcompressor.modifiers.quantization import QuantizationModifier

        if not self.is_mix_precision:
            modifier = QuantizationModifier(
                targets=self.targets,
                ignore=self.ignore,
                scheme=self.scheme,
                **(self.algo if isinstance(self.algo, dict) else {}),
            )
            return [modifier]

        else:
            logger.info("Detected Mixed Precision Mode. Recipe will be handled by the pipeline.")
            return None

    def run(self):
        logger.info(f"Starting compression with scheme: {self.scheme}")

        if self.is_mix_precision:
            try:
                import flagscale.compress.pipelines.mix_precision_pipeline  # noqa: F401

                logger.info("Successfully registered MixPrecisionPipeline.")
            except ImportError as e:
                raise ImportError(
                    f"Failed to import mix_precision_pipeline: {e}. Please check your PYTHONPATH."
                )

        recipe = self._prepare_recipe()

        oneshot_args = {
            "model": self.model,
            "dataset": self.dataset,
            "output_dir": self.output_dir,
            "num_calibration_batches": self.num_calibration_steps,
        }

        if self.is_mix_precision:
            from llmcompressor.pipelines.registry import CalibrationPipeline

            # pipeline_cls = CalibrationPipeline.load("mix_precision_search")
            pipeline_cls = CalibrationPipeline.load_from_registry("mix_precision_search")

            logger.info("Invoking MixPrecisionPipeline manually...")
            pipeline_cls(
                model=self.model,
                dataloader=self.dataset,
                dataset_args=None,
                output_dir=self.output_dir,
            )

        else:
            oneshot_args["recipe"] = recipe
            oneshot(**oneshot_args)

        self.save_artifacts()

    def save_artifacts(self):

        if self.tokenizer:
            self.tokenizer.save_pretrained(self.output_dir)
        logger.info(f"Artifacts saved to {self.output_dir}")
