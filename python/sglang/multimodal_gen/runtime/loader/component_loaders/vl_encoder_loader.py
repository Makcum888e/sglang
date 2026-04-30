import atexit
import logging
import multiprocessing
import os
from typing import Any

from sglang.multimodal_gen.runtime.loader.component_loaders.component_loader import (
    ComponentLoader,
)
from sglang.multimodal_gen.runtime.server_args import ServerArgs

logger = logging.getLogger(__name__)


class VisionLanguageEncoderLoader(ComponentLoader):
    """Loader for vision language encoder via SGLang Engine."""

    component_names = ["vision_language_encoder"]
    expected_library = "transformers"
    engine = None

    def load_customized(
        self,
        component_model_path: str,
        server_args: ServerArgs,
        transformers_or_diffusers: str = "vision_language_encoder",
    ) -> Any:
        if transformers_or_diffusers == "vision_language_encoder":

            model_root = os.path.dirname(component_model_path)
            processor_path = os.path.join(model_root, "processor")

            # sglang.Engine spawns scheduler/detokenizer subprocesses, but
            # multimodal_gen GPU workers run as daemon processes which cannot
            # create children. Temporarily clear the daemon flag to allow
            # subprocess creation, then restore it.
            current = multiprocessing.current_process()
            was_daemon = current._config.get("daemon", False)
            if was_daemon:
                logger.info(
                    "Temporarily clearing daemon flag to allow "
                    "sglang.Engine subprocess creation"
                )
                current._config["daemon"] = False
            try:
                # engine = Engine(
                #    model_path=component_model_path,
                #    tokenizer_path=processor_path,
                #    mem_fraction_static=0.6,
                #    enable_multimodal=True,
                #    disable_cuda_graph=True,
                #    tp_size=server_args.tp_size if server_args.tp_size > 0 else 1,
                #    port=8764,
                # )
                # command = [
                #    "sglang",
                #    "serve",
                #    "--model-path",
                #    component_model_path,
                #    "--tokenizer-path",
                #    processor_path,
                #    "--mem-fraction-static",
                #    0.8,
                #    "--enable-multimodal",
                #    "--disable-cuda-graph",
                #    "--port",
                #    8764,
                # ]
                from sglang.test.test_utils import popen_launch_server

                self.engine = popen_launch_server(
                    component_model_path,
                    "http://127.0.0.1:8764",
                    timeout=600,
                    other_args=(
                        "--tokenizer-path",
                        processor_path,
                        "--mem-fraction-static",
                        0.5,
                        "--enable-multimodal",
                        # "--disable-cuda-graph",
                        "--cuda-graph-bs",
                        "1",
                        "--device",
                        "npu",
                        "--attention-backend",
                        "ascend",
                        # "--base-gpu-id",
                        # 2,
                        "--disable-fast-image-processor",
                        "--tp-size",
                        4,
                    ),
                )
                atexit.register(self.cleanup)
            except Exception as e:
                print(e)
            finally:
                current._config["daemon"] = was_daemon
            return self.engine
        else:
            raise ValueError(
                f"Unsupported library for VisionLanguageEncoder: {transformers_or_diffusers}"
            )

    def cleanup(self):
        if self.engine.poll() is None:
            self.engine.terminate()
            try:
                self.engine.wait(timeout=10)
            except self.engine.TimeoutExpired:
                self.engine.kill()
