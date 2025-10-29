from .base_llm import BaseLLM
import time
import inspect
import torch
import multiprocessing as mp

# could be dynamically imported similar to other models

from pyopenagi.utils.chat_template import Response
from ...utils.utils import get_from_env

from transformers import AutoTokenizer

from huggingface_hub import login
#login(token="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")


class vLLM(BaseLLM):

    def __init__(self, llm_name: str,
                 max_gpu_memory: dict = None,
                 eval_device: str = None,
                 max_new_tokens: int = 256,
                 log_mode: str = "console"):

        super().__init__(llm_name,
                         max_gpu_memory,
                         eval_device,
                         max_new_tokens,
                         log_mode)

    def load_llm_and_tokenizer(self) -> None:
        """ fetch the model from huggingface and run it """
        if self.max_gpu_memory:
            self.available_gpus = list(self.max_gpu_memory.keys())
            self.gpu_nums = len(self.available_gpus)
        else:
            if torch.cuda.is_available():
                self.gpu_nums = torch.cuda.device_count()
                self.available_gpus = list(range(self.gpu_nums))
            else:
                self.available_gpus = []
                self.gpu_nums = 0
        try:
            import vllm
        except Exception as exc:
            raise ImportError(
                "Could not import vllm python package. "
                "Please install it with `pip install vllm`."
            ) from exc

        """ only casual lms for now """
        llm_kwargs = {
            "model": self.model_name,
        }

        try:
            download_dir = get_from_env("HF_HOME")
            llm_kwargs["download_dir"] = download_dir
        except ValueError:
            # Allow running without HF_HOME; vLLM will fallback to default cache
            pass

        if self.gpu_nums:
            llm_kwargs["tensor_parallel_size"] = self.gpu_nums

        try:
            mp.set_start_method("spawn")
        except RuntimeError:
            pass

        self.model = vllm.LLM(**llm_kwargs)

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
        )
        self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        self.sampling_params = vllm.SamplingParams(
            temperature=0.8,
            top_p=0.95,
            max_tokens=self.max_new_tokens,
        )

    def process(self,
                agent_process,
                temperature=0.0) -> None:
        agent_process.set_status("executing")
        agent_process.set_start_time(time.time())
        self.logger.log(
            f"{agent_process.agent_name} is switched to executing.\n",
            level = "executing"
        )

        messages = agent_process.query.messages
        tools = agent_process.query.tools
        message_return_type = agent_process.query.message_return_type

        if tools:
            messages = self.tool_calling_input_format(messages, tools)
            # print(messages)
            prompt = self.tokenizer.apply_chat_template(
                messages,
                # tools = tools,
                tokenize = False
            )
            # prompt = self.parse_messages(messages)
            response = self.model.generate(
                prompt, self.sampling_params
            )
            # print(response)
            result = response[0].outputs[0].text

            import ipdb; ipdb.set_trace()

            print(f"***** Result: {result} *****")

            tool_calls = self.parse_tool_calls(
                result
            )
            if tool_calls:
                agent_process.set_response(
                    Response(
                        response_message = None,
                        tool_calls = tool_calls
                    )
                )
            else:
                agent_process.set_response(
                    Response(
                        response_message = result
                    )
                )

        else:
            prompt = self.tokenizer.apply_chat_template(
                messages,
                tokenize = False
            )

            # prompt = self.parse_messages(messages)
            response = self.model.generate(
                prompt, self.sampling_params
            )

            result = response[0].outputs[0].text
            if message_return_type == "json":
                result = self.parse_json_format(result)

            agent_process.set_response(
                Response(
                    response_message=result
                )
            )

        agent_process.set_status("done")

        agent_process.set_end_time(time.time())
