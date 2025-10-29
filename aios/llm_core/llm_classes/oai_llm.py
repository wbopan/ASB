# Wrapper around OpenAI-compatible endpoints exposed locally.

import os
import time

from openai import OpenAI as Open

from .base_llm import BaseLLM
from pyopenagi.utils.chat_template import Response


class OAILLM(BaseLLM):

    def __init__(
        self,
        llm_name: str,
        max_gpu_memory: dict = None,
        eval_device: str = None,
        max_new_tokens: int = 256,
        log_mode: str = "console",
    ):
        super().__init__(
            llm_name,
            max_gpu_memory,
            eval_device,
            max_new_tokens,
            log_mode,
        )

    def load_llm_and_tokenizer(self) -> None:
        api_key = os.getenv("OPENAI_API_KEY", "EMPTY")
        # Use remote vllm server
        self.model = Open(base_url="http://e.wenbo.io:8123/v1", api_key=api_key)
        self.tokenizer = None

    def process(
        self,
        agent_process,
        temperature=0.0,
    ):

        # Hard code temperature to 0.7 for qwen3
        temperature = 0.7

        agent_process.set_status("executing")
        agent_process.set_start_time(time.time())

        messages = agent_process.query.messages
        tools = agent_process.query.tools
        message_return_type = agent_process.query.message_return_type

        self.logger.log(
            f"{agent_process.agent_name} is switched to executing.\n",
            level="executing",
        )

        model_identifier = (
            self.model_name.split("/", 1)[1]
            if self.model_name.startswith("oai/")
            else self.model_name
        )

        if tools:
            formatted_messages = self.tool_calling_input_format(
                [message.copy() for message in messages],
                tools,
            )
            try:
                response = self.model.chat.completions.create(
                    model=model_identifier,
                    messages=formatted_messages,
                    max_tokens=self.max_new_tokens,
                    temperature=temperature,
                )
                raw_message = response.choices[0].message
                content = raw_message.content
                tool_calls = self.parse_tool_calls(content)

                if tool_calls:
                    agent_process.set_response(
                        Response(
                            response_message=None,
                            tool_calls=tool_calls,
                            raw_response=raw_message,
                        )
                    )
                else:
                    agent_process.set_response(
                        Response(
                            response_message=content,
                            raw_response=raw_message,
                        )
                    )
            except Exception as e:
                agent_process.set_response(
                    Response(
                        response_message=f"An unexpected error occurred: {e}",
                    )
                )
        else:
            try:
                response = self.model.chat.completions.create(
                    model=model_identifier,
                    messages=messages,
                    max_tokens=self.max_new_tokens,
                    temperature=temperature,
                )
                raw_message = response.choices[0].message
                result = raw_message.content

                if message_return_type == "json":
                    result = self.parse_json_format(result)

                agent_process.set_response(
                    Response(
                        response_message=result,
                        raw_response=raw_message,
                    )
                )
            except Exception as e:
                agent_process.set_response(
                    Response(
                        response_message=f"An unexpected error occurred: {e}",
                    )
                )

        agent_process.set_status("done")
        agent_process.set_end_time(time.time())
        return
