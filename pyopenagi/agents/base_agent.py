import os
import copy

import json

from .agent_process import (
    AgentProcess
)

import time

from threading import Thread

from ..utils.logger import AgentLogger

from ..utils.chat_template import Query

import importlib

from ..queues.llm_request_queue import LLMRequestQueue

from pyopenagi.tools.simulated_tool import SimulatedTool

class CustomizedThread(Thread):
    def __init__(self, target, args=()):
        super().__init__()
        self.target = target
        self.args = args
        self.result = None

    def run(self):
        self.result = self.target(*self.args)

    def join(self):
        super().join()
        return self.result

class BaseAgent:
    def __init__(self,
                 agent_name,
                 task_input,
                 agent_process_factory,
                 log_mode: str
        ):
        self.agent_name = agent_name
        self.config = self.load_config()
        self.tool_names = self.config["tools"]

        self.agent_process_factory = agent_process_factory

        self.tool_list = dict()
        self.tools = []
        # self.load_tools(self.tool_names)

        self.start_time = None
        self.end_time = None
        self.request_waiting_times: list = []
        self.request_turnaround_times: list = []
        self.task_input = task_input
        self.messages = []
        self.workflow_mode = "manual" # (mannual, automatic)
        self.rounds = 0

        self.log_mode = log_mode
        self.logger = self.setup_logger()
        # self.logger.log("Initialized. \n", level="info")

        self.set_status("active")
        self.set_created_time(time.time())


    def _convert_raw_message(self, raw_message, fallback_content=None):
        if raw_message is None:
            return None

        if hasattr(raw_message, "model_dump"):
            message_dict = raw_message.model_dump()
        elif hasattr(raw_message, "to_dict"):
            message_dict = raw_message.to_dict()
        elif isinstance(raw_message, dict):
            message_dict = copy.deepcopy(raw_message)
        else:
            message_dict = {
                "role": getattr(raw_message, "role", "assistant"),
            }
            content = getattr(raw_message, "content", None)
            if content is not None:
                message_dict["content"] = content

        if "role" not in message_dict or message_dict["role"] is None:
            message_dict["role"] = "assistant"

        if not message_dict.get("content") and fallback_content is not None:
            message_dict["content"] = fallback_content

        return message_dict

    def _append_response_message(self, response):
        if response is None:
            return None

        fallback_content = getattr(response, "response_message", None)
        raw_message = getattr(response, "raw_response", None)

        message_dict = self._convert_raw_message(raw_message, fallback_content=fallback_content)

        if message_dict is None and fallback_content is not None:
            message_dict = {
                "role": "assistant",
                "content": fallback_content,
            }

        if message_dict is not None:
            self.messages.append(message_dict)

        return message_dict

    def run(self):
        '''Execute each step to finish the task.'''
        pass

    # can be customization
    def build_system_instruction(self):
        pass

    def check_workflow(self, message):
        def parse_and_validate(text):
            """Helper function to parse and validate workflow JSON"""
            try:
                workflow = json.loads(text)

                if not isinstance(workflow, list):
                    workflow = [workflow]

                for step in workflow:
                    if "message" not in step or "tool_use" not in step:
                        return None

                return workflow

            except json.JSONDecodeError:
                return None

        # First, try to decode the full message
        result = parse_and_validate(message)
        if result is not None:
            return result

        # If that fails, split by \n\n and try the last component
        components = message.split('\n\n')
        if len(components) > 1:
            last_component = components[-1].strip()
            result = parse_and_validate(last_component)
            if result is not None:
                return result

        return None

    def automatic_workflow(self):
        last_response = None
        for i in range(self.plan_max_fail_times):
            response, start_times, end_times, waiting_times, turnaround_times = self.get_response(
                query=Query(
                    messages=self.messages,
                    tools=None,
                    message_return_type="json"
                )
            )

            last_response = response

            if self.rounds == 0:
                self.set_start_time(start_times[0])

            self.request_waiting_times.extend(waiting_times)
            self.request_turnaround_times.extend(turnaround_times)

            print(f'workflow before check: {response.response_message}')
            workflow = self.check_workflow(response.response_message)
            print(f'workflow after check: {workflow}')

            self.rounds += 1

            if workflow:
                return workflow, response

            llm_name = getattr(getattr(self, "args", None), "llm_name", None)
            fail_message = f"Fail {i+1} times to generate a valid plan. I need to regenerate a plan."

            if llm_name == 'claude-3-5-sonnet-20240620':
                self.messages.append(
                    {
                        "role": "assistant",
                        "content": fail_message
                    }
                )
                self.messages.append(
                    {
                        "role": "user",
                        "content": f"Please try again. {fail_message}"
                    }
                )
            else:
                self.messages.append(
                    {
                        "role": "assistant",
                        "content": fail_message
                    }
                )

            if i == self.plan_max_fail_times - 1:
                appended = self._append_response_message(response)
                if appended is None and response is not None:
                    self.messages.append(
                        {
                            "role": "assistant",
                            "thinking": f"{response.response_message}"
                        }
                    )
                return None, response

        return None, last_response

    def manual_workflow(self):
        pass

    def snake_to_camel(self, snake_str):
        components = snake_str.split('_')
        return ''.join(x.title() for x in components)

    def load_tools(self, tool_names):
        for tool_name in tool_names:
            org, name = tool_name.split("/")
            module_name = ".".join(["pyopenagi", "tools", org, name])
            class_name = self.snake_to_camel(name)

            tool_module = importlib.import_module(module_name)
            tool_class = getattr(tool_module, class_name)

            self.tool_list[name] = tool_class()
            self.tools.append(tool_class().get_tool_call_format())

    def load_tools_from_file(self, tool_names, tools_info):
        for tool_name in tool_names:
            org, name = tool_name.split("/")
            tool_instance = SimulatedTool(name, tools_info)
            self.tool_list[name] = tool_instance
            self.tools.append(tool_instance.get_tool_call_format())


    def pre_select_tools(self, tool_names):
        pre_selected_tools = []
        for tool_name in tool_names:
            for tool in self.tools:
                if tool["function"]["name"] == tool_name:
                    pre_selected_tools.append(tool)
                    break

        return pre_selected_tools

    def setup_logger(self):
        logger = AgentLogger(self.agent_name, self.log_mode)
        return logger

    def load_config(self):
        script_path = os.path.abspath(__file__)
        script_dir = os.path.dirname(script_path)
        config_file = os.path.join(script_dir, self.agent_name, "config.json")
        with open(config_file, "r") as f:
            config = json.load(f)
            return config

    # the default method used for getting response from AIOS
    def get_response(self,
            query,
            temperature=0.0
        ):
        thread = CustomizedThread(target=self.query_loop, args=(query, ))
        thread.start()
        return thread.join()

    def query_loop(self, query):
        agent_process = self.create_agent_request(query)

        completed_response, start_times, end_times, waiting_times, turnaround_times = "", [], [], [], []

        while agent_process.get_status() != "done":
            thread = Thread(target=self.listen, args=(agent_process, ))
            current_time = time.time()
            # reinitialize agent status
            agent_process.set_created_time(current_time)
            agent_process.set_response(None)
            LLMRequestQueue.add_message(agent_process)

            thread.start()
            thread.join()

            completed_response = agent_process.get_response()
            if agent_process.get_status() != "done":
                self.logger.log(
                    f"Suspended due to the reach of time limit ({agent_process.get_time_limit()}s). Current result is: {completed_response.response_message}\n",
                    level="suspending"
                )
            start_time = agent_process.get_start_time()
            end_time = agent_process.get_end_time()
            waiting_time = start_time - agent_process.get_created_time()
            turnaround_time = end_time - agent_process.get_created_time()

            start_times.append(start_time)
            end_times.append(end_time)
            waiting_times.append(waiting_time)
            turnaround_times.append(turnaround_time)
            # Re-start the thread if not done

        # self.agent_process_factory.deactivate_agent_process(agent_process.get_pid())

        return completed_response, start_times, end_times, waiting_times, turnaround_times

    def create_agent_request(self, query):
        agent_process = self.agent_process_factory.activate_agent_process(
            agent_name = self.agent_name,
            query = query
        )
        agent_process.set_created_time(time.time())
        # print("Already put into the queue")
        return agent_process

    def listen(self, agent_process: AgentProcess):
        """Response Listener for agent

        Args:
            agent_process (AgentProcess): Listened AgentProcess

        Returns:
            str: LLM response of Agent Process
        """
        while agent_process.get_response() is None:
            time.sleep(0.2)

        return agent_process.get_response()

    def set_aid(self, aid):
        self.aid = aid

    def get_aid(self):
        return self.aid

    def get_agent_name(self):
        return self.agent_name

    def set_status(self, status):

        """
        Status type: Waiting, Running, Done, Inactive
        """
        self.status = status

    def get_status(self):
        return self.status

    def set_created_time(self, time):
        self.created_time = time

    def get_created_time(self):
        return self.created_time

    def set_start_time(self, time):
        self.start_time = time

    def get_start_time(self):
        return self.start_time

    def set_end_time(self, time):
        self.end_time = time

    def get_end_time(self):
        return self.end_time
