# Copyright 2023 The Qwen team, Alibaba Group. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import copy
import logging
from typing import Dict, Iterator, List, Literal, Optional, Union

from qwen_agent import Agent
from qwen_agent.llm import BaseChatModel
from qwen_agent.llm.schema import ASSISTANT, DEFAULT_SYSTEM_MESSAGE, FUNCTION, USER, ContentItem, Message
from qwen_agent.memory import Memory
from qwen_agent.settings import MAX_LLM_CALL_PER_RUN
from qwen_agent.tools import BaseTool
from qwen_agent.utils.utils import extract_files_from_messages

logger = logging.getLogger(__name__)

# --- LOCAL MOD (sandbox_agent) -----------------------------------------------
# Graceful tool-call budget exhaustion, modeled on Hermes Agent's `budget_pressure`
# + `_handle_max_iterations`. Stock qwen-agent just exits `_run`'s loop when
# MAX_LLM_CALL_PER_RUN is spent and yields a possibly-dangling tool call with no
# concluding message. Two additions, both inside `_run` (so a subclass override
# isn't clean — hence the in-file mod):
#   1. When fewer than `_BUDGET_PRESSURE_AT` LLM calls remain, prepend a one-line
#      budget note to each tool result so the model self-consolidates.
#   2. When the budget runs out mid-chain (loop exited via the `while` condition,
#      not via `break`), make one final tool-less LLM call asking for a wrap-up,
#      and prepend a deterministic marker line that (a) shows clearly in the UI and
#      (b) matches sandbox_agent's `stage_runner._detect_part_completion` greps so
#      pipeline stages that hit the cap continue next run instead of restarting.
_BUDGET_PRESSURE_AT = 10
_BUDGET_EXHAUSTED_MARKER = "⚠️ I ran out of tool calls for this run (part-completion). Here's where I am:"
# -----------------------------------------------------------------------------


class FnCallAgent(Agent):
    """This is a widely applicable function call agent integrated with llm and tool use ability."""

    def __init__(self,
                 function_list: Optional[List[Union[str, Dict, BaseTool]]] = None,
                 llm: Optional[Union[Dict, BaseChatModel]] = None,
                 system_message: Optional[str] = DEFAULT_SYSTEM_MESSAGE,
                 name: Optional[str] = None,
                 description: Optional[str] = None,
                 files: Optional[List[str]] = None,
                 **kwargs):
        """Initialization the agent.

        Args:
            function_list: One list of tool name, tool configuration or Tool object,
              such as 'code_interpreter', {'name': 'code_interpreter', 'timeout': 10}, or CodeInterpreter().
            llm: The LLM model configuration or LLM model object.
              Set the configuration as {'model': '', 'api_key': '', 'model_server': ''}.
            system_message: The specified system message for LLM chat.
            name: The name of this agent.
            description: The description of this agent, which will be used for multi_agent.
            files: A file url list. The initialized files for the agent.
        """
        super().__init__(function_list=function_list,
                         llm=llm,
                         system_message=system_message,
                         name=name,
                         description=description)

        if not hasattr(self, 'mem'):
            # Default to use Memory to manage files
            if 'qwq' in self.llm.model.lower() or 'qvq' in self.llm.model.lower() or 'qwen3' in self.llm.model.lower():
                if 'dashscope' in self.llm.model_type:
                    mem_llm = {
                        'model': 'qwen-turbo',
                        'model_type': 'qwen_dashscope',
                        'generate_cfg': {
                            'max_input_tokens': 30000
                        }
                    }
                else:
                    mem_llm = None
            else:
                mem_llm = self.llm
            self.mem = Memory(llm=mem_llm, files=files, **kwargs)

    def _run(self, messages: List[Message], lang: Literal['en', 'zh'] = 'en', **kwargs) -> Iterator[List[Message]]:
        messages = copy.deepcopy(messages)
        num_llm_calls_available = MAX_LLM_CALL_PER_RUN
        response = []
        completed_with_answer = False  # True only if we exit via `break` (model gave a final answer)
        while True and num_llm_calls_available > 0:
            num_llm_calls_available -= 1

            extra_generate_cfg = {'lang': lang}
            if kwargs.get('seed') is not None:
                extra_generate_cfg['seed'] = kwargs['seed']
            output_stream = self._call_llm(messages=messages,
                                           functions=[func.function for func in self.function_map.values()],
                                           extra_generate_cfg=extra_generate_cfg)
            output: List[Message] = []
            for output in output_stream:
                if output:
                    yield response + output
            if output:
                response.extend(output)
                messages.extend(output)
                used_any_tool = False
                for out in output:
                    use_tool, tool_name, tool_args, _ = self._detect_tool(out)
                    if use_tool:
                        tool_result = self._call_tool(tool_name, tool_args, messages=messages, **kwargs)
                        # LOCAL MOD: budget pressure — when few LLM calls remain, tell the
                        # model to consolidate. Prepend (not append): long tool results get
                        # truncated from the end, so a leading note survives.
                        if num_llm_calls_available <= _BUDGET_PRESSURE_AT:
                            _note = (f'[budget: {num_llm_calls_available} tool calls left this run — '
                                    f'start consolidating; deliver your answer before you run out]')
                            if isinstance(tool_result, str):
                                tool_result = _note + '\n\n' + tool_result
                            elif isinstance(tool_result, list):
                                # Prepend budget note to the first text item
                                for _ci in tool_result:
                                    if isinstance(_ci, ContentItem) and getattr(_ci, 'text', None):
                                        _ci.text = _note + '\n\n' + _ci.text
                                        break
                        # LOCAL MOD: split multimodal tool results. Function messages must have
                        # string content for the OpenAI/vLLM API. Image items are emitted as a
                        # separate USER message so convert_messages_to_dicts routes them through
                        # _multimodal_to_oai_dict which produces proper {type:image_url,...} parts.
                        _function_id = out.extra.get('function_id', '1')
                        if isinstance(tool_result, list) and any(
                                isinstance(ci, ContentItem) and getattr(ci, 'image', None)
                                for ci in tool_result):
                            # Split into text-only (FUNCTION msg) and image (USER msg)
                            _text_items = [ci for ci in tool_result if isinstance(ci, ContentItem) and getattr(ci, 'text', None)]
                            _image_items = [ci for ci in tool_result if isinstance(ci, ContentItem) and getattr(ci, 'image', None)]
                            _fn_content = '\n'.join(ci.text for ci in _text_items) if _text_items else ''
                            fn_msg = Message(role=FUNCTION,
                                              name=tool_name,
                                              content=_fn_content,
                                              extra={'function_id': _function_id})
                            messages.append(fn_msg)
                            response.append(fn_msg)
                            if _image_items:
                                img_msg = Message(role=USER, content=_image_items)
                                messages.append(img_msg)
                                response.append(img_msg)
                        else:
                            fn_msg = Message(role=FUNCTION,
                                              name=tool_name,
                                              content=tool_result,
                                              extra={'function_id': _function_id})
                            messages.append(fn_msg)
                            response.append(fn_msg)
                        yield response
                        used_any_tool = True
                if not used_any_tool:
                    completed_with_answer = True
                    break
        # LOCAL MOD: budget ran out mid-chain (loop exited via the `while` condition, not
        # `break`) — make a graceful wrap-up call so the turn ends with a real message
        # carrying a part-completion marker, instead of a dangling tool result.
        if not completed_with_answer and response:
            try:
                wrapup = self._wrap_up_after_budget(messages, lang)
                if wrapup:
                    response = response + wrapup
            except Exception:  # noqa: BLE001 — never let the wrap-up break the run
                logger.exception('budget wrap-up call failed; emitting fallback marker')
                response = response + [Message(
                    role=ASSISTANT,
                    content=("⚠️ I ran out of tool calls for this run (part-completion). "
                             "Reply 'continue' to resume."),
                )]
        yield response

    def _wrap_up_after_budget(self, messages: List[Message], lang: str) -> List[Message]:
        """One final tool-less LLM call asking the model to summarize what it
        accomplished and what's left. Returns a single assistant message whose
        content is prefixed with `_BUDGET_EXHAUSTED_MARKER` (a deterministic
        line that shows clearly in the UI AND is recognized by
        `stage_runner._detect_part_completion`)."""
        instruction = (
            "You have used your entire tool-call budget for this run. Do NOT attempt any "
            "further tool calls. In 3-6 sentences, summarize what you accomplished, what is "
            "still outstanding, and end with: reply 'continue' and I'll resume."
        )
        msgs2 = list(messages) + [Message(role=USER, content=instruction)]
        extra_generate_cfg = {'lang': lang}
        summary_chunks: List[Message] = []
        for summary_chunks in self._call_llm(messages=msgs2, functions=None,
                                             extra_generate_cfg=extra_generate_cfg):
            pass  # take the final accumulated chunk
        summary_text = ''
        for m in (summary_chunks or []):
            if getattr(m, 'role', None) == ASSISTANT and isinstance(getattr(m, 'content', None), str):
                summary_text += m.content
        summary_text = summary_text.strip() or "(no summary produced)"
        return [Message(role=ASSISTANT, content=f'{_BUDGET_EXHAUSTED_MARKER}\n\n{summary_text}')]

    def _call_tool(self, tool_name: str, tool_args: Union[str, dict] = '{}', **kwargs) -> str:
        if tool_name not in self.function_map:
            return f'Tool {tool_name} does not exists.'
        # Temporary plan: Check if it is necessary to transfer files to the tool
        # Todo: This should be changed to parameter passing, and the file URL should be determined by the model
        if self.function_map[tool_name].file_access:
            assert 'messages' in kwargs
            files = extract_files_from_messages(kwargs['messages'], include_images=True) + self.mem.system_files
            return super()._call_tool(tool_name, tool_args, files=files, **kwargs)
        else:
            return super()._call_tool(tool_name, tool_args, **kwargs)
