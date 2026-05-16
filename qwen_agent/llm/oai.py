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
import os
import threading
from pprint import pformat
from typing import Callable, Dict, Iterator, List, Optional

import openai

from qwen_agent.utils.utils import format_as_text_message

if openai.__version__.startswith('0.'):
    from openai.error import OpenAIError  # noqa
else:
    from openai import OpenAIError

from qwen_agent.llm.base import ModelServiceError, register_llm
from qwen_agent.llm.function_calling import BaseFnCallModel
from qwen_agent.llm.schema import ASSISTANT, FunctionCall, Message
from qwen_agent.log import logger


# --- LOCAL MOD (sandbox_agent) -----------------------------------------------
# vLLM/OpenAI usage capture: emit per-call prompt/completion token counts to
# (a) sandbox_agent's activity log (audit trail) and (b) an optional thread-
# keyed hook (chat_app registers one per chat turn so the UI can show "last
# turn: X in / Y out · chat total: Z"). For streaming, this needs
# `stream_options={"include_usage": True}` on the request so vLLM emits a
# trailing usage chunk — added below in _chat_stream.
_usage_hooks: "Dict[int, Callable[[dict], None]]" = {}


def register_usage_hook(fn: "Callable[[dict], None]") -> None:
    """chat_app calls this on the worker thread that runs agent.run() so each
    _chat_stream call's final usage event flows back into that turn's queue.
    Hook receives {'model', 'prompt_tokens', 'completion_tokens', 'total_tokens'}."""
    _usage_hooks[threading.get_ident()] = fn


def unregister_usage_hook() -> None:
    _usage_hooks.pop(threading.get_ident(), None)


def _capture_usage(usage, model: str) -> None:
    """Fire the activity-log + hook for a freshly received usage payload."""
    if not usage:
        return
    info = {
        'model': model,
        'prompt_tokens': int(getattr(usage, 'prompt_tokens', 0) or 0),
        'completion_tokens': int(getattr(usage, 'completion_tokens', 0) or 0),
    }
    info['total_tokens'] = info['prompt_tokens'] + info['completion_tokens']
    try:
        from sandbox_agent.activity_log import log_event
        log_event('llm_usage', **info)
    except Exception:  # noqa: BLE001 — qwen_agent must stay usable without sandbox_agent
        pass
    hook = _usage_hooks.get(threading.get_ident())
    if hook is not None:
        try:
            hook(info)
        except Exception:  # noqa: BLE001
            pass


# -----------------------------------------------------------------------------


def _extract_reasoning(obj) -> str:
    """Pull reasoning/thinking text off a streaming delta or a message.

    OpenAI-compatible servers haven't standardized the field name: vLLM emits
    `reasoning`, DeepSeek and some others emit `reasoning_content`. Check both
    so the model's thinking isn't silently dropped.
    """
    if obj is None:
        return ''
    for attr in ('reasoning_content', 'reasoning'):
        val = getattr(obj, attr, None)
        if val:
            return val
    return ''


@register_llm('oai')
class TextChatAtOAI(BaseFnCallModel):

    def __init__(self, cfg: Optional[Dict] = None):
        super().__init__(cfg)
        self.model = self.model or 'gpt-4o-mini'
        cfg = cfg or {}

        api_base = cfg.get('api_base')
        api_base = api_base or cfg.get('base_url')
        api_base = api_base or cfg.get('model_server')
        api_base = (api_base or '').strip()

        api_key = cfg.get('api_key')
        api_key = api_key or os.getenv('OPENAI_API_KEY')
        api_key = (api_key or 'EMPTY').strip()

        if openai.__version__.startswith('0.'):
            if api_base:
                openai.api_base = api_base
            if api_key:
                openai.api_key = api_key
            self._complete_create = openai.Completion.create
            self._chat_complete_create = openai.ChatCompletion.create
        else:
            api_kwargs = {}
            if api_base:
                api_kwargs['base_url'] = api_base
            if api_key:
                api_kwargs['api_key'] = api_key

            def _chat_complete_create(*args, **kwargs):
                # OpenAI API v1 does not allow the following args, must pass by extra_body
                extra_params = ['top_k', 'repetition_penalty']
                if any((k in kwargs) for k in extra_params):
                    kwargs['extra_body'] = copy.deepcopy(kwargs.get('extra_body', {}))
                    for k in extra_params:
                        if k in kwargs:
                            kwargs['extra_body'][k] = kwargs.pop(k)
                if 'request_timeout' in kwargs:
                    kwargs['timeout'] = kwargs.pop('request_timeout')

                client = openai.OpenAI(**api_kwargs)
                return client.chat.completions.create(*args, **kwargs)

            def _complete_create(*args, **kwargs):
                # OpenAI API v1 does not allow the following args, must pass by extra_body
                extra_params = ['top_k', 'repetition_penalty']
                if any((k in kwargs) for k in extra_params):
                    kwargs['extra_body'] = copy.deepcopy(kwargs.get('extra_body', {}))
                    for k in extra_params:
                        if k in kwargs:
                            kwargs['extra_body'][k] = kwargs.pop(k)
                if 'request_timeout' in kwargs:
                    kwargs['timeout'] = kwargs.pop('request_timeout')

                client = openai.OpenAI(**api_kwargs)
                return client.completions.create(*args, **kwargs)

            self._complete_create = _complete_create
            self._chat_complete_create = _chat_complete_create

    def _chat_stream(
        self,
        messages: List[Message],
        delta_stream: bool,
        generate_cfg: dict,
    ) -> Iterator[List[Message]]:
        messages = self.convert_messages_to_dicts(messages)
        logger.debug(f'LLM Input generate_cfg: \n{generate_cfg}')
        # LOCAL MOD: request a trailing usage chunk so we can capture prompt /
        # completion token counts on every streamed call. vLLM honors this.
        gc = dict(generate_cfg)
        opts = dict(gc.get('stream_options') or {})
        opts.setdefault('include_usage', True)
        gc['stream_options'] = opts
        try:
            response = self._chat_complete_create(model=self.model, messages=messages, stream=True, **gc)
            if delta_stream:
                for chunk in response:
                    if getattr(chunk, 'usage', None):
                        _capture_usage(chunk.usage, self.model)
                    if chunk.choices:
                        delta_reasoning = _extract_reasoning(chunk.choices[0].delta)
                        if delta_reasoning:
                            yield [
                                Message(role=ASSISTANT,
                                        content='',
                                        reasoning_content=delta_reasoning)
                            ]
                        if hasattr(chunk.choices[0].delta, 'content') and chunk.choices[0].delta.content:
                            yield [Message(role=ASSISTANT, content=chunk.choices[0].delta.content)]
            else:
                full_response = ''
                full_reasoning_content = ''
                full_tool_calls = []
                for chunk in response:
                    if getattr(chunk, 'usage', None):
                        _capture_usage(chunk.usage, self.model)
                    if chunk.choices:
                        delta_reasoning = _extract_reasoning(chunk.choices[0].delta)
                        if delta_reasoning:
                            full_reasoning_content += delta_reasoning
                        if hasattr(chunk.choices[0].delta, 'content') and chunk.choices[0].delta.content:
                            full_response += chunk.choices[0].delta.content
                        if hasattr(chunk.choices[0].delta, 'tool_calls') and chunk.choices[0].delta.tool_calls:
                            for tc in chunk.choices[0].delta.tool_calls:
                                if full_tool_calls and (not tc.id or
                                                        tc.id == full_tool_calls[-1]['extra']['function_id']):
                                    if tc.function.name:
                                        full_tool_calls[-1].function_call['name'] += tc.function.name
                                    if tc.function.arguments:
                                        full_tool_calls[-1].function_call['arguments'] += tc.function.arguments
                                else:
                                    full_tool_calls.append(
                                        Message(role=ASSISTANT,
                                                content='',
                                                function_call=FunctionCall(name=tc.function.name,
                                                                           arguments=tc.function.arguments),
                                                extra={'function_id': tc.id}))

                        res = []
                        if full_reasoning_content:
                            res.append(Message(role=ASSISTANT, content='', reasoning_content=full_reasoning_content))
                        if full_response:
                            res.append(Message(
                                role=ASSISTANT,
                                content=full_response,
                            ))
                        if full_tool_calls:
                            res += full_tool_calls
                        yield res
        except OpenAIError as ex:
            raise ModelServiceError(exception=ex)

    def _chat_no_stream(
        self,
        messages: List[Message],
        generate_cfg: dict,
    ) -> List[Message]:
        messages = self.convert_messages_to_dicts(messages)
        try:
            response = self._chat_complete_create(model=self.model, messages=messages, stream=False, **generate_cfg)
            _capture_usage(getattr(response, 'usage', None), self.model)
            msg_reasoning = _extract_reasoning(response.choices[0].message)
            if msg_reasoning:
                return [
                    Message(role=ASSISTANT,
                            content=response.choices[0].message.content,
                            reasoning_content=msg_reasoning)
                ]
            else:
                return [Message(role=ASSISTANT, content=response.choices[0].message.content)]
        except OpenAIError as ex:
            raise ModelServiceError(exception=ex)

    def convert_messages_to_dicts(self, messages: List[Message]) -> List[dict]:
        # TODO: Change when the VLLM deployed model needs to pass reasoning_complete.
        #  At this time, in order to be compatible with lower versions of vLLM,
        #  and reasoning content is currently not useful
        messages = [format_as_text_message(msg, add_upload_info=False) for msg in messages]
        messages = [msg.model_dump() for msg in messages]
        messages = self._conv_qwen_agent_messages_to_oai(messages)

        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f'LLM Input: \n{pformat(messages, indent=2)}')
        return messages
