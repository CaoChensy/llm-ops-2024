import os
import logging

from typing import Any, Union, Callable, Iterable, Optional
from dataclasses import dataclass

from zlai.llms import TypeLLM
from zlai.schema import SystemMessage
from zlai.agent.base import AgentMixin
from zlai.agent.prompt.tasks import TaskCompletion
from zlai.agent.prompt.knowledge import *

__all__ = ["KnowledgeAgent"]


@dataclass
class ErrorMessage:
    """"""
    not_find_content: Optional[str] = "未在知识库中找到相关信息..."


class KnowledgeAgent(AgentMixin):
    """"""

    def __init__(
            self,
            agent_name: Optional[str] = "Knowledge Agent",
            llm: Optional[TypeLLM] = None,
            system_message: Optional[SystemMessage] = PromptKnowledge.system_message,
            prompt_template: Optional[PromptTemplate] = PromptKnowledge.summary_prompt,
            search: Optional[str] = None,
            recall_nums: Optional[int] = None,
            stream: Optional[bool] = False,
            error_message: Optional[ErrorMessage] = ErrorMessage(),
            logger: Optional[Callable] = None,
            log_path: Optional[str] = None,
            verbose: Optional[bool] = False,
            *args: Any,
            **kwargs: Any
    ):
        """"""
        self.agent_name = agent_name
        self.llm = llm
        self.system_message = system_message
        self.prompt_template = prompt_template
        self.search = search
        self.recall_nums = recall_nums
        self.stream = stream
        self.error_message = error_message
        self.logger = logger
        self.verbose = verbose
        self.args = args
        self.kwargs = kwargs
        self.rag_logger = None
        if log_path:
            log_dir = os.path.dirname(log_path)
            os.makedirs(log_dir, exist_ok=True)

            rag_logger = logging.getLogger('RAG')
            rag_logger.setLevel(logging.INFO)
            file_handler = logging.FileHandler(log_path)
            rag_logger.addHandler(file_handler)
            self.rag_logger = rag_logger

    def search_content(
            self,
            question: Optional[str] = None
    ):
        """"""
        results = self.search.mix_search(question, top=self.recall_nums)
        # text_results = [re[1] for re in results]
        # text_scores = [re[0] for re in results]
        text_results = list(results.keys())
        text_scores = list(results.values())
        if self.rag_logger:
            self.rag_logger.info(f"\nquestion: {question}"
                                 f"\nretrieve:  total_num:{len(text_results)}  retrieve score:{text_scores}"
                                 f"\n{text_results} ")
        return "------".join(list(set(text_results)))

    #         self._logger(msg=f"[{self.agent_name}] Find Knowledge Title: {title}, Score: {score:.4f}.", color="green")
    #         if score < thresh:
    #             self._logger(msg=f"[{self.agent_name}] Not enough score: {score:.4f}.", color="red")
    #             return None
    #         else:
    #             return content

    def generate(
            self,
            query: Union[str, TaskCompletion],
            *args: Any,
            **kwargs: Any,
    ) -> TaskCompletion:
        """"""
        task_completion = self._make_task_completion(query=query, **kwargs)
        task_completion.observation = self.search_content(question=task_completion.query)
        if task_completion.observation is None:
            task_completion.content = self.error_message.not_find_content
        else:
            messages = self._make_messages(question=task_completion.query, content=task_completion.observation, )
            self._show_messages(messages=messages, logger_name=self.agent_name)
            completion = self.llm.generate(messages=messages)
            task_completion.content = completion.choices[0].message.content
        if self.rag_logger:
            self.rag_logger.info(f"answers: \n{task_completion.content}")
        self._logger_agent_final_answer(name=self.agent_name, content=task_completion.content)
        self._logger_agent_end(name=self.agent_name)
        return task_completion

    def stream_generate(
            self,
            query: Union[str, TaskCompletion],
            *args: Any,
            **kwargs: Any,
    ) -> Iterable[TaskCompletion]:
        """"""
        task_completion = self._make_task_completion(query=query, **kwargs)
        task_completion.observation = self.search_content(question=task_completion.query)
        if task_completion.observation is None:
            task_completion.content = self.error_message.not_find_content
            task_completion.delta = self.error_message.not_find_content
            yield task_completion
        else:
            messages = self._make_messages(question=task_completion.query, content=task_completion.observation, )
            self._show_messages(messages=messages, logger_name=self.agent_name)
            stream_task_instance = self.stream_task_completion(messages=messages, task_completion=task_completion)
            for task_completion in stream_task_instance:
                yield task_completion
        self._logger_agent_final_answer(name=self.agent_name, content=task_completion.content)
        self._logger_agent_end(name=self.agent_name)
