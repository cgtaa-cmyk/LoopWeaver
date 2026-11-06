from LoopWaver_model.Agents.PlanningAgent import PlanningAgent
from LoopWaver_model.Agents.GenerationAgent import GenerationAgent
from LoopWaver_model.BaselineGen import BaselineGen

class LoopWeaver(BaselineGen):

    @staticmethod
    async def async_generate(plan_model, gen_model, example, semaphore):
        example = await PlanningAgent.async_create_hierarchy(plan_model, example, semaphore)
        example = await GenerationAgent.async_generate(gen_model, example, semaphore)
        return example
