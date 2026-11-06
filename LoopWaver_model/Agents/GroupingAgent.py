import logging
import asyncio
from llms.llms import async_call_llm
from json_repair import repair_json
import json
import re

class GroupingAgent:
    """Groups generated content by task position and builds step-level groups.
    Each group corresponds to a specific context state (e.g. the same floor, or the same week),
    and all generated versions under that state form a step-level group."""

    @staticmethod
    async def async_create_groups(model, example, semaphore):
        """Dispatch to the appropriate handler based on the content type."""
        if example["type"] == "Week":
            example = await GroupingAgent.async_group_weeks(model, example, semaphore)
        elif example["type"] == "Floor":
            example = await GroupingAgent.async_group_floors(model, example, semaphore)
        elif example["type"] == "Menu Week":
            example = await GroupingAgent.async_group_menu_weeks(model, example, semaphore)
        elif example["type"] == "Block":
            example = await GroupingAgent.async_group_blocks(model, example, semaphore)
        return example

    @staticmethod
    async def async_group_weeks(model, example, semaphore):
        """Group weekly contents by week."""
        logging.info("Grouping weekly contents")

        weeks = example.get('weekly_plan', [])
        if not weeks:
            logging.error("No weekly plan found")
            return example

        groups = {}
        for week in weeks:
            week_id = week.get('week_id')
            if not week_id:
                continue

            if week_id not in groups:
                groups[week_id] = {
                    'week_id': week_id,
                    'events': week.get('events', ''),
                    'versions': []
                }

            if 'diary_entry' in week:
                groups[week_id]['versions'].append({
                    'content': week['diary_entry'],
                    'metadata': {'source': 'initial_generation'}
                })

        example['step_groups'] = list(groups.values())

        tasks = []
        for group in example['step_groups']:
            tasks.append(GroupingAgent._generate_group_summary(model, group, example['prompt'], semaphore))

        summaries = await asyncio.gather(*tasks)

        for i, summary in enumerate(summaries):
            if i < len(example['step_groups']):
                example['step_groups'][i]['summary'] = summary

        return example

    @staticmethod
    async def async_group_floors(model, example, semaphore):
        """Group floor contents by floor."""
        logging.info("Grouping floor contents")

        floors = example.get('floor_plan', [])
        if not floors:
            logging.error("No floor plan found")
            return example

        groups = {}
        for floor in floors:
            floor_id = floor.get('floor_id')
            if not floor_id:
                continue

            if floor_id not in groups:
                groups[floor_id] = {
                    'floor_id': floor_id,
                    'purpose': floor.get('purpose', ''),
                    'versions': []
                }

            if 'plan' in floor:
                groups[floor_id]['versions'].append({
                    'content': floor['plan'],
                    'metadata': {'source': 'initial_generation'}
                })

        example['step_groups'] = list(groups.values())

        tasks = []
        for group in example['step_groups']:
            tasks.append(GroupingAgent._generate_group_summary(model, group, example['prompt'], semaphore))

        summaries = await asyncio.gather(*tasks)

        for i, summary in enumerate(summaries):
            if i < len(example['step_groups']):
                example['step_groups'][i]['summary'] = summary

        return example

    @staticmethod
    async def async_group_menu_weeks(model, example, semaphore):
        """Group menu contents by week."""
        logging.info("Grouping menu contents")

        weeks = example.get('weekly_plan', [])
        if not weeks:
            logging.error("No weekly menu plan found")
            return example

        groups = {}
        for week in weeks:
            week_id = week.get('week_id')
            if not week_id:
                continue

            if week_id not in groups:
                groups[week_id] = {
                    'week_id': week_id,
                    'dishes': week.get('dishes', ''),
                    'versions': []
                }

            if 'week_menu' in week:
                groups[week_id]['versions'].append({
                    'content': week['week_menu'],
                    'metadata': {'source': 'initial_generation'}
                })

        example['step_groups'] = list(groups.values())

        tasks = []
        for group in example['step_groups']:
            tasks.append(GroupingAgent._generate_group_summary(model, group, example['prompt'], semaphore))

        summaries = await asyncio.gather(*tasks)

        for i, summary in enumerate(summaries):
            if i < len(example['step_groups']):
                example['step_groups'][i]['summary'] = summary

        return example

    @staticmethod
    async def async_group_blocks(model, example, semaphore):
        """Group city block contents by block."""
        logging.info("Grouping block contents")

        blocks = example.get('block_plan', [])
        if not blocks:
            logging.error("No block plan found")
            return example

        groups = {}
        for block in blocks:
            block_id = block.get('block_id')
            if not block_id:
                continue

            if block_id not in groups:
                groups[block_id] = {
                    'block_id': block_id,
                    'use': block.get('use', ''),
                    'versions': []
                }

            if 'plan' in block:
                groups[block_id]['versions'].append({
                    'content': block['plan'],
                    'metadata': {'source': 'initial_generation'}
                })

        example['step_groups'] = list(groups.values())

        tasks = []
        for group in example['step_groups']:
            tasks.append(GroupingAgent._generate_group_summary(model, group, example['prompt'], semaphore))

        summaries = await asyncio.gather(*tasks)

        for i, summary in enumerate(summaries):
            if i < len(example['step_groups']):
                example['step_groups'][i]['summary'] = summary

        return example

    @staticmethod
    async def _generate_group_summary(model, group, user_prompt, semaphore):
        """Generate a descriptive summary for each group."""
        group_type = ""
        group_id = ""

        if 'week_id' in group:
            group_type = "week"
            group_id = group['week_id']
            context = group.get('events', '') if 'events' in group else group.get('dishes', '')
        elif 'floor_id' in group:
            group_type = "floor"
            group_id = group['floor_id']
            context = group.get('purpose', '')
        elif 'block_id' in group:
            group_type = "block"
            group_id = group['block_id']
            context = group.get('use', '')
        else:
            return "No summary available"

        versions_content = ""
        for i, version in enumerate(group.get('versions', [])):
            versions_content += f"Version {i+1}: {version.get('content', '')}\n\n"

        prompt = f"""
You are an expert analyst. Create a concise summary (around 30-50 words) for the following {group_type} content:

{group_type.title()} ID: {group_id}
{group_type.title()} Context: {context}
User Requirements: {user_prompt}

Content Versions:
{versions_content}

Create a concise summary that captures the essence of what this {group_type} represents based on its context and content versions.
Return only the summary text, without any prefixes or additional explanations.
"""

        try:
            async with semaphore:
                response = await async_call_llm(model, prompt)
            summary = response.strip()
            if len(summary) > 200:
                summary = summary[:197] + "..."
            return summary
        except Exception as e:
            logging.error(f"Error generating summary: {e}")
            return f"Summary for {group_id}"
