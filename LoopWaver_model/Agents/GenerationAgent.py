import os
import json
import logging
import re
import asyncio
import copy
from llms.llms import async_call_llm
from utils.wordCounter import count_words
from json_repair import repair_json

class GenerationAgent:

    @staticmethod
    async def async_generate(model, example, semaphore):
        example = copy.deepcopy(example)

        if 'weekly_plan' in example:
            if not isinstance(example['weekly_plan'], list):
                if isinstance(example['weekly_plan'], str):
                    example['weekly_plan'] = [{"week_id": "Week 1", "events": example['weekly_plan']}]
                elif isinstance(example['weekly_plan'], dict):
                    example['weekly_plan'] = [example['weekly_plan']]
                else:
                    example['weekly_plan'] = []

        if example['type'] == 'Week':
            example = await GenerationAgent.async_generate_week(model, example, semaphore)
        elif example["type"] == "Floor":
            example = await GenerationAgent.async_generate_floor(model, example, semaphore)
        elif example["type"] == "Menu Week":
            example = await GenerationAgent.async_generate_menu(model, example, semaphore)
        elif example["type"] == "Block":
            example = await GenerationAgent.async_generate_block(model, example, semaphore)
        return example

    @staticmethod
    async def async_generate_week(model, example, semaphore):
        if 'weekly_plan' not in example or not example['weekly_plan']:
            example['weekly_plan'] = []
        elif not isinstance(example['weekly_plan'], list):
            if isinstance(example['weekly_plan'], str):
                example['weekly_plan'] = [{"week_id": "Week 1", "events": example['weekly_plan']}]
            elif isinstance(example['weekly_plan'], dict):
                example['weekly_plan'] = [example['weekly_plan']]
            else:
                example['weekly_plan'] = []

        processed_weeks = []

        for i, week in enumerate(example['weekly_plan']):
            if not isinstance(week, dict):
                processed_week = {"week_id": f"Week {i+1}", "events": str(week)}
            else:
                processed_week = copy.deepcopy(week)

                if 'week_id' not in processed_week:
                    processed_week['week_id'] = f"Week {i+1}"
                if 'events' not in processed_week:
                    processed_week['events'] = ""

            processed_weeks.append(processed_week)

        example['weekly_plan'] = processed_weeks

        async def process_week(week):
            week_id = week.get('week_id', "Unknown Week")
            events = week.get('events', "")

            if not isinstance(events, str):
                if isinstance(events, list):
                    events = ", ".join([str(event) for event in events])
                else:
                    events = str(events)

            prompt = f"""You are an expert writer.
Write a 200-word weekly diary entry for the week of {week_id}.
The events for this week are: {events}
You should consider the coherence of the diary entry referring to the plan of the whole year:
{example['weekly_plan']}
You should consider the user requirements: {example['prompt']}
Check from the user requirements if there are any special events that should be included in the diary entry. If there are, include them in the diary entry. If there are no special events, write a general diary entry for the week.
Return the diary entry in the following json format:
{{
    "week_id": "{week_id}",
    "check": "reason and check if the user requirements are met",
    "diary_entry": "Your 200-word diary entry here"
}}
"""
            logging.info(f"Generating initial diary entry for week {week_id}")

            max_attempts = 3
            for attempt in range(max_attempts):
                async with semaphore:
                    response = await async_call_llm(model, prompt)

                try:
                    response = repair_json(response)

                    match = re.search(r"\{.*\}", response, re.DOTALL)
                    if match:
                        response_json = match.group(0)

                        try:
                            diary_entry = json.loads(response_json)

                            if 'diary_entry' in diary_entry:
                                week['diary_entry'] = diary_entry['diary_entry']
                                if not isinstance(week['diary_entry'], str):
                                    week['diary_entry'] = str(week['diary_entry'])
                                week['length_requirement'] = 200
                                logging.info(f"Diary entry word count: {count_words(week['diary_entry'])}")
                                break
                            else:
                                for possible_key in ['entry', 'content', 'text', 'diary']:
                                    if possible_key in diary_entry:
                                        week['diary_entry'] = diary_entry[possible_key]
                                        if not isinstance(week['diary_entry'], str):
                                            week['diary_entry'] = str(week['diary_entry'])
                                        week['length_requirement'] = 200
                                        logging.info(f"Used alternative key '{possible_key}' for diary entry")
                                        break

                                if 'diary_entry' not in week and attempt == max_attempts - 1:
                                    text_content = re.sub(r'[{}"\'week_id[^:]*:|check[^:]*:|\s*]', '', response_json)
                                    week['diary_entry'] = text_content.strip() or f"Weekly diary for {week_id}."
                                    week['length_requirement'] = 200
                                    logging.warning(f"Created fallback diary entry from JSON text for {week_id}")
                        except json.JSONDecodeError:
                            if attempt == max_attempts - 1:
                                week['diary_entry'] = f"Weekly diary for {week_id}."
                                week['length_requirement'] = 200
                                logging.warning(f"JSON decode error, created fallback diary entry for {week_id}")
                    else:
                        if attempt == max_attempts - 1:
                            week['diary_entry'] = response.strip() or f"Weekly diary for {week_id}."
                            week['length_requirement'] = 200
                            logging.warning(f"No JSON found, using full response as diary entry for {week_id}")
                except Exception as e:
                    logging.error(f"Error processing response for {week_id}: {e}")
                    if attempt == max_attempts - 1:
                        week['diary_entry'] = f"Weekly diary for {week_id}."
                        week['length_requirement'] = 200
                        logging.warning(f"Exception occurred, created fallback diary entry for {week_id}")

            if 'diary_entry' not in week:
                week['diary_entry'] = f"Weekly diary for {week_id}."
                week['length_requirement'] = 200
                logging.warning(f"Created fallback diary entry for {week_id}")

            try:
                current_length = count_words(week['diary_entry'])
                required_length = week['length_requirement']

                word_diff = 0 if os.environ.get("LOOPWEAVER_DISABLE_REFINE") == "1" else abs(required_length - current_length)

                while word_diff > (required_length * 0.1):
                    refinement_prompt = f"""You are an expert editor. The provided text need to be {"shorten" if current_length > required_length else "lengthen"} by {word_diff} words while maintaining the original meaning and coherence.
                    Text:
                    {week['diary_entry']}
                    Return only the refined text."""

                    logging.info(f"Refining text for week {week_id}")
                    async with semaphore:
                        refined_text = await async_call_llm(model, refinement_prompt)

                    if refined_text and len(refined_text) > 10:
                        week['diary_entry'] = refined_text.strip()
                        logging.info(f"Refined text: {week['diary_entry'][:50]}...")
                    else:
                        break

                    current_length = count_words(week['diary_entry'])
                    word_diff = 0 if os.environ.get("LOOPWEAVER_DISABLE_REFINE") == "1" else abs(required_length - current_length)

                    if word_diff <= (required_length * 0.1):
                        break
            except Exception as e:
                logging.error(f"Error during refinement for {week_id}: {e}")

            logging.info(f"Final diary entry word count: {count_words(week['diary_entry'])}")
            return week

        tasks = [process_week(week) for week in example['weekly_plan']]

        try:
            processed_results = await asyncio.gather(*tasks)
            example['weekly_plan'] = processed_results
        except Exception as e:
            logging.error(f"Error during concurrent processing: {e}")
            processed_results = []
            for week in example['weekly_plan']:
                try:
                    if 'diary_entry' not in week:
                        week['diary_entry'] = f"Weekly diary for {week.get('week_id', 'Unknown Week')}."
                        week['length_requirement'] = 200
                    processed_results.append(week)
                except Exception as inner_e:
                    logging.error(f"Error processing individual week: {inner_e}")
                    processed_results.append({
                        "week_id": "Unknown Week",
                        "diary_entry": "Weekly diary entry placeholder.",
                        "length_requirement": 200
                    })

        example['final_text'] = GenerationAgent.get_final_week_text(example['weekly_plan'])
        return example

    @staticmethod
    def get_final_week_text(weekly_plan):
        """
        Get the final text from the generated text with respect to the plan.
        """
        text = ""

        for week in weekly_plan:
            if isinstance(week, dict):
                week_id = str(week.get('week_id', "Unknown Week"))
                diary_entry = week.get('diary_entry', "")
                if not isinstance(diary_entry, str):
                    diary_entry = str(diary_entry)
                text += '#*# ' + week_id + ':' + diary_entry
            else:
                text += '#*# Unknown Week:' + str(week)

        text += '*** finished ***'
        return text

    @staticmethod
    async def async_generate_floor(model, example, semaphore):
        if 'floor_plan' not in example or not example['floor_plan']:
            example['floor_plan'] = []
        elif not isinstance(example['floor_plan'], list):
            if isinstance(example['floor_plan'], str):
                example['floor_plan'] = [{"floor_id": "Floor 1", "purpose": example['floor_plan']}]
            elif isinstance(example['floor_plan'], dict):
                example['floor_plan'] = [example['floor_plan']]
            else:
                example['floor_plan'] = []

        processed_floors = []

        for i, floor in enumerate(example['floor_plan']):
            if not isinstance(floor, dict):
                processed_floor = {"floor_id": f"Floor {i+1}", "purpose": str(floor)}
            else:
                processed_floor = copy.deepcopy(floor)

                if 'floor_id' not in processed_floor:
                    processed_floor['floor_id'] = f"Floor {i+1}"
                if 'purpose' not in processed_floor:
                    processed_floor['purpose'] = ""

            processed_floors.append(processed_floor)

        example['floor_plan'] = processed_floors

        async def process_floor(floor):
            floor_id = floor.get('floor_id', "Unknown Floor")
            purpose = floor.get('purpose', "")

            if not isinstance(purpose, str):
                if isinstance(purpose, list):
                    purpose = ", ".join([str(p) for p in purpose])
                else:
                    purpose = str(purpose)

            prompt = f"""You are an expert disigner.
    Write a 150-word skyscraper floor plan for the floor of {floor_id}.
    The purpose for this floor is: {purpose}
    You should consider the coherence of the floor plan by referring to the plan of the whole skyscraper:
    {example['floor_plan']}

    You should consider the user requirements: {example['prompt']}
    Check from the user requirements if there are any special requirement that should be included in the floor plan. If there are, include them in the floor plan. If there are no special events, write a general floor plan.
    Return the floor plan for the floor of {floor_id} in the following json format:
    {{
        "floor_id": "{floor_id}",
        "check": "reason and check if the user requirements are met",
        "plan": "Your 150-word floor plan here"
    }}
    """
            logging.info(f"Generating initial floor plan for floor {floor_id}")

            max_attempts = 3
            for attempt in range(max_attempts):
                try:
                    async with semaphore:
                        response = await async_call_llm(model, prompt)

                    response = repair_json(response)

                    match = re.search(r"\{.*\}", response, re.DOTALL)
                    if match:
                        json_str = match.group(0)

                        try:
                            floor_plan = json.loads(json_str)

                            if 'plan' in floor_plan:
                                floor['plan'] = floor_plan['plan']
                                if not isinstance(floor['plan'], str):
                                    floor['plan'] = str(floor['plan'])
                            elif 'floor_plan' in floor_plan:
                                floor['plan'] = floor_plan['floor_plan']
                                if not isinstance(floor['plan'], str):
                                    floor['plan'] = str(floor['plan'])
                            elif 'design' in floor_plan:
                                floor['plan'] = floor_plan['design']
                                if not isinstance(floor['plan'], str):
                                    floor['plan'] = str(floor['plan'])
                            else:
                                content = re.sub(r'[\{\}"\'floor_id[^:]*:|check[^:]*:|\s*]', '', json_str)
                                floor['plan'] = content.strip()
                                if not isinstance(floor['plan'], str):
                                    floor['plan'] = str(floor['plan'])
                        except json.JSONDecodeError as e:
                            logging.error(f"JSON decode error for {floor_id}: {e}, attempt {attempt+1}/{max_attempts}")
                            if attempt == max_attempts - 1:
                                floor['plan'] = f"Skyscraper floor plan for {floor_id} with purpose: {purpose}"
                                floor['length_requirement'] = 150
                    else:
                        logging.warning(f"No JSON found in response for {floor_id}, attempt {attempt+1}/{max_attempts}")
                        if attempt == max_attempts - 1:
                            floor['plan'] = response.strip() or f"Skyscraper floor plan for {floor_id} with purpose: {purpose}"
                            floor['length_requirement'] = 150
                except Exception as e:
                    logging.error(f"Error processing floor for {floor_id}: {e}, attempt {attempt+1}/{max_attempts}")
                    if attempt == max_attempts - 1:
                        floor['plan'] = f"Skyscraper floor plan for {floor_id} with purpose: {purpose}"
                        floor['length_requirement'] = 150

            if 'plan' not in floor:
                floor['plan'] = f"Skyscraper floor plan for {floor_id} with purpose: {purpose}"
                floor['length_requirement'] = 150
                logging.warning(f"Created fallback floor plan for {floor_id}")

            try:
                current_length = count_words(floor['plan'])
                required_length = floor['length_requirement']

                word_diff = 0 if os.environ.get("LOOPWEAVER_DISABLE_REFINE") == "1" else abs(required_length - current_length)

                if word_diff > (required_length * 0.1):
                    refinement_prompt = f"""You are an expert editor. The provided text need to be {"shorten" if current_length > required_length else "lengthen"} by {word_diff} words while maintaining the original meaning and coherence.
                    Text:
                    {floor['plan']}
                    Return only the refined text."""

                    logging.info(f"Refining text for floor {floor_id}")
                    async with semaphore:
                        refined_text = await async_call_llm(model, refinement_prompt)

                    if refined_text and len(refined_text) > 10:
                        floor['plan'] = refined_text.strip()
                        logging.info(f"Refined text: {floor['plan'][:50]}...")

                    current_length = count_words(floor['plan'])
                    logging.info(f"Final floor plan word count: {current_length}")
            except Exception as e:
                logging.error(f"Error during refinement for {floor_id}: {e}")

            return floor

        tasks = [process_floor(floor) for floor in example['floor_plan']]

        try:
            processed_results = await asyncio.gather(*tasks)
            example['floor_plan'] = processed_results
        except Exception as e:
            logging.error(f"Error during concurrent floor processing: {e}")
            processed_results = []
            for floor in example['floor_plan']:
                try:
                    if 'plan' not in floor:
                        floor['plan'] = f"Floor plan for {floor.get('floor_id', 'Unknown Floor')}"
                        floor['length_requirement'] = 150
                    processed_results.append(floor)
                except Exception as inner_e:
                    logging.error(f"Error processing individual floor: {inner_e}")
                    processed_results.append({
                        "floor_id": "Unknown Floor",
                        "plan": "Skyscraper floor plan placeholder.",
                        "length_requirement": 150
                    })

            example['floor_plan'] = processed_results

        example['final_text'] = GenerationAgent.get_final_floor_text(example['floor_plan'])

        return example


    @staticmethod
    def get_final_floor_text(floor_plan):
        """
        Get the final text from the generated text with respect to the plan.
        """
        text = ""

        for floor in floor_plan:
            if isinstance(floor, dict):
                floor_id = str(floor.get('floor_id', "Unknown Floor"))
                plan_text = floor.get('plan', "")
                if not isinstance(plan_text, str):
                    plan_text = str(plan_text)
                text += '#*# ' + floor_id + ':' + plan_text
            else:
                text += '#*# Unknown Floor:' + str(floor)

        text += '*** finished'
        return text

    @staticmethod
    async def async_generate_menu(model, example, semaphore):
        if 'weekly_plan' not in example or not example['weekly_plan']:
            example['weekly_plan'] = []
        elif not isinstance(example['weekly_plan'], list):
            if isinstance(example['weekly_plan'], str):
                example['weekly_plan'] = [{"week_id": "Week 1", "dishes": example['weekly_plan']}]
            elif isinstance(example['weekly_plan'], dict):
                example['weekly_plan'] = [example['weekly_plan']]
            else:
                example['weekly_plan'] = []

        processed_weeks = []

        for i, week in enumerate(example['weekly_plan']):
            if not isinstance(week, dict):
                processed_week = {"week_id": f"Menu Week {i+1}", "dishes": str(week)}
            else:
                processed_week = copy.deepcopy(week)

                if 'week_id' not in processed_week:
                    processed_week['week_id'] = f"Menu Week {i+1}"
                if 'dishes' not in processed_week:
                    processed_week['dishes'] = ""

            processed_weeks.append(processed_week)

        example['weekly_plan'] = processed_weeks

        async def process_menu(week):
            week_id = week.get('week_id', "Unknown Menu Week")
            dishes = week.get('dishes', [])

            dishes_str = ""
            if isinstance(dishes, list):
                dishes_str = ", ".join([str(dish) for dish in dishes])
            else:
                dishes_str = str(dishes)

            prompt = f"""You are an expert chef.
    Write a 200-word weekly menu plan for the week of {week_id}.
    The dishes for this week are: {dishes_str}
    You should consider the coherence of the menu plan referring to the plan of the whole year:
    {example['weekly_plan']}
    You should consider the user requirements: {example['prompt']}
    Check from the user requirements if there are any special dishes that should be included in the menu plan. If there are, include them in the menu plan. If there are no special dishes, write a general menu plan for the week.
    Return the menu plan in the following json format:
    {{
        "week_id": "{week_id}",
        "check": "reason and check if the user requirements are met",
        "week_menu": "Your 200-word menu plan here"
    }}
    """
            logging.info(f"Generating initial menu plan for week {week_id}")

            max_attempts = 3
            for attempt in range(max_attempts):
                try:
                    async with semaphore:
                        response = await async_call_llm(model, prompt)

                    response = repair_json(response)

                    match = re.search(r"\{.*\}", response, re.DOTALL)
                    if match:
                        json_str = match.group(0)

                        try:
                            week_menu_json = json.loads(json_str)

                            if 'week_menu' in week_menu_json:
                                week['week_menu'] = week_menu_json['week_menu']
                                if not isinstance(week['week_menu'], str):
                                    week['week_menu'] = str(week['week_menu'])
                            else:
                                content = re.sub(r'[\{\}"\'week_id[^:]*:|check[^:]*:|\s*]', '', json_str)
                                week['week_menu'] = content.strip()
                                if not isinstance(week['week_menu'], str):
                                    week['week_menu'] = str(week['week_menu'])

                            week['length_requirement'] = 200
                            logging.info(f"Menu plan word count: {count_words(week['week_menu'])}")
                            break
                        except json.JSONDecodeError as e:
                            logging.error(f"JSON decode error for {week_id}: {e}, attempt {attempt+1}/{max_attempts}")
                            if attempt == max_attempts - 1:
                                week['week_menu'] = f"Weekly menu for {week_id} featuring {dishes_str}"
                                week['length_requirement'] = 200
                    else:
                        logging.warning(f"No JSON found in response for {week_id}, attempt {attempt+1}/{max_attempts}")
                        if attempt == max_attempts - 1:
                            week['week_menu'] = response.strip() or f"Weekly menu for {week_id} featuring {dishes_str}"
                            week['length_requirement'] = 200
                except Exception as e:
                    logging.error(f"Error processing menu for {week_id}: {e}, attempt {attempt+1}/{max_attempts}")
                    if attempt == max_attempts - 1:
                        week['week_menu'] = f"Weekly menu for {week_id} featuring {dishes_str}"
                        week['length_requirement'] = 200

            if 'week_menu' not in week:
                week['week_menu'] = f"Weekly menu for {week_id} featuring {dishes_str}"
                week['length_requirement'] = 200

            return week

        tasks = [process_menu(week) for week in example['weekly_plan']]

        try:
            processed_results = await asyncio.gather(*tasks)
            example['weekly_plan'] = processed_results
        except Exception as e:
            logging.error(f"Error during concurrent menu processing: {e}")
            processed_results = []
            for week in example['weekly_plan']:
                try:
                    if 'week_menu' not in week:
                        week['week_menu'] = f"Weekly menu for {week.get('week_id', 'Unknown Week')}"
                        week['length_requirement'] = 200
                    processed_results.append(week)
                except Exception as inner_e:
                    logging.error(f"Error processing individual menu week: {inner_e}")
                    processed_results.append({
                        "week_id": "Unknown Menu Week",
                        "week_menu": "Weekly menu placeholder.",
                        "length_requirement": 200
                    })

            example['weekly_plan'] = processed_results

        example['final_text'] = GenerationAgent.get_final_menu_text(example['weekly_plan'])
        return example

    @staticmethod
    def get_final_menu_text(weekly_plan):
        """
        Get the final text from the generated text with respect to the plan.
        """
        text = ""

        for week in weekly_plan:
            if isinstance(week, dict):
                week_id = str(week.get('week_id', "Unknown Menu Week"))
                menu = week.get('week_menu', "")
                if not isinstance(menu, str):
                    menu = str(menu)
                text += '#*# ' + week_id + ':' + menu
            else:
                text += '#*# Unknown Menu Week:' + str(week)

        text += '*** finished ***'
        return text

    @staticmethod
    async def async_generate_block(model, example, semaphore):
        if 'block_plan' not in example or not example['block_plan']:
            example['block_plan'] = []
        elif not isinstance(example['block_plan'], list):
            if isinstance(example['block_plan'], str):
                example['block_plan'] = [{"block_id": "Block 1", "use": example['block_plan']}]
            elif isinstance(example['block_plan'], dict):
                example['block_plan'] = [example['block_plan']]
            else:
                example['block_plan'] = []

        processed_blocks = []

        for i, block in enumerate(example['block_plan']):
            if not isinstance(block, dict):
                processed_block = {"block_id": f"Block {i+1}", "use": str(block)}
            else:
                processed_block = copy.deepcopy(block)

                if 'block_id' not in processed_block:
                    processed_block['block_id'] = f"Block {i+1}"

                if 'use' not in processed_block:
                    if any(processed_block.values()):
                        for key, value in processed_block.items():
                            if value and key != 'block_id':
                                processed_block['use'] = f"{key}: {value}"
                                break

                    if 'use' not in processed_block:
                        processed_block['use'] = ""

            processed_blocks.append(processed_block)

        example['block_plan'] = processed_blocks

        async def process_block(block):
            block_id = block.get('block_id', "Unknown Block")
            use = block.get('use', "")

            if not isinstance(use, str):
                if isinstance(use, list):
                    use = ", ".join([str(item) for item in use])
                else:
                    use = str(use)

            prompt = f"""You are an expert disigner.
    Write a 150-word city block plan for the block of {block_id}.
    The use for this block is: {use}
    You should consider the coherence of the block plan by referring to the plan of the whole city:
    {example['block_plan']}

    You should consider the user requirements: {example['prompt']}
    Check from the user requirements if there are any special requirement that should be included in the block plan. If there are, include them in the block plan. If there are no special events, write a general block plan.
    Return the block plan for the block of {block_id} in the following json format:
    {{
        "block_id": "{block_id}",
        "check": "reason and check if the user requirements are met",
        "plan": "Your 150-word block plan here"
    }}
    """
            logging.info(f"Generating initial block plan for block {block_id}")

            max_attempts = 3
            for attempt in range(max_attempts):
                try:
                    async with semaphore:
                        response = await async_call_llm(model, prompt)

                    response = repair_json(response)

                    match = re.search(r"\{.*\}", response, re.DOTALL)
                    if match:
                        json_str = match.group(0)

                        try:
                            block_plan = json.loads(json_str)

                            if 'plan' in block_plan:
                                block['plan'] = block_plan['plan']
                                if not isinstance(block['plan'], str):
                                    block['plan'] = str(block['plan'])
                            elif 'design' in block_plan:
                                block['plan'] = block_plan['design']
                                if not isinstance(block['plan'], str):
                                    block['plan'] = str(block['plan'])
                            elif 'block_plan' in block_plan:
                                block['plan'] = block_plan['block_plan']
                                if not isinstance(block['plan'], str):
                                    block['plan'] = str(block['plan'])
                            else:
                                content = re.sub(r'[\{\}"\'block_id[^:]*:|check[^:]*:|\s*]', '', json_str)
                                block['plan'] = content.strip()
                                if not isinstance(block['plan'], str):
                                    block['plan'] = str(block['plan'])

                            block['length_requirement'] = 150
                            logging.info(f"Block plan word count: {count_words(block['plan'])}")
                            break
                        except json.JSONDecodeError as e:
                            logging.error(f"JSON decode error for {block_id}: {e}, attempt {attempt+1}/{max_attempts}")
                            if attempt == max_attempts - 1:
                                block['plan'] = f"City block plan for {block_id} with use: {use}"
                                block['length_requirement'] = 150
                    else:
                        logging.warning(f"No JSON found in response for {block_id}, attempt {attempt+1}/{max_attempts}")
                        if attempt == max_attempts - 1:
                            block['plan'] = response.strip() or f"City block plan for {block_id} with use: {use}"
                            block['length_requirement'] = 150
                except Exception as e:
                    logging.error(f"Error processing block plan for {block_id}: {e}, attempt {attempt+1}/{max_attempts}")
                    if attempt == max_attempts - 1:
                        block['plan'] = f"City block plan for {block_id} with use: {use}"
                        block['length_requirement'] = 150

            if 'plan' not in block:
                block['plan'] = f"City block plan for {block_id} with use: {use}"
                block['length_requirement'] = 150
                logging.warning(f"Created fallback block plan for {block_id}")

            try:
                current_length = count_words(block['plan'])
                required_length = block['length_requirement']

                word_diff = 0 if os.environ.get("LOOPWEAVER_DISABLE_REFINE") == "1" else abs(required_length - current_length)

                if word_diff > (required_length * 0.1):
                    refinement_prompt = f"""You are an expert editor. The provided text need to be {"shorten" if current_length > required_length else "lengthen"} by {word_diff} words while maintaining the original meaning and coherence.
                    Text:
                    {block['plan']}
                    Return only the refined text."""

                    logging.info(f"Refining text for block {block_id}")
                    async with semaphore:
                        refined_text = await async_call_llm(model, refinement_prompt)

                    if refined_text and len(refined_text) > 10:
                        block['plan'] = refined_text.strip()
                        logging.info(f"Refined text: {block['plan'][:50]}...")

                    current_length = count_words(block['plan'])
                    logging.info(f"Final block plan word count: {current_length}")
            except Exception as e:
                logging.error(f"Error during refinement for {block_id}: {e}")

            return block

        tasks = [process_block(block) for block in example['block_plan']]

        try:
            processed_results = await asyncio.gather(*tasks)
            example['block_plan'] = processed_results
        except Exception as e:
            logging.error(f"Error during concurrent block processing: {e}")
            processed_results = []
            for block in example['block_plan']:
                try:
                    if 'plan' not in block:
                        block['plan'] = f"Block plan for {block.get('block_id', 'Unknown Block')}"
                        block['length_requirement'] = 150
                    processed_results.append(block)
                except Exception as inner_e:
                    logging.error(f"Error processing individual block: {inner_e}")
                    processed_results.append({
                        "block_id": "Unknown Block",
                        "plan": "City block plan placeholder.",
                        "length_requirement": 150
                    })

            example['block_plan'] = processed_results

        example['final_text'] = GenerationAgent.get_final_block_text(example['block_plan'])

        return example

    @staticmethod
    def get_final_block_text(block_plan):
        """
        Get the final text from the generated text with respect to the plan.
        """
        text = ""

        for block in block_plan:
            if isinstance(block, dict):
                block_id = str(block.get('block_id', "Unknown Block"))
                plan_text = block.get('plan', "")
                if not isinstance(plan_text, str):
                    plan_text = str(plan_text)
                text += '#*# ' + block_id + ':' + plan_text
            else:
                text += '#*# Unknown Block:' + str(block)

        text += '*** finished'
        return text
