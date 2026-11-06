import logging
import numpy as np
import re
from rouge import Rouge
from difflib import SequenceMatcher

class RolloutRewardAgent:
    """Compute rollout rewards for sub-plans that passed filtering, supporting multiple reward
    evaluation methods. Adapted to the planning_results.json structure, using
    checks_once/checks_range/checks_periodic for evaluation."""
    @staticmethod
    def _rougel_score(prediction, ground_truth):
        """Compute the ROUGE-L score."""
        rouge = Rouge()
        try:
            scores = rouge.get_scores(prediction, ground_truth, avg=True)
        except ValueError:
            return 0.0
        return scores["rouge-l"]["f"]

    @staticmethod
    def _acc_score(prediction, ground_truth):
        """Compute the containment accuracy."""
        if not prediction or not ground_truth:
            return 0.0

        if isinstance(prediction, list):
            prediction = " ".join(prediction)

        if ground_truth in prediction or ground_truth.lower() in prediction or ground_truth.capitalize() in prediction:
            return 1.0
        else:
            return 0.0

    @staticmethod
    def _soft_match(text, pattern, threshold=0.7):
        """Use similarity rather than exact match."""
        if not text or not pattern:
            return 0.0

        if isinstance(text, list):
            text = " ".join(str(t) for t in text)

        text = str(text).lower().strip()
        pattern = str(pattern).lower().strip()

        if pattern in text:
            return 1.0

        similarity = SequenceMatcher(None, text, pattern).ratio()

        return similarity if similarity >= threshold else 0.0

    @staticmethod
    def _extract_keywords(text):
        """Extract keywords from text."""
        if not text:
            return []

        words = str(text).lower().split()
        keywords = [w.strip('.,;:?!()[]{}') for w in words if len(w) > 3]
        return list(set(keywords))

    @staticmethod
    def _extract_content(step):
        """Extract the content text from a step."""
        if isinstance(step, str):
            return step

        content = ""
        if "content" in step:
            content = step["content"]
        elif "events" in step:
            events = step["events"]
            if isinstance(events, list):
                if all(isinstance(e, dict) for e in events):
                    content = " ".join([
                        f"{e.get('event_name', '')} {e.get('event', '')} {e.get('details', '')}"
                        for e in events if isinstance(e, dict)
                    ])
                else:
                    content = " ".join(str(e) for e in events)
            else:
                content = str(events)
        elif "dishes" in step:
            dishes = step["dishes"]
            if isinstance(dishes, list):
                content = " ".join(str(d) for d in dishes)
            else:
                content = str(dishes)
        elif "purpose" in step:
            content = step["purpose"]
        elif "use" in step:
            content = step["use"]

        for id_field in ["week_id", "floor_id", "block_id"]:
            if id_field in step:
                content = f"{step[id_field]} {content}"

        return content

    @staticmethod
    def _calculate_relevance(content, prompt):
        """Compute the relevance between content and prompt."""
        if not content or not prompt:
            return 0.0

        prompt_keywords = RolloutRewardAgent._extract_keywords(prompt)
        content = content.lower()

        if not prompt_keywords:
            return 0.0

        matches = sum(1 for keyword in prompt_keywords if keyword in content)
        return min(1.0, matches / max(len(prompt_keywords) / 3, 1))

    @staticmethod
    def _check_date_match(date_str, target_date):
        """Check whether a date matches the target date."""
        date_match_target = re.search(r'(\d{1,2})[-/](\d{1,2})', target_date)
        date_match = re.search(r'(\d{1,2})[-/](\d{1,2})', date_str)

        if not date_match or not date_match_target:
            return False

        target_month, target_day = int(date_match_target.group(1)), int(date_match_target.group(2))
        month, day = int(date_match.group(1)), int(date_match.group(2))

        return month == target_month and day == target_day

    @staticmethod
    def _check_week_match(week_str, target_week):
        """Check whether a week matches the target week."""
        week_range_match = re.search(r'Week\s+(\d+)[-\s]+(\d+)', week_str)
        if week_range_match:
            start_week = int(week_range_match.group(1))
            end_week = int(week_range_match.group(2))
            return start_week <= target_week <= end_week

        week_match = re.search(r'Week\s+(\d+)', week_str)
        if week_match:
            week_num = int(week_match.group(1))
            return week_num == target_week

        return False

    @staticmethod
    def _get_floor_reward(floor, example):
        """Scoring logic for floor_plan."""
        reward = 0.0
        floor_id = floor.get("floor_id", "")
        purpose = floor.get("purpose", "")

        if not floor_id or not purpose:
            return 0.0

        floor_num_match = re.search(r'Floor\s+(\d+)', floor_id)
        if not floor_num_match:
            return 0.0

        floor_num = int(floor_num_match.group(1))

        for check_floor, expected_use in example.get("checks_once", {}).items():
            if str(floor_num) == str(check_floor):
                match_score = RolloutRewardAgent._soft_match(purpose, expected_use)
                if match_score > 0:
                    reward += match_score
                    break

        for range_str, expected_use in example.get("checks_range", {}).items():
            try:
                check_floor = int(range_str)
                if check_floor == floor_num:
                    match_score = RolloutRewardAgent._soft_match(purpose, expected_use)
                    if match_score > 0:
                        reward += match_score
                        break
            except ValueError:
                range_match = re.search(r'(\d+)[-](\d+)', range_str)
                if range_match:
                    start_floor = int(range_match.group(1))
                    end_floor = int(range_match.group(2))
                    if start_floor <= floor_num <= end_floor:
                        match_score = RolloutRewardAgent._soft_match(purpose, expected_use)
                        if match_score > 0:
                            reward += match_score
                            break

        for period_str, expected_use in example.get("checks_periodic", {}).items():
            try:
                check_floor = int(period_str)
                if check_floor == floor_num:
                    match_score = RolloutRewardAgent._soft_match(purpose, expected_use)
                    if match_score > 0:
                        reward += match_score
                        break
            except ValueError:
                pass

        prompt = example.get("prompt", "")
        if prompt and purpose:
            relevance_score = RolloutRewardAgent._calculate_relevance(purpose, prompt)
            reward += relevance_score * 0.5

        return min(reward, 1.0)

    @staticmethod
    def _get_block_reward(block, example):
        """Scoring logic for block_plan."""
        reward = 0.0
        block_id = block.get("block_id", "")
        use = block.get("use", "")

        if not block_id or not use:
            return 0.0

        coords_match = re.search(r'\((\d+),\s*(\d+)\)', block_id)
        if coords_match:
            x = int(coords_match.group(1))
            y = int(coords_match.group(2))

            for check_block, expected_use in example.get("checks_once", {}).items():
                try:
                    block_num = int(check_block)
                    grid_size = 10
                    check_x = (block_num - 1) % grid_size
                    check_y = (block_num - 1) // grid_size

                    if x == check_x and y == check_y:
                        match_score = RolloutRewardAgent._soft_match(use, expected_use)
                        if match_score > 0:
                            reward += match_score
                            break
                except ValueError:
                    coords_check = re.search(r'\((\d+),\s*(\d+)\)', check_block)
                    if coords_check:
                        check_x = int(coords_check.group(1))
                        check_y = int(coords_check.group(2))
                        if x == check_x and y == check_y:
                            match_score = RolloutRewardAgent._soft_match(use, expected_use)
                            if match_score > 0:
                                reward += match_score
                                break

            for range_str, expected_use in example.get("checks_range", {}).items():
                try:
                    range_parts = range_str.split('-')
                    if len(range_parts) == 2:
                        start_block = int(range_parts[0])
                        end_block = int(range_parts[1])

                        block_num = y * grid_size + x + 1
                        if start_block <= block_num <= end_block:
                            match_score = RolloutRewardAgent._soft_match(use, expected_use)
                            if match_score > 0:
                                reward += match_score
                                break
                except ValueError:
                    pass

            for period_str, expected_use in example.get("checks_periodic", {}).items():
                try:
                    period_block = int(period_str)
                    block_num = y * grid_size + x + 1

                    if block_num == period_block:
                        match_score = RolloutRewardAgent._soft_match(use, expected_use)
                        if match_score > 0:
                            reward += match_score
                            break
                except ValueError:
                    pass

            prompt = example.get("prompt", "")
            if prompt and use:
                relevance_score = RolloutRewardAgent._calculate_relevance(use, prompt)
                reward += relevance_score * 0.5

        return min(reward, 1.0)

    @staticmethod
    def _get_weekly_menu_reward(menu_item, example):
        """Scoring logic for weekly_plan/menu."""
        reward = 0.0
        week_id = menu_item.get("week_id", "")

        dishes = menu_item.get("dishes", "")
        if isinstance(dishes, list):
            dishes = " ".join(str(dish) for dish in dishes)
        elif not isinstance(dishes, str):
            dishes = str(dishes)

        events = menu_item.get("events", "")
        if isinstance(events, list):
            events = " ".join(str(event) for event in events)
        elif not isinstance(events, str):
            events = str(events)

        content = f"{dishes} {events}".lower()

        week_num = None
        week_match = re.search(r'Week\s+(\d+)', week_id)
        if week_match:
            week_num = int(week_match.group(1))

        date_match = re.search(r'(\d{1,2})[-/](\d{1,2})', week_id)
        date_str = None
        if date_match:
            date_str = f"{date_match.group(1)}-{date_match.group(2)}"

        for check_week, expected in example.get("checks_once", {}).items():
            try:
                check_week_num = int(check_week)

                if week_num and week_num == check_week_num:
                    match_score = RolloutRewardAgent._soft_match(content, expected)
                    if match_score > 0:
                        reward += match_score
                        break

                expected_date_match = re.search(r'(\d{1,2})[-/](\d{1,2})', expected)
                if expected_date_match and date_str:
                    expected_date = f"{expected_date_match.group(1)}-{expected_date_match.group(2)}"
                    if date_str == expected_date:
                        keywords = RolloutRewardAgent._extract_keywords(expected)
                        if any(keyword in content for keyword in keywords):
                            reward += 0.8
                            break
            except ValueError:
                pass

        if week_num:
            for range_str, expected in example.get("checks_range", {}).items():
                try:
                    check_week = int(range_str)
                    if week_num == check_week:
                        match_score = RolloutRewardAgent._soft_match(content, expected)
                        if match_score > 0:
                            reward += match_score
                            break
                except ValueError:
                    range_match = re.search(r'(\d+)[-](\d+)', range_str)
                    if range_match:
                        start_week = int(range_match.group(1))
                        end_week = int(range_match.group(2))
                        if start_week <= week_num <= end_week:
                            match_score = RolloutRewardAgent._soft_match(content, expected)
                            if match_score > 0:
                                reward += match_score
                                break

        if week_num:
            for period_str, expected in example.get("checks_periodic", {}).items():
                try:
                    check_week = int(period_str)
                    if week_num == check_week:
                        match_score = RolloutRewardAgent._soft_match(content, expected)
                        if match_score > 0:
                            reward += match_score
                            break
                except ValueError:
                    pass

        prompt = example.get("prompt", "")
        if prompt and content:
            relevance_score = RolloutRewardAgent._calculate_relevance(content, prompt)
            reward += relevance_score * 0.5

        return min(reward, 1.0)

    @staticmethod
    def _get_weekly_diary_reward(week_item, example):
        """Scoring logic for weekly_plan/diary."""
        reward = 0.0
        week_id = week_item.get("week_id", "")

        events = week_item.get("events", "")
        if isinstance(events, list):
            if all(isinstance(e, dict) for e in events):
                events_text = " ".join([
                    f"{e.get('event_name', '')} {e.get('event', '')} {e.get('details', '')}"
                    for e in events if isinstance(e, dict)
                ])
            else:
                events_text = " ".join(str(e) for e in events)
        elif isinstance(events, dict):
            events_text = " ".join(f"{k}: {v}" for k, v in events.items())
        else:
            events_text = str(events)

        content = f"{week_id} {events_text}".lower()

        week_num = None
        week_match = re.search(r'Week\s+(\d+)', week_id)
        if week_match:
            week_num = int(week_match.group(1))

        date_range = re.findall(r'(\w+)\s+(\d+)', week_id)

        for check_week, expected in example.get("checks_once", {}).items():
            try:
                check_week_num = int(check_week)

                if week_num and week_num == check_week_num:
                    match_score = RolloutRewardAgent._soft_match(content, expected)
                    if match_score > 0:
                        reward += match_score
                        break

                birthday_match = re.search(r'birthday', expected.lower())
                if birthday_match and "birthday" in content:
                    family_members = ["husband", "wife", "child", "father", "mother"]
                    for member in family_members:
                        if member in expected.lower() and member in content:
                            reward += 0.8
                            break
            except ValueError:
                pass

        if week_num:
            for range_str, expected in example.get("checks_range", {}).items():
                try:
                    check_week = int(range_str)
                    if week_num == check_week:
                        keywords = RolloutRewardAgent._extract_keywords(expected)
                        if keywords and any(keyword in content for keyword in keywords):
                            reward += 0.7
                            break
                except ValueError:
                    range_match = re.search(r'(\d+)[-](\d+)', range_str)
                    if range_match:
                        start_week = int(range_match.group(1))
                        end_week = int(range_match.group(2))
                        if start_week <= week_num <= end_week:
                            keywords = RolloutRewardAgent._extract_keywords(expected)
                            if keywords and any(keyword in content for keyword in keywords):
                                reward += 0.7
                                break

        if week_num:
            for period_str, expected in example.get("checks_periodic", {}).items():
                try:
                    check_week = int(period_str)
                    if week_num == check_week:
                        keywords = RolloutRewardAgent._extract_keywords(expected)
                        keyword_matches = sum(1 for keyword in keywords if keyword in content)
                        if keyword_matches:
                            reward += min(0.8, keyword_matches / len(keywords) * 0.8)
                            break
                except ValueError:
                    pass

        prompt = example.get("prompt", "")
        if prompt and content:
            relevance_score = RolloutRewardAgent._calculate_relevance(content, prompt)
            reward += relevance_score * 0.5

        return min(reward, 1.0)

    @staticmethod
    def _get_reward(step, example, plan_key):
        """Automatically select the reward function by type."""
        if "floor_plan" in plan_key:
            return RolloutRewardAgent._get_floor_reward(step, example)
        elif "block_plan" in plan_key:
            return RolloutRewardAgent._get_block_reward(step, example)
        elif "weekly_plan" in plan_key and example.get("type", "").lower().startswith("menu"):
            return RolloutRewardAgent._get_weekly_menu_reward(step, example)
        elif "weekly_plan" in plan_key:
            return RolloutRewardAgent._get_weekly_diary_reward(step, example)

        gt = example.get("answer", "")
        text = RolloutRewardAgent._extract_content(step)
        if example.get("reward_type") == "rouge":
            return RolloutRewardAgent._rougel_score(text, gt)
        else:
            return RolloutRewardAgent._acc_score(text, gt)

    @staticmethod
    def _apply_fallback_rewards(plan, example):
        """Fallback reward computation when all rewards are zero."""
        rewards = []
        prompt = example.get('prompt', '')

        for step in plan:
            content = RolloutRewardAgent._extract_content(step)
            if not content:
                rewards.append(0.1)
                continue

            length_reward = min(len(content) / 500, 0.5)

            structure_reward = sum(0.1 for field in ['events', 'dishes', 'purpose', 'use', 'week_id', 'floor_id', 'block_id']
                                  if field in step and step[field])

            relevance_words = RolloutRewardAgent._extract_keywords(prompt)
            relevance_score = sum(1 for word in relevance_words
                                 if word in content.lower()) / max(len(relevance_words), 1)
            relevance_reward = relevance_score * 0.3

            total_reward = length_reward + structure_reward + relevance_reward
            normalized_reward = 0.1 + (total_reward * 0.8)
            rewards.append(min(0.9, normalized_reward))

        if len(rewards) > 1:
            min_r, max_r = min(rewards), max(rewards)
            if max_r - min_r < 0.2:
                rewards = [0.1 + (r - min_r) / (max_r - min_r + 1e-6) * 0.8 for r in rewards]

        return rewards

    @staticmethod
    def rollout_rewards(example):
        """Compute rollout rewards for valid sub-plans (synchronous version)."""
        if not (example.get("is_valid") or example.get("filter_response", "").upper() == "YES"):
            logging.info(f"Example {example.get('id', '')} skipped: not valid.")
            return example

        plan_key = None
        candidate_keys = [
            "weekly_plan", "floor_plan", "block_plan",
            "filtered_weekly_plan", "filtered_floor_plan", "filtered_block_plan"
        ]

        for key in candidate_keys:
            if key in example and isinstance(example[key], list) and example[key]:
                plan_key = key
                break

        if plan_key and plan_key.startswith("filtered_"):
            filtered_plan = example[plan_key]
            if isinstance(filtered_plan, dict) and "original_plan" in filtered_plan:
                plan = filtered_plan["original_plan"]
            else:
                plan = filtered_plan
        elif plan_key:
            plan = example[plan_key]
        else:
            logging.warning(f"Example {example.get('id', '')} has no valid plan to rollout.")
            return example

        if not isinstance(plan, list):
            logging.warning(f"Plan for {example.get('id', '')} is not a list.")
            return example

        rewards = []
        processed_plan = []

        for step in plan:
            if not isinstance(step, dict):
                if isinstance(step, str):
                    step = {"content": step}
                else:
                    try:
                        step = {"content": str(step)}
                    except:
                        continue

            step_dict = dict(step)

            reward = RolloutRewardAgent._get_reward(step_dict, example, plan_key)
            step_dict["reward"] = reward
            rewards.append(reward)
            processed_plan.append(step_dict)

        plan = processed_plan

        if rewards and all(r == 0 for r in rewards):
            logging.info(f"All rewards are zero for {example.get('id', '')}, applying fallback rewards.")
            fallback_rewards = RolloutRewardAgent._apply_fallback_rewards(plan, example)

            for i, r in enumerate(fallback_rewards):
                if i < len(plan):
                    plan[i]["reward"] = r
                    rewards[i] = r

        if not rewards:
            return example

        max_reward = max(rewards)
        min_reward = min(rewards)

        max_steps = [step for step in plan if step.get("reward") == max_reward]
        min_steps = [step for step in plan if step.get("reward") == min_reward]

        example["max_reward_steps"] = max_steps
        example["min_reward_steps"] = min_steps

        rewards_np = np.array(rewards, dtype=np.float32)
        mean = rewards_np.mean()
        std = rewards_np.std() if rewards_np.std() > 1e-6 else 1.0

        for step in plan:
            step["advantage"] = float((step.get("reward", 0) - mean) / std)

        if plan_key.startswith("filtered_"):
            if isinstance(example[plan_key], dict) and "original_plan" in example[plan_key]:
                example[plan_key]["original_plan"] = plan
            else:
                example[plan_key] = plan
        else:
            example[plan_key] = plan

        example["reward_stats"] = {
            "mean": float(mean),
            "std": float(std),
            "max": float(max_reward),
            "min": float(min_reward),
            "positive_rewards": sum(1 for r in rewards if r > 0),
            "total_steps": len(rewards),
            "using_fallback": all(r == 0 for r in rewards)
        }

        return example

    @staticmethod
    async def async_rollout_rewards(model, example, semaphore):
        """Asynchronous version of rollout_rewards, compatible with async calls."""
        return RolloutRewardAgent.rollout_rewards(example)
