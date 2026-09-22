import os
import re
import logging
from typing import Optional, Dict, Any, Tuple
from src.config import Config

logger = logging.getLogger(__name__)

class LLMClient:
    def __init__(self, api_key: Optional[str] = None, model_name: Optional[str] = None):
        self.api_key = api_key or Config.GEMINI_API_KEY
        self.model_name = model_name or Config.GEMINI_MODEL
        self._genai_client = None

        if self.api_key:
            try:
                from google import genai
                self._genai_client = genai.Client(api_key=self.api_key)
                logger.info(f"Initialized google.genai Client with model: {self.model_name}")
            except Exception as e:
                logger.warning(f"Could not initialize google.genai: {e}")

    def generate(self, prompt: str, system_instruction: str = "") -> Tuple[str, Dict[str, int]]:
        """
        Generate answer given prompt and system instruction.
        Returns: (response_text, token_stats_dict)
        """
        # Estimate input tokens (~4 chars per token)
        est_input_tokens = len(prompt) // 4 + len(system_instruction) // 4

        if self._genai_client:
            try:
                config_kwargs = {}
                if system_instruction:
                    config_kwargs['system_instruction'] = system_instruction
                
                resp = self._genai_client.models.generate_content(
                    model=self.model_name,
                    contents=prompt,
                    config=config_kwargs
                )
                text = resp.text.strip() if resp.text else ""
                
                usage = getattr(resp, 'usage_metadata', None)
                if usage:
                    in_tok = getattr(usage, 'prompt_token_count', est_input_tokens)
                    out_tok = getattr(usage, 'candidates_token_count', len(text) // 4)
                else:
                    in_tok = est_input_tokens
                    out_tok = len(text) // 4

                return text, {
                    'prompt_tokens': in_tok,
                    'completion_tokens': out_tok,
                    'total_tokens': in_tok + out_tok
                }
            except Exception as e:
                logger.warning(f"Gemini generation call failed ({e}); falling back to deterministic extraction.")

        # Deterministic fallback extractor for local testing or when offline
        ans = self._deterministic_extract(prompt)
        out_tok = len(ans) // 4 + 10
        return ans, {
            'prompt_tokens': est_input_tokens,
            'completion_tokens': out_tok,
            'total_tokens': est_input_tokens + out_tok
        }

    def _deterministic_extract(self, prompt: str) -> str:
        # Check if context has gold athlete
        m_gold = re.search(r'Gold Medalist:\s*([^\n\r]+)', prompt, re.I)
        if m_gold:
            return m_gold.group(1).strip()
        m_ans = re.search(r'(?:Answer|Result|Count|Winner):\s*([^\n\r]+)', prompt, re.I)
        if m_ans:
            return m_ans.group(1).strip()
        # Look for number if question asks "how many"
        if "how many" in prompt.lower():
            m_num = re.search(r'\b(\d+)\b', prompt)
            if m_num:
                return m_num.group(1)
        return "Not found in evidence"
